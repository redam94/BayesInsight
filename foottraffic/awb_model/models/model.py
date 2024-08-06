from foottraffic.awb_model.models.dataloading import MFF
from foottraffic.awb_model.models.variablemodels import ExogVariableDetails, ControlVariableDetails, MediaVariableDetails
from foottraffic.awb_model.utils import var_dims
from foottraffic.awb_model.constants import MFFCOLUMNS

from pydantic import BaseModel, FilePath, Field, ConfigDict, DirectoryPath
from arviz import InferenceData
import arviz as az
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
from typing import List, Optional, Union, Annotated, Any, Literal
from pathlib import Path
import os

Variable = Annotated[
    Union[ControlVariableDetails, MediaVariableDetails, ExogVariableDetails],
    Field(discriminator="variable_type")]

class FoottrafficModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: MFF
    variable_details: List[Variable]
    artifact: Optional[DirectoryPath] = None 
    fitted: bool = False
    trace: Optional[InferenceData] = None
    VOF: Optional[pd.DataFrame] =None
    

    def fit(self, draws=1000, tune=1000, chains=4, overwrite=False):
        if not overwrite:
            if self.fitted:
                raise UserWarning("Model was already fitted! If you ment to call fit again set overwrite to True")
        model = self.build()
        with model:
            trace = pm.sample(draws, tune=tune, chains=chains)
        self.fitted = True
        self.trace = trace

    def build(self):
    
        data = self.data
        
        coords = self.get_coords()
        media_variables = self.return_media_variables()
        control_variables = self.return_control_variables()
        exog_variables = self.return_exog_variables()
        
        assert len(exog_variables) == 1, "Only one exog variable is supported"

        exog_variable = exog_variables[0]
        with pm.Model(coords=coords) as model:
            var_dim = var_dims(model)
            contributions = exog_variable.build_intercept_prior()
            shape_ = np.ones([len(model.coords[col]) for col in var_dim])
            if isinstance(contributions, int) or isinstance(contributions, float):
                
                contributions = pm.Deterministic('intercept_contribution', contributions*shape_, dims=var_dim)
            else:
                contributions = pm.Deterministic("intercept_contribution", contributions[..., None]*shape_, dims=var_dim)
            
            for var in media_variables:
                contributions_ = var.get_contributions(data)
                
                contributions = contributions + contributions_
            for var in control_variables:
                contributions_ = var.get_contributions(data)
                
                contributions = contributions + contributions_
            if exog_variable.likelihood.type == 'Normal':
                mu = pm.Deterministic('mu', contributions, dims=var_dim)
            else:
                mu = pm.Deterministic('mu', pm.math.exp(contributions), dims=var_dim)
            like = exog_variable.build_likelihood(mu, exog_variable.get_observation(data))
            

        return model
    
    def save(self, folder):
        
        if isinstance(folder, str):
            folder = Path(folder)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        self.artifact = folder
        self.data.to_bundle(self.artifact/'data')
        if self.fitted:
            self.trace.to_netcdf(self.artifact/'model.nc')

        with open(folder/'model_def.json', 'w') as f:
            f.write(self.model_dump_json(exclude=['data', 'trace']))

    def get_variable(self, varname):
        for var in self.variable_details:
            if var.variable_name == varname:
                return var
        raise ValueError(f"{varname} not in variable details")
    
    def return_media_variables(self) -> List[MediaVariableDetails]:
        media_vars = []
        for var in self.variable_details:
            if var.variable_type == 'media':
                media_vars.append(var)
        return media_vars
    
    def return_exog_variables(self) -> List[ExogVariableDetails]:
        exog_vars = []
        for var in self.variable_details:
            if var.variable_type == 'exog':
                exog_vars.append(var)
        return exog_vars
    
    def return_control_variables(self)-> List[ControlVariableDetails]:
        control_vars = []
        for var in self.variable_details:
            if var.variable_type == 'control':
                control_vars.append(var)
        return control_vars
    
    def get_coords(self):
        meta_data = self.data.metadata
        af = self.data.analytic_dataframe()
        row_ids = meta_data.row_ids.copy()
        coords = {
            col: af[col].unique() for col in row_ids
        }
        return coords
    
    def check_prior(self, varname):
        variable = self.get_variable(varname)
        coords = self.get_coords()
        with pm.Model(coords=coords) as model:
            coeff_prior = variable.build_coeff_prior()
        
        coeff_draws = pm.draw(coeff_prior, 4000).reshape((4, 1000, -1))
        axs = az.plot_trace(coeff_draws, figsize=(16, 9))
        axs[0][0].set_title(f"{coeff_prior.name}")
        axs[0][1].set_title(f"{coeff_prior.name}")
        return axs

    def check_media_transform_prior(self, varname):
        media_variable = self.get_variable(varname)
        coords = self.get_coords()
        with pm.Model(coords=coords) as model:
            media_prior = media_variable.build_media_priors()

        for i, var in enumerate(pm.draw(media_prior, 4000)):
            axs = az.plot_trace(var.reshape(4, 1000, -1), figsize=(16, 9))
            axs[0][0].set_title(f"{media_prior[i].name}")
            axs[0][1].set_title(f"{media_prior[i].name}")

    def _plot_posterior(self, varname):
        if not self.fitted:
            raise ValueError("Model has not been fitted")
        trace = self.trace
        
        if varname == 'intercept':
            return az.plot_trace(trace, var_names=['intercept'], figsize=(16, 5), kind='rank_bars')
            
        variable = self.get_variable(varname)

        if variable.variable_type == 'media':
            return az.plot_trace(trace, var_names=[f"{varname}_coeff_estimate", f"{varname}_K", f"{varname}_n"], figsize=(16, 15), kind='rank_bars')
           
        return az.plot_trace(trace, var_names=[f"{varname}_coeff_estimate"], figsize=(16, 5), kind='rank_bars')
    
    def plot_posterior(self, varname: Union[str, List[str]]):
        if isinstance(varname, str):
            return self._plot_posterior(varname)
        else:
            return [self._plot_posterior(var) for var in varname]
        
    def plot_posterior_curve(self, media_var):
        if not self.fitted:
            raise ValueError("Model has not been fitted")
        if media_var not in self.return_media_variables():
            raise ValueError(f"{media_var} is not a media variable")
        trace = self.trace
        plt.scatter(
            self.data.analytic_dataframe()[media_var],
            trace.posterior[f"{media_var}_contribution"].to_dataframe().groupby(self.data.metadata.row_ids).mean()
        )
    
    def plot_posterior_predictive(self, kind: Literal['kde', 'cumulative', 'scatter']='cumulative', coords=None):
        if not self.fitted:
            raise ValueError("Model has not been fitted")
        posterior = self.get_posterior_predictive()
        return az.plot_ppc(posterior, kind=kind, coords=coords)

    def get_contributions(self)->pd.DataFrame:
        row_ids = self.data.metadata.row_ids
        contributions = self.data.analytic_dataframe()[row_ids]

        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        media_variables = self.return_media_variables()
        control_variables = self.return_control_variables()
        
        var_names = ['intercept']

        for control_variable in control_variables:
            var_names.append(control_variable.variable_name)
        
        for media_variable in media_variables:
            var_names.append(media_variable.variable_name)
        
        for var in var_names:
            contributions = contributions.merge(self.get_var_con(var), on=row_ids)

        return contributions
        
    def get_posterior_predictive(self):
        with self.build():
            posterior = pm.sample_posterior_predictive(self.trace)
        return posterior
    
    def avm(self, agg='mean'):
        posterior = self.get_posterior_predictive()
        posterior_df = (posterior.posterior_predictive.to_dataframe().reset_index()
                .groupby(self.data.metadata.row_ids)
                .agg(agg).drop(columns=["chain", "draw"])
                .reset_index()
                )
        mu_df = (self.trace.posterior.mu.to_dataframe()
                .reset_index().groupby(self.data.metadata.row_ids)
                .agg(agg).drop(columns=["chain", "draw"])
                .reset_index())
        var_name = self.return_exog_variables()[0].variable_name
        af = self.data.analytic_dataframe()[self.data.metadata.row_ids + [var_name]]
        af = af.merge(posterior_df, on=self.data.metadata.row_ids)

        return af.merge(mu_df, on=self.data.metadata.row_ids)
        

    def get_var_con(self, varname:str, agg='mean'):
        trace = self.trace.posterior[f'{varname}_contribution']

        mean_trace = trace.to_dataframe().reset_index()
        groupby_cols = [col for col in mean_trace.columns if (col not in ['chain', 'draw']) and (col in MFFCOLUMNS)]

        df = mean_trace.groupby(groupby_cols).agg(agg).reset_index().drop(columns=['chain', 'draw'])
        return df


    
    @classmethod
    def load(cls, folder):
        if isinstance(folder, str):
            folder = Path(folder)
        file_list = os.listdir(folder)
        data = MFF.from_bundle(folder/'data')
        with open(folder/'model_def.json', 'r') as f:
            model_def = json.load(f)
        if "model.nc" in file_list:
            trace = az.from_netcdf(folder/'model.nc')
            return cls(data=data, variable_details=model_def['variable_details'], trace=trace, fitted=True)
        
        return cls(data=data, variable_details=model_def['variable_details'])

    @classmethod
    def new_from_dataset(cls, folder, output="new_model"):
        mff = MFF.from_bundle(folder)
        #print(mff.metadata.necessary_variables)
        cls(data=mff, variable_details=[ControlVariableDetails(variable_name="Placeholder Change Me"), MediaVariableDetails(variable_name="Media Placeholder Change Me")]).save(output)
