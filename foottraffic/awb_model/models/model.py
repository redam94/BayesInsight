from foottraffic.awb_model.models.dataloading import MFF
from foottraffic.awb_model.models.variablemodels import VariableDetails, ControlVariableDetails, MediaVariableDetails

from pydantic import BaseModel, FilePath, Field, ConfigDict, DirectoryPath
from arviz import InferenceData
import arviz as az
import pymc as pm
import numpy as np
import pandas as pd

import json
from typing import List, Optional, Union, Annotated, Any
from pathlib import Path
import os

Variable = Annotated[
    Union[ControlVariableDetails, MediaVariableDetails],
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
        
        with pm.Model(coords=coords) as model:
            x = pm.Normal('x', mu=3, tau=.1)
            pm.Normal('obs', mu=x, tau=10, observed=np.array([2, 2.1, 1.9, 1.98]))

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

    def check_media_transform_prior(self, varname):
        pass
    
    

    
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
