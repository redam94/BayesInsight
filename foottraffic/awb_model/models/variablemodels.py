from foottraffic.awb_model.types.variable_types import VarType
from foottraffic.awb_model.types.transform_types import MediaTransform, Adstock
from foottraffic.awb_model.models.transformsmodel import DeterministicTransform, Normilization, TimeTransformer
from foottraffic.awb_model.models.priormodel import MediaCoeffPrior, HillPrior, SShapedPrior, ControlCoeffPrior
from foottraffic.awb_model.models.likelihood import Likelihood
from foottraffic.awb_model.models.dataloading import MFF
from foottraffic.awb_model.constants import MFFCOLUMNS, TRANSFOMER_MAP

from typing import Optional, Union, Literal

from pydantic import BaseModel, PositiveFloat, model_validator
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

def _row_ids_to_ind_map(row_ids: list[str]) -> list[int]:
    ind_map = {
        row_ids[i]: i for i in range(len(row_ids))
    }
    return ind_map

class VariableDetails(BaseModel):
    """
    VariableDetails class for tracking variables in the model.

    Required Attributes:
    variable_name: Name of the variable in the analytic dataset
    variable_type: Defines how the variable behaves in the model

    Optional Attributes:
    deterministic_transform: Transform that with parameters known in advance
    normalization: Normalization applied after deterministic_transform
    time_transform: Not implemented
    std: Standard deviation assigned when normalize is called for first time
    mean: Mean assigned when normalize is called for first time
    """

    variable_name: str
    variable_type: Literal["control", "exog", 'media', 'base']
    deterministic_transform: DeterministicTransform = DeterministicTransform(functional_form='linear', params=None)
    normalization: Normilization = Normilization.none
    std: Optional[float] = None
    mean: Optional[float] = None
    time_transform: Optional[TimeTransformer] = None
    sign: Optional[Literal['positive', 'negative']] = None
    
    def normalize(self, data: Union[MFF, np.ndarray]) -> np.ndarray:
        """Apply normilazation to data"""
        if isinstance(data, MFF):
            data = self.as_numpy(data)

        if self.normalization == Normilization.none:
            return data
        
        if self.normalization == Normilization.global_standardize:
            var = data
            if self.mean is None:
                self.mean = np.mean(var)
            demeaned = (var - self.mean)
            if self.std is None:
                self.std = np.std(demeaned)
            standardized = demeaned/self.std
            return standardized
        
        raise NotImplementedError("Only global_standardize is implemented. :(")
    
    def transform(self, data: Union[MFF,np.ndarray], time_first=True, normalize_first=False) -> np.ndarray:
        """Apply deterministic transform to data"""
        
        variable = data
        if isinstance(variable, MFF):
            variable = self.as_numpy(variable)
        
        if normalize_first:
            variable = self.normalize(variable)
          
        if self.time_transform is None:
            if normalize_first:
                return self.deterministic_transform(variable)
            return self.normalize(self.deterministic_transform(variable))
        
        if time_first:
            transformed_variable = self.deterministic_transform(self.time_transform(variable))
            if normalize_first:
                return transformed_variable
            return self.normalize(transformed_variable)
        
        transformed_variable = self.time_transform(self.deterministic_transform(variable))
        if normalize_first:
            return transformed_variable
        return self.normalize(transformed_variable)
    
    def get_variable_values(self, data: MFF) -> pd.Series:
        analytic_dataframe = data.analytic_dataframe(indexed=True)
        try:
            return analytic_dataframe[self.variable_name]
        except KeyError:
            raise ValueError(f"{self.variable_name} not in AF. Check spelling.")
        
    def as_numpy(self, data: MFF) -> np.ndarray:
        row_dims = (data._info[index_col]["# Unique"] for index_col in data.metadata.row_ids)
        return self.get_variable_values(data).to_numpy().reshape(tuple(row_dims))
    
    def enforce_dim_order(self, dims: Union[list, tuple, None], drop_period: bool = True) -> list:
        if dims is None:
            return None
        ordered_dims = [dim for dim in MFFCOLUMNS if dim in dims and dim != "Period"]
        if 'Period' in dims and not drop_period:
            return ordered_dims + ["Period"]
        return ordered_dims
    
    def _check_coord(self, col: str, include_period:bool = False) -> bool:
        if col == 'Period':
            return include_period
        if "_" in col:
            return False
        return True
    
    def _check_dim(self, col: str, dims: Union[None, list, tuple]) -> bool:
        if dims is None:
            return self._check_coord(col)
        if col in dims:
            return False
        return self._check_coord(col)
    
    def _var_dims(self, model: Optional[pm.Model]=None) -> list:
        model = pm.modelcontext(model)
        var_dims = self.enforce_dim_order(list(model.coords.keys()), drop_period=False)
        return var_dims
    
    def build_coeff_prior(self, model: Optional[pm.Model]=None):
        """Build a prior for the coefficients of the control variable
           Grabs the model on the context stack if model is None."""
        
        model = pm.modelcontext(model)

        index_map = _row_ids_to_ind_map(self.enforce_dim_order(list(model.coords.keys())))
        
        model_dims = {col: len(model.coords[col]) for col in index_map.keys() if self._check_dim(col, None)}

        fixed_ind_coeff_dims = self.enforce_dim_order(self.fixed_ind_coeff_dims)
        fixed_dims_project = dict(
            repeats=tuple([model_dims[col] for col, index in index_map.items() if self._check_dim(col, None)]),
            axis=tuple([index for col, index in index_map.items() if self._check_dim(col, fixed_ind_coeff_dims)])
        )
        
        random_coeff_dims = self.enforce_dim_order(self.random_coeff_dims)
        random_dims_project = dict(
            repeats=tuple([model_dims[col] for col, index in index_map.items() if self._check_dim(col, None)]),
            axis=tuple([index for col, index in index_map.items() if self._check_dim(col, random_coeff_dims)])
        )

        with model:
            coeff_dist = getattr(pm, self.coeff_prior.coeff_dist) # Get coeff distribution
            coeff_params = self.coeff_prior.coef_params # Get coeff dist params
            ## Fixed Coefficients
            coeff_fixed = coeff_dist(f"{self.variable_name}_fixed_coeff", **coeff_params, dims=fixed_ind_coeff_dims)
            
            # Enforce Shape
            expanded_fixed = pt.expand_dims(coeff_fixed, axis=fixed_dims_project['axis'])
            repeats_fixed = np.ones(shape=fixed_dims_project['repeats'])
            if self.random_coeff_dims is None: 
                
                coeff_est = pm.Deterministic(
                    f'{self.variable_name}_coeff_estimate',
                    expanded_fixed*repeats_fixed, 
                    dims=tuple(model_dims.keys())
                    )
            
                return coeff_est
            
            sigma = pm.HalfCauchy(f'{self.variable_name}_rand_coeff_sigma', self.partial_pooling_sigma)
            #random_coeff_mu = coeff_dist(f"{self.variable_name}_random_coeff_mu", **coeff_params)
            random_fixed = pm.Normal(f"{self.variable_name}_rand_coeff", 0, 1, dims=self.random_coeff_dims)

            expanded_random = pt.expand_dims(random_fixed, axis=random_dims_project['axis'])
            repeats_random = np.ones(shape=random_dims_project['repeats'])
            coeff_est = pm.Deterministic(
                f'{self.variable_name}_coeff_estimate',
                expanded_fixed*repeats_fixed + (expanded_random*repeats_random*sigma), 
                dims=tuple(model_dims.keys())
            )
            return coeff_est 
    
    def enforce_sign(self, model=None):
        model = pm.modelcontext(model)
        pot = lambda constraint: (pm.math.log(pm.math.switch(constraint, 1, 1e-10)))
        with model:
            if self.sign is None:
                return model
            coeff_est = getattr(model, f"{self.variable_name}_coeff_estimate")
            if self.sign == 'positive':
                pm.Potential(f"{self.variable_name}_positive_sign", pot(coeff_est >= 0))
            if self.sign == 'negative':
                pm.Potential(f"{self.variable_name}_negative_sign", pot(coeff_est <= 0))
        return model
    
    def register_variable(self, data: MFF | np.ndarray, model=None):
        variable = data
        if isinstance(data, MFF):
            variable = self.as_numpy(variable)
        
        model = pm.modelcontext(model)
        with model:
            dims = self._var_dims()
            var = pm.Data(f"{self.variable_name}", variable, dims=dims)
            var = pm.Deterministic(f"{self.variable_name}_transformed", self.transform(var), dims=dims)
        return var
    
    def contributions(self, model=None):
        model = pm.modelcontext(model)
        with model:
            dims = self._var_dims()
            try:
                var = getattr(model, f"{self.variable_name}_transformed")
            except AttributeError:
                raise AttributeError("Variable must be register before it can be used")
            
            try:
                coef = getattr(model, f"{self.variable_name}_coeff_estimate")
            except AttributeError:
                coef = self.build_coeff_prior()

            contributions = pm.Deterministic(
                f"{self.variable_name}_contributions",
                coef[:, :, None]*var, dims=dims
                )
        return contributions
    

class ControlVariableDetails(VariableDetails):
    variable_type: Literal['control'] = 'control'
    coeff_prior: ControlCoeffPrior = ControlCoeffPrior()
    fixed_ind_coeff_dims: Optional[list[str]] = None
    random_coeff_dims: Optional[list[str]] = None
    partial_pooling_sigma: PositiveFloat = 1
    

    @model_validator(mode="after")
    def validate_effects(self):
        if self.random_coeff_dims is None or self.fixed_ind_coeff_dims is None:
            return self
        if set(self.fixed_ind_coeff_dims).intersection(set(self.random_coeff_dims)):
            raise ValueError("Fixed and Random Coefficients must be orthogonal.")
        return self

    def build_coeff_prior(self, model=None):
        model = pm.modelcontext(model)
        with model:
            estimate = super().build_coeff_prior()
            self.enforce_sign()
        return estimate
    
    def get_contributions(self, model=None):
        model = pm.modelcontext(model)
        with model:
            estimate = self.build_coeff_prior()
            dims = self._var_dims()
            contributions = pm.Deterministic(f"{self.variable_name}_contribution", estimate*getattr(model, f"{self.variable_name}_transformed"), dims=dims)
        return contributions                                 
    
            
class MediaVariableDetails(VariableDetails):
    variable_type: Literal['media']= 'media'
    time_transform: None = None
    adstock: Adstock = Adstock.delayed
    media_transform: MediaTransform = MediaTransform.hill
    coeff_prior: MediaCoeffPrior = MediaCoeffPrior()
    fixed_ind_coeff_dims: Optional[list[str]] = None
    random_coeff_dims: Optional[list[str]] = None
    media_transform_prior: Union[HillPrior, SShapedPrior] = HillPrior()
    sign: Literal['positive'] = 'positive'

    @model_validator(mode="after")
    def validate_transform_and_prior(self):
        if self.media_transform == MediaTransform.hill:
            assert isinstance(self.media_transform_prior, HillPrior)
        if self.media_transform == MediaTransform.sorigin or self.media_transform == MediaTransform.sshaped:
            assert isinstance(self.media_transform_prior, SShapedPrior)
        return self
    
    def build_coeff_prior(self, model=None):
        model = pm.modelcontext(model)
        with model:
            priors = super().build_coeff_prior()
            self.enforce_sign()
        return priors
    
    def build_hill_priors(self, model=None):
        model = pm.modelcontext(model)
        # Half saturation prior helper terms
        K_ave = self.media_transform_prior.K_ave
        K_std = self.media_transform_prior.K_std
        K_ratio = K_std/K_ave + 1
        # Shape prior helper terms
        n_ave = self.media_transform_prior.n_ave
        n_std = self.media_transform_prior.n_std
        n_ratio = n_std/n_ave + 1
        with model:

            K_ = pm.Normal(
                f"{self.variable_name}_K_", 
                np.log(K_ave), 
                np.log(K_ratio))
            n_ = pm.Normal(
                f"{self.variable_name}_n_",
                np.log(n_ave),
                np.log(n_ratio)
            )

            K = pm.Deterministic(
                f"{self.variable_name}_K",
                pm.math.exp(K_)
            )
            n = pm.Deterministic(
                f"{self.variable_name}_n",
                pm.math.exp(n_)
            )
        return K, n
    
    def build_sshaped_priors(self, model=None):
        model = pm.modelcontext(model)
        alpha_mu = self.media_transform_prior.alpha_ave
        alpha_sigma = self.media_transform_prior.alpha_std
        beta_mu = np.log(self.media_transform_prior.beta_ave)/8
        beta_sigma = self.media_transform_prior.beta_std/10e8
        with model:
            alpha = pm.Beta(
                f"{self.variable_name}_alpha", 
                mu=alpha_mu,
                alpha_sigma = alpha_sigma
            )
            beta_ = pm.Beta(
                f"{self.variable_name}_beta_",
                mu=beta_mu,
                sigma=beta_sigma
            )
            beta = pm.Deterministic(
                f"{self.variable_name}_beta",
                10**(beta_*8)
            )
        return alpha, beta

    def build_media_priors(self, model=None):
        model = pm.modelcontext(model)
        with model:
            if self.media_transform == MediaTransform.hill:
                K, n = self.build_hill_priors()
                return K, n
            if self.media_transform == MediaTransform.sorigin or self.media_transform == MediaTransform.sshaped:
                alpha, beta = self.build_sshaped_priors()
                return alpha, beta
        
        raise NotImplementedError(f"{self.media_transform} is not implemented yet.")
    
    def build_delayed_adstock_priors(self, model=None):
        model = pm.modelcontext(model)
        with model:
            pass

    def get_contributions(self, model=None):
        model = pm.modelcontext(model)
        with model:
            estimate = self.build_coeff_prior()
            media_priors = self.build_media_priors()
            dims = self._var_dims()
            transformed_variable = getattr(model, f"{self.variable_name}_transformed")
            media_transformed = pm.Deterministic(f"{self.variable_name}_media_transform", TRANSFOMER_MAP[self.media_transform](transformed_variable, *media_priors), dims=dims)

            contributions = pm.Deterministic(f"{self.variable_name}_contribution", estimate*getattr(model, f"{self.variable_name}_transformed"), dims=dims)
        return contributions  

class ExogVariableDetails(VariableDetails):
    variable_type: Literal['exog'] = 'exog'
    intercept: bool = True
    fixed_ind_coeff_dims: Optional[list[str]] = None
    random_coeff_dims: Optional[list[str]] = None
    likelihood: Likelihood 
