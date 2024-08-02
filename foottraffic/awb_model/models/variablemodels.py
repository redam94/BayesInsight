from foottraffic.awb_model.types.variable_types import VarType
from foottraffic.awb_model.types.transform_types import MediaTransform, Adstock
from foottraffic.awb_model.models.transformsmodel import DeterministicTransform, Normilization, TimeTransformer
from foottraffic.awb_model.models.priormodel import MediaCoeffPrior, HillPrior, SShapedPrior, ControlCoeffPrior
from foottraffic.awb_model.models.dataloading import MFF

from typing import Optional, Union, Literal

from pydantic import BaseModel, PositiveFloat, model_validator
import numpy as np
import pandas as pd
import pymc as pm

INDEX_MAP = {
    "Geography": 'i',
    "Product": 'j',
    "Outlet": 'k',
    "Campaign": 'l',
    'Creative': 'm'
}
class VariableDetails(BaseModel):
    variable_name: str
    variable_type: Literal["control", "exog", 'media', 'base']
    deterministic_transform: DeterministicTransform = DeterministicTransform(functional_form='linear', params=None)
    normalization: Normilization = Normilization.none
    time_transform: Optional[TimeTransformer] = None
    
    def normalize(self, data: Union[MFF, np.ndarray]) -> np.ndarray:
        if isinstance(data, MFF):
            data = self.as_numpy(data)

        if self.normalization == Normilization.none:
            return data
        
        if self.normalization == Normilization.global_standardize:
            var = data
            demeaned = (var - np.mean(var))
            standardized = demeaned/np.std(demeaned)
            return standardized
        
        raise NotImplementedError("Only global_standardize is implemented. :(")
    
    def transform(self, data: MFF, time_first=True, normalize_first=False) -> np.ndarray:
        """Apply transforms to """
        
        variable = self.as_numpy(data)
        
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
    

    def build_coeff_prior(self):
        raise NotImplementedError
    
class ControlVariableDetails(VariableDetails):
    variable_type: Literal['control'] = 'control'
    coeff_prior: ControlCoeffPrior = ControlCoeffPrior()
    fixed_ind_coeff_dims: Optional[list[str]] = None
    random_coeff_dims: Optional[list[str]] = None

    def build_coeff_prior(self, model=None):
        model = pm.modelcontext(model)
        with model:
            coeff_fixed = getattr(pm, self.coeff_prior.coeff_dist)(f"{self.variable_name}_fixed_coeff", **self.coeff_prior.coef_params, dims=self.fixed_coeff_dims)
            if self.random_coeff_dims is None:
                coeff_est = pm.Deterministic(f'{self.variable_name}_coeff_estimate', coeff_fixed, dims=self.fixed_coeff_dims)
                return coeff_est
            sigma = pm.HalfCauchy(f'{self.variable_name}_rand_coeff_sigma', 1)
            random_fixed = pm.Normal(f"{self.variable_name}_rand_coeff", 0, 1, dims=self.random_coeff_dims)
            
class MediaVariableDetails(VariableDetails):
    variable_type: Literal['media']= 'media'
    time_transform: None = None
    adstock: Adstock = Adstock.delayed
    media_transform: MediaTransform = MediaTransform.hill
    coeff_prior: MediaCoeffPrior = MediaCoeffPrior()
    fixed_ind_coeff_dims: Optional[list[str]] = None
    random_coeff_dims: Optional[list[str]] = None
    media_transform_prior: Union[HillPrior, SShapedPrior] = HillPrior()

    @model_validator(mode="after")
    def validate_transform_and_prior(self):
        if self.media_transform == MediaTransform.hill:
            assert isinstance(self.media_transform_prior, HillPrior)
        if self.media_transform == MediaTransform.sorigin or self.media_transform == MediaTransform.sshaped:
            assert isinstance(self.media_transform_prior, SShapedPrior)
        return self
    