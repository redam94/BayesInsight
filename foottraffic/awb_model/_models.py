from pydantic import BaseModel, PositiveFloat, model_validator, field_validator, computed_field, Field, ValidationError, NaiveDatetime
import pandas as pd
import numpy as np
import pymc as pm

from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Union, Literal, Tuple, Set, Callable, Dict, Any, Iterable, TypeVar
import warnings
import json
import os


from foottraffic.awb_model.types import (
    Normilization, FunctionalForms,
    VarType, ModelType, Dim,
    MFF_ROW, INDEXCOL, Distribution, 
    PosDist, ContDist, Adstock)
from foottraffic.awb_model.constants import MFFCOLUMNS

class Parameters(BaseModel):
    normalizationMethod: Normilization = Normilization.none
    functionalForm: FunctionalForms = FunctionalForms.linear
    alpha: PositiveFloat = 0.0
    beta: PositiveFloat = 0.0
    lag: int = 0
    decay: float = 1
    combineWith: Optional[List[str]] = None

class VariableDef(BaseModel):
    variableName: str
    transformParams: Parameters
    coeff: Optional[float] = None
    splitVariables: Optional[List[str]] = None
    #varType: VarType = VarType.none

    @classmethod
    def load_from_csv(cls, file: Union[str, Path]):
        file_extention = str(file).split('.')[-1]
        try:
            assert file_extention == 'csv'
        except AssertionError:
            raise ValueError(f"Must provide a csv file not a {file_extention} type file")

class ModelDef(BaseModel):
    variables: List[VariableDef]
    modelType: ModelType = ModelType.ols
    data: MFF
    groupingDims: Optional[List[Dim]] = None
    timeDim: Optional[Literal["Period"]] = None

    @model_validator(mode = 'after')
    def varify_variables_in_data(self):

        if not all(var.variableName in self.data.get_unique_varnames() for var in self.variables):
            raise ValueError("Data must contain all variables in the variable list")
        return self
        
    @model_validator(mode='after')
    def varify_necessary_attributes_for_model_type(self):

        if self.modelType == ModelType.ols or self.modelType == ModelType.timeseries:
            if not self.groupingDims is None:
                warnings.warn(f"Grouping dimentions are ignored if model is of type {self.modelType}. Try using FixedEffects or MixedEffects model.")
        if self.modelType == ModelType.ols:
            if not self.timeDim is None:
                warnings.warn(f"Time dimention is ignored if model is of type OLS. Try using a timeseries model.")
        if self.modelType == ModelType.timeseries:
            if self.timeDim is None:
                raise AttributeError("timeDim must be defined in order to use timeseries model")     
        if self.modelType == ModelType.mixedEffect or self.modelType == ModelType.fixedEffect:
            if self.groupingDims is None:
                raise AttributeError(f"Grouping dims must be defined for models of type {self.modelType}")
            
        return self


class Prior(BaseModel):
    dist: Union[Distribution, PosDist] = Field(default=Distribution.normal, validate_default=True)
    params: Dict[str, Union[float, List[Union[float, List]]]] = Field(default_factory=lambda:{'mu': 0.0, 'tau': 1.0})
    dims: Optional[Tuple[Dim, ...]] = None

    def get_pymc_prior(self, name: str):
        params = {key: value if isinstance(value, float) else np.array(value) for key, value in self.params.items()}
        return getattr(pm, self.dist)(name, **params, dims=self.dims)
    
class PositivePrior(Prior):
    
    @field_validator('dist')
    @classmethod
    def only_positive_dists(cls, v)->PosDist:
        try:
            return PosDist(v)
        except ValueError:
            raise ValueError(f"{v} is not a valid positive distribution {PosDist._member_names_}")
        





    


    