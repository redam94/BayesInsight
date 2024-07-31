from foottraffic.awb_model.types.variable_types import VarType
from foottraffic.awb_model.models.transformsmodel import DeterministicTransform, Normilization
from foottraffic.awb_model.models.dataloading import MFF

from typing import Optional

from pydantic import BaseModel
import numpy as np
import pandas as pd


class VariableDetails(BaseModel):
    variable_name: str
    variable_type: VarType
    deterministic_transform: DeterministicTransform = DeterministicTransform(functional_form='linear', params=None)
    normalization: Normilization = Normilization.none
    variable_max: Optional[float] = None
    variable_min: Optional[float] = None

    def transform(self, data: MFF) -> pd.Series:
        analytic_dataframe = data.analytic_dataframe()
        try:
            variable = analytic_dataframe[self.variable_name]
        except KeyError:
            raise ValueError(f"{self.variable_name} not in MFF. Check spelling.")
        
        
        return self.deterministic_transform(variable)
    
    def fit(self, data: MFF):
        if self.deterministic_transform.functional_form in ['s-shape', 's-origin', ]:
            pass
