from foottraffic.awb_model.types.variable_types import VarType
from foottraffic.awb_model.models.transformsmodel import DeterministicTransform, Normilization, TimeTransformer
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
    time_transform: Optional[TimeTransformer] = None
    def transform(self, data: MFF) -> pd.Series:

        variable = self.get_variable_values(data)
        return self.deterministic_transform(variable)
    
    def get_variable_values(self, data: MFF) -> pd.Series:
        analytic_dataframe = data.analytic_dataframe(indexed=True)
        try:
            return analytic_dataframe[self.variable_name]
        except KeyError:
            raise ValueError(f"{self.variable_name} not in MFF. Check spelling.")
        
    def as_numpy(self, data: MFF) -> np.ndarray:
        analytic_dataframe = data.analytic_dataframe(indexed=True)
        row_dims = (data._info[index_col]["# Unique"] for index_col in data.metadata.row_ids)

        return analytic_dataframe[self.variable_name].to_numpy().reshape(tuple(row_dims))
        