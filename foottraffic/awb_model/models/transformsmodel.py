from typing import Dict, Optional

from pydantic import BaseModel, model_validator

from foottraffic.awb_model.types.transform_types import FunctionalForms, Normilization, TimeTransforms
from foottraffic.awb_model.constants import TRANSFOMER_MAP

class DeterministicTransform(BaseModel):
    functional_form: FunctionalForms = FunctionalForms.linear
    params: Optional[Dict[str, float]] = None

    
    @model_validator(mode='after')
    def params_defined(self):
        annotations = TRANSFOMER_MAP[self.functional_form].__annotations__

        errors = []
        if not annotations:
            if self.params is None:
                return self
            raise ValueError("No parameters needed yet parameters given")
        for key, _ in annotations.items():
            try:
                assert key in list(self.params.keys())
            except AssertionError:
                errors.append(f"{key} not found in parameters")
        for key, _ in self.params.items():
            try:
                assert key in list(annotations.keys())
            except AssertionError:
                errors.append(f"{key} not a parameter for {self.functional_form}")
        if errors:
            raise ExceptionGroup("Parameters failed to validate", errors)
        return self

    def __call__(self, value):
        try:
            if self.params is None:
                return TRANSFOMER_MAP[self.functional_form](value)
            return TRANSFOMER_MAP[self.functional_form](value, **self.params)
        except KeyError:
            raise NotImplementedError(f"{self.functional_form} not implemented yet sorry.")
        
class TimeTransformer(BaseModel):
    transform_type: Optional[TimeTransforms] = None

    def __call__(self, value):
        pass