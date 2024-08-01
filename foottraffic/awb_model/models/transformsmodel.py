from typing import Dict, Optional

from pydantic import BaseModel

from foottraffic.awb_model.types.transform_types import FunctionalForms, Normilization, TimeTransforms
from foottraffic.awb_model.constants import TRANSFOMER_MAP

class DeterministicTransform(BaseModel):
    functional_form: FunctionalForms = FunctionalForms.linear
    params: Optional[Dict[str, float]] = None

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