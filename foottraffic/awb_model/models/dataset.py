from pydantic import BaseModel
from foottraffic.awb_model.models.dataloading import MFF
from foottraffic.awb_model.models.variablemodels import VariableDetails


class DataSet(BaseModel):
    mff: MFF
    variables: VariableDetails