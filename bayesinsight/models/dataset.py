from pydantic import BaseModel
from bayesinsight.models.dataloading import MFF
from bayesinsight.models.variablemodels import VariableDetails


class DataSet(BaseModel):
    mff: MFF
    variables: VariableDetails