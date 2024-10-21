import numpy as np
import xarray as xr
from bayesinsight import BayesInsightModel
from bayesinsight.models.variablemodels import VariableDetails

from typing import Tuple, Dict, List, Union, Iterable

def compute_model_contributions(
  model: BayesInsightModel,
):
  """Compute the contributions of each variable to the model"""
  variable_details = model.variable_details
  variable_contributions = {}
  
  try:
    posterior = model.trace.posterior
  except KeyError:
    raise ValueError("Model needs to be fitted")
  
  for variable in variable_details:
    var_name = variable.var_name
    contribution_name = f"{var_name}_contribution"
    variable_contributions[variable.variable_name] = posterior[var_name].mean(dim=("chain", "draw"))
  
  
  return variable_contributions