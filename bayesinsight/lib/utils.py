from typing import Union, Optional

import pymc as pm
import numpy as np
import pandas as pd
from patsy import dmatrix
import xarray as xr

from bayesinsight.lib.constants import MFFCOLUMNS

def row_ids_to_ind_map(row_ids: list[str]) -> list[int]:
    ind_map = {
        row_ids[i]: i for i in range(len(row_ids))
    }
    return ind_map

def check_coord(col: str, include_period:bool = False) -> bool:
    if col == 'Period':
        return include_period
    if "_" in col:
        return False
    return True

def check_dim(col: str, dims: Union[None, list, tuple]) -> bool:
    if dims is None:
        return check_coord(col)
    if col in dims:
        return False
    return check_coord(col)

def enforce_dim_order(dims: Union[list, tuple, None], drop_period: bool = True) -> list:
    if dims is None:
        return None
    ordered_dims = [dim for dim in MFFCOLUMNS if dim in dims and dim != "Period"]
    if 'Period' in dims and not drop_period:
        return ordered_dims + ["Period"]
    return ordered_dims

def var_dims(model: Optional[pm.Model]=None) -> list:
    model = pm.modelcontext(model)
    var_dims = enforce_dim_order(list(model.coords.keys()), drop_period=False)
    return var_dims

def spline_matrix(data: pd.DataFrame, geo: str, n_knots: int=6, order: int=3):
  """Outputs spline matrix from data"""
  t = np.linspace(0, 1, len(data[geo].unique())) # Time is scaled to between 0 and 1
  knots = np.linspace(0, 1, n_knots+2)[1:-1] # Knots are taken at even intervals
  
  B0 = dmatrix("bs(t, knots=knots, degree=order, include_intercept=True) - 1", 
             {"t": t, "knots": knots, "order": order})
  return np.asarray(B0)

def sum_over_vars(dataset: xr.Dataset, name: str):
  data_vars = list(dataset.data_vars)
  accum = dataset[data_vars[0]]
  for var in data_vars[1:]:
    accum = accum + dataset[var]
  accum = accum.to_dataset(name=name)
  return accum

def calculate_mult_contributions(avm: pd.DataFrame, contributions: pd.DataFrame, row_ids: list[str], base_vars: list, inc_vars: list, exog_var: str, trace=None):
    
    contributions = contributions.set_index(row_ids)
    avm = xr.Dataset.from_dataframe(avm.set_index(row_ids))
    contributions = xr.Dataset.from_dataframe(contributions)
  
    exog_var_name = exog_var
    weekly_error = np.log(avm[exog_var_name]) - avm['mu']
  
    media_vars_name = [f"{var}_contribution" for var in inc_vars]
    media_vars_contributions = contributions[media_vars_name]
    base_var_contributions = contributions[[var for var in list(contributions.data_vars) if var not in media_vars_name]]
    base_var_contributions["weekly_error"] = weekly_error
    base_vars = [f"{var}_contribution" for var in base_vars]
    do_not_breakout = [var for var in list(base_var_contributions.data_vars) if var not in base_vars]
    base_var_contributions = base_var_contributions.assign(sum_over_vars(base_var_contributions[do_not_breakout], "total_intercept"))
    base_var_contributions = base_var_contributions[["total_intercept"] + base_vars]
    
    inc_contributions = sum_over_vars(media_vars_contributions, "inc_contributions")
    base_contributions = sum_over_vars(base_var_contributions,"base_contributions")
    
    total_contributions = inc_contributions.assign(base_contributions)
    total_contributions["total_contributions"] = total_contributions["inc_contributions"] + total_contributions["base_contributions"]
    total_contributions_exp = np.exp(total_contributions)
    total_contributions_exp["inc_contributions_syn"] =total_contributions_exp["total_contributions"] - total_contributions_exp["base_contributions"]
    total_contributions_exp['base_contributions_syn'] = total_contributions_exp['total_contributions']-total_contributions_exp['inc_contributions']
    total_contributions_exp['base_contributions_norm'] = total_contributions_exp['base_contributions_syn']/(total_contributions_exp['base_contributions_syn']+total_contributions_exp['inc_contributions_syn'])*total_contributions_exp['total_contributions']
    total_contributions_exp['inc_contributions_norm'] = total_contributions_exp['inc_contributions_syn']/(total_contributions_exp['base_contributions_syn']+total_contributions_exp['inc_contributions_syn'])*total_contributions_exp['total_contributions']
    
    inc_contributions_exp = np.exp(sum_over_vars(media_vars_contributions, "total")['total']) - np.exp(sum_over_vars(media_vars_contributions, "total")['total'] - media_vars_contributions)
    base_contributions_exp = np.exp(sum_over_vars(base_var_contributions, 'total')['total']) - np.exp(sum_over_vars(base_var_contributions, 'total')['total'] - base_var_contributions)
    
    inc_contributions_exp = inc_contributions_exp/sum_over_vars(inc_contributions_exp, 'total')['total']*total_contributions_exp['inc_contributions_norm']
    base_contributions_exp = base_contributions_exp/sum_over_vars(base_contributions_exp, 'total')['total']*total_contributions_exp['base_contributions_norm']
    
    return  inc_contributions_exp, base_contributions_exp, total_contributions_exp