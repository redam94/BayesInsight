from typing import Union, Optional

import pymc as pm

from foottraffic.awb_model.constants import MFFCOLUMNS

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