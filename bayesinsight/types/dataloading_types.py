from pydantic import NaiveDatetime

from typing import Dict, Union, Literal

MFF_ROW = Dict[str, Union[str, float, NaiveDatetime]]
METADATA = Dict[Literal['Periodicity'], Literal['Weekly', "Daily"]]
INDEXCOL = Literal['Geography', 'Period', "Product", "Outlet", "Campaign", "Creative"]
