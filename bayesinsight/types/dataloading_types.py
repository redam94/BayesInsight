from pydantic import NaiveDatetime

from typing import Dict, Union, Literal

__all__ = ["MFF_ROW", "METADATA", "INDEXCOL"]

MFF_ROW = Dict[str, Union[str, float, NaiveDatetime]]
METADATA = Dict[Literal["Periodicity"], Literal["Weekly", "Daily"]]
INDEXCOL = Literal["Geography", "Period", "Product", "Outlet", "Campaign", "Creative"]
