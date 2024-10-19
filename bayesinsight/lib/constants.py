"""File contains constants used throughout the project"""
from bayesinsight.lib.transforms import (
    log, linear, exp,
    indexp, power, 
    hill
)
from bayesinsight.lib.adstock import (
  delayed_adstock, weibull_adstock
)
from bayesinsight.types.transform_types import FunctionalForms, MediaTransform, Adstock


MFFCOLUMNS = [
    "Period",
    "Geography",
    "Product",
    "Outlet",
    "Campaign",
    "Creative",
    "VariableValue",
    "VariableName"
]

TRANSFOMER_MAP = {
    FunctionalForms.exp: exp,
    FunctionalForms.linear: linear,
    FunctionalForms.log: log,
    FunctionalForms.indexp: indexp,
    FunctionalForms.power: power
}

ADSTOCK_MAP = {
  Adstock.delayed: delayed_adstock,
  Adstock.weibull: weibull_adstock
}

MEDIA_TRANSFORM_MAP = {
  MediaTransform.hill: hill,
  MediaTransform.linear: linear
}