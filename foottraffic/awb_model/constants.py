"""File contains constants used throughout the project"""
from foottraffic.awb_model.transforms import (
    log, linear, exp,
    indexp, power, 
    s_origin, s_shaped, 
    hill
)
from foottraffic.bayes.media_transforms.adstock import (
  delayed_adstock, weibull_adstock
)
from foottraffic.awb_model.types.transform_types import FunctionalForms, MediaTransform


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
    FunctionalForms.power: power,
    FunctionalForms.sshape: s_shaped,
    FunctionalForms.sorigin: s_origin,
}

MEDIA_TRANSFORM_MAP = {
  MediaTransform.hill: hill,
  MediaTransform.linear: linear,
  MediaTransform.sorigin: s_origin,
  MediaTransform.sshaped: s_shaped
}