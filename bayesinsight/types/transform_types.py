from enum import StrEnum

__all__ = [
    "Normilization",
    "FunctionalForms",
    "TimeTransforms",
    "Adstock",
    "MediaTransform",
]

INDEXED_TRANSFORMS = []


class Normilization(StrEnum):
    # Grouped Normilization Not Implemented
    # around1 = 'around1'
    # around0 = 'around0'
    global_standardize = "Global Standardize"
    none = "none"


class FunctionalForms(StrEnum):
    linear = "linear"
    log = "log"
    exp = "exp"
    power = "power"
    indexp = "indexp"


class TimeTransforms(StrEnum):
    movingAverage = "movingAverage"


class Adstock(StrEnum):
    geometric = "geometric"
    weibull = "weibull"
    delayed = "delayed"
    none = "none"


class MediaTransform(StrEnum):
    hill = "hill"
    linear = "linear"
