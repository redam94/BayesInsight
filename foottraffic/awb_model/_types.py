from pydantic import NaiveDatetime

from enum import Enum, StrEnum
from typing import Dict, Union, Literal


MFF_ROW = Dict[str, Union[str, float, NaiveDatetime]]
METADATA = Dict[Literal['Periodicity'], Literal['Weekly', "Daily"]]
INDEXCOL = Literal['Geography', 'Period', "Product", "Outlet", "Campaign", "Creative"]

class Normilization(str, Enum):
    around1 = 'around1'
    around0 = 'around0'
    standardize = 'standardize'
    none = 'none'

class FunctionalForms(str, Enum):
    sshape = 's-shape'
    sorigin = 's-origin'
    linear = 'linear'
    log = 'log'
    exp = 'exp'
    power = 'power'
    indexp =  'indexp'
    movingAverage = 'movingAverage'

class Adstock(StrEnum):
    geometric = "geometric"
    weibull = "weibull"
    delayed = "delayed"
    none = 'none'

class MediaTransform(StrEnum):
    hill = "hill"
    sorigin = "s-origin"
    sshaped = 's-shape'
    linear = 'linear'

class VarType(str, Enum):
    exog = 'exog'
    media = 'media'
    control = 'control'
    treatment = 'treatment'
    base = 'base'

class ModelType(str, Enum):
    ols = 'OLS'
    mixedEffect = 'MixedEffect'
    fixedEffect = 'FixedEffect'
    timeseries = 'Timeseries'

class Dim(str, Enum):
    geography = 'Geography'
    product = 'Product'
    outlet = 'Outlet'
    campaign = "Campaign"
    creative = "Creative"
    period = 'Period'

class _ContUniDist(str, Enum):
    normal = 'Normal'
    laplace = 'Laplace'
    uniform = 'Uniform'

class DiscreteUniDist(str, Enum):
    poisson = 'Poisson'
    geometric = "Geometric"
    binomial = "Binomial"

class DiscreteMultDist(str, Enum):
    multinomial = "MultiNomial"

class _ContMultDist(str, Enum):
    mvnormal = "MvNormal"

class PosContUniDist(str, Enum):
    halfnormal = "HalfNormal"
    halfcauchy = "HalfCauchy"
    beta = "Beta"
    exponential = 'Exponetial'

class PosContMultDist(str, Enum):
    dirichlet = "Dirichlet"
    lkjcholeskycov = "LKJCholeskyCov"

def extend_flag(inherited,_type):
   def wrapper(final):
     joined = {}
     inherited.append(final)
     for i in inherited:
        for j in i:
           joined[j.name] = j.value
     return _type(final.__name__, joined)
   return wrapper

@extend_flag([_ContUniDist, PosContUniDist], StrEnum)
class ContUniDist(StrEnum):
    pass

@extend_flag([_ContMultDist, PosContMultDist], StrEnum)
class ContMultDist(StrEnum):
    pass

@extend_flag([_ContUniDist, PosContUniDist] + [_ContMultDist, PosContMultDist], StrEnum)
class ContDist(StrEnum):
    pass

@extend_flag([PosContUniDist, PosContMultDist], StrEnum)
class PosDist(StrEnum):
    pass

@extend_flag([DiscreteMultDist, DiscreteUniDist], StrEnum)
class DiscreteDist(StrEnum):
    pass

@extend_flag([_ContUniDist, PosContUniDist] + [_ContMultDist, PosContMultDist] + [DiscreteMultDist, DiscreteUniDist], StrEnum)
class Distribution(StrEnum):
    pass

