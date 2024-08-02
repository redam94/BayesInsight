from enum import StrEnum

class _ContUniDist(StrEnum):
    normal = 'Normal'
    laplace = 'Laplace'
    uniform = 'Uniform'

class DiscreteUniDist(StrEnum):
    poisson = 'Poisson'
    geometric = "Geometric"
    binomial = "Binomial"

class DiscreteMultDist(StrEnum):
    multinomial = "MultiNomial"

class _ContMultDist(StrEnum):
    mvnormal = "MvNormal"

class PosContUniDist(StrEnum):
    halfnormal = "HalfNormal"
    halfcauchy = "HalfCauchy"
    beta = "Beta"
    exponential = 'Exponetial'
    lognormal = "LogNormal"

class PosContMultDist(StrEnum):
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

