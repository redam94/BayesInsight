from enum import StrEnum

INDEXED_TRANSFORMS = []

class Normilization(StrEnum):
    around1 = 'around1'
    around0 = 'around0'
    standardize = 'standardize'
    none = 'none'

class FunctionalForms(StrEnum):
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