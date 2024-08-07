from enum import StrEnum

class VarType(StrEnum):
    exog = 'exog'
    media = 'media'
    control = 'control'
    treatment = 'treatment'
    base = 'base'
    none = 'none'
