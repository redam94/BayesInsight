from enum import StrEnum

__all__ = ["VarType"]


class VarType(StrEnum):
    exog = "exog"
    media = "media"
    control = "control"
    treatment = "treatment"
    base = "base"
    none = "none"
