from foottraffic.awb_model.types.distribution_types import PosDist, Distribution, ContDist

from pydantic import BaseModel, PositiveFloat
import numpy as np

from typing import Dict, Literal

class MediaCoeffPrior(BaseModel):
    coeff_dist: PosDist = PosDist.lognormal
    coeff_params: Dict[str, float] = {
        'mu': np.log(.05),
        'sigma': np.log(1.3)
    }

class ControlCoeffPrior(BaseModel):
    coeff_dist: Distribution = Distribution.normal
    coef_params: Dict[str, float] = {
        'mu': 0,
        'sigma': 3
    }

class HillPrior(BaseModel):
    type: Literal['Hill'] = "Hill"
    K_ave: PositiveFloat = .85
    K_std: PositiveFloat = .6
    n_ave: PositiveFloat = 1.5
    n_std: PositiveFloat = 1.2


    

class SShapedPrior(BaseModel):
    type: Literal['SShaped'] = "SShaped"
    alpha_ave: PositiveFloat = .96
    alpha_std: PositiveFloat = .2
    beta_ave: PositiveFloat = 1e2
    beta_std: PositiveFloat = 3

    
