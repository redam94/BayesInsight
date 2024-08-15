from foottraffic.awb_model.types.distribution_types import PosDist, Distribution, ContDist
from foottraffic.awb_model.utils import row_ids_to_ind_map, var_dims, enforce_dim_order, check_coord, check_dim

from pydantic import BaseModel, PositiveFloat
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from typing import Dict, Literal, Optional

class Prior(BaseModel):
    def build(self, varname, model=None, **kwargs):
        raise NotImplementedError
    


class CoeffPrior(Prior):

    coeff_dist: Distribution
    coeff_params: Dict[str, float]

    def build(self, varname, random_dims=None, fixed_dims=None, pooling_sigma=1, model=None):
        """Build a prior for the coefficients of the control variable
           Grabs the model on the context stack if model is None."""
        
        model = pm.modelcontext(model)

        index_map = row_ids_to_ind_map(enforce_dim_order(list(model.coords.keys())))
        
        model_dims = {col: len(model.coords[col]) for col in index_map.keys() if check_dim(col, None)}

        fixed_ind_coeff_dims = enforce_dim_order(fixed_dims)
        fixed_dims_project = dict(
            repeats=tuple([model_dims[col] for col, index in index_map.items() if check_dim(col, None)]),
            axis=tuple([index for col, index in index_map.items() if check_dim(col, fixed_ind_coeff_dims)])
        )
        
        random_coeff_dims = enforce_dim_order(random_dims)
        random_dims_project = dict(
            repeats=tuple([model_dims[col] for col, index in index_map.items() if check_dim(col, None)]),
            axis=tuple([index for col, index in index_map.items() if check_dim(col, random_coeff_dims)])
        )

        with model:
            coeff_dist = getattr(pm, self.coeff_dist) # Get coeff distribution
            coeff_params = self.coeff_params # Get coeff dist params
            ## Fixed Coefficients
            coeff_fixed = coeff_dist(f"{varname}_fixed_coeff", **coeff_params, dims=fixed_ind_coeff_dims)
            
            # Enforce Shape
            expanded_fixed = pt.expand_dims(coeff_fixed, axis=fixed_dims_project['axis'])
            repeats_fixed = np.ones(shape=fixed_dims_project['repeats'])
            if random_dims is None: 
                
                coeff_est = pm.Deterministic(
                    f'{varname}_coeff_estimate',
                    expanded_fixed*repeats_fixed, 
                    dims=tuple(model_dims.keys())
                    )
            
                return coeff_est
            
            sigma = pm.HalfCauchy(f'{varname}_rand_coeff_sigma', pooling_sigma)
            #random_coeff_mu = coeff_dist(f"{self.variable_name}_random_coeff_mu", **coeff_params)
            random_fixed = pm.Normal(f"{varname}_rand_coeff", 0, 1, dims=random_coeff_dims)

            expanded_random = pt.expand_dims(random_fixed, axis=random_dims_project['axis'])
            repeats_random = np.ones(shape=random_dims_project['repeats'])
            coeff_est = pm.Deterministic(
                f'{varname}_coeff_estimate',
                expanded_fixed*repeats_fixed + (expanded_random*repeats_random*sigma), 
                dims=tuple(model_dims.keys())
            )
        return coeff_est

class MediaCoeffPrior(CoeffPrior):
    coeff_dist: PosDist = PosDist.lognormal
    coeff_params: Dict[str, float] = {
        'mu': np.log(.05),
        'sigma': np.log(1.3)
    }


class ControlCoeffPrior(CoeffPrior):
    coeff_dist: Distribution = Distribution.normal
    coeff_params: Dict[str, float] = {
        'mu': 0,
        'sigma': 3
    }

class DelayedAdStockPrior(Prior):
    type: Literal['Delayed'] = "Delayed"
    retention_rate_mean: PositiveFloat = .05
    retention_rate_std: PositiveFloat = 1.2
    lag_mean: PositiveFloat = 1
    lag_std: PositiveFloat = .5

    def build(self, var_name, model=None):
        model = pm.modelcontext(model)
        retention_rate = pm.Normal(
            f"{var_name}_retention_rate",
            np.log(self.retention_rate_mean),
            np.log(self.retention_rate_std)
        )
        


class HillPrior(Prior):
    type: Literal['Hill'] = "Hill"
    K_ave: Optional[PositiveFloat] = .85
    K_std: Optional[PositiveFloat] = .6
    n_ave: Optional[PositiveFloat] = 1.5
    n_std: Optional[PositiveFloat] = 1.2

    def build(self, var_name, model=None):
        model = pm.modelcontext(model)
        # Half saturation prior helper terms
        K_ave = self.K_ave
        K_std = self.K_std
        K_ratio = K_std/K_ave + 1
        # Shape prior helper terms
        n_ave = self.n_ave
        n_std = self.n_std
        n_ratio = n_std/n_ave + 1
        with model:

            K_ = pm.Normal(
                f"{var_name}_K_", 
                np.log(K_ave), 
                np.log(K_ratio))
            n_ = pm.Normal(
                f"{var_name}_n_",
                np.log(n_ave),
                np.log(n_ratio)
            )

            K = pm.Deterministic(
                f"{var_name}_K",
                pm.math.exp(K_)
            )
            n = pm.Deterministic(
                f"{var_name}_n",
                pm.math.exp(n_)
            )
        return K, n

class SShapedPrior(Prior):
    type: Literal['SShaped'] = "SShaped"
    alpha_ave: Optional[PositiveFloat] = .88
    alpha_std: Optional[PositiveFloat] = .03
    beta_ave: Optional[PositiveFloat] = 1e2
    beta_std: Optional[PositiveFloat] = 1000

    def build(self, var_name, model=None):
        model = pm.modelcontext(model)
        alpha_mu = self.alpha_ave
        alpha_sigma = self.alpha_std
        beta_mu = np.log(self.beta_ave)/8/np.log(10)
        beta_sigma = np.log(self.beta_std)/np.log(10)/8
        with model:
            alpha = pm.Beta(
                f"{var_name}_alpha", 
                mu=alpha_mu,
                sigma = alpha_sigma
            )
            beta_ = pm.Beta(
                f"{var_name}_beta_",
                mu=beta_mu,
                sigma=beta_sigma
            )
            beta = pm.Deterministic(
                f"{var_name}_beta",
                10**(beta_*8)
            )
        return alpha, beta
    
class InterceptPrior(ControlCoeffPrior):
    pass