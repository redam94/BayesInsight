from typing import Optional, List
from bayesinsight.types.likelihood_types import LikelihoodType
from bayesinsight.lib.utils import var_dims

from pydantic import BaseModel
import pymc as pm

__all__ = ["Likelihood"]


def requires_dispersion(likelihood: LikelihoodType):
    return likelihood in ["NegativeBinomial", "ZeroInflatedNegativeBinomial"]


def requires_prob(likelihood: LikelihoodType):
    return likelihood in ["ZeroInflatedPoisson", "ZeroInflatedNegativeBinomial"]


def requires_sd(likelihood: LikelihoodType):
    return likelihood in ["Normal", "StudentT", "LogNormal"]


def requires_nu(likelihood: LikelihoodType):
    return likelihood in ["StudentT"]


class Likelihood(BaseModel):
    type: LikelihoodType
    likelihood_params: Optional[dict] = None
    dispersion_dims: Optional[List[str]] = None

    def build(self, varname, estimate, obs, model=None):
        model = pm.modelcontext(model)
        likelihood = getattr(pm, self.type)
        dispersion_params = dict()
        var_dim = var_dims(model)
        with model:
            if requires_dispersion(self.type):
                if self.likelihood_params is None:
                    alpha = pm.Gamma(f"{varname}_alpha", 2, 1.0 / 1000)
                else:
                    alpha = pm.Gamma(f"{varname}_alpha", **self.likelihood_params)
                dispersion_params["alpha"] = alpha
            elif requires_prob(self.type):
                if self.likelihood_params is None:
                    psi = pm.Beta(f"{varname}_psi", alpha=1, beta=1)
                else:
                    psi = pm.Beta(f"{varname}_psi", **self.likelihood_params)
                
                dispersion_params["psi"] = psi
            elif requires_sd(self.type):
                if self.likelihood_params is None:
                    sigma = pm.HalfNormal(f"{varname}_sigma", 1.0, dims=self.dispersion_dims)
                else:
                    sigma = pm.HalfNormal(f"{varname}_sigma", **self.likelihood_params, dims=self.dispersion_dims)
                
                dispersion_params["sigma"] = sigma

            lik = likelihood(
                f"{varname}_likelihood",
                mu=estimate,
                **dispersion_params,
                observed=obs,
                dims=var_dim,
            )
        return lik
