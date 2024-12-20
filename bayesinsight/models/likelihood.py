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
    dispersion_dims: Optional[List[str]] = None

    def build(self, varname, estimate, obs, model=None):
        model = pm.modelcontext(model)
        likelihood = getattr(pm, self.type)
        dispersion_params = dict()
        var_dim = var_dims(model)
        with model:
            if requires_dispersion(self.type):
                alpha = pm.Gamma(f"{varname}_alpha", 2, 1.0 / 1000)
                dispersion_params["alpha"] = alpha
            elif requires_prob(self.type):
                psi = pm.Beta(
                    f"{varname}_psi", alpha=1, beta=1, dims=self.dispersion_dims
                )
                dispersion_params["psi"] = psi
            elif requires_sd(self.type):
                sigma = pm.Exponential(
                    f"{varname}_sigma", lam=1, dims=self.dispersion_dims
                )
                dispersion_params["sigma"] = sigma

            lik = likelihood(
                f"{varname}_likelihood",
                mu=estimate,
                **dispersion_params,
                observed=obs,
                dims=var_dim,
            )
        return lik
