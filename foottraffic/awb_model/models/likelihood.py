from foottraffic.awb_model.types.likelihood_types import LikelihoodType
from foottraffic.awb_model.utils import row_ids_to_ind_map, var_dims, enforce_dim_order, check_coord, check_dim

from pydantic import BaseModel
import pymc as pm

def requires_dispersion(likelihood: LikelihoodType):
  return likelihood in ['NegativeBinomial', 'ZeroInflatedNegativeBinomial']

def requires_prob(likelihood: LikelihoodType):
  return likelihood in ['ZeroInflatedPoisson', 'ZeroInflatedNegativeBinomial']

def requires_sd(likelihood: LikelihoodType):
  return likelihood in ['Normal', 'StudentT', 'LogNormal']

def requires_nu(likelihood: LikelihoodType):
  return likelihood in ['StudentT']
class Likelihood(BaseModel):
  type: LikelihoodType

  def build(self, varname, estimate, obs, model=None):
    model = pm.modelcontext(model)
    likelihood = getattr(pm, self.type)
    dispersion_params = dict()
    var_dim = var_dims(model)
    with model:
      if requires_dispersion(self.type):
        alpha = pm.Exponential(f"{varname}_alpha", lam=.2)
        dispersion_params['alpha'] = alpha
      elif requires_prob(self.type):
        psi = pm.Beta(f"{varname}_psi", alpha=1, beta=1)
        dispersion_params['psi'] = psi
      elif requires_sd(self.type):
        sigma = pm.Exponential(f"{varname}_sigma", lam=1)
        dispersion_params['sigma'] = sigma
      
      lik = likelihood(f"{varname}_likelihood", mu=estimate, **dispersion_params, observed=obs, dims=var_dim)
    return lik
      