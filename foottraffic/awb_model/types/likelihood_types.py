from enum import StrEnum

class LikelihoodType(StrEnum):
  poisson = "Poisson" # Good choice for count data
  normal = "Normal" # Standard regression independence assumptions apply
  mvnormal = "MvNormal" # Please make good choice for Covariance matrix mostlikely an autoregressive
  negative_binomial = "NegativeBinomial" # Good choice if count data is overdispersed
  zero_inflated_poisson = "ZeroInflatedPoisson" # Good choice if count data is zero often
  zero_inflated_negative_binomial = "ZeroInflatedNegativeBinomial" # Good choice if count data is often zero and overdispersed