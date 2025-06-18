from bayesinsight.types.distribution_types import PosDist, Distribution
from bayesinsight.lib.utils import (
    row_ids_to_ind_map,
    var_dims,
    enforce_dim_order,
    check_dim,
)

from pydantic import BaseModel, PositiveFloat
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from typing import Dict, Literal, Optional, Union, List

__all__ = [
    "Prior",
    "CoeffPrior",
    "MediaCoeffPrior",
    "ControlCoeffPrior",
    "DelayedAdStockPrior",
    "HillPrior",
    "SShapedPrior",
    "InterceptPrior",
    "LocalTrendPrior",
    "EventPrior",
    "SeasonPrior",
]


class Prior(BaseModel):
    def build(self, varname, model=None, **kwargs):
        raise NotImplementedError


class CoeffPrior(Prior):
    coeff_dist: Distribution
    coeff_params: Dict[str, float]

    def build(
        self, varname, random_dims=None, fixed_dims=None, pooling_sigma=1, model=None
    ):
        """Build a prior for the coefficients of the control variable
        Grabs the model on the context stack if model is None."""

        model = pm.modelcontext(model)

        index_map = row_ids_to_ind_map(enforce_dim_order(list(model.coords.keys())))

        model_dims = {
            col: len(model.coords[col])
            for col in index_map.keys()
            if check_dim(col, None)
        }

        fixed_ind_coeff_dims = enforce_dim_order(fixed_dims)
        fixed_dims_project = dict(
            repeats=tuple(
                [
                    model_dims[col]
                    for col, index in index_map.items()
                    if check_dim(col, None)
                ]
            ),
            axis=tuple(
                [
                    index
                    for col, index in index_map.items()
                    if check_dim(col, fixed_ind_coeff_dims)
                ]
            ),
        )

        random_coeff_dims = enforce_dim_order(random_dims)
        random_dims_project = dict(
            repeats=tuple(
                [
                    model_dims[col]
                    for col, index in index_map.items()
                    if check_dim(col, None)
                ]
            ),
            axis=tuple(
                [
                    index
                    for col, index in index_map.items()
                    if check_dim(col, random_coeff_dims)
                ]
            ),
        )

        with model:
            coeff_dist = getattr(pm, self.coeff_dist)  # Get coeff distribution
            coeff_params = self.coeff_params  # Get coeff dist params
            ## Fixed Coefficients
            coeff_fixed = coeff_dist(
                f"{varname}_fixed_coeff", **coeff_params, dims=fixed_ind_coeff_dims
            )

            # Enforce Shape
            expanded_fixed = pt.expand_dims(
                coeff_fixed, axis=fixed_dims_project["axis"]
            )
            repeats_fixed = np.ones(shape=fixed_dims_project["repeats"])
            if random_dims is None:
                coeff_est = pm.Deterministic(
                    f"{varname}_coeff_estimate",
                    expanded_fixed * repeats_fixed,
                    dims=tuple(model_dims.keys()),
                )

                return coeff_est
            if pooling_sigma == 0:
                sigma = 0
                random_fixed = 0
            else:
                sigma = pm.HalfNormal(f"{varname}_rand_coeff_sigma", pooling_sigma)
                # random_coeff_mu = coeff_dist(f"{self.variable_name}_random_coeff_mu", **coeff_params)
                random_fixed = pm.Normal(
                    f"{varname}_rand_coeff", 0, 1, dims=random_coeff_dims
                )

            expanded_random = pt.expand_dims(
                random_fixed, axis=random_dims_project["axis"]
            )
            repeats_random = np.ones(shape=random_dims_project["repeats"])
            coeff_est = pm.Deterministic(
                f"{varname}_coeff_estimate",
                expanded_fixed * repeats_fixed
                + (expanded_random * repeats_random * sigma),
                dims=tuple(model_dims.keys()),
            )
        return coeff_est


class MediaCoeffPrior(CoeffPrior):
    coeff_dist: PosDist = PosDist.lognormal
    coeff_params: Dict[str, float] = {"mu": np.log(0.05), "sigma": np.log(1.3)}


class ControlCoeffPrior(CoeffPrior):
    coeff_dist: Distribution = Distribution.normal
    coeff_params: Dict[str, float] = {"mu": 0, "sigma": 3}


class DelayedAdStockPrior(Prior):
    type: Literal["Delayed"] = "Delayed"
    retention_rate_mean: PositiveFloat = 0.2
    retention_rate_std: PositiveFloat = 10
    lag_min: PositiveFloat = 1e-4
    lag_max: PositiveFloat = 3

    def build(self, var_name, model=None):
        model = pm.modelcontext(model)
        retention_rate = pm.Beta(
            f"{var_name}_retention_rate_log",
            mu=self.retention_rate_mean,
            nu=self.retention_rate_std,
        )
        # retention_rate = pm.Deterministic(
        #    f"{var_name}_retention_rate",
        #    pm.math.exp(retention_rate_log)
        # )
        # lag = pm.Uniform(
        #    f"{var_name}_lag",
        #    self.lag_min,
        #    self.lag_max
        # )
        lag = pm.Exponential(f"{var_name}_lag", self.lag_max)
        return retention_rate, lag


class HillPrior(Prior):
    type: Literal["Hill"] = "Hill"
    K_ave: Optional[PositiveFloat] = 0.85
    K_std: Optional[PositiveFloat] = 0.6
    n_ave: Optional[PositiveFloat] = 1.5
    n_std: Optional[PositiveFloat] = 1.2

    def build(self, var_name, model=None):
        model = pm.modelcontext(model)
        # Half saturation prior helper terms
        K_ave = self.K_ave
        K_std = self.K_std
        K_ratio = K_std / K_ave + 1
        # Shape prior helper terms
        n_ave = self.n_ave
        n_std = self.n_std
        n_ratio = n_std / n_ave + 1
        with model:
            K = pm.LogNormal(f"{var_name}_K", np.log(K_ave), np.log(K_ratio))
            n_ = pm.Exponential(f"{var_name}_n_", 1 / (n_ave-1))

            # K = pm.Deterministic(f"{var_name}_K", pm.math.exp(K_))
            n = pm.Deterministic(f"{var_name}_n", n_ + 1)
        return K, n


class SShapedPrior(Prior):
    type: Literal["SShaped"] = "SShaped"
    alpha_ave: Optional[PositiveFloat] = 0.88
    alpha_std: Optional[PositiveFloat] = 0.03
    beta_ave: Optional[PositiveFloat] = 1e2
    beta_std: Optional[PositiveFloat] = 1000

    def build(self, var_name: str, model=None):
        model = pm.modelcontext(model)
        alpha_mu = self.alpha_ave
        alpha_sigma = self.alpha_std
        beta_mu = np.log(self.beta_ave) / 8 / np.log(10)
        beta_sigma = np.log(self.beta_std) / np.log(10) / 8
        with model:
            alpha = pm.Beta(f"{var_name}_alpha", mu=alpha_mu, sigma=alpha_sigma)
            beta_ = pm.Beta(f"{var_name}_beta_", mu=beta_mu, sigma=beta_sigma)
            beta = pm.Deterministic(f"{var_name}_beta", 10 ** (beta_ * 8))
        return alpha, beta


class InterceptPrior(ControlCoeffPrior):
    pass


class LocalTrendPrior(Prior):
    type: Literal["LocalTrend"] = "LocalTrend"
    variability: Optional[PositiveFloat] = 0.2
    group_variablility: Optional[PositiveFloat] = 0.01
    partial_pooling: Optional[PositiveFloat] = 0.01
    initial_dist_var: Optional[PositiveFloat] = 1.0

    def build(
        self,
        var_name: str,
        n_splines: int,
        random_dims: Optional[List[str]] = None,
        grouping_map: Optional[Union[Dict[str, List[str]], str]] = None,
        grouping_name: Optional[str] = None,
        model=None,
    ):
        model = pm.modelcontext(model)

        coords = {"splines": np.arange(n_splines)}

        if isinstance(grouping_map, dict):
            try:
                assert isinstance(grouping_name, str)
            except AssertionError:
                raise ValueError(
                    f"grouping_name must be defined in LocalTrendPrior for {var_name}"
                )

            try:
                assert isinstance(random_dims, list)
                assert len(random_dims) == 1
            except AssertionError:
                raise ValueError(
                    f"Random_dims for LocalTrendPrior for {var_name} must be defined with len 1"
                )

            coords |= {grouping_name: set(grouping_map.keys())}

            group_array = np.array(
                [
                    [1 if name in group else 0 for key, group in grouping_map.items()]
                    for name in model.coords[random_dims[0]]
                ]
            )

        with model:
            with pm.Model(name=f"LLT_{var_name}", coords=coords):
                if random_dims is None:
                    tau = pm.HalfNormal("tau", self.variability)
                    trends_betas = pm.GaussianRandomWalk(
                        "splines_betas",
                        mu=0,
                        sigma=tau,
                        init_dist=pm.Normal.dist(0, self.initial_dist_var),
                        dims=("splines"),
                    )
                    # trends_betas = pm.Normal("splines_betas", mu=0, sigma=self.variability, dims=("splines",))
                    return trends_betas

                if grouping_name is not None:
                    tau = pm.HalfNormal("tau", self.variability)
                    trends_betas_mu = pm.GaussianRandomWalk(
                        "splines_betas_mu",
                        mu=0,
                        sigma=tau,
                        init_dist=pm.Normal.dist(0, self.initial_dist_var),
                        dims=("splines"),
                    )
                    # trends_betas_mu = pm.Normal("splines_betas_mu", mu=0, sigma=self.variability, dims=("splines",))
                    trends_betas_sd = pm.HalfNormal(
                        "splines_betas_sd",
                        sigma=self.group_variablility,
                        dims=("splines",),
                    )
                    trends_betas_group = pm.Normal(
                        "splines_betas_group",
                        mu=trends_betas_mu,
                        sigma=trends_betas_sd,
                        dims=(grouping_name, "splines"),
                    )
                    trends_betas_geo_sd = pm.HalfNormal(
                        "splines_betas_group_sd",
                        sigma=self.partial_pooling,
                        dims=(grouping_name),
                    )
                    trends_betas = pm.Normal(
                        "splines_betas",
                        mu=group_array @ trends_betas_group,
                        sigma=group_array @ trends_betas_geo_sd[:, None],
                        dims=(random_dims[0], "splines"),
                    )
                    return trends_betas

                tau = pm.HalfNormal("tau", self.variability)
                trends_betas_mu = pm.GaussianRandomWalk(
                    "splines_beta_mu",
                    mu=0,
                    sigma=tau,
                    init_dist=pm.Normal.dist(0, self.initial_dist_var),
                    dims=("splines"),
                )
                # trends_betas_mu = pm.Normal("splines_betas_mu", mu=0, sigma=self.variability, dims=("splines",))
                trends_betas_sd = pm.HalfNormal(
                    "splines_betas_sd", sigma=self.group_variablility, dims=("splines",)
                )
                trends_betas = pm.Normal(
                    "splines_betas",
                    mu=trends_betas_mu,
                    sigma=trends_betas_sd,
                    dims=(*random_dims, "splines"),
                )

                return trends_betas


class EventPrior(Prior):
    type: Literal["Event"] = "Event"
    tau_sd: Optional[PositiveFloat] = 1.0
    holiday_sd: Optional[PositiveFloat] = 1.0
    partial_pooling: Optional[PositiveFloat] = 0.75

    def build(
        self,
        var_name: str,
        n_splines: int,
        random_dims: Optional[List[str]] = None,
        grouping_map: Optional[Union[Dict[str, List[str]], str]] = None,
        grouping_name: Optional[str] = None,
        model=None,
    ): ...


class SeasonPrior(ControlCoeffPrior):
    type: Literal["Season"] = "Season"

    def build(
        self,
        variable_name,
        n_components,
        random_dims=None,
        fixed_dims=None,
        pooling_sigma=1,
        model=None,
    ):
        model = pm.modelcontext(model)
        super_ = super()
        model.add_coord(variable_name, np.arange(n_components))
        with model:
            dims = var_dims()
            coeffs = pm.Deterministic(
                f"{variable_name}_coeff_estimate",
                pt.concatenate(
                    [
                        super_.build(
                            f"{variable_name}_{comp}",
                            random_dims=random_dims,
                            fixed_dims=fixed_dims,
                            pooling_sigma=pooling_sigma,
                            model=model,
                        )[..., None]
                        for comp in range(n_components)
                    ],
                    axis=-1,
                ),
                dims=(*dims[:-1], f"{variable_name}"),
            )
        return coeffs
