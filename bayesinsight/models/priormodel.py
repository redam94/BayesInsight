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

from typing import Dict, Literal, Optional, Union, List, Any, TypeVar, Tuple

T = TypeVar('T')

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
    """Base class for all prior models.
    
    Prior models define the prior distributions for model parameters.
    Each prior model must implement a build method that constructs
    the prior distribution in a PyMC model context.
    """
    def build(self, var_name: str, model: Optional[pm.Model] = None, **kwargs: Any) -> Any:
        """Build the prior distribution in a PyMC model.
        
        Args:
            var_name: Name of the variable to create
            model: PyMC model to add the prior to (uses current context if None)
            **kwargs: Additional arguments specific to the prior type
            
        Returns:
            The created prior variable(s)
        """
        raise NotImplementedError("Subclasses must implement build method")
            
    @staticmethod
    def get_model_context(model: Optional[pm.Model] = None) -> pm.Model:
        """Get the model context, using the provided model or current context.
        
        Args:
            model: Optional PyMC model
            
        Returns:
            PyMC model context
        """
        return pm.modelcontext(model)


class CoeffPrior(Prior):
    """Prior for coefficient parameters.
    
    Defines a prior distribution for coefficients with fixed and random effects.
    """
    coeff_dist: Distribution
    coeff_params: Dict[str, float]

    def build(
        self, 
        var_name: str, 
        random_dims: Optional[List[str]] = None, 
        fixed_dims: Optional[List[str]] = None, 
        pooling_sigma: float = 1, 
        model: Optional[pm.Model] = None
    ) -> pt.TensorVariable:
        """Build a prior for coefficients with optional random effects.
        
        Args:
            var_name: Name of the variable
            random_dims: Dimensions for random effects
            fixed_dims: Dimensions for fixed effects
            pooling_sigma: Standard deviation for partial pooling
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            PyMC deterministic variable representing the coefficient estimates
        """
        model = self.get_model_context(model)

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
                f"{var_name}_fixed_coeff", **coeff_params, dims=fixed_ind_coeff_dims
            )

            # Enforce Shape
            expanded_fixed = pt.expand_dims(
                coeff_fixed, axis=fixed_dims_project["axis"]
            )
            repeats_fixed = np.ones(shape=fixed_dims_project["repeats"])
            if random_dims is None:
                coeff_est = pm.Deterministic(
                    f"{var_name}_coeff_estimate",
                    expanded_fixed * repeats_fixed,
                    dims=tuple(model_dims.keys()),
                )

                return coeff_est

            sigma = pm.HalfCauchy(f"{var_name}_rand_coeff_sigma", pooling_sigma)
            random_fixed = pm.Normal(
                f"{var_name}_rand_coeff", 0, 1, dims=random_coeff_dims
            )

            expanded_random = pt.expand_dims(
                random_fixed, axis=random_dims_project["axis"]
            )
            repeats_random = np.ones(shape=random_dims_project["repeats"])
            coeff_est = pm.Deterministic(
                f"{var_name}_coeff_estimate",
                expanded_fixed * repeats_fixed
                + (expanded_random * repeats_random * sigma),
                dims=tuple(model_dims.keys()),
            )
        return coeff_est


class MediaCoeffPrior(CoeffPrior):
    """Prior for media coefficient parameters.
    
    Uses lognormal distribution to ensure positive coefficients for media variables.
    """
    coeff_dist: PosDist = PosDist.lognormal
    coeff_params: Dict[str, float] = {"mu": np.log(0.05), "sigma": np.log(1.3)}


class ControlCoeffPrior(CoeffPrior):
    """Prior for control variable coefficient parameters.
    
    Uses normal distribution for control variable coefficients.
    """
    coeff_dist: Distribution = Distribution.normal
    coeff_params: Dict[str, float] = {"mu": 0, "sigma": 3}


class DelayedAdStockPrior(Prior):
    """Prior for delayed adstock parameters.
    
    Defines priors for retention rate and lag parameters
    of the delayed adstock transformation.
    """
    type: Literal["Delayed"] = "Delayed"
    retention_rate_mean: PositiveFloat = 0.2
    retention_rate_std: PositiveFloat = 10
    lag_min: PositiveFloat = 1e-4
    lag_max: PositiveFloat = 3

    def build(self, var_name: str, model: Optional[pm.Model] = None) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
        """Build priors for delayed adstock parameters.
        
        Args:
            var_name: Name prefix for the variables
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            Tuple of (retention_rate, lag) parameters as PyMC variables
        """
        model = self.get_model_context(model)
        
        with model:
            retention_rate = pm.Beta(
                f"{var_name}_retention_rate_log",
                mu=self.retention_rate_mean,
                nu=self.retention_rate_std,
            )
            lag = pm.Exponential(f"{var_name}_lag", self.lag_max)
        
        return retention_rate, lag


class HillPrior(Prior):
    """Hill function prior for media transformations.
    
    Defines prior distributions for K (half-saturation) and n (Hill coefficient)
    parameters of the Hill function.
    """
    type: Literal["Hill"] = "Hill"
    K_ave: PositiveFloat = 0.85
    K_std: PositiveFloat = 0.6
    n_ave: PositiveFloat = 1.5
    n_std: PositiveFloat = 1.2

    def build(self, var_name: str, model: Optional[pm.Model] = None) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
        """Build priors for Hill function parameters.
        
        Args:
            var_name: Name prefix for the variables
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            Tuple of (K, n) parameters as PyMC deterministic variables
        """
        model = self.get_model_context(model)
        
        # Half saturation prior helper terms
        K_ave = self.K_ave
        K_std = self.K_std
        K_ratio = K_std / K_ave + 1
        
        # Shape prior helper terms
        n_ave = self.n_ave
        n_std = self.n_std
        n_ratio = n_std / n_ave + 1
        
        with model:
            K_ = pm.Normal(f"{var_name}_K_", np.log(K_ave), np.log(K_ratio))
            n_ = pm.Normal(f"{var_name}_n_", np.log(n_ave), np.log(n_ratio))

            K = pm.Deterministic(f"{var_name}_K", pm.math.exp(K_))
            n = pm.Deterministic(f"{var_name}_n", pm.math.exp(n_))
        
        return K, n


class SShapedPrior(Prior):
    """S-Shaped function prior for media transformations.
    
    Defines prior distributions for alpha and beta parameters 
    of the S-shaped function.
    """
    type: Literal["SShaped"] = "SShaped"
    alpha_ave: PositiveFloat = 0.88
    alpha_std: PositiveFloat = 0.03
    beta_ave: PositiveFloat = 1e2
    beta_std: PositiveFloat = 1000

    def build(self, var_name: str, model: Optional[pm.Model] = None) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
        """Build priors for S-shaped function parameters.
        
        Args:
            var_name: Name prefix for the variables
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            Tuple of (alpha, beta) parameters as PyMC variables
        """
        model = self.get_model_context(model)
        
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
    """Prior for intercept parameters.
    
    Inherits from ControlCoeffPrior to use normal distribution
    for intercept coefficients.
    """
    pass


class LocalTrendPrior(Prior):
    """Local trend prior for time series components.
    
    Defines priors for spline coefficients with optional random effects
    and grouping structure.
    """
    type: Literal["LocalTrend"] = "LocalTrend"
    variability: PositiveFloat = 2.0
    group_variablility: PositiveFloat = 1.0
    partial_pooling: PositiveFloat = 0.75
    initial_dist_var: PositiveFloat = 3.0

    def build(
        self,
        var_name: str,
        n_splines: int,
        random_dims: Optional[List[str]] = None,
        grouping_map: Optional[Union[Dict[str, List[str]], str]] = None,
        grouping_name: Optional[str] = None,
        model: Optional[pm.Model] = None,
    ) -> pt.TensorVariable:
        """Build priors for local trend spline coefficients.
        
        Args:
            var_name: Name prefix for the variables
            n_splines: Number of spline basis functions
            random_dims: Dimensions for random effects
            grouping_map: Mapping of groups to dimension values
            grouping_name: Name of the grouping dimension
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            PyMC variable for spline coefficients
        """
        model = self.get_model_context(model)

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
                    tau = pm.HalfCauchy("tau", self.variability)
                    trends_betas = pm.GaussianRandomWalk(
                        "splines_betas",
                        mu=0,
                        sigma=tau,
                        init_dist=pm.Normal.dist(0, self.initial_dist_var),
                        dims=("splines"),
                    )
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
    """Event prior for modeling discrete events and holidays.
    
    Implements horseshoe priors for event effects to allow for sparse but large changes
    that may or may not repeat yearly. This is appropriate for modeling infrequent but
    potentially high-impact events.
    
    The horseshoe prior uses a global shrinkage parameter to shrink all coefficients towards zero
    and local scale parameters that allow some coefficients to escape shrinkage.
    """
    type: Literal["Event"] = "Event"
    global_scale: PositiveFloat = 0.05  # Global shrinkage parameter (smaller = stronger regularization)
    df: PositiveFloat = 1.0  # Degrees of freedom for the half-t prior
    partial_pooling: PositiveFloat = 0.5  # Amount of partial pooling between groups

    def build(
        self,
        var_name: str,
        n_splines: int,
        random_dims: Optional[List[str]] = None,
        grouping_map: Optional[Union[Dict[str, List[str]], str]] = None,
        grouping_name: Optional[str] = None,
        model: Optional[pm.Model] = None,
    ) -> pt.TensorVariable:
        """Build horseshoe priors for event effects.
        
        Args:
            var_name: Name prefix for the variables
            n_splines: Number of event types or basis functions
            random_dims: Dimensions for random effects
            grouping_map: Mapping of groups to dimension values
            grouping_name: Name of the grouping dimension
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            PyMC variable for event effects with horseshoe prior
        """
        model = self.get_model_context(model)
        
        coords = {"event_types": np.arange(n_splines)}
        
        if isinstance(grouping_map, dict):
            if not isinstance(grouping_name, str):
                raise ValueError(f"grouping_name must be defined in EventPrior for {var_name}")
                
            if not isinstance(random_dims, list) or len(random_dims) != 1:
                raise ValueError(f"random_dims for EventPrior for {var_name} must be a list with len 1")
                
            coords |= {grouping_name: set(grouping_map.keys())}
            
            group_array = np.array([
                [1 if name in group else 0 for key, group in grouping_map.items()]
                for name in model.coords[random_dims[0]]
            ])
        
        with model:
            with pm.Model(name=f"Event_{var_name}", coords=coords):
                # Base case: no random effects, just horseshoe for events
                if random_dims is None:
                    # Global scale parameter
                    tau = pm.HalfCauchy(f"global_scale", self.global_scale)
                    
                    # Local scale parameters for each event type
                    lambda_vars = pm.HalfCauchy(
                        f"local_scale", 
                        1.0, 
                        dims=("event_types",)
                    )
                    
                    # Compute the effective scale
                    lambda_tilde = tau * lambda_vars
                    
                    # Event effects with horseshoe prior
                    event_effects = pm.Normal(
                        "event_effects", 
                        mu=0, 
                        sigma=lambda_tilde,
                        dims=("event_types",)
                    )
                    
                    return event_effects
                
                # With grouping - hierarchical horseshoe
                if grouping_name is not None:
                    # Global scale parameter
                    tau = pm.HalfCauchy(f"global_scale", self.global_scale)
                    
                    # Group-level local scales
                    group_lambda = pm.HalfCauchy(
                        f"group_local_scale", 
                        1.0, 
                        dims=(grouping_name,)
                    )
                    
                    # Event-type local scales
                    event_lambda = pm.HalfCauchy(
                        f"event_local_scale", 
                        1.0, 
                        dims=("event_types",)
                    )
                    
                    # Group-event interaction local scales
                    group_event_lambda = pm.HalfCauchy(
                        f"group_event_local_scale", 
                        1.0, 
                        dims=(grouping_name, "event_types")
                    )
                    
                    # Global shrinkage
                    shrinkage = tau * group_event_lambda
                    
                    # Group-level effects
                    group_effects = pm.Normal(
                        f"group_effects",
                        mu=0, 
                        sigma=tau * group_lambda[:, None] * event_lambda,
                        dims=(grouping_name, "event_types")
                    )
                    
                    # Compute individual region effects with horseshoe prior
                    # and partial pooling to the group effects
                    pooling_scale = pm.HalfCauchy(f"pooling_scale", self.partial_pooling)
                    
                    # Apply group effects with partial pooling
                    event_effects = pm.Normal(
                        "event_effects",
                        mu=group_array @ group_effects,
                        sigma=pooling_scale,
                        dims=(random_dims[0], "event_types")
                    )
                    
                    return event_effects
                
                # With random dimensions but no grouping - multilevel horseshoe
                # Global scale parameter
                tau = pm.HalfCauchy(f"global_scale", self.global_scale)
                
                # Dimension-specific scales
                dim_lambda = pm.HalfCauchy(
                    f"dim_local_scale", 
                    1.0, 
                    dims=(*random_dims,)
                )
                
                # Event-type local scales
                event_lambda = pm.HalfCauchy(
                    f"event_local_scale", 
                    1.0, 
                    dims=("event_types",)
                )
                
                # Compute the effective scales for each dimension-event combination
                lambda_tilde = pt.reshape(dim_lambda, dim_lambda.shape + (1,)) * event_lambda
                
                # Event effects with horseshoe prior
                event_effects = pm.Normal(
                    "event_effects",
                    mu=0,
                    sigma=tau * lambda_tilde,
                    dims=(*random_dims, "event_types")
                )
                
                return event_effects


class SeasonPrior(ControlCoeffPrior):
    """Season prior for modeling seasonal components.
    
    Defines priors for seasonal components using Fourier series.
    """
    type: Literal["Season"] = "Season"

    def build(
        self,
        var_name: str,
        n_components: int,
        random_dims: Optional[List[str]] = None,
        fixed_dims: Optional[List[str]] = None,
        pooling_sigma: float = 1,
        model: Optional[pm.Model] = None,
    ) -> pt.TensorVariable:
        """Build priors for seasonal components.
        
        Args:
            var_name: Name prefix for the variables
            n_components: Number of Fourier components
            random_dims: Dimensions for random effects
            fixed_dims: Dimensions for fixed effects
            pooling_sigma: Standard deviation for partial pooling
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            PyMC deterministic variable for seasonal coefficients
        """
        model = self.get_model_context(model)
        super_ = super()
        model.add_coord(var_name, np.arange(n_components))
        
        with model:
            dims = var_dims()
            coeffs = pm.Deterministic(
                f"{var_name}_coeff_estimate",
                pt.concatenate(
                    [
                        super_.build(
                            f"{var_name}_{comp}",
                            random_dims=random_dims,
                            fixed_dims=fixed_dims,
                            pooling_sigma=pooling_sigma,
                            model=model,
                        )[..., None]
                        for comp in range(n_components)
                    ],
                    axis=-1,
                ),
                dims=(*dims[:-1], f"{var_name}"),
            )
        
        return coeffs
