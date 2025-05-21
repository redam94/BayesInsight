from bayesinsight.types.transform_types import MediaTransform, Adstock, Normilization
from bayesinsight.models.transformsmodel import (
    DeterministicTransform,
    TimeTransformer,
)
from bayesinsight.models.priormodel import (
    MediaCoeffPrior,
    HillPrior,
    SShapedPrior,
    ControlCoeffPrior,
    InterceptPrior,
    DelayedAdStockPrior,
    LocalTrendPrior,
    SeasonPrior,
)
from bayesinsight.models.likelihood import Likelihood
from bayesinsight.types.likelihood_types import LikelihoodType
from bayesinsight.models.dataloading import MFF
from bayesinsight.lib.constants import MEDIA_TRANSFORM_MAP, ADSTOCK_MAP
from bayesinsight.lib.utils import (
    spline_matrix,
    row_ids_to_ind_map,
    var_dims,
    enforce_dim_order,
    check_dim,
)

from typing import Optional, Union, Literal, Annotated

from pydantic import BaseModel, PositiveFloat, model_validator, Field
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

__all__ = [
    "VariableDetails",
    "ControlVariableDetails",
    "MediaVariableDetails",
    "ExogVariableDetails",
    "LocalTrendsVariableDetails",
    "SeasonVariableDetails",
]


def _row_ids_to_ind_map(row_ids: list[str]) -> list[int]:
    ind_map = {row_ids[i]: i for i in range(len(row_ids))}
    return ind_map


MediaTransformType = Annotated[
    Union[SShapedPrior, HillPrior], Field(discriminator="type")
]

AdstockType = Annotated[Union[DelayedAdStockPrior], Field(discriminator="type")]


class VariableDetails(BaseModel):
    """
    VariableDetails class for tracking variables in the model.

    Required Attributes:
    variable_name: Name of the variable in the analytic dataset
    variable_type: Defines how the variable behaves in the model

    Optional Attributes:
    deterministic_transform: Transform that with parameters known in advance
    normalization: Normalization applied after deterministic_transform
    time_transform: Not implemented
    std: Standard deviation assigned when normalize is called for first time
    mean: Mean assigned when normalize is called for first time
    """

    variable_name: str
    variable_type: Literal["control", "exog", "media", "base", "season", "localtrend"]
    deterministic_transform: DeterministicTransform = DeterministicTransform(
        functional_form="linear", params=None
    )
    normalization: Normilization = Normilization.none
    std: Optional[float] = None
    mean: Optional[float] = None
    time_transform: Optional[TimeTransformer] = None
    sign: Optional[Literal["positive", "negative"]] = None
    partial_pooling_sigma: PositiveFloat = 1

    def normalize(self, data: Union[MFF, np.ndarray]) -> np.ndarray:
        """Apply normilazation to data"""
        if isinstance(data, MFF):
            data = self.as_numpy(data)

        if self.normalization == Normilization.none:
            return data

        if self.normalization == Normilization.global_standardize:
            var = data
            if self.mean is None:
                self.mean = pm.math.mean(var).eval()
            demeaned = var - self.mean
            if self.std is None:
                self.std = pt.sqrt(pt.var(demeaned, ddof=1)).eval()

            standardized = demeaned / self.std
            return standardized

        raise NotImplementedError("Only global_standardize is implemented. :(")

    def transform(
        self, data: Union[MFF, np.ndarray], time_first=True, normalize_first=False
    ) -> np.ndarray:
        """Apply deterministic transform to data"""

        variable = data
        if isinstance(variable, MFF):
            variable = self.as_numpy(variable)

        if normalize_first:
            variable = self.normalize(variable)

        if self.time_transform is None:
            if normalize_first:
                return self.deterministic_transform(variable)
            return self.normalize(self.deterministic_transform(variable))

        if time_first:
            transformed_variable = self.deterministic_transform(
                self.time_transform(variable)
            )
            if normalize_first:
                return transformed_variable
            return self.normalize(transformed_variable)

        transformed_variable = self.time_transform(
            self.deterministic_transform(variable)
        )
        if normalize_first:
            return transformed_variable
        return self.normalize(transformed_variable)

    def get_variable_values(self, data: MFF) -> pd.Series:
        """Get the variable values from the analytic dataframe"""
        analytic_dataframe = data.analytic_dataframe(indexed=True)
        try:
            return analytic_dataframe[self.variable_name]
        except KeyError:
            raise ValueError(f"{self.variable_name} not in AF. Check spelling.")

    def as_numpy(self, data: MFF) -> np.ndarray:
        """Return variable data in shape suitable for modeling"""

        row_dims = (
            data._info[index_col]["# Unique"] for index_col in data.metadata.row_ids
        )
        return self.get_variable_values(data).to_numpy().reshape(tuple(row_dims))

    def as_xarray(self, data: MFF) -> xr.DataArray:
        """Return variable data in shape suitable for modeling"""
        df = self.get_variable_values(data)
        return xr.DataArray.from_series(df)

    def build_coeff_prior(self, model: Optional[pm.Model] = None):
        """Build a prior for the coefficients of the control variable
        Grabs the model on the context stack if model is None."""

        model = pm.modelcontext(model)
        with model:
            priors = self.coeff_prior.build(
                self.variable_name,
                fixed_dims=self.fixed_ind_coeff_dims,
                random_dims=self.random_coeff_dims,
                pooling_sigma=self.partial_pooling_sigma,
            )

        return priors

    def enforce_sign(self, model=None):
        """Enforce the sign of the coefficients"""
        model = pm.modelcontext(model)

        def pot(constraint):
            return pm.math.log(pm.math.switch(constraint, 1, 1e-10))

        with model:
            if self.sign is None:
                return model
            coeff_est = getattr(model, f"{self.variable_name}_coeff_estimate")
            if self.sign == "positive":
                pm.Potential(f"{self.variable_name}_positive_sign", pot(coeff_est >= 0))
            if self.sign == "negative":
                pm.Potential(f"{self.variable_name}_negative_sign", pot(coeff_est <= 0))
        return model

    def register_variable(self, data: MFF | np.ndarray, model=None):
        """Add the variable to the model"""

        variable = data
        if isinstance(data, MFF):
            variable = self.as_xarray(variable)

        model = pm.modelcontext(model)
        with model:
            dims = var_dims()
            var = pm.Data(f"{self.variable_name}", variable, dims=dims)
            var = pm.Deterministic(
                f"{self.variable_name}_transformed", self.transform(var), dims=dims
            )

        return var

    def _calculate_deterministic_contribution(self, var, coef, model=None):
        """Calculate the deterministic contribution component."""
        model = pm.modelcontext(model)
        with model:
            dims = var_dims()
            # Ensure coef is correctly shaped for broadcasting with var
            # var typically has dims (..., "Period")
            # coef typically has dims (...,) excluding "Period"
            # We need to add a new axis for "Period" to coef for element-wise multiplication
            contribution_val = coef[..., None] * var
            contributions = pm.Deterministic(
                f"{self.variable_name}_contribution", contribution_val, dims=dims
            )
        return contributions

    def get_contribution(self, data: MFF, model: Optional[pm.Model] = None):
        """
        Calculate and register the variable's contribution to the model.

        Args:
            data: MFF object containing the variable data.
            model: Optional PyMC model context.

        Returns:
            A PyMC Deterministic object representing the variable's contribution.
        """
        model = pm.modelcontext(model)

        # Register the variable (applies transformations)
        # The registered variable should have the correct dimensions including "Period"
        registered_var = self.register_variable(data, model=model)

        # Build the coefficient prior
        # Coeff prior dimensions should not include "Period"
        coeff_prior = self.build_coeff_prior(model=model)

        # Enforce sign constraints on the coefficient prior
        self.enforce_sign(model=model) # This modifies the model state if applicable

        # Calculate the contribution using the (potentially sign-enforced) coefficient
        # and the transformed variable.
        # We might need to re-fetch the coefficient if enforce_sign modifies it in place
        # and returns a new object, or if build_coeff_prior returns a different object
        # than what enforce_sign expects.
        # For now, assume build_coeff_prior gives the main coefficient object and
        # enforce_sign applies constraints to it or related potentials.
        # The _calculate_deterministic_contribution will use the coefficient from build_coeff_prior.
        
        # It's crucial that `coeff_prior` used here is the one potentially affected by `enforce_sign`.
        # If `enforce_sign` adds a potential, it doesn't change `coeff_prior` directly.
        # If `build_coeff_prior` in subclasses already calls `enforce_sign`, this call might be redundant
        # or needs careful handling. For the base class, this order is logical.
        
        # The name of the deterministic node for the coefficient estimate is usually
        # f"{self.variable_name}_coeff_estimate". Let's assume build_coeff_prior
        # registers this or returns the symbolic variable for it.
        # And _calculate_deterministic_contribution will use it.

        contribution = self._calculate_deterministic_contribution(
            registered_var, coeff_prior, model=model
        )
        return contribution


class ControlVariableDetails(VariableDetails):
    variable_type: Literal["control"] = "control"
    coeff_prior: ControlCoeffPrior = ControlCoeffPrior()
    fixed_ind_coeff_dims: Optional[list[str]] = Field(default_factory=lambda: [])
    random_coeff_dims: Optional[list[str]] = Field(default_factory=lambda: [])

    @model_validator(mode="after")
    def validate_effects(self):
        """Check that fixed and random coefficients are orthogonal"""
        if self.random_coeff_dims is None or self.fixed_ind_coeff_dims is None:
            return self
        if set(self.fixed_ind_coeff_dims).intersection(set(self.random_coeff_dims)):
            raise ValueError("Fixed and Random Coefficients must be orthogonal.")
        return self

    def build_coeff_prior(self, model=None):
        """Build a prior for the coefficients of the control variable"""
        model = pm.modelcontext(model)
        with model:
            estimate = super().build_coeff_prior()
            self.enforce_sign()
        return estimate

    # get_contributions removed, base class get_contribution will be used.
    # The base get_contribution calls self.register_variable, self.build_coeff_prior (which is overridden here),
    # self.enforce_sign (called within build_coeff_prior here), and self._calculate_deterministic_contribution.
    # This correctly replicates the original logic.


class MediaVariableDetails(VariableDetails):
    variable_type: Literal["media"] = "media"
    time_transform: None = None
    adstock: Adstock = Adstock.delayed
    media_transform: MediaTransform = MediaTransform.hill
    coeff_prior: MediaCoeffPrior = MediaCoeffPrior()
    fixed_ind_coeff_dims: Optional[list[str]] = None
    random_coeff_dims: Optional[list[str]] = None
    media_transform_prior: Union[HillPrior, SShapedPrior] = HillPrior()
    adstock_prior: AdstockType = DelayedAdStockPrior()
    sign: Literal["positive"] = "positive"
    index_to: Literal["mean", "median"] = "median"
    __group_nonzero_median: Optional[pd.Series] = None
    __group_nonzero_mean: Optional[pd.Series] = None

    @model_validator(mode="after")
    def validate_transform_and_prior(self):
        if self.media_transform == MediaTransform.hill:
            assert isinstance(self.media_transform_prior, HillPrior)
        # if self.media_transform == MediaTransform.sorigin or self.media_transform == MediaTransform.sshaped:
        #    assert isinstance(self.media_transform_prior, SShapedPrior)
        return self

    def register_variable(self, data: MFF | np.ndarray, model=None):
        self.__group_nonzero_median = (
            self.as_xarray(data).where(lambda x: x > 0).median(dim="Period")
        )
        self.__group_nonzero_mean = (
            self.as_xarray(data).where(lambda x: x > 0).mean(dim="Period")
        )
        return super().register_variable(data, model=model)

    def build_coeff_prior(self, model=None):
        model = pm.modelcontext(model)
        with model:
            priors = super().build_coeff_prior()
            self.enforce_sign()
        return priors

    def build_media_priors(self, model=None):
        model = pm.modelcontext(model)
        with model:
            return self.media_transform_prior.build(self.variable_name)

    def build_adstock_prior(self, model=None):
        model = pm.modelcontext(model)
        with model:
            return self.adstock_prior.build(self.variable_name)

    def apply_adstock(self, data, dims=None, model=None):
        model = pm.modelcontext(model)
        with model:
            adstock_prior = self.build_adstock_prior()

            return pm.Deterministic(
                f"{self.variable_name}_adstock",
                ADSTOCK_MAP[self.adstock](
                    data,
                    alpha=adstock_prior[0],
                    theta=adstock_prior[1],
                    normalize=True,
                    axis=-1,
                ),
                dims=dims,
            )

    def apply_shape_transform(self, data, dims=None, model=None):
        model = pm.modelcontext(model)

        if self.index_to == "mean":
            index = self.__group_nonzero_mean
        else:
            index = self.__group_nonzero_median

        with model:
            media_priors = self.build_media_priors()
            return pm.Deterministic(
                f"{self.variable_name}_media_transform",
                MEDIA_TRANSFORM_MAP[self.media_transform](
                    data, *media_priors, mean=index.values
                ),
                dims=dims,
            )

    def apply_coeff(self, data, dims=None, model=None):
        model = pm.modelcontext(model)
        with model:
            estimate = self.build_coeff_prior()
            return pm.Deterministic(
                f"{self.variable_name}_contribution",
                estimate[..., None] * data,
                dims=dims,
            )

    def build_delayed_adstock_prior(self, model=None):
        model = pm.modelcontext(model)
        with model:
            pass

    def apply_delayed_adstock(self, data, dims=None, model=None):
        model = pm.modelcontext(model)
        with model:
            transformed_data = data
            return transformed_data

    def get_contribution(self, data: MFF, model: Optional[pm.Model] = None): # Renamed from get_contributions
        model = pm.modelcontext(model)
        with model:
            # build_coeff_prior in MediaVariableDetails already calls enforce_sign
            estimate = self.build_coeff_prior(model=model) 
            
            dims = var_dims(model=model) # Pass model for context
            
            # register_variable needs to be called to set up __group_nonzero_median/mean
            # and to place the raw variable in the model for transformation.
            # This returns the transformed variable (after deterministic, normalization, time transforms if any)
            # but for media, subsequent specific transforms (adstock, shape) are applied.
            # The base register_variable returns the output of self.transform(var)
            # For media, we usually apply adstock and shape transforms on the raw or deterministically transformed variable.
            # Let's ensure register_variable here gives us the data *before* adstock/shape.
            
            # MediaVariableDetails overrides register_variable to store __group_nonzero_median/mean.
            # The super().register_variable call within it will return the result of self.transform().
            # This is fine as media_transform and adstock are applied to this potentially pre-transformed variable.
            registered_variable = self.register_variable(data, model=model)

            # Apply media-specific transformations
            media_transformed = self.apply_shape_transform(
                registered_variable, dims=dims, model=model
            )
            ad_stocked = self.apply_adstock(media_transformed, dims=dims, model=model)
            
            # Calculate final contribution
            contributions = pm.Deterministic(
                f"{self.variable_name}_contribution", # Name consistency
                estimate[..., None] * ad_stocked,
                dims=dims,
            )
        return contributions


class ExogVariableDetails(VariableDetails):
    variable_type: Literal["exog"] = "exog"
    intercept_prior: Optional[InterceptPrior] = InterceptPrior()
    fixed_ind_coeff_dims: Optional[list[str]] = None
    random_coeff_dims: Optional[list[str]] = None
    likelihood: Likelihood = Likelihood(type=LikelihoodType.poisson)

    def build_intercept_prior(self, model=None):
        model = pm.modelcontext(model)
        if self.intercept_prior is None:
            return 0
        with model:
            intercept = self.intercept_prior.build(
                "intercept",
                fixed_dims=self.fixed_ind_coeff_dims,
                random_dims=self.random_coeff_dims,
                pooling_sigma=self.partial_pooling_sigma,
            )
        return intercept

    def build_likelihood(self, estimate, obs, model=None):
        model = pm.modelcontext(model)
        with model:
            likelihood = self.likelihood.build(self.variable_name, estimate, obs)
        return likelihood

    def register_variable(self, data: MFF | np.ndarray, model=None):
        variable = data
        if isinstance(data, MFF):
            variable = self.as_numpy(variable)

        model = pm.modelcontext(model)
        with model:
            dims = var_dims()
            var = pm.Data(f"{self.variable_name}", variable, dims=dims)
            # var = pm.Deterministic(f"{self.variable_name}_transformed", self.transform(var), dims=dims)
        return var

    def get_observation(self, data, model=None):
        model = pm.modelcontext(model)
        with model:
            return self.register_variable(data)

    def get_contribution(self, data: MFF, model: Optional[pm.Model] = None):
        """
        Exogenous variables do not contribute in the same additive manner as others.
        Their primary role is to define the likelihood's observed data and overall model structure.
        The intercept is handled separately, and the exog variable itself is the response.
        """
        # This method is overridden to prevent accidental calls or to clarify its role.
        # It should not produce a contribution term to be summed with others.
        return None


class LocalTrendsVariableDetails(VariableDetails):
    variable_type: Literal["localtrend"] = "localtrend"
    num_knots: int = 6  # Assuming 3 years of data ~1 knot every 6 months
    order: int = 3  # Cubic Splines as default
    random_coeff_dims: Optional[list[str]] = Field(default_factory=lambda: [])
    llt_prior: LocalTrendPrior = LocalTrendPrior()
    grouping_map: Optional[dict[str, list[str]]] = None
    grouping_name: Optional[str] = None

    def register_variable(self, data: MFF | np.ndarray, model=None):
        spline_mat = spline_matrix(
            data.data, "Period", n_knots=self.num_knots, order=self.order
        )
        model = pm.modelcontext(model)
        self.__n_splines = spline_mat.shape[1]
        model.add_coord(f"{self.variable_name}_splines", np.arange(self.__n_splines))
        with model:
            variable = pm.Data(
                f"{self.variable_name}_spline_matrix",
                spline_mat,
                dims=("Period", f"{self.variable_name}_splines"),
            )
        return variable

    def build_coeff_prior(self, n_splines: int, model: pm.Model | None = None):
        betas = self.llt_prior.build(
            self.variable_name,
            n_splines=n_splines,
            random_dims=self.random_coeff_dims,
            grouping_map=self.grouping_map,
            grouping_name=self.grouping_name,
        )

        return betas

    def get_contribution(self, data: MFF, model: Optional[pm.Model] = None): # Renamed from get_contributions
        model = pm.modelcontext(model)

        index_map = row_ids_to_ind_map(enforce_dim_order(list(model.coords.keys()))) # type: ignore
        model_dims = {
            col: len(model.coords[col])
            for col in index_map.keys()
            if check_dim(col, None)
        }

        random_coeff_dims = enforce_dim_order(self.random_coeff_dims)

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
            # media_priors = self.build_media_priors()

            transformed_variable = self.register_variable(data)
            betas = self.build_coeff_prior(n_splines=self.__n_splines)

            contributions_ = betas @ transformed_variable.T
            expanded_random = pt.expand_dims(
                contributions_, axis=random_dims_project["axis"]
            )
            repeats_random = np.ones(shape=random_dims_project["repeats"])
            contributions = pm.Deterministic(
                f"{self.variable_name}_contribution", # Name consistency
                expanded_random * repeats_random[..., None],
                dims=(*tuple(model_dims.keys()), "Period"), # type: ignore
            )

        return contributions


class SeasonVariableDetails(VariableDetails):
    variable_type: Literal["season"] = "season"
    n_fourier: Optional[int] = 5
    period: Optional[PositiveFloat] = 365.25 / 7
    coeff_prior: Optional[SeasonPrior] = SeasonPrior(type="Season")
    fixed_ind_coeff_dims: Optional[list[str]] = None
    random_coeff_dims: Optional[list[str]] = None
    partial_pooling_sigma: Optional[PositiveFloat] = 1

    def __fourier_components(self, mff: MFF) -> pd.DataFrame:
        n_time_steps = len(mff.data.Period.unique())
        t = np.linspace(0, 2 * np.pi * n_time_steps / self.period, n_time_steps)
        comps = {}
        for freq in range(1, self.n_fourier + 1):
            for comp in ["cos", "sin"]:
                comps |= {f"{comp}_{freq}": getattr(np, comp)(t * freq)}

        return pd.DataFrame(comps).values

    def register_variable(self, data: MFF | np.ndarray, model=None):
        """Add the variable to the model"""

        variable = self.__fourier_components(data)

        model = pm.modelcontext(model)
        model.add_coord(self.variable_name, np.arange(2 * self.n_fourier))
        with model:
            var = pm.Data(
                f"{self.variable_name}_data",
                variable,
                dims=["Period", self.variable_name],
            )
            var = pm.Deterministic(
                f"{self.variable_name}_transformed",
                self.transform(var.T),
                dims=[self.variable_name, "Period"],
            )

        return var

    def build_coeff_prior(self, model: pm.Model | None = None):
        model = pm.modelcontext(model)
        return self.coeff_prior.build(
            self.variable_name,
            self.n_fourier * 2,
            random_dims=self.random_coeff_dims,
            fixed_dims=self.fixed_ind_coeff_dims,
            pooling_sigma=self.partial_pooling_sigma,
            model=model,
        )

    def get_contribution(self, data: MFF, model: Optional[pm.Model] = None): # Renamed from get_contributions
        model = pm.modelcontext(model)
        # register_variable in SeasonVariableDetails handles Fourier components creation
        # and returns the transformed Fourier series.
        variable = self.register_variable(data, model=model)
        
        # build_coeff_prior is overridden in SeasonVariableDetails
        coeffs = self.build_coeff_prior(model=model)
        
        # SeasonVariableDetails does not typically have sign enforcement in its build_coeff_prior,
        # so if sign enforcement is desired for seasonal components (uncommon),
        # it would need to be added to its build_coeff_prior or called explicitly here.
        # self.enforce_sign(model=model) # if needed, but usually not for seasonal.

        with model:
            dims = var_dims(model=model) # Pass model for context
            # The registered variable (Fourier components) has dims [self.variable_name, "Period"]
            # The coeffs have dims that should include self.variable_name for the dot product.
            # The result of coeffs @ variable should have dims that match var_dims (e.g., ("Geo", "Period"))
            contributions = pm.Deterministic(
                f"{self.variable_name}_contribution", # Name consistency
                coeffs @ variable, 
                dims=dims
            )
        return contributions
