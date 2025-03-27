from bayesinsight.types.transform_types import MediaTransform, Adstock, Normalization
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

from typing import Optional, Union, Literal, Annotated, List, Dict, TypeVar, Any, Tuple, cast

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


def _row_ids_to_ind_map(row_ids: list[str]) -> Dict[str, int]:
    ind_map = {row_ids[i]: i for i in range(len(row_ids))}
    return ind_map


MediaTransformType = Annotated[
    Union[SShapedPrior, HillPrior], Field(discriminator="type")
]

AdstockType = Annotated[Union[DelayedAdStockPrior], Field(discriminator="type")]


class VariableDetails(BaseModel):
    """Base class for variable details in Bayesian models.

    Provides common functionality for handling different types of variables in the model.

    Required Attributes:
        variable_name: Name of the variable in the analytic dataset
        variable_type: Defines how the variable behaves in the model

    Optional Attributes:
        deterministic_transform: Transform with parameters known in advance
        normalization: Normalization applied after deterministic_transform
        time_transform: Transform applied to time dimension
        std: Standard deviation assigned when normalize is called for first time
        mean: Mean assigned when normalize is called for first time
        sign: Enforced sign constraint on coefficients
        partial_pooling_sigma: Standard deviation for partial pooling
    """

    variable_name: str
    variable_type: Literal["control", "exog", "media", "base", "season", "localtrend"]
    deterministic_transform: DeterministicTransform = DeterministicTransform(
        functional_form="linear", params=None
    )
    normalization: Normalization = Normalization.none
    std: Optional[float] = None
    mean: Optional[float] = None
    time_transform: Optional[TimeTransformer] = None
    sign: Optional[Literal["positive", "negative"]] = None
    partial_pooling_sigma: PositiveFloat = 1
    fixed_ind_coeff_dims: Optional[List[str]] = None
    random_coeff_dims: Optional[List[str]] = None

    @staticmethod
    def get_model_context(model: Optional[pm.Model] = None) -> pm.Model:
        """Get the model context, using the provided model or current context.
        
        Args:
            model: Optional PyMC model
            
        Returns:
            PyMC model context
        """
        return pm.modelcontext(model)

    def normalize(self, data: Union[MFF, np.ndarray]) -> np.ndarray:
        """Apply normalization to data.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
            
        Raises:
            NotImplementedError: If normalization method is not implemented
        """
        if isinstance(data, MFF):
            data = self.as_numpy(data)

        if self.normalization == Normalization.none:
            return data

        if self.normalization == Normalization.global_standardize:
            var = data
            if self.mean is None:
                self.mean = pm.math.mean(var).eval()
            demeaned = var - self.mean
            if self.std is None:
                self.std = pt.sqrt(pt.var(demeaned, ddof=1)).eval()

            standardized = demeaned / self.std
            return standardized

        raise NotImplementedError("Only global_standardize is implemented.")

    def transform(
        self, data: Union[MFF, np.ndarray], time_first: bool = True, normalize_first: bool = False
    ) -> np.ndarray:
        """Apply deterministic transform to data.
        
        Args:
            data: Input data to transform
            time_first: Whether to apply time transform before deterministic transform
            normalize_first: Whether to normalize before transforms
            
        Returns:
            Transformed data
        """
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
        """Get the variable values from the analytic dataframe.
        
        Args:
            data: Model-friendly format data
            
        Returns:
            Series containing variable values
            
        Raises:
            ValueError: If variable not found in analytic dataframe
        """
        analytic_dataframe = data.analytic_dataframe(indexed=True)
        try:
            return analytic_dataframe[self.variable_name]
        except KeyError:
            raise ValueError(f"{self.variable_name} not in analytic dataframe. Check spelling.")

    def as_numpy(self, data: MFF) -> np.ndarray:
        """Return variable data in shape suitable for modeling.
        
        Args:
            data: Model-friendly format data
            
        Returns:
            NumPy array with correct dimensions
        """
        row_dims = (
            data._info[index_col]["# Unique"] for index_col in data.metadata.row_ids
        )
        return self.get_variable_values(data).to_numpy().reshape(tuple(row_dims))

    def as_xarray(self, data: MFF) -> xr.DataArray:
        """Return variable data as xarray DataArray.
        
        Args:
            data: Model-friendly format data
            
        Returns:
            xarray DataArray with correct dimensions
        """
        df = self.get_variable_values(data)
        return xr.DataArray.from_series(df)

    def build_coeff_prior(self, model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Build a prior for the coefficients of the variable.
        
        Args:
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            PyMC variable representing coefficient prior
            
        Raises:
            AttributeError: If coeff_prior is not defined
        """
        model = self.get_model_context(model)
        with model:
            priors = self.coeff_prior.build(
                self.variable_name,
                fixed_dims=self.fixed_ind_coeff_dims,
                random_dims=self.random_coeff_dims,
                pooling_sigma=self.partial_pooling_sigma,
            )

        return priors

    def enforce_sign(self, model: Optional[pm.Model] = None) -> pm.Model:
        """Enforce the sign constraint on the coefficients.
        
        Args:
            model: PyMC model to add the constraint to (uses current context if None)
            
        Returns:
            PyMC model with sign constraint added
        """
        model = self.get_model_context(model)

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

    def register_variable(self, data: Union[MFF, np.ndarray], model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Add the variable to the model.
        
        Args:
            data: Input data
            model: PyMC model to add the variable to (uses current context if None)
            
        Returns:
            PyMC variable representing the registered variable
        """
        variable = data
        if isinstance(data, MFF):
            variable = self.as_xarray(variable)

        model = self.get_model_context(model)
        with model:
            dims = var_dims()
            var = pm.Data(f"{self.variable_name}", variable, dims=dims)
            var = pm.Deterministic(
                f"{self.variable_name}_transformed", self.transform(var), dims=dims
            )

        return var

    def contributions(self, model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Get the contributions of the variable to the model.
        
        Args:
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable for contributions
            
        Raises:
            AttributeError: If variable not registered or coefficients not available
        """
        model = self.get_model_context(model)
        with model:
            dims = var_dims()
            try:
                var = getattr(model, f"{self.variable_name}_transformed")
            except AttributeError:
                raise AttributeError("Variable must be registered before it can be used")

            try:
                coef = getattr(model, f"{self.variable_name}_coeff_estimate")
            except AttributeError:
                coef = self.build_coeff_prior()

            contributions = pm.Deterministic(
                f"{self.variable_name}_contributions", coef[..., None] * var, dims=dims
            )
        return contributions


class ControlVariableDetails(VariableDetails):
    """Details for control variables in Bayesian models.
    
    Control variables represent factors that influence the dependent variable
    but are not of primary interest in the analysis.
    """
    variable_type: Literal["control"] = "control"
    coeff_prior: ControlCoeffPrior = ControlCoeffPrior()
    fixed_ind_coeff_dims: Optional[List[str]] = Field(default_factory=lambda: [])
    random_coeff_dims: Optional[List[str]] = Field(default_factory=lambda: [])

    @model_validator(mode="after")
    def validate_effects(self) -> "ControlVariableDetails":
        """Check that fixed and random coefficients are orthogonal.
        
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If fixed and random dimensions overlap
        """
        if self.random_coeff_dims is None or self.fixed_ind_coeff_dims is None:
            return self
        if set(self.fixed_ind_coeff_dims).intersection(set(self.random_coeff_dims)):
            raise ValueError("Fixed and Random Coefficients must be orthogonal.")
        return self

    def build_coeff_prior(self, model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Build a prior for the coefficients of the control variable.
        
        Args:
            model: PyMC model to add the prior to (uses current context if None)
            
        Returns:
            PyMC variable representing coefficient prior
        """
        model = self.get_model_context(model)
        with model:
            estimate = super().build_coeff_prior()
            self.enforce_sign()
        return estimate

    def get_contributions(self, data: Union[MFF, np.ndarray], model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Get the contributions of the variable to the model.
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable for contributions
        """
        model = self.get_model_context(model)
        with model:
            estimate = self.build_coeff_prior()
            dims = var_dims()
            variable = self.register_variable(data)
            contributions = pm.Deterministic(
                f"{self.variable_name}_contribution",
                estimate[..., None] * variable,
                dims=dims,
            )
        return contributions


class MediaVariableDetails(VariableDetails):
    """Details for media variables in Bayesian models.
    
    Media variables represent marketing channels or interventions
    and include adstock and media response transformations.
    """
    variable_type: Literal["media"] = "media"
    time_transform: None = None
    adstock: Adstock = Adstock.delayed
    media_transform: MediaTransform = MediaTransform.hill
    coeff_prior: MediaCoeffPrior = MediaCoeffPrior()
    fixed_ind_coeff_dims: Optional[List[str]] = None
    random_coeff_dims: Optional[List[str]] = None
    media_transform_prior: Union[HillPrior, SShapedPrior] = HillPrior()
    adstock_prior: AdstockType = DelayedAdStockPrior()
    sign: Literal["positive"] = "positive"
    index_to: Literal["mean", "median"] = "median"
    __group_nonzero_median: Optional[xr.DataArray] = None
    __group_nonzero_mean: Optional[xr.DataArray] = None

    @model_validator(mode="after")
    def validate_transform_and_prior(self) -> "MediaVariableDetails":
        """Validate that media transform and prior are compatible.
        
        Returns:
            Self for chaining
            
        Raises:
            AssertionError: If transform and prior are incompatible
        """
        if self.media_transform == MediaTransform.hill:
            assert isinstance(self.media_transform_prior, HillPrior), "Hill transform requires HillPrior"
        #if self.media_transform in [MediaTransform.sorigin, MediaTransform.sshaped]:
        #    assert isinstance(self.media_transform_prior, SShapedPrior), "S-shaped transform requires SShapedPrior"
        return self

    def register_variable(self, data: Union[MFF, np.ndarray], model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Register the media variable in the model and compute statistics.
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC variable representing the registered variable
        """
        if isinstance(data, MFF):
            xarray_data = self.as_xarray(data)
            self.__group_nonzero_median = xarray_data.where(lambda x: x > 0).median(dim="Period")
            self.__group_nonzero_mean = xarray_data.where(lambda x: x > 0).mean(dim="Period")
        
        return super().register_variable(data, model=model)

    def build_coeff_prior(self, model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Build a prior for media coefficients.
        
        Args:
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC variable representing coefficient prior
        """
        model = self.get_model_context(model)
        with model:
            priors = super().build_coeff_prior()
            self.enforce_sign()
        return priors

    def build_media_priors(self, model: Optional[pm.Model] = None) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
        """Build priors for media transformation parameters.
        
        Args:
            model: PyMC model (uses current context if None)
            
        Returns:
            Media transformation parameter priors (e.g., K, n for Hill)
        """
        model = self.get_model_context(model)
        with model:
            return self.media_transform_prior.build(self.variable_name)

    def build_adstock_prior(self, model: Optional[pm.Model] = None) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
        """Build priors for adstock parameters.
        
        Args:
            model: PyMC model (uses current context if None)
            
        Returns:
            Adstock parameter priors (e.g., retention rate, lag)
        """
        model = self.get_model_context(model)
        with model:
            return self.adstock_prior.build(self.variable_name)

    def apply_adstock(self, 
                      data: pt.TensorVariable, 
                      dims: Optional[Tuple[str, ...]] = None, 
                      model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Apply adstock transformation to the variable.
        
        Args:
            data: Input data tensor
            dims: Dimensions for the resulting variable
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable with adstock applied
        """
        model = self.get_model_context(model)
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

    def apply_shape_transform(self, 
                             data: pt.TensorVariable, 
                             dims: Optional[Tuple[str, ...]] = None, 
                             model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Apply media shape transformation (e.g., Hill, S-shaped).
        
        Args:
            data: Input data tensor
            dims: Dimensions for the resulting variable
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable with shape transform applied
        """
        model = self.get_model_context(model)

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

    def apply_coeff(self, 
                   data: pt.TensorVariable, 
                   dims: Optional[Tuple[str, ...]] = None, 
                   model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Apply coefficients to transformed data.
        
        Args:
            data: Input data tensor
            dims: Dimensions for the resulting variable
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable with coefficients applied
        """
        model = self.get_model_context(model)
        with model:
            estimate = self.build_coeff_prior()
            return pm.Deterministic(
                f"{self.variable_name}_contribution",
                estimate[..., None] * data,
                dims=dims,
            )

    def get_contributions(self, 
                         data: Union[MFF, np.ndarray], 
                         model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Get the contributions of the media variable to the model.
        
        This applies the full media modeling pipeline:
        1. Register the variable
        2. Apply shape transform (e.g., Hill)
        3. Apply adstock transformation 
        4. Apply coefficients
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable for contributions
        """
        model = self.get_model_context(model)
        with model:
            estimate = self.build_coeff_prior()
            dims = var_dims()
            transformed_variable = self.register_variable(data)
            media_transformed = self.apply_shape_transform(
                transformed_variable, dims=dims
            )
            ad_stocked = self.apply_adstock(media_transformed, dims=dims)
            contributions = pm.Deterministic(
                f"{self.variable_name}_contribution",
                estimate[..., None] * ad_stocked,
                dims=dims,
            )
        return contributions


class ExogVariableDetails(VariableDetails):
    """Details for exogenous variables in Bayesian models.
    
    Exogenous variables represent dependent variables that are modeled
    with a specific likelihood function.
    """
    variable_type: Literal["exog"] = "exog"
    intercept_prior: Optional[InterceptPrior] = InterceptPrior()
    fixed_ind_coeff_dims: Optional[List[str]] = None
    random_coeff_dims: Optional[List[str]] = None
    likelihood: Likelihood = Likelihood(type=LikelihoodType.poisson)

    def build_intercept_prior(self, model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Build a prior for the intercept parameter.
        
        Args:
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC variable representing intercept prior or 0 if no prior
        """
        model = self.get_model_context(model)
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

    def build_likelihood(self, 
                         estimate: pt.TensorVariable, 
                         obs: pt.TensorVariable, 
                         model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Build the likelihood function for the dependent variable.
        
        Args:
            estimate: Estimated value (linear predictor)
            obs: Observed values
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC likelihood variable
        """
        model = self.get_model_context(model)
        with model:
            likelihood = self.likelihood.build(self.variable_name, estimate, obs)
        return likelihood

    def register_variable(self, 
                         data: Union[MFF, np.ndarray], 
                         model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Register the exogenous variable in the model.
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC data variable
        """
        variable = data
        if isinstance(data, MFF):
            variable = self.as_numpy(variable)

        model = self.get_model_context(model)
        with model:
            dims = var_dims()
            var = pm.Data(f"{self.variable_name}", variable, dims=dims)
        return var

    def get_observation(self, 
                       data: Union[MFF, np.ndarray], 
                       model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Get the observation variable for the dependent variable.
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC data variable representing observations
        """
        model = self.get_model_context(model)
        with model:
            return self.register_variable(data)


class LocalTrendsVariableDetails(VariableDetails):
    """Details for local trend variables in Bayesian models.
    
    Local trend variables represent time-varying effects modeled
    using spline basis functions.
    """
    variable_type: Literal["localtrend"] = "localtrend"
    num_knots: int = 6  # Assuming 3 years of data ~1 knot every 6 months
    order: int = 3  # Cubic Splines as default
    random_coeff_dims: Optional[List[str]] = Field(default_factory=lambda: [])
    llt_prior: LocalTrendPrior = LocalTrendPrior()
    grouping_map: Optional[Dict[str, List[str]]] = None
    grouping_name: Optional[str] = None
    __n_splines: Optional[int] = None

    def register_variable(self, 
                         data: Union[MFF, np.ndarray], 
                         model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Register the local trend variable in the model.
        
        Creates the spline basis matrix for the time dimension.
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC data variable for spline matrix
        """
        if not isinstance(data, MFF):
            raise TypeError("LocalTrendsVariableDetails requires MFF data")
            
        spline_mat = spline_matrix(
            data.data, "Period", n_knots=self.num_knots, order=self.order
        )
        model = self.get_model_context(model)
        self.__n_splines = spline_mat.shape[1]
        model.add_coord(f"{self.variable_name}_splines", np.arange(self.__n_splines))
        with model:
            variable = pm.Data(
                f"{self.variable_name}_spline_matrix",
                spline_mat,
                dims=("Period", f"{self.variable_name}_splines"),
            )
        return variable

    def build_coeff_prior(self, 
                         n_splines: int, 
                         model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Build priors for local trend coefficients.
        
        Args:
            n_splines: Number of spline basis functions
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC variable for spline coefficients
        """
        model = self.get_model_context(model)
        betas = self.llt_prior.build(
            self.variable_name,
            n_splines=n_splines,
            random_dims=self.random_coeff_dims,
            grouping_map=self.grouping_map,
            grouping_name=self.grouping_name,
            model=model,
        )

        return betas

    def get_contributions(self, 
                         data: MFF, 
                         model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Get the contributions of the local trend to the model.
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable for contributions
        """
        model = self.get_model_context(model)

        index_map = row_ids_to_ind_map(enforce_dim_order(list(model.coords.keys())))
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
            transformed_variable = self.register_variable(data)
            betas = self.build_coeff_prior(n_splines=self.__n_splines)

            contributions_ = betas @ transformed_variable.T
            expanded_random = pt.expand_dims(
                contributions_, axis=random_dims_project["axis"]
            )
            repeats_random = np.ones(shape=random_dims_project["repeats"])
            contributions = pm.Deterministic(
                f"{self.variable_name}_contribution",
                expanded_random * repeats_random[..., None],
                dims=(*tuple(model_dims.keys()), "Period"),
            )

        return contributions


class SeasonVariableDetails(VariableDetails):
    """Details for seasonal variables in Bayesian models.
    
    Seasonal variables represent cyclic patterns modeled using
    Fourier series components.
    """
    variable_type: Literal["season"] = "season"
    n_fourier: int = 5
    period: PositiveFloat = 365.25 / 7  # Default weekly seasonality (52 weeks per year)
    coeff_prior: SeasonPrior = SeasonPrior(type="Season")
    fixed_ind_coeff_dims: Optional[List[str]] = None
    random_coeff_dims: Optional[List[str]] = None
    partial_pooling_sigma: PositiveFloat = 1

    def __fourier_components(self, mff: MFF) -> np.ndarray:
        """Calculate Fourier components for the seasonal pattern.
        
        Args:
            mff: Model-friendly format data
            
        Returns:
            Array of Fourier components (sin and cos terms)
        """
        n_time_steps = len(mff.data.Period.unique())
        t = np.linspace(0, 2 * np.pi * n_time_steps / self.period, n_time_steps)
        comps = {}
        for freq in range(1, self.n_fourier + 1):
            for comp in ["cos", "sin"]:
                comps |= {f"{comp}_{freq}": getattr(np, comp)(t * freq)}

        return pd.DataFrame(comps).values

    def register_variable(self, 
                         data: Union[MFF, np.ndarray], 
                         model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Register the seasonal variable in the model.
        
        Computes Fourier components and registers them in the model.
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable for transformed Fourier components
        """
        if not isinstance(data, MFF):
            raise TypeError("SeasonVariableDetails requires MFF data")
            
        variable = self.__fourier_components(data)

        model = self.get_model_context(model)
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

    def build_coeff_prior(self, model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Build priors for seasonal component coefficients.
        
        Args:
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC variable for seasonal coefficients
        """
        model = self.get_model_context(model)
        return self.coeff_prior.build(
            self.variable_name,
            self.n_fourier * 2,
            random_dims=self.random_coeff_dims,
            fixed_dims=self.fixed_ind_coeff_dims,
            pooling_sigma=self.partial_pooling_sigma,
            model=model,
        )

    def get_contributions(self, 
                         data: MFF, 
                         model: Optional[pm.Model] = None) -> pt.TensorVariable:
        """Get the contributions of the seasonal component to the model.
        
        Args:
            data: Input data
            model: PyMC model (uses current context if None)
            
        Returns:
            PyMC deterministic variable for contributions
        """
        model = self.get_model_context(model)
        variable = self.register_variable(data, model=model)
        coeffs = self.build_coeff_prior(model=model)
        with model:
            dims = var_dims()
            contributions = pm.Deterministic(
                f"{self.variable_name}_contribution", coeffs @ variable, dims=dims
            )
        return contributions
