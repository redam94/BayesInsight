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

    def contributions(self, model=None):
        """Get the contributions of the variable to the model"""
        model = pm.modelcontext(model)
        with model:
            dims = var_dims()
            try:
                var = getattr(model, f"{self.variable_name}_transformed")
            except AttributeError:
                raise AttributeError("Variable must be register before it can be used")

            try:
                coef = getattr(model, f"{self.variable_name}_coeff_estimate")
            except AttributeError:
                coef = self.build_coeff_prior()

            contributions = pm.Deterministic(
                f"{self.variable_name}_contributions", coef[..., None] * var, dims=dims
            )
        return contributions