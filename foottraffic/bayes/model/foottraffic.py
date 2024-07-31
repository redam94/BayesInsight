from pymc.util import RandomState
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from typing import Any
import jax
import numpyro
import warnings
from sklearn.utils.validation import check_array, check_X_y
numpyro.set_host_device_count(4)


from foottraffic.awb_model.models import DataSet
from pymc_experimental.model_builder import ModelBuilder


class FoottrafficModel(ModelBuilder):
    _model_type = "FoottrafficModel"

    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        pass

    def fit(
        self,
        dataset: DataSet, 
        progressbar: bool = True,
        predictor_names: list[str] | None = None,
        random_seed: RandomState = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Fit a model using the data passed as a parameter.
        Sets attrs to inference data of the model.


        Parameters
        ----------
        X : array-like if sklearn is available, otherwise array, shape (n_obs, n_features)
            The training input samples.
        y : array-like if sklearn is available, otherwise array, shape (n_obs,)
            The target values (real numbers).
        progressbar : bool
            Specifies whether the fit progressbar should be displayed
        predictor_names: List[str] = None,
            Allows for custom naming of predictors given in a form of 2dArray
            allows for naming of predictors when given in a form of np.ndarray, if not provided the predictors will be named like predictor1, predictor2...
        random_seed : RandomState
            Provides sampler with initial random seed for obtaining reproducible samples
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : az.InferenceData
            returns inference data of the fitted model.

        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(data)
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...
        """
        if predictor_names is None:
            predictor_names = []
        if y is None:
            y = np.zeros(X.shape[0])
        y = pd.DataFrame({self.output_var: y})
        self._generate_and_preprocess_model_data(X, y.values.flatten())
        self.build_model(self.X, self.y)

        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        self.idata = self.sample_model(**sampler_config)

        X_df = pd.DataFrame(X, columns=X.columns)
        combined_data = pd.concat([X_df, y], axis=1)
        assert all(combined_data.columns), "All columns must have non-empty names"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=combined_data.to_xarray())  # type: ignore

        return self.idata 
