import pytest
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from unittest.mock import MagicMock, patch

from bayesinsight.models.variablemodels import (
    VariableDetails,
    ControlVariableDetails,
    MediaVariableDetails,
    ExogVariableDetails,
    LocalTrendsVariableDetails,
    SeasonVariableDetails,
)
from bayesinsight.types.transform_types import MediaTransform, Adstock, Normalization
from bayesinsight.models.priormodel import (
    MediaCoeffPrior,
    HillPrior,
    SShapedPrior,
    ControlCoeffPrior,
)
from bayesinsight.models.likelihood import Likelihood
from bayesinsight.types.likelihood_types import LikelihoodType

# Mock the MFF class for testing
class MockMFF:
    def __init__(self, data_dict, row_ids=None):
        # Create mock data
        self.data = xr.Dataset.from_dict(data_dict)
        self.metadata = MagicMock()
        self.metadata.row_ids = row_ids or ["Region", "Period"]
        self._info = {
            "Region": {"# Unique": 3},
            "Period": {"# Unique": 10},
        }
    
    def analytic_dataframe(self, indexed=True):
        # Return DataFrame with variable data
        df = self.data.to_dataframe()
        return df


@pytest.fixture
def mock_data():
    # Create mock data for testing
    # 3 regions, 10 time periods
    regions = ["Region1", "Region2", "Region3"]
    periods = list(range(10))
    
    # Create coordinates and data
    coords = {
        "Region": regions,
        "Period": periods,
    }
    
    # Create test variables
    control_var = np.random.normal(0, 1, (3, 10))
    media_var = np.random.gamma(1, 2, (3, 10))  # Media is always positive
    response_var = np.random.poisson(10, (3, 10))
    
    # Create dataset
    data_dict = {
        "coords": coords,
        "dims": ["Region", "Period"],
        "data_vars": {
            "control_var": {
                "dims": ["Region", "Period"],
                "data": control_var,
            },
            "media_var": {
                "dims": ["Region", "Period"],
                "data": media_var,
            },
            "response_var": {
                "dims": ["Region", "Period"],
                "data": response_var,
            },
        },
    }
    
    return MockMFF(data_dict)


@pytest.fixture
def mock_pm_model():
    with pm.Model() as model:
        # Add mock coordinates
        model.add_coord("Region", ["Region1", "Region2", "Region3"])
        model.add_coord("Period", list(range(10)))
        yield model


# Test base VariableDetails class
class TestVariableDetails:
    def test_init_with_defaults(self):
        # Test initialization with default parameters
        var_details = VariableDetails(
            variable_name="test_var",
            variable_type="control",
        )
        
        assert var_details.variable_name == "test_var"
        assert var_details.variable_type == "control"
        assert var_details.normalization == Normalization.none
        assert var_details.sign is None
        assert var_details.partial_pooling_sigma == 1
    
    def test_get_variable_values(self, mock_data):
        # Test getting variable values from MFF
        var_details = VariableDetails(
            variable_name="control_var",
            variable_type="control",
        )
        
        values = var_details.get_variable_values(mock_data)
        assert isinstance(values, pd.Series)
        assert values.name == "control_var"
    
    def test_get_variable_values_error(self, mock_data):
        # Test error when variable not found
        var_details = VariableDetails(
            variable_name="nonexistent_var",
            variable_type="control",
        )
        
        with pytest.raises(ValueError) as excinfo:
            var_details.get_variable_values(mock_data)
        assert "not in analytic dataframe" in str(excinfo.value)
    
    def test_as_numpy(self, mock_data):
        # Test converting to numpy array
        var_details = VariableDetails(
            variable_name="control_var",
            variable_type="control",
        )
        
        np_array = var_details.as_numpy(mock_data)
        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (3, 10)  # 3 regions, 10 periods
    
    def test_as_xarray(self, mock_data):
        # Test converting to xarray DataArray
        var_details = VariableDetails(
            variable_name="control_var",
            variable_type="control",
        )
        
        xr_array = var_details.as_xarray(mock_data)
        assert isinstance(xr_array, xr.DataArray)
    
    def test_normalize_none(self, mock_data):
        # Test no normalization
        var_details = VariableDetails(
            variable_name="control_var",
            variable_type="control",
            normalization=Normalization.none,
        )
        
        data = var_details.as_numpy(mock_data)
        normalized = var_details.normalize(data)
        np.testing.assert_array_equal(normalized, data)
    
    @patch('pymc.math.mean')
    @patch('pytensor.tensor.var')
    def test_normalize_standardize(self, mock_var, mock_mean, mock_data):
        # Test standardization
        mock_mean.return_value.eval.return_value = 0
        mock_var.return_value = 1

        var_details = VariableDetails(
            variable_name="control_var",
            variable_type="control",
            normalization=Normalization.global_standardize,
        )
        
        data = var_details.as_numpy(mock_data)
        normalized = var_details.normalize(data)
        assert var_details.mean == 0
        assert var_details.std is not None
    
    def test_register_variable(self, mock_data, mock_pm_model):
        # Test registering variable in PyMC model
        with patch('bayesinsight.lib.utils.var_dims', return_value=("Region", "Period")):
            var_details = VariableDetails(
                variable_name="control_var",
                variable_type="control",
            )
            
            result = var_details.register_variable(mock_data, model=mock_pm_model)
            assert f"control_var" in mock_pm_model.named_vars
            assert f"control_var_transformed" in mock_pm_model.named_vars


# Test ControlVariableDetails class
class TestControlVariableDetails:
    def test_init_with_defaults(self):
        # Test initialization with default parameters
        var_details = ControlVariableDetails(
            variable_name="test_control",
            # variable_type is fixed for ControlVariableDetails
        )
        
        assert var_details.variable_name == "test_control"
        assert var_details.variable_type == "control"
        assert isinstance(var_details.coeff_prior, ControlCoeffPrior)
        assert var_details.fixed_ind_coeff_dims == []
        assert var_details.random_coeff_dims == []
    
    def test_validate_effects(self):
        # Test validation of fixed and random dimensions
        # Should pass with non-overlapping dimensions
        var_details = ControlVariableDetails(
            variable_name="test_control",
            fixed_ind_coeff_dims=["Region"],
            random_coeff_dims=["Period"],
        )
        
        # Should raise error with overlapping dimensions
        with pytest.raises(ValueError) as excinfo:
            ControlVariableDetails(
                variable_name="test_control",
                fixed_ind_coeff_dims=["Region"],
                random_coeff_dims=["Region", "Period"],
            )
        assert "must be orthogonal" in str(excinfo.value)
    
    @patch('bayesinsight.models.variablemodels.ControlVariableDetails.build_coeff_prior')
    @patch('bayesinsight.lib.utils.var_dims', return_value=("Region", "Period"))
    def test_get_contributions(self, mock_var_dims, mock_build_prior, mock_data, mock_pm_model):
        # Test getting contributions
        mock_build_prior.return_value = pm.Normal.dist(0, 1).eval()
        
        var_details = ControlVariableDetails(
            variable_name="control_var",
        )
        
        with patch.object(var_details, 'register_variable') as mock_register:
            mock_register.return_value = pm.Normal.dist(0, 1).eval()
            result = var_details.get_contributions(mock_data, model=mock_pm_model)
            assert f"control_var_contribution" in mock_pm_model.named_vars


# Test MediaVariableDetails class
class TestMediaVariableDetails:
    def test_init_with_defaults(self):
        # Test initialization with default parameters
        var_details = MediaVariableDetails(
            variable_name="test_media",
            # variable_type is fixed for MediaVariableDetails
        )
        
        assert var_details.variable_name == "test_media"
        assert var_details.variable_type == "media"
        assert isinstance(var_details.coeff_prior, MediaCoeffPrior)
        assert var_details.adstock == Adstock.delayed
        assert var_details.media_transform == MediaTransform.hill
        assert isinstance(var_details.media_transform_prior, HillPrior)
        assert var_details.sign == "positive"
    
    def test_validate_transform_and_prior(self):
        # Test validation of transform and prior compatibility
        # Test with compatible transform and prior
        var_details = MediaVariableDetails(
            variable_name="test_media",
            media_transform=MediaTransform.hill,
            media_transform_prior=HillPrior(),
        )
        
        # Test with incompatible transform and prior
        with pytest.raises(AssertionError):
            MediaVariableDetails(
                variable_name="test_media",
                media_transform=MediaTransform.hill,
                media_transform_prior=SShapedPrior(),
            )
    
    @patch('bayesinsight.lib.constants.ADSTOCK_MAP')
    @patch('bayesinsight.models.variablemodels.MediaVariableDetails.build_adstock_prior')
    def test_apply_adstock(self, mock_build_adstock, mock_adstock_map, mock_pm_model):
        # Test applying adstock transformation
        mock_build_adstock.return_value = (0.5, 0.2)  # Alpha, theta
        mock_adstock_fn = MagicMock()
        mock_adstock_map.__getitem__.return_value = mock_adstock_fn
        
        var_details = MediaVariableDetails(
            variable_name="test_media",
        )
        
        data = np.random.normal(0, 1, (3, 10))
        result = var_details.apply_adstock(data, dims=("Region", "Period"), model=mock_pm_model)
        
        mock_adstock_fn.assert_called_once()
        assert "test_media_adstock" in mock_pm_model.named_vars
    
    @patch('bayesinsight.lib.constants.MEDIA_TRANSFORM_MAP')
    @patch('bayesinsight.models.variablemodels.MediaVariableDetails.build_media_priors')
    def test_apply_shape_transform(self, mock_build_media, mock_transform_map, mock_pm_model):
        # Test applying shape transformation
        mock_build_media.return_value = (0.5, 0.2)  # K, n for Hill function
        mock_transform_fn = MagicMock()
        mock_transform_map.__getitem__.return_value = mock_transform_fn
        
        var_details = MediaVariableDetails(
            variable_name="test_media",
        )
        var_details._MediaVariableDetails__group_nonzero_median = xr.DataArray([1, 1, 1])
        var_details._MediaVariableDetails__group_nonzero_mean = xr.DataArray([1, 1, 1])
        
        data = np.random.normal(0, 1, (3, 10))
        result = var_details.apply_shape_transform(data, dims=("Region", "Period"), model=mock_pm_model)
        
        mock_transform_fn.assert_called_once()
        assert "test_media_media_transform" in mock_pm_model.named_vars
    
    @patch('bayesinsight.models.variablemodels.MediaVariableDetails.build_coeff_prior')
    @patch('bayesinsight.models.variablemodels.MediaVariableDetails.apply_shape_transform')
    @patch('bayesinsight.models.variablemodels.MediaVariableDetails.apply_adstock')
    @patch('bayesinsight.lib.utils.var_dims', return_value=("Region", "Period"))
    def test_get_contributions(self, mock_var_dims, mock_adstock, mock_shape, mock_build_prior, mock_data, mock_pm_model):
        # Test getting contributions
        mock_build_prior.return_value = pm.Normal.dist(0, 1).eval()
        mock_shape.return_value = pm.Normal.dist(0, 1).eval()
        mock_adstock.return_value = pm.Normal.dist(0, 1).eval()
        
        var_details = MediaVariableDetails(
            variable_name="media_var",
        )
        
        with patch.object(var_details, 'register_variable') as mock_register:
            mock_register.return_value = pm.Normal.dist(0, 1).eval()
            result = var_details.get_contributions(mock_data, model=mock_pm_model)
            
            mock_register.assert_called_once()
            mock_shape.assert_called_once()
            mock_adstock.assert_called_once()
            assert "media_var_contribution" in mock_pm_model.named_vars


# Test ExogVariableDetails class
class TestExogVariableDetails:
    def test_init_with_defaults(self):
        # Test initialization with default parameters
        var_details = ExogVariableDetails(
            variable_name="test_response",
            # variable_type is fixed for ExogVariableDetails
        )
        
        assert var_details.variable_name == "test_response"
        assert var_details.variable_type == "exog"
        assert var_details.likelihood.type == LikelihoodType.poisson
    
    @patch('bayesinsight.models.likelihood.Likelihood.build')
    def test_build_likelihood(self, mock_likelihood_build, mock_pm_model):
        # Test building likelihood function
        var_details = ExogVariableDetails(
            variable_name="response_var",
        )
        
        estimate = pm.Normal.dist(0, 1).eval()
        obs = pm.Normal.dist(0, 1).eval()
        
        var_details.build_likelihood(estimate, obs, model=mock_pm_model)
        mock_likelihood_build.assert_called_once()
    
    def test_register_variable(self, mock_data, mock_pm_model):
        # Test registering exogenous variable
        with patch('bayesinsight.lib.utils.var_dims', return_value=("Region", "Period")):
            var_details = ExogVariableDetails(
                variable_name="response_var",
            )
            
            result = var_details.register_variable(mock_data, model=mock_pm_model)
            assert "response_var" in mock_pm_model.named_vars
            # Should not create transformed variable for exog
            assert "response_var_transformed" not in mock_pm_model.named_vars


# Test LocalTrendsVariableDetails class
class TestLocalTrendsVariableDetails:
    def test_init_with_defaults(self):
        # Test initialization with default parameters
        var_details = LocalTrendsVariableDetails(
            variable_name="test_trend",
            # variable_type is fixed for LocalTrendsVariableDetails
        )
        
        assert var_details.variable_name == "test_trend"
        assert var_details.variable_type == "localtrend"
        assert var_details.num_knots == 6
        assert var_details.order == 3  # Cubic splines
        assert var_details.random_coeff_dims == []
    
    @patch('bayesinsight.models.variablemodels.spline_matrix')
    def test_register_variable(self, mock_spline_matrix, mock_data, mock_pm_model):
        # Test registering local trend variable
        mock_spline_mat = np.random.normal(0, 1, (10, 6))  # 10 periods, 6 knots
        mock_spline_matrix.return_value = mock_spline_mat
        
        var_details = LocalTrendsVariableDetails(
            variable_name="trend_var",
        )
        
        result = var_details.register_variable(mock_data, model=mock_pm_model)
        assert "trend_var_spline_matrix" in mock_pm_model.named_vars
        assert "trend_var_splines" in mock_pm_model.coords
    
    def test_register_variable_type_error(self):
        # Test error when data is not MFF
        var_details = LocalTrendsVariableDetails(
            variable_name="trend_var",
        )
        
        with pytest.raises(TypeError) as excinfo:
            var_details.register_variable(np.array([1, 2, 3]))
        assert "requires MFF data" in str(excinfo.value)
    
    @patch('bayesinsight.models.priormodel.LocalTrendPrior.build')
    def test_build_coeff_prior(self, mock_prior_build, mock_pm_model):
        # Test building coefficient priors
        var_details = LocalTrendsVariableDetails(
            variable_name="trend_var",
            random_coeff_dims=["Region"],
        )
        
        var_details.build_coeff_prior(n_splines=6, model=mock_pm_model)
        mock_prior_build.assert_called_once_with(
            "trend_var", 
            n_splines=6, 
            random_dims=["Region"], 
            grouping_map=None, 
            grouping_name=None,
            model=mock_pm_model
        )


# Test SeasonVariableDetails class
class TestSeasonVariableDetails:
    def test_init_with_defaults(self):
        # Test initialization with default parameters
        var_details = SeasonVariableDetails(
            variable_name="test_season",
            # variable_type is fixed for SeasonVariableDetails
        )
        
        assert var_details.variable_name == "test_season"
        assert var_details.variable_type == "season"
        assert var_details.n_fourier == 5
        assert var_details.period == 365.25 / 7  # Weekly seasonality
    
    def test_fourier_components(self, mock_data):
        # Test calculation of Fourier components
        var_details = SeasonVariableDetails(
            variable_name="season_var",
            n_fourier=3,  # Use fewer components for test
        )
        
        components = var_details._SeasonVariableDetails__fourier_components(mock_data)
        assert isinstance(components, np.ndarray)
        assert components.shape == (10, 6)  # 10 periods, 6 components (cos/sin for 3 frequencies)
    
    def test_register_variable(self, mock_data, mock_pm_model):
        # Test registering seasonal variable
        var_details = SeasonVariableDetails(
            variable_name="season_var",
            n_fourier=2,  # Use fewer components for test
        )
        
        with patch('bayesinsight.lib.utils.var_dims', return_value=("Region", "Period")):
            result = var_details.register_variable(mock_data, model=mock_pm_model)
            assert "season_var_data" in mock_pm_model.named_vars
            assert "season_var_transformed" in mock_pm_model.named_vars
            assert "season_var" in mock_pm_model.coords
            assert len(mock_pm_model.coords["season_var"]) == 4  # 2 frequencies * 2 components
    
    def test_register_variable_type_error(self):
        # Test error when data is not MFF
        var_details = SeasonVariableDetails(
            variable_name="season_var",
        )
        
        with pytest.raises(TypeError) as excinfo:
            var_details.register_variable(np.array([1, 2, 3]))
        assert "requires MFF data" in str(excinfo.value)
    
    @patch('bayesinsight.models.priormodel.SeasonPrior.build')
    def test_build_coeff_prior(self, mock_prior_build, mock_pm_model):
        # Test building coefficient priors
        var_details = SeasonVariableDetails(
            variable_name="season_var",
            n_fourier=3,
            random_coeff_dims=["Region"],
        )
        
        var_details.build_coeff_prior(model=mock_pm_model)
        mock_prior_build.assert_called_once_with(
            "season_var", 
            6,  # 3 frequencies * 2 components
            random_dims=["Region"], 
            fixed_dims=None,
            pooling_sigma=1,
            model=mock_pm_model
        ) 