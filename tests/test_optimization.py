"""
Unit tests for hyperparameter optimization module.

Tests Optuna-based hyperparameter tuning functionality.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.optimization import HyperparameterOptimizer


class TestHyperparameterOptimizer:
    """Test suite for HyperparameterOptimizer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')

        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })

        y = pd.Series(
            2 * X['feature1'] + 3 * X['feature2'] + np.random.randn(n_samples) * 0.5
        )

        return X, y

    def test_optimizer_initialization(self):
        """Test optimizer initialization with default parameters."""
        optimizer = HyperparameterOptimizer()

        assert optimizer.n_trials == 50
        assert optimizer.cv_splits == 3
        assert optimizer.random_state == 42
        assert isinstance(optimizer.best_params, dict)

    def test_optimizer_custom_parameters(self):
        """Test optimizer initialization with custom parameters."""
        optimizer = HyperparameterOptimizer(n_trials=20, cv_splits=5, random_state=123)

        assert optimizer.n_trials == 20
        assert optimizer.cv_splits == 5
        assert optimizer.random_state == 123

    def test_optimize_xgboost(self, sample_data):
        """Test XGBoost hyperparameter optimization."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(n_trials=5, cv_splits=2)

        result = optimizer.optimize_xgboost(X, y, metric='rmse')

        assert 'best_params' in result
        assert 'best_score' in result
        assert 'n_trials' in result
        assert result['n_trials'] == 5
        assert isinstance(result['best_params'], dict)
        assert 'n_estimators' in result['best_params']
        assert 'max_depth' in result['best_params']
        assert 'learning_rate' in result['best_params']

    def test_optimize_lightgbm(self, sample_data):
        """Test LightGBM hyperparameter optimization."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(n_trials=5, cv_splits=2)

        result = optimizer.optimize_lightgbm(X, y, metric='rmse')

        assert 'best_params' in result
        assert 'best_score' in result
        assert isinstance(result['best_params'], dict)
        assert 'n_estimators' in result['best_params']
        assert 'learning_rate' in result['best_params']

    def test_optimize_random_forest(self, sample_data):
        """Test Random Forest hyperparameter optimization."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(n_trials=5, cv_splits=2)

        result = optimizer.optimize_random_forest(X, y, metric='rmse')

        assert 'best_params' in result
        assert 'best_score' in result
        assert isinstance(result['best_params'], dict)
        assert 'n_estimators' in result['best_params']
        assert 'max_depth' in result['best_params']

    def test_optimize_model_wrapper(self, sample_data):
        """Test the general optimize_model method."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(n_trials=3, cv_splits=2)

        # Test XGBoost
        result = optimizer.optimize_model('xgboost', X, y, metric='mae')
        assert 'best_params' in result

        # Test invalid model type
        result = optimizer.optimize_model('invalid_model', X, y)
        assert 'error' in result

    def test_different_metrics(self, sample_data):
        """Test optimization with different metrics."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(n_trials=3, cv_splits=2)

        # Test RMSE
        result_rmse = optimizer.optimize_xgboost(X, y, metric='rmse')
        assert result_rmse['best_score'] > 0

        # Test MAE
        result_mae = optimizer.optimize_xgboost(X, y, metric='mae')
        assert result_mae['best_score'] > 0

        # Test R2 (should be negative for minimization)
        result_r2 = optimizer.optimize_xgboost(X, y, metric='r2')
        assert 'best_score' in result_r2

    def test_optimization_summary(self, sample_data):
        """Test getting optimization summary."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(n_trials=3, cv_splits=2)

        optimizer.optimize_xgboost(X, y)
        optimizer.optimize_lightgbm(X, y)

        summary = optimizer.get_optimization_summary()

        assert 'best_parameters' in summary
        assert 'models_optimized' in summary
        assert 'total_models' in summary
        assert len(summary['models_optimized']) == 2
        assert 'xgboost' in summary['best_parameters']
        assert 'lightgbm' in summary['best_parameters']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])