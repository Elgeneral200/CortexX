"""
Unit tests for prediction intervals and backtesting modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.intervals import PredictionIntervals
from src.models.backtesting import Backtester
from sklearn.ensemble import RandomForestRegressor


class TestPredictionIntervals:
    """Test suite for PredictionIntervals class."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randn(n) * 10 + 50
        y_pred = y_true + np.random.randn(n) * 2
        future_pred = np.random.randn(30) * 10 + 50

        return y_true, y_pred, future_pred

    def test_intervals_initialization(self):
        """Test interval calculator initialization."""
        pi = PredictionIntervals(confidence_level=0.95)

        assert pi.confidence_level == 0.95
        # Use approximate comparison for floating point
        assert abs(pi.alpha - 0.05) < 1e-10

    def test_residual_intervals(self, sample_predictions):
        """Test residual-based interval calculation."""
        y_true, y_pred, future_pred = sample_predictions
        pi = PredictionIntervals()

        result = pi.calculate_residual_intervals(y_true, y_pred, future_pred)

        assert 'predictions' in result
        assert 'lower_bound' in result
        assert 'upper_bound' in result
        assert 'interval_width' in result
        assert len(result['lower_bound']) == len(future_pred)
        assert np.all(result['lower_bound'] < result['upper_bound'])

    def test_quantile_intervals(self, sample_predictions):
        """Test quantile-based interval calculation."""
        y_true, y_pred, future_pred = sample_predictions
        pi = PredictionIntervals()

        result = pi.calculate_quantile_intervals(y_true, y_pred, future_pred)

        assert 'predictions' in result
        assert 'lower_bound' in result
        assert 'upper_bound' in result
        assert len(result['predictions']) == len(future_pred)

    def test_bootstrap_intervals(self, sample_predictions):
        """Test bootstrap interval calculation."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 3), columns=['f1', 'f2', 'f3'])
        y_train = pd.Series(np.random.randn(100))
        X_future = pd.DataFrame(np.random.randn(30, 3), columns=['f1', 'f2', 'f3'])

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        pi = PredictionIntervals()
        result = pi.calculate_bootstrap_intervals(
            model, X_train, y_train, X_future, n_iterations=10
        )

        assert 'predictions' in result
        assert 'lower_bound' in result
        assert 'upper_bound' in result
        assert len(result['predictions']) == len(X_future)

    def test_interval_coverage_evaluation(self, sample_predictions):
        """Test interval coverage evaluation."""
        y_true, y_pred, _ = sample_predictions
        pi = PredictionIntervals()

        intervals = pi.calculate_residual_intervals(y_true[:80], y_pred[:80], y_true[80:])

        coverage = pi.evaluate_interval_coverage(
            y_true[80:],
            intervals['lower_bound'],
            intervals['upper_bound']
        )

        assert 'coverage_percentage' in coverage
        assert 'expected_coverage' in coverage
        assert 'average_interval_width' in coverage
        assert 0 <= coverage['coverage_percentage'] <= 100

    def test_create_interval_dataframe(self):
        """Test interval dataframe creation."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        intervals = {
            'predictions': np.arange(10),
            'lower_bound': np.arange(10) - 1,
            'upper_bound': np.arange(10) + 1
        }

        pi = PredictionIntervals()
        df = pi.create_interval_dataframe(dates, intervals)

        assert len(df) == 10
        assert 'date' in df.columns
        assert 'prediction' in df.columns
        assert 'lower_bound' in df.columns
        assert 'upper_bound' in df.columns


class TestBacktester:
    """Test suite for Backtester class."""

    @pytest.fixture
    def sample_ts_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')

        X = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
            'feature3': np.random.randn(n)
        })

        y = pd.Series(2 * X['feature1'] + np.random.randn(n) * 0.5)

        # Return dates as a Series, not DatetimeIndex
        return X, y, pd.Series(dates)

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        bt = Backtester(initial_train_size=100, test_size=20, step_size=10)

        assert bt.initial_train_size == 100
        assert bt.test_size == 20
        assert bt.step_size == 10
        assert bt.window_type == 'expanding'

    def test_generate_windows_expanding(self):
        """Test expanding window generation."""
        bt = Backtester(initial_train_size=100, test_size=20, step_size=20, 
                       window_type='expanding')

        windows = bt.generate_windows(200)

        assert len(windows) > 0
        # Check first window
        train_slice, test_slice = windows[0]
        assert train_slice.start == 0
        assert train_slice.stop == 100
        assert test_slice.start == 100

    def test_generate_windows_rolling(self):
        """Test rolling window generation."""
        bt = Backtester(initial_train_size=100, test_size=20, step_size=20,
                       window_type='rolling')

        windows = bt.generate_windows(200)

        assert len(windows) > 0
        # In rolling window, train window size should be constant
        train_slice, test_slice = windows[-1]
        assert train_slice.stop - train_slice.start == 100

    def test_backtest_model(self, sample_ts_data):
        """Test walk-forward backtesting."""
        X, y, dates = sample_ts_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)

        bt = Backtester(initial_train_size=100, test_size=30, step_size=30)
        results = bt.backtest_model(model, X, y, dates)

        assert 'window_type' in results
        assert 'n_windows' in results
        assert 'iteration_results' in results
        assert 'aggregate_metrics' in results
        assert 'overall_rmse' in results['aggregate_metrics']
        assert 'overall_mae' in results['aggregate_metrics']
        assert 'overall_r2' in results['aggregate_metrics']

    def test_compare_models(self, sample_ts_data):
        """Test model comparison with backtesting."""
        X, y, dates = sample_ts_data

        models = {
            'RF_10': RandomForestRegressor(n_estimators=10, random_state=42),
            'RF_20': RandomForestRegressor(n_estimators=20, random_state=42)
        }

        bt = Backtester(initial_train_size=100, test_size=30, step_size=50)
        comparison = bt.compare_models(models, X, y, dates)

        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'overall_rmse' in comparison.columns
        assert 'overall_mae' in comparison.columns

    def test_forecast_plot_data_creation(self, sample_ts_data):
        """Test forecast plot data creation."""
        X, y, dates = sample_ts_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)

        bt = Backtester(initial_train_size=100, test_size=30, step_size=50)
        results = bt.backtest_model(model, X, y, dates)

        plot_df = bt.create_forecast_plot_data(results)

        assert not plot_df.empty
        assert 'date' in plot_df.columns
        assert 'actual' in plot_df.columns
        assert 'predicted' in plot_df.columns

    def test_evaluate_forecast_horizon(self, sample_ts_data):
        """Test forecast horizon evaluation."""
        X, y, dates = sample_ts_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)

        bt = Backtester(initial_train_size=100, test_size=20, step_size=20)
        results = bt.backtest_model(model, X, y, dates)

        horizon_df = bt.evaluate_forecast_accuracy_by_horizon(results)

        assert not horizon_df.empty
        assert 'iteration' in horizon_df.columns
        assert 'rmse' in horizon_df.columns
        assert 'mae' in horizon_df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])