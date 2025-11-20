"""
Backtesting module for CortexX sales forecasting platform.

Provides walk-forward validation and time series backtesting capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class Backtester:
    """
    A class to perform walk-forward backtesting for time series forecasting models.

    Implements expanding window and rolling window strategies to evaluate
    model performance on historical data in a realistic manner.
    """

    def __init__(self, initial_train_size: int = 100, test_size: int = 30,
                 step_size: int = 30, window_type: str = 'expanding'):
        """
        Initialize backtester with configuration.

        Args:
            initial_train_size (int): Initial training window size
            test_size (int): Number of periods to forecast in each iteration
            step_size (int): Number of periods to step forward
            window_type (str): 'expanding' (growing window) or 'rolling' (fixed window)
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.window_type = window_type
        self.logger = logging.getLogger(__name__)
        self.results = []

    def generate_windows(self, data_length: int) -> List[Tuple[slice, slice]]:
        """
        Generate train/test window splits for backtesting.

        Args:
            data_length (int): Total length of dataset

        Returns:
            List[Tuple[slice, slice]]: List of (train_slice, test_slice) tuples
        """
        windows = []
        train_end = self.initial_train_size

        while train_end + self.test_size <= data_length:
            test_end = min(train_end + self.test_size, data_length)

            if self.window_type == 'expanding':
                train_slice = slice(0, train_end)
            else:  # rolling
                train_start = max(0, train_end - self.initial_train_size)
                train_slice = slice(train_start, train_end)

            test_slice = slice(train_end, test_end)
            windows.append((train_slice, test_slice))

            train_end += self.step_size

        self.logger.info(f"Generated {len(windows)} backtesting windows")
        return windows

    def backtest_model(self, model, X: pd.DataFrame, y: pd.Series,
                      date_col: Optional[pd.Series] = None,
                      refit: bool = True) -> Dict[str, Any]:
        """
        Perform walk-forward backtesting on a model.

        Args:
            model: Model object with fit() and predict() methods
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            date_col (pd.Series): Optional date column for tracking (can be Series or DatetimeIndex)
            refit (bool): Whether to refit model in each window

        Returns:
            Dict[str, Any]: Backtesting results with metrics
        """
        try:
            self.logger.info(f"Starting {self.window_type} window backtesting...")

            windows = self.generate_windows(len(X))
            iteration_results = []

            all_actuals = []
            all_predictions = []
            all_dates = []

            # Convert date_col to list if it's a DatetimeIndex
            if date_col is not None:
                if isinstance(date_col, pd.DatetimeIndex):
                    date_list = date_col.tolist()
                else:
                    date_list = date_col.tolist()
            else:
                date_list = None

            for i, (train_slice, test_slice) in enumerate(windows):
                # Split data
                X_train, X_test = X.iloc[train_slice], X.iloc[test_slice]
                y_train, y_test = y.iloc[train_slice], y.iloc[test_slice]

                # Refit model if required
                if refit or i == 0:
                    from sklearn.base import clone
                    model_iter = clone(model)
                    model_iter.fit(X_train, y_train)
                else:
                    model_iter = model

                # Make predictions
                y_pred = model_iter.predict(X_test)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100

                # Store results
                iter_result = {
                    'iteration': i + 1,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'train_start': train_slice.start,
                    'train_end': train_slice.stop,
                    'test_start': test_slice.start,
                    'test_end': test_slice.stop
                }

                if date_list is not None:
                    iter_result['test_start_date'] = date_list[test_slice.start]
                    iter_result['test_end_date'] = date_list[test_slice.stop - 1]
                    all_dates.extend(date_list[test_slice.start:test_slice.stop])

                iteration_results.append(iter_result)
                all_actuals.extend(y_test.tolist())
                all_predictions.extend(y_pred.tolist())

                self.logger.info(f"Window {i+1}/{len(windows)}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

            # Calculate aggregate metrics
            all_actuals = np.array(all_actuals)
            all_predictions = np.array(all_predictions)

            aggregate_metrics = {
                'overall_rmse': np.sqrt(mean_squared_error(all_actuals, all_predictions)),
                'overall_mae': mean_absolute_error(all_actuals, all_predictions),
                'overall_r2': r2_score(all_actuals, all_predictions),
                'overall_mape': np.mean(np.abs((all_actuals - all_predictions) / np.where(all_actuals != 0, all_actuals, 1))) * 100,
                'avg_rmse': np.mean([r['rmse'] for r in iteration_results]),
                'avg_mae': np.mean([r['mae'] for r in iteration_results]),
                'avg_r2': np.mean([r['r2'] for r in iteration_results]),
                'std_rmse': np.std([r['rmse'] for r in iteration_results]),
                'std_mae': np.std([r['mae'] for r in iteration_results]),
                'std_r2': np.std([r['r2'] for r in iteration_results])
            }

            results = {
                'window_type': self.window_type,
                'n_windows': len(windows),
                'iteration_results': iteration_results,
                'aggregate_metrics': aggregate_metrics,
                'all_actuals': all_actuals,
                'all_predictions': all_predictions,
                'all_dates': all_dates if date_list is not None else None
            }

            self.results.append(results)
            self.logger.info(f"Backtesting complete. Overall RMSE: {aggregate_metrics['overall_rmse']:.4f}")

            return results

        except Exception as e:
            self.logger.error(f"Error in backtesting: {str(e)}")
            return {'error': str(e)}

    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, 
                      y: pd.Series, date_col: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compare multiple models using backtesting.

        Args:
            models (Dict[str, Any]): Dictionary of model_name: model_object
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            date_col (pd.Series): Optional date column

        Returns:
            pd.DataFrame: Comparison table with metrics for each model
        """
        try:
            self.logger.info(f"Comparing {len(models)} models with backtesting...")

            comparison_results = []

            for model_name, model in models.items():
                self.logger.info(f"Backtesting {model_name}...")
                result = self.backtest_model(model, X, y, date_col)

                if 'error' not in result:
                    metrics = result['aggregate_metrics']
                    comparison_results.append({
                        'model': model_name,
                        'overall_rmse': metrics['overall_rmse'],
                        'overall_mae': metrics['overall_mae'],
                        'overall_r2': metrics['overall_r2'],
                        'overall_mape': metrics['overall_mape'],
                        'avg_rmse': metrics['avg_rmse'],
                        'std_rmse': metrics['std_rmse'],
                        'n_windows': result['n_windows']
                    })

            comparison_df = pd.DataFrame(comparison_results)
            if not comparison_df.empty:
                comparison_df = comparison_df.sort_values('overall_rmse')

            self.logger.info("Model comparison complete")
            return comparison_df

        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()

    def create_forecast_plot_data(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create DataFrame for plotting backtesting forecasts.

        Args:
            results (Dict[str, Any]): Backtesting results from backtest_model()

        Returns:
            pd.DataFrame: DataFrame with dates, actuals, and predictions
        """
        try:
            if 'error' in results:
                return pd.DataFrame()

            if results.get('all_dates') is not None:
                df = pd.DataFrame({
                    'date': results['all_dates'],
                    'actual': results['all_actuals'],
                    'predicted': results['all_predictions']
                })
            else:
                df = pd.DataFrame({
                    'actual': results['all_actuals'],
                    'predicted': results['all_predictions']
                })
                df['index'] = df.index

            return df

        except Exception as e:
            self.logger.error(f"Error creating plot data: {str(e)}")
            return pd.DataFrame()

    def evaluate_forecast_accuracy_by_horizon(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate forecast accuracy by forecast horizon.

        Analyzes how accuracy changes as we forecast further into the future.

        Args:
            results (Dict[str, Any]): Backtesting results

        Returns:
            pd.DataFrame: Metrics by forecast horizon
        """
        try:
            if 'error' in results or 'iteration_results' not in results:
                return pd.DataFrame()

            horizon_metrics = []

            for iter_result in results['iteration_results']:
                horizon_metrics.append({
                    'iteration': iter_result['iteration'],
                    'horizon_days': iter_result['test_size'],
                    'rmse': iter_result['rmse'],
                    'mae': iter_result['mae'],
                    'r2': iter_result['r2'],
                    'mape': iter_result['mape']
                })

            df = pd.DataFrame(horizon_metrics)

            # Calculate correlation between iteration number and errors
            if len(df) > 2:
                rmse_trend = np.corrcoef(df['iteration'], df['rmse'])[0, 1]
                self.logger.info(f"RMSE trend correlation: {rmse_trend:.3f}")

            return df

        except Exception as e:
            self.logger.error(f"Error evaluating forecast horizon: {str(e)}")
            return pd.DataFrame()

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate summary report of all backtesting runs.

        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not self.results:
            return {'message': 'No backtesting results available'}

        summary = {
            'total_backtests': len(self.results),
            'window_type': self.window_type,
            'initial_train_size': self.initial_train_size,
            'test_size': self.test_size,
            'step_size': self.step_size,
            'results': []
        }

        for i, result in enumerate(self.results):
            if 'error' not in result:
                metrics = result['aggregate_metrics']
                summary['results'].append({
                    'backtest_id': i + 1,
                    'overall_rmse': metrics['overall_rmse'],
                    'overall_mae': metrics['overall_mae'],
                    'overall_r2': metrics['overall_r2'],
                    'n_windows': result['n_windows']
                })

        return summary