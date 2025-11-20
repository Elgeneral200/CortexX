"""
Prediction intervals module for CortexX sales forecasting platform.

Provides confidence bands and uncertainty quantification for forecasts.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.metrics import mean_squared_error
from scipy import stats

logger = logging.getLogger(__name__)


class PredictionIntervals:
    """
    A class to calculate prediction intervals and confidence bands for forecasts.

    Supports multiple methods:
    - Residual-based intervals (assumes normal distribution)
    - Quantile regression intervals
    - Bootstrap intervals
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize prediction intervals calculator.

        Args:
            confidence_level (float): Confidence level for intervals (default: 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.logger = logging.getLogger(__name__)

    def calculate_residual_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    future_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals based on residual distribution.

        Assumes residuals are normally distributed. Uses standard error of residuals
        to construct confidence bands around predictions.

        Args:
            y_true (np.ndarray): Actual values (training/validation)
            y_pred (np.ndarray): Predicted values (training/validation)
            future_pred (np.ndarray): Future predictions to add intervals to

        Returns:
            Dict[str, np.ndarray]: Dictionary with 'lower', 'upper', 'width' bounds
        """
        try:
            # Calculate residuals
            residuals = y_true - y_pred

            # Calculate standard error
            se = np.std(residuals, ddof=1)

            # Calculate z-score for confidence level
            z_score = stats.norm.ppf(1 - self.alpha / 2)

            # Calculate margin of error
            margin = z_score * se

            # Calculate intervals
            lower_bound = future_pred - margin
            upper_bound = future_pred + margin
            interval_width = upper_bound - lower_bound

            self.logger.info(f"Residual-based intervals calculated with {self.confidence_level*100}% confidence")

            return {
                'predictions': future_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': interval_width,
                'standard_error': se,
                'confidence_level': self.confidence_level
            }

        except Exception as e:
            self.logger.error(f"Error calculating residual intervals: {str(e)}")
            return {
                'predictions': future_pred,
                'lower_bound': future_pred,
                'upper_bound': future_pred,
                'error': str(e)
            }

    def calculate_bootstrap_intervals(self, model, X_train: pd.DataFrame, 
                                     y_train: pd.Series, X_future: pd.DataFrame,
                                     n_iterations: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals using bootstrap resampling.

        Trains multiple models on bootstrap samples and uses prediction distribution
        to construct confidence intervals.

        Args:
            model: Fitted model object (must have fit and predict methods)
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_future (pd.DataFrame): Future features for prediction
            n_iterations (int): Number of bootstrap iterations

        Returns:
            Dict[str, np.ndarray]: Dictionary with interval bounds
        """
        try:
            self.logger.info(f"Calculating bootstrap intervals with {n_iterations} iterations...")

            n_samples = len(X_train)
            predictions = []

            for i in range(n_iterations):
                # Bootstrap sample
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_train.iloc[indices]
                y_boot = y_train.iloc[indices]

                # Clone and train model
                from sklearn.base import clone
                model_boot = clone(model)
                model_boot.fit(X_boot, y_boot)

                # Predict on future data
                pred = model_boot.predict(X_future)
                predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate percentiles for confidence intervals
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100

            lower_bound = np.percentile(predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(predictions, upper_percentile, axis=0)
            mean_pred = np.mean(predictions, axis=0)

            self.logger.info(f"Bootstrap intervals calculated successfully")

            return {
                'predictions': mean_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': upper_bound - lower_bound,
                'confidence_level': self.confidence_level,
                'n_iterations': n_iterations
            }

        except Exception as e:
            self.logger.error(f"Error calculating bootstrap intervals: {str(e)}")
            # Fallback to point predictions
            pred = model.predict(X_future)
            return {
                'predictions': pred,
                'lower_bound': pred,
                'upper_bound': pred,
                'error': str(e)
            }

    def calculate_quantile_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    future_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals using quantile-based method.

        Uses empirical quantiles of residuals to construct asymmetric intervals.
        Better for non-normal residual distributions.

        Args:
            y_true (np.ndarray): Actual values
            y_pred (np.ndarray): Predicted values
            future_pred (np.ndarray): Future predictions

        Returns:
            Dict[str, np.ndarray]: Dictionary with interval bounds
        """
        try:
            # Calculate residuals
            residuals = y_true - y_pred

            # Calculate quantiles
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100

            lower_error = np.percentile(residuals, lower_percentile)
            upper_error = np.percentile(residuals, upper_percentile)

            # Apply to future predictions (residuals are y_true - y_pred, so add to pred)
            lower_bound = future_pred + lower_error
            upper_bound = future_pred + upper_error

            self.logger.info(f"Quantile-based intervals calculated")

            return {
                'predictions': future_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': upper_bound - lower_bound,
                'confidence_level': self.confidence_level
            }

        except Exception as e:
            self.logger.error(f"Error calculating quantile intervals: {str(e)}")
            return {
                'predictions': future_pred,
                'lower_bound': future_pred,
                'upper_bound': future_pred,
                'error': str(e)
            }

    def calculate_intervals(self, method: str, y_true: np.ndarray, 
                          y_pred: np.ndarray, future_pred: np.ndarray,
                          model=None, X_train: pd.DataFrame = None,
                          y_train: pd.Series = None, X_future: pd.DataFrame = None,
                          **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals using specified method.

        Args:
            method (str): Method to use ('residual', 'bootstrap', 'quantile')
            y_true (np.ndarray): Actual values
            y_pred (np.ndarray): Predicted values
            future_pred (np.ndarray): Future predictions
            model: Model object (required for bootstrap)
            X_train, y_train, X_future: Data (required for bootstrap)
            **kwargs: Additional method-specific parameters

        Returns:
            Dict[str, np.ndarray]: Interval bounds and metadata
        """
        if method.lower() == 'residual':
            return self.calculate_residual_intervals(y_true, y_pred, future_pred)

        elif method.lower() == 'bootstrap':
            if model is None or X_train is None or y_train is None or X_future is None:
                self.logger.warning("Bootstrap requires model and data. Falling back to residual method.")
                return self.calculate_residual_intervals(y_true, y_pred, future_pred)
            return self.calculate_bootstrap_intervals(model, X_train, y_train, X_future, 
                                                     **kwargs)

        elif method.lower() == 'quantile':
            return self.calculate_quantile_intervals(y_true, y_pred, future_pred)

        else:
            self.logger.error(f"Unknown method: {method}. Using residual method.")
            return self.calculate_residual_intervals(y_true, y_pred, future_pred)

    def create_interval_dataframe(self, dates: pd.DatetimeIndex, 
                                 intervals: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Create a formatted DataFrame with predictions and intervals.

        Args:
            dates (pd.DatetimeIndex): Date index for predictions
            intervals (Dict[str, np.ndarray]): Interval dictionary from calculate methods

        Returns:
            pd.DataFrame: Formatted dataframe with date, prediction, and bounds
        """
        try:
            df = pd.DataFrame({
                'date': dates,
                'prediction': intervals['predictions'],
                'lower_bound': intervals['lower_bound'],
                'upper_bound': intervals['upper_bound'],
                'interval_width': intervals.get('interval_width', 
                                               intervals['upper_bound'] - intervals['lower_bound'])
            })

            return df

        except Exception as e:
            self.logger.error(f"Error creating interval dataframe: {str(e)}")
            return pd.DataFrame()

    def evaluate_interval_coverage(self, y_true: np.ndarray, 
                                   lower_bound: np.ndarray,
                                   upper_bound: np.ndarray) -> Dict[str, float]:
        """
        Evaluate prediction interval coverage.

        Calculates what percentage of actual values fall within the predicted intervals.
        For 95% confidence intervals, coverage should be approximately 95%.

        Args:
            y_true (np.ndarray): Actual values
            lower_bound (np.ndarray): Lower interval bounds
            upper_bound (np.ndarray): Upper interval bounds

        Returns:
            Dict[str, float]: Coverage statistics
        """
        try:
            # Check if actuals fall within intervals
            within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            coverage = np.mean(within_interval) * 100

            # Calculate average interval width
            avg_width = np.mean(upper_bound - lower_bound)

            # Calculate interval quality score (coverage close to confidence level is good)
            expected_coverage = self.confidence_level * 100
            coverage_error = abs(coverage - expected_coverage)

            result = {
                'coverage_percentage': coverage,
                'expected_coverage': expected_coverage,
                'coverage_error': coverage_error,
                'average_interval_width': avg_width,
                'n_observations': len(y_true),
                'n_within_interval': int(np.sum(within_interval))
            }

            self.logger.info(f"Interval coverage: {coverage:.2f}% (expected: {expected_coverage:.2f}%)")

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating interval coverage: {str(e)}")
            return {'error': str(e)}