"""
Model evaluation module for CortexX sales forecasting platform.
Handles model performance evaluation and comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A class to handle model evaluation and performance comparison.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        try:
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {
                    'rmse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'mape': np.nan,
                    'mse': np.nan
                }
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'r2': r2_score(y_true_clean, y_pred_clean),
                'mse': mean_squared_error(y_true_clean, y_pred_clean)
            }
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            try:
                metrics['mape'] = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
            except:
                metrics['mape'] = np.nan
            
            # Calculate additional business metrics
            metrics['bias'] = np.mean(y_pred_clean - y_true_clean)
            metrics['std_error'] = np.std(y_pred_clean - y_true_clean)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'mse': np.nan,
                'bias': np.nan,
                'std_error': np.nan
            }
    
    def evaluate_model(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single model's performance.
        
        Args:
            model_results (Dict[str, Any]): Model training results
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        try:
            evaluation = {
                'model_name': model_results.get('model', 'Unknown'),
                'training_time': model_results.get('training_time', 0)
            }
            
            # Calculate metrics if actual and predictions are available
            if 'actual' in model_results and 'predictions' in model_results:
                y_true = np.array(model_results['actual'])
                y_pred = np.array(model_results['predictions'])
                
                metrics = self.calculate_metrics(y_true, y_pred)
                evaluation.update(metrics)
                
                # Additional analysis
                evaluation['residuals'] = y_pred - y_true
                evaluation['residual_stats'] = {
                    'mean': np.mean(evaluation['residuals']),
                    'std': np.std(evaluation['residuals']),
                    'min': np.min(evaluation['residuals']),
                    'max': np.max(evaluation['residuals'])
                }
            
            self.logger.info(f"Evaluated model: {evaluation['model_name']}")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {
                'model_name': model_results.get('model', 'Unknown'),
                'error': str(e)
            }
    
    def compare_models(self, models_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models and create performance comparison.
        
        Args:
            models_results (Dict[str, Dict[str, Any]]): Dictionary of model results
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        try:
            comparison_data = []
            
            for model_name, results in models_results.items():
                evaluation = self.evaluate_model(results)
                
                if 'error' not in evaluation:
                    comparison_data.append({
                        'Model': evaluation['model_name'],
                        'RMSE': evaluation.get('rmse', np.nan),
                        'MAE': evaluation.get('mae', np.nan),
                        'R²': evaluation.get('r2', np.nan),
                        'MAPE (%)': evaluation.get('mape', np.nan),
                        'Training Time (s)': evaluation.get('training_time', np.nan),
                        'Bias': evaluation.get('bias', np.nan),
                        'Std Error': evaluation.get('std_error', np.nan)
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Sort by RMSE (lower is better)
            if not comparison_df.empty and 'RMSE' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('RMSE')
            
            self.logger.info(f"Compared {len(comparison_data)} models")
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()
    
    def create_residual_analysis(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform residual analysis for model diagnostics.
        
        Args:
            model_results (Dict[str, Any]): Model training results
            
        Returns:
            Dict[str, Any]: Residual analysis results
        """
        try:
            if 'actual' not in model_results or 'predictions' not in model_results:
                return {'error': 'No actual/predictions data available'}
            
            y_true = np.array(model_results['actual'])
            y_pred = np.array(model_results['predictions'])
            
            residuals = y_pred - y_true
            
            analysis = {
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': self._calculate_skewness(residuals),
                'residual_kurtosis': self._calculate_kurtosis(residuals),
                'normality_test': self._test_normality(residuals),
                'autocorrelation_test': self._test_autocorrelation(residuals)
            }
            
            # Residuals by predicted values
            if len(y_pred) > 0:
                analysis['heteroscedasticity'] = self._test_heteroscedasticity(y_pred, residuals)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in residual analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            return np.nan
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            return np.nan
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, float]:
        """Test normality of residuals using Shapiro-Wilk test."""
        try:
            from scipy.stats import shapiro
            stat, p_value = shapiro(data)
            return {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
        except:
            return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
    
    def _test_autocorrelation(self, data: np.ndarray, max_lag: int = 10) -> Dict[str, Any]:
        """Test autocorrelation in residuals."""
        try:
            from statsmodels.tsa.stattools import acf
            acf_values = acf(data, nlags=max_lag)
            significant_lags = [i for i, val in enumerate(acf_values) 
                              if abs(val) > 1.96/np.sqrt(len(data)) and i > 0]
            return {
                'acf_values': acf_values.tolist(),
                'significant_lags': significant_lags,
                'has_autocorrelation': len(significant_lags) > 0
            }
        except:
            return {'acf_values': [], 'significant_lags': [], 'has_autocorrelation': False}
    
    def _test_heteroscedasticity(self, predictions: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for heteroscedasticity in residuals."""
        try:
            # Group predictions into bins and check variance of residuals
            n_bins = min(10, len(predictions) // 10)
            if n_bins < 2:
                return {'has_heteroscedasticity': False, 'variance_ratio': 1.0}
            
            bins = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
            bin_indices = np.digitize(predictions, bins)
            
            variances = []
            for i in range(1, n_bins + 1):
                bin_residuals = residuals[bin_indices == i]
                if len(bin_residuals) > 1:
                    variances.append(np.var(bin_residuals))
            
            if len(variances) < 2:
                return {'has_heteroscedasticity': False, 'variance_ratio': 1.0}
            
            variance_ratio = max(variances) / min(variances)
            return {
                'has_heteroscedasticity': variance_ratio > 2.0,
                'variance_ratio': variance_ratio
            }
        except:
            return {'has_heteroscedasticity': False, 'variance_ratio': 1.0}
    
    def generate_evaluation_report(self, models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for all models.
        
        Args:
            models_results (Dict[str, Dict[str, Any]]): Dictionary of model results
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation report
        """
        try:
            report = {
                'model_comparison': self.compare_models(models_results).to_dict('records'),
                'best_model': None,
                'detailed_evaluations': {},
                'recommendations': []
            }
            
            # Find best model based on RMSE
            if report['model_comparison']:
                best_model = min(report['model_comparison'], key=lambda x: x.get('RMSE', float('inf')))
                report['best_model'] = best_model['Model']
            
            # Detailed evaluation for each model
            for model_name, results in models_results.items():
                evaluation = self.evaluate_model(results)
                residual_analysis = self.create_residual_analysis(results)
                
                report['detailed_evaluations'][model_name] = {
                    'basic_metrics': evaluation,
                    'residual_analysis': residual_analysis
                }
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            self.logger.info("Generated comprehensive evaluation report")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate business recommendations based on model evaluation."""
        recommendations = []
        
        try:
            if not report['model_comparison']:
                return ["No model data available for recommendations"]
            
            best_model = report['best_model']
            if best_model:
                recommendations.append(f"**Recommended model**: {best_model} based on lowest RMSE")
            
            # Check model performance
            for model in report['model_comparison']:
                r2 = model.get('R²', 0)
                mape = model.get('MAPE (%)', 100)
                
                if r2 > 0.8:
                    recommendations.append(f"{model['Model']} shows excellent explanatory power (R² = {r2:.3f})")
                elif r2 > 0.6:
                    recommendations.append(f"{model['Model']} shows good explanatory power (R² = {r2:.3f})")
                
                if mape < 10:
                    recommendations.append(f"{model['Model']} has high accuracy (MAPE = {mape:.1f}%)")
                elif mape < 20:
                    recommendations.append(f"{model['Model']} has reasonable accuracy (MAPE = {mape:.1f}%)")
            
            # General recommendations
            recommendations.extend([
                "**Inventory Management**: Use forecasts to optimize stock levels",
                "**Promotional Planning**: Align promotions with predicted high-demand periods",
                "**Staffing Optimization**: Adjust staffing based on weekly seasonality patterns"
            ])
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]