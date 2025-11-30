"""
Model evaluation module for CortexX sales forecasting platform.
Handles model performance evaluation and comparison.

FIXED: Proper NaN handling, safe metric extraction, and hybrid model support.
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
    Includes comprehensive error handling and NaN-safe operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics with robust NaN handling.
        """
        try:
            # Ensure arrays are numpy arrays
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            
            # Remove any NaN or infinite values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                self.logger.warning("No valid data points after cleaning NaN/Inf values")
                return {
                    'rmse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'mape': np.nan,
                    'mse': np.nan,
                    'bias': np.nan,
                    'std_error': np.nan
                }
            
            # Calculate basic metrics
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))),
                'mae': float(mean_absolute_error(y_true_clean, y_pred_clean)),
                'r2': float(r2_score(y_true_clean, y_pred_clean)),
                'mse': float(mean_squared_error(y_true_clean, y_pred_clean))
            }
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            try:
                # Avoid division by zero
                mask_zeros = y_true_clean != 0
                if np.any(mask_zeros):
                    mape_values = np.abs((y_true_clean[mask_zeros] - y_pred_clean[mask_zeros]) / y_true_clean[mask_zeros])
                    metrics['mape'] = float(np.mean(mape_values) * 100)
                else:
                    metrics['mape'] = np.nan
            except Exception as e:
                self.logger.warning(f"Could not calculate MAPE: {e}")
                metrics['mape'] = np.nan
            
            # Calculate additional business metrics
            metrics['bias'] = float(np.mean(y_pred_clean - y_true_clean))
            metrics['std_error'] = float(np.std(y_pred_clean - y_true_clean))
            
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
        Evaluate a single model's performance with comprehensive error handling.
        """
        try:
            evaluation = {
                'model_name': model_results.get('model', 'Unknown'),
                'training_time': model_results.get('training_time', 0)
            }
            
            # Try multiple keys for actual and predicted values
            y_true = None
            y_pred = None
            
            # Check for test predictions first (preferred)
            if 'y_test' in model_results and 'test_predictions' in model_results:
                y_true = np.array(model_results['y_test'])
                y_pred = np.array(model_results['test_predictions'])
            # Fallback to 'actual' and 'predictions'
            elif 'actual' in model_results and 'predictions' in model_results:
                y_true = np.array(model_results['actual'])
                y_pred = np.array(model_results['predictions'])
            
            if y_true is not None and y_pred is not None:
                # Ensure same length
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_true, y_pred)
                evaluation.update(metrics)
                
                # Additional analysis
                residuals = y_pred - y_true
                evaluation['residuals'] = residuals
                evaluation['residual_stats'] = {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'min': float(np.min(residuals)),
                    'max': float(np.max(residuals))
                }
            else:
                self.logger.warning(f"No prediction data found for {evaluation['model_name']}")
                # Return NaN metrics
                evaluation.update({
                    'rmse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'mape': np.nan,
                    'mse': np.nan,
                    'bias': np.nan,
                    'std_error': np.nan
                })
            
            self.logger.info(f"Evaluated model: {evaluation['model_name']}")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {
                'model_name': model_results.get('model', 'Unknown'),
                'error': str(e),
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'training_time': np.nan
            }
    
    def compare_models(self, models_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models - FIXED with comprehensive NaN handling.
        """
        try:
            comparison_data = []
            
            for model_name, results in models_results.items():
                evaluation = self.evaluate_model(results)
                
                # Always add model, even with NaN metrics
                comparison_data.append({
                    'Model': evaluation.get('model_name', model_name),
                    'RMSE': evaluation.get('rmse', np.nan),
                    'MAE': evaluation.get('mae', np.nan),
                    'R2': evaluation.get('r2', np.nan),
                    'MAPE': evaluation.get('mape', np.nan),
                    'Training Time (s)': evaluation.get('training_time', np.nan),
                    'Bias': evaluation.get('bias', np.nan),
                    'Std Error': evaluation.get('std_error', np.nan)
                })
            
            if not comparison_data:
                self.logger.warning("No comparison data available")
                return pd.DataFrame()
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Log warnings for NaN metrics
            for metric in ['RMSE', 'MAE', 'R2']:
                if metric in comparison_df.columns:
                    valid_count = comparison_df[metric].notna().sum()
                    total_count = len(comparison_df)
                    if valid_count == 0:
                        self.logger.warning(f"All {metric} values are NaN - models may have failed")
                    elif valid_count < total_count:
                        self.logger.warning(f"Only {valid_count}/{total_count} models have valid {metric}")
            
            # Sort by RMSE (lower is better), handling NaN values
            if not comparison_df.empty and 'RMSE' in comparison_df.columns:
                # Sort with NaN values at the end
                comparison_df = comparison_df.sort_values('RMSE', na_position='last')
            
            self.logger.info(f"Compared {len(comparison_data)} models")
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()
    
    def create_residual_analysis(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform residual analysis for model diagnostics.
        """
        try:
            # Try multiple keys for actual and predicted values
            y_true = None
            y_pred = None
            
            if 'y_test' in model_results and 'test_predictions' in model_results:
                y_true = np.array(model_results['y_test'])
                y_pred = np.array(model_results['test_predictions'])
            elif 'actual' in model_results and 'predictions' in model_results:
                y_true = np.array(model_results['actual'])
                y_pred = np.array(model_results['predictions'])
            
            if y_true is None or y_pred is None:
                return {'error': 'No actual/predictions data available'}
            
            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            residuals = y_pred - y_true
            
            analysis = {
                'residual_mean': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
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
            return float(skew(data))
        except Exception as e:
            self.logger.warning(f"Could not calculate skewness: {e}")
            return np.nan
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except Exception as e:
            self.logger.warning(f"Could not calculate kurtosis: {e}")
            return np.nan
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """Test normality of residuals using Shapiro-Wilk test."""
        try:
            from scipy.stats import shapiro
            # Shapiro test requires at least 3 samples
            if len(data) < 3:
                return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
            stat, p_value = shapiro(data)
            return {
                'statistic': float(stat), 
                'p_value': float(p_value), 
                'is_normal': bool(p_value > 0.05)
            }
        except Exception as e:
            self.logger.warning(f"Could not test normality: {e}")
            return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
    
    def _test_autocorrelation(self, data: np.ndarray, max_lag: int = 10) -> Dict[str, Any]:
        """Test autocorrelation in residuals."""
        try:
            from statsmodels.tsa.stattools import acf
            if len(data) <= max_lag:
                return {'acf_values': [], 'significant_lags': [], 'has_autocorrelation': False}
            
            acf_values = acf(data, nlags=max_lag, fft=False)
            threshold = 1.96 / np.sqrt(len(data))
            significant_lags = [i for i, val in enumerate(acf_values) 
                                if abs(val) > threshold and i > 0]
            return {
                'acf_values': [float(x) for x in acf_values],
                'significant_lags': significant_lags,
                'has_autocorrelation': len(significant_lags) > 0
            }
        except Exception as e:
            self.logger.warning(f"Could not test autocorrelation: {e}")
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
            
            variance_ratio = max(variances) / (min(variances) + 1e-6)
            return {
                'has_heteroscedasticity': bool(variance_ratio > 2.0),
                'variance_ratio': float(variance_ratio)
            }
        except Exception as e:
            self.logger.warning(f"Could not test heteroscedasticity: {e}")
            return {'has_heteroscedasticity': False, 'variance_ratio': 1.0}
    
    def generate_evaluation_report(self, models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for all models.
        Includes hybrid model creation if component models are available.
        """
        try:
            # Create hybrid model if components are available
            hybrid_components = ['Lasso', 'LightGBM', 'RandomForest']
            if all(comp in models_results for comp in hybrid_components):
                self.logger.info("Creating Hybrid Model (Lasso + LightGBM + RF) predictions...")
                try:
                    # Get predictions from each component
                    lasso_preds = self._get_predictions(models_results['Lasso'])
                    lgbm_preds = self._get_predictions(models_results['LightGBM'])
                    rf_preds = self._get_predictions(models_results['RandomForest'])
                    
                    if lasso_preds is not None and lgbm_preds is not None and rf_preds is not None:
                        # Ensure same length
                        min_len = min(len(lasso_preds), len(lgbm_preds), len(rf_preds))
                        
                        # Average predictions
                        hybrid_preds = (
                            lasso_preds[:min_len] + 
                            lgbm_preds[:min_len] + 
                            rf_preds[:min_len]
                        ) / 3
                        
                        # Get actual values
                        y_true = self._get_actuals(models_results['Lasso'])
                        if y_true is not None:
                            y_true = y_true[:min_len]
                            
                            # Add hybrid model to results
                            models_results['Hybrid (Lasso+LGBM+RF)'] = {
                                'model': 'Hybrid (Lasso+LGBM+RF)',
                                'training_time': 0,
                                'model_object': 'ensemble_average',
                                'y_test': y_true,
                                'test_predictions': hybrid_preds,
                                'dates': models_results['Lasso'].get('dates', [])[:min_len]
                            }
                            
                            self.logger.info("Hybrid model created successfully")
                except Exception as e:
                    self.logger.warning(f"Could not create hybrid model: {str(e)}")
            
            # Generate report
            report = {
                'model_comparison': [],
                'best_model': None,
                'detailed_evaluations': {},
                'recommendations': []
            }
            
            # Model comparison
            comparison_df = self.compare_models(models_results)
            if not comparison_df.empty:
                report['model_comparison'] = comparison_df.to_dict('records')
                
                # Find best model based on RMSE (filtering NaN values)
                valid_models = [m for m in report['model_comparison'] 
                              if not np.isnan(m.get('RMSE', np.nan))]
                
                if valid_models:
                    best_model = min(valid_models, key=lambda x: x.get('RMSE', float('inf')))
                    report['best_model'] = best_model['Model']
                    self.logger.info(f"Best model: {report['best_model']} with RMSE={best_model.get('RMSE'):.4f}")
                else:
                    report['best_model'] = 'No valid model found'
                    self.logger.warning("No valid models with RMSE scores")
            
            # Detailed evaluation for each model
            for model_name, results in models_results.items():
                try:
                    evaluation = self.evaluate_model(results)
                    residual_analysis = self.create_residual_analysis(results)
                    
                    report['detailed_evaluations'][model_name] = {
                        'basic_metrics': evaluation,
                        'residual_analysis': residual_analysis
                    }
                except Exception as e:
                    self.logger.error(f"Error evaluating {model_name}: {e}")
                    continue
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            self.logger.info("Generated comprehensive evaluation report")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            return {'error': str(e)}
    
    def _get_predictions(self, model_results: Dict[str, Any]) -> Optional[np.ndarray]:
        """Helper to safely extract predictions from model results."""
        try:
            if 'test_predictions' in model_results:
                return np.array(model_results['test_predictions'])
            elif 'predictions' in model_results:
                return np.array(model_results['predictions'])
            return None
        except Exception as e:
            self.logger.warning(f"Could not extract predictions: {e}")
            return None
    
    def _get_actuals(self, model_results: Dict[str, Any]) -> Optional[np.ndarray]:
        """Helper to safely extract actual values from model results."""
        try:
            if 'y_test' in model_results:
                return np.array(model_results['y_test'])
            elif 'actual' in model_results:
                return np.array(model_results['actual'])
            return None
        except Exception as e:
            self.logger.warning(f"Could not extract actuals: {e}")
            return None
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate business recommendations based on model evaluation."""
        recommendations = []
        
        try:
            if not report['model_comparison']:
                return ["No model data available for recommendations"]
            
            best_model = report['best_model']
            if best_model and best_model != 'No valid model found':
                recommendations.append(f"**Recommended model**: {best_model} based on lowest RMSE")
            
            # Check model performance
            for model in report['model_comparison']:
                r2 = model.get('R2', np.nan)
                mape = model.get('MAPE', np.nan)
                model_name = model.get('Model', 'Unknown')
                
                # Skip NaN values
                if pd.isna(r2) or pd.isna(mape):
                    continue
                
                # R² recommendations
                if r2 > 0.8:
                    recommendations.append(f"✅ {model_name} shows excellent explanatory power (R² = {r2:.3f})")
                elif r2 > 0.6:
                    recommendations.append(f"✓ {model_name} shows good explanatory power (R² = {r2:.3f})")
                
                # MAPE recommendations
                if mape < 10:
                    recommendations.append(f"✅ {model_name} has high accuracy (MAPE = {mape:.1f}%)")
                elif mape < 20:
                    recommendations.append(f"✓ {model_name} has reasonable accuracy (MAPE = {mape:.1f}%)")
            
            # General business recommendations
            if best_model and best_model != 'No valid model found':
                recommendations.extend([
                    "\n**Business Applications:**",
                    "📦 **Inventory Management**: Use forecasts to optimize stock levels and reduce holding costs",
                    "📢 **Promotional Planning**: Align marketing campaigns with predicted high-demand periods",
                    "👥 **Staffing Optimization**: Adjust workforce scheduling based on forecasted demand",
                    "💰 **Budget Allocation**: Allocate resources efficiently based on demand predictions",
                    "📊 **Performance Monitoring**: Track actual vs predicted values to detect anomalies"
                ])
            
            # Add warning if no good models
            valid_models = [m for m in report['model_comparison'] 
                          if not np.isnan(m.get('RMSE', np.nan))]
            if not valid_models:
                recommendations.append("⚠️ **Warning**: No models produced valid results. Check data quality and model training process.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return [f"Error generating recommendations: {str(e)}"]
