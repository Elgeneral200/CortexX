"""
Model deployment module for CortexX sales forecasting platform.
Handles model saving, loading, and deployment functionalities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import pickle
import json
import os
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)

class ModelDeployer:
    """
    A class to handle model deployment and management.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ModelDeployer with model directory.
        
        Args:
            model_dir (str): Directory to save/load models
        """
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save trained model to disk with metadata.
        
        Args:
            model (Any): Trained model object
            model_name (str): Name for the model
            metadata (Dict[str, Any]): Additional model metadata
            
        Returns:
            str: Path to saved model file
        """
        try:
            # Create model filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}"
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{filename}.pkl")
            
            # Use joblib for sklearn-based models, pickle for others
            try:
                joblib.dump(model, model_path)
            except:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'model_name': model_name,
                'saved_at': timestamp,
                'model_type': type(model).__name__,
                'model_path': model_path
            })
            
            metadata_path = os.path.join(self.model_dir, f"{filename}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved model {model_name} to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
            raise ValueError(f"Model saving failed: {str(e)}")
    
    def load_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load trained model from disk.
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Loaded model and its metadata
        """
        try:
            # Load model
            try:
                model = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Load metadata
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {'model_path': model_path}
            
            self.logger.info(f"Loaded model from {model_path}")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")
    
    def deploy_model_api(self, model: Any, model_name: str, port: int = 8000) -> Dict[str, Any]:
        """
        Deploy model as a REST API (simplified version).
        
        Args:
            model (Any): Trained model to deploy
            model_name (str): Name for the API
            port (int): Port for API server
            
        Returns:
            Dict[str, Any]: Deployment information
        """
        try:
            # This is a simplified version - in production, you'd use FastAPI or Flask
            deployment_info = {
                'model_name': model_name,
                'deployed_at': datetime.now().isoformat(),
                'port': port,
                'status': 'simulated',
                'endpoints': {
                    'predict': f'http://localhost:{port}/predict',
                    'health': f'http://localhost:{port}/health',
                    'metrics': f'http://localhost:{port}/metrics'
                }
            }
            
            self.logger.info(f"Simulated API deployment for {model_name} on port {port}")
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Error deploying model API: {str(e)}")
            return {'error': str(e), 'status': 'failed'}
    
    def create_model_card(self, model: Any, model_name: str, 
                         training_data_info: Dict[str, Any],
                         performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Create a model card with comprehensive information.
        
        Args:
            model (Any): Trained model
            model_name (str): Model name
            training_data_info (Dict[str, Any]): Information about training data
            performance_metrics (Dict[str, float]): Model performance metrics
            
        Returns:
            Dict[str, Any]: Model card information
        """
        try:
            model_card = {
                'model_name': model_name,
                'version': '1.0.0',
                'created_date': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'training_data': training_data_info,
                'performance_metrics': performance_metrics,
                'intended_use': 'Sales forecasting and demand prediction',
                'limitations': [
                    'Assumes historical patterns will continue',
                    'Performance may degrade during unprecedented events',
                    'Requires regular retraining with new data'
                ],
                'ethical_considerations': [
                    'Should not be used for discriminatory pricing',
                    'Transparent about uncertainty in predictions',
                    'Regular monitoring for bias required'
                ]
            }
            
            # Add model-specific information
            if hasattr(model, 'feature_importances_'):
                model_card['key_features'] = self._get_important_features(model, training_data_info)
            
            self.logger.info(f"Created model card for {model_name}")
            return model_card
            
        except Exception as e:
            self.logger.error(f"Error creating model card: {str(e)}")
            return {'error': str(e)}
    
    def _get_important_features(self, model: Any, training_data_info: Dict[str, Any]) -> Dict[str, float]:
        """Extract important features from model if available."""
        try:
            if hasattr(model, 'feature_importances_') and 'feature_names' in training_data_info:
                feature_names = training_data_info['feature_names']
                importances = model.feature_importances_
                
                # Get top 10 features
                indices = np.argsort(importances)[::-1][:10]
                important_features = {
                    feature_names[i]: float(importances[i]) 
                    for i in indices if i < len(feature_names)
                }
                return important_features
            return {}
        except:
            return {}
    
    def monitor_model_performance(self, model_name: str, 
                                current_metrics: Dict[str, float],
                                baseline_metrics: Dict[str, float],
                                threshold: float = 0.1) -> Dict[str, Any]:
        """
        Monitor model performance and detect degradation.
        
        Args:
            model_name (str): Model name
            current_metrics (Dict[str, float]): Current performance metrics
            baseline_metrics (Dict[str, float]): Baseline performance metrics
            threshold (float): Performance degradation threshold
            
        Returns:
            Dict[str, Any]: Monitoring results and alerts
        """
        try:
            monitoring_results = {
                'model_name': model_name,
                'monitored_at': datetime.now().isoformat(),
                'alerts': [],
                'status': 'healthy',
                'performance_changes': {}
            }
            
            # Check for performance degradation
            for metric_name, current_value in current_metrics.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    
                    # Calculate percentage change
                    if baseline_value != 0:
                        change_pct = abs(current_value - baseline_value) / abs(baseline_value)
                        
                        monitoring_results['performance_changes'][metric_name] = {
                            'current': current_value,
                            'baseline': baseline_value,
                            'change_pct': change_pct
                        }
                        
                        if change_pct > threshold:
                            alert_msg = (
                                f"Significant degradation in {metric_name}: "
                                f"{change_pct:.1%} change from baseline"
                            )
                            monitoring_results['alerts'].append(alert_msg)
                            monitoring_results['status'] = 'degraded'
            
            # Check for data drift (simplified)
            if len(monitoring_results['alerts']) > 0:
                monitoring_results['recommendation'] = 'Consider retraining the model'
            else:
                monitoring_results['recommendation'] = 'Model performance is stable'
            
            self.logger.info(f"Monitored model {model_name}, status: {monitoring_results['status']}")
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"Error monitoring model performance: {str(e)}")
            return {'error': str(e), 'status': 'monitoring_failed'}



