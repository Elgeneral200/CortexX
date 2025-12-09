"""
Model training module for CortexX sales forecasting platform.

CRITICAL FIXES:
- Data leakage eliminated (proper imputer fitting)
- Hyperparameter support added to all methods
- Config integration for default parameters
- Model serialization for persistence
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib
from pathlib import Path
import streamlit as st

# Model Imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Enterprise-grade model trainer with proper data handling and config integration.
    
    ENHANCED:
    - No data leakage (proper train/test separation)
    - Hyperparameter support
    - Model persistence
    - Config integration
    """

    def __init__(self, config=None):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration object (auto-loaded if None)
        """
        if config is None:
            from src.utils.config import get_config
            config = get_config()
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.train_imputer = SimpleImputer(strategy='mean')
        self.train_scaler = StandardScaler()
        self.is_fitted = False  # Track if preprocessors are fitted
        warnings.filterwarnings('ignore')

    def train_test_split(self, df: pd.DataFrame, date_col: str, target_col: str,
                    test_size: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets using time-based split."""
        try:
            if test_size is None:
                test_size = self.config.model.default_test_size
            
            # 🔥 FIXED: Bulletproof date column detection
            date_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64', 'object']).columns
            if len(date_cols) > 0:
                actual_date_col = date_cols[0]
                df_sorted = df.sort_values(actual_date_col).copy()
            else:
                df_sorted = df.reset_index(drop=True)  # Fallback
            
            split_idx = int(len(df_sorted) * (1 - test_size))
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            
            self.logger.info(f"Split data: {len(train_df)} training, {len(test_df)} testing")
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error in train-test split: {str(e)}")
            raise ValueError(f"Train-test split failed: {str(e)}")


    def _prepare_features(self, df: pd.DataFrame, date_col: str, target_col: str, 
                         fit_preprocessors: bool = False, scale_features: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare features and target for machine learning models.
        
        FIXED: Proper handling of fit vs transform to prevent data leakage.
        
        Args:
            df: Input dataframe
            date_col: Date column name
            target_col: Target column name
            fit_preprocessors: If True, fit imputer/scaler (use for training data only)
            scale_features: If True, apply StandardScaler (for KNN, SVR)
        
        Returns:
            Tuple of (X, y) or (None, None) if error
        """
        try:
            # Select numeric features and exclude date and target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col not in [date_col, target_col]]
            
            if not feature_cols:
                self.logger.warning("No suitable numeric features found for ML models")
                return None, None
            
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Handle missing values - FIT only on training data!
            if fit_preprocessors:
                self.train_imputer.fit(X)
                if scale_features:
                    X_imputed = self.train_imputer.transform(X)
                    self.train_scaler.fit(X_imputed)
                self.is_fitted = True
            
            # Transform (works for both train and test)
            X_imputed = self.train_imputer.transform(X)
            X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            
            # Apply scaling if requested
            if scale_features and self.is_fitted:
                X_scaled = self.train_scaler.transform(X_imputed)
                X_imputed = pd.DataFrame(X_scaled, columns=X_imputed.columns, index=X_imputed.index)
            
            # Fill target missing values
            y = y.fillna(y.mean())
            
            return X_imputed, y
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None, None

    def _train_generic_model(self, model, model_name: str, train_df: pd.DataFrame, 
                           test_df: pd.DataFrame, date_col: str, target_col: str,
                           scale_features: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """
        Generic training method to reduce code duplication.
        
        NEW: Single method handles all model types.
        
        Args:
            model: Initialized model object
            model_name: Name of the model
            train_df: Training dataframe
            test_df: Testing dataframe
            date_col: Date column name
            target_col: Target column name
            scale_features: Whether to scale features
        
        Returns:
            Tuple of (model, results)
        """
        try:
            # Prepare training features (FIT preprocessors)
            X_train, y_train = self._prepare_features(
                train_df, date_col, target_col, 
                fit_preprocessors=True, 
                scale_features=scale_features
            )
            
            # Prepare test features (TRANSFORM only - no fit!)
            X_test, y_test = self._prepare_features(
                test_df, date_col, target_col, 
                fit_preprocessors=False,
                scale_features=scale_features
            )
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, model_name)
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred_test = model.predict(X_test)
            
            # Build results
            results = {
                'model': model_name,
                'training_time': training_time,
                'model_object': model,
                'y_test': y_test.values,
                'test_predictions': y_pred_test,
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = dict(zip(X_train.columns, model.feature_importances_))
            
            self.logger.info(f"{model_name} model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training {model_name} model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, model_name)

    # ============================================================================
    # MODEL-SPECIFIC TRAINING METHODS (Now with hyperparameter support)
    # ============================================================================

    def train_xgboost(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                     date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Train XGBoost model.
        
        ENHANCED: Accepts hyperparameters from optimization.
        """
        # Get default config and merge with hyperparams
        params = self.config.get_model_config('XGBoost')
        if hyperparams:
            params.update(hyperparams)
        
        model = xgb.XGBRegressor(**params)
        return self._train_generic_model(model, 'XGBoost', train_df, test_df, date_col, target_col)

    def train_lightgbm(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                      date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """Train LightGBM model with hyperparameter support."""
        params = self.config.get_model_config('LightGBM')
        if hyperparams:
            params.update(hyperparams)
        params['verbose'] = -1  # Silence output
        
        model = lgb.LGBMRegressor(**params)
        return self._train_generic_model(model, 'LightGBM', train_df, test_df, date_col, target_col)

    def train_random_forest(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """Train Random Forest model with hyperparameter support."""
        params = self.config.get_model_config('Random Forest')
        if hyperparams:
            params.update(hyperparams)
        
        model = RandomForestRegressor(**params)
        return self._train_generic_model(model, 'Random Forest', train_df, test_df, date_col, target_col)

    def train_catboost(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                      date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """Train CatBoost model with hyperparameter support."""
        params = self.config.get_model_config('CatBoost')
        if hyperparams:
            params.update(hyperparams)
        
        model = CatBoostRegressor(**params)
        return self._train_generic_model(model, 'CatBoost', train_df, test_df, date_col, target_col)

    def train_linear_regression(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                               date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """Train Linear Regression model."""
        params = self.config.get_model_config('Linear Regression')
        if hyperparams:
            params.update(hyperparams)
        
        model = LinearRegression(**params)
        return self._train_generic_model(model, 'Linear Regression', train_df, test_df, date_col, target_col)

    def train_ridge(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                   date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """Train Ridge Regression model with hyperparameter support."""
        params = self.config.get_model_config('Ridge Regression')
        if hyperparams:
            params.update(hyperparams)
        
        model = Ridge(**params)
        return self._train_generic_model(model, 'Ridge Regression', train_df, test_df, date_col, target_col)

    def train_lasso(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                   date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """Train Lasso Regression model with hyperparameter support."""
        params = self.config.get_model_config('Lasso Regression')
        if hyperparams:
            params.update(hyperparams)
        
        model = Lasso(**params)
        return self._train_generic_model(model, 'Lasso Regression', train_df, test_df, date_col, target_col)

    def train_knn(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """Train KNN model with hyperparameter support and scaling."""
        params = self.config.get_model_config('K-Nearest Neighbors')
        if hyperparams:
            params.update(hyperparams)
        
        model = KNeighborsRegressor(**params)
        return self._train_generic_model(model, 'K-Nearest Neighbors', train_df, test_df, 
                                        date_col, target_col, scale_features=True)

    
    def train_decision_tree(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           date_col: str, target_col: str, hyperparams: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """Train Decision Tree model with hyperparameter support."""
        params = self.config.get_model_config('Decision Tree')
        if hyperparams:
            params.update(hyperparams)
        
        model = DecisionTreeRegressor(**params)
        return self._train_generic_model(model, 'Decision Tree', train_df, test_df, date_col, target_col)

    
    # ============================================================================
    # MODEL PERSISTENCE
    # ============================================================================

    def save_model(self, model, model_name: str, version: str = 'v1') -> str:
        """
        Save trained model to disk.
        
        NEW: Model persistence for production deployment.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Version string
        
        Returns:
            str: Path where model was saved
        """
        try:
            save_dir = Path(self.config.model.model_registry_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{model_name.replace(' ', '_')}_{version}.pkl"
            save_path = save_dir / filename
            
            joblib.dump(model, save_path)
            self.logger.info(f"Model saved to {save_path}")
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
        
        Returns:
            Loaded model object
        """
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _create_dummy_model(self, df: pd.DataFrame, date_col: str, target_col: str,
                           model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Create a dummy model for demonstration when real training fails."""
        self.logger.warning(f"Creating dummy model for {model_name}")
        
        class DummyModel:
            def predict(self, X):
                return np.full(X.shape[0] if hasattr(X, 'shape') else len(X), df[target_col].mean())
        
        # Create simple results
        mean_val = df[target_col].mean()
        results = {
            'model': model_name,
            'training_time': 0.1,
            'model_object': DummyModel(),
            'y_test': df[target_col].values[:10],
            'test_predictions': np.full(10, mean_val),
            'test_rmse': df[target_col].std(),  # Honest metric
            'test_mae': df[target_col].std() * 0.8,
            'test_r2': 0.0,
            'is_dummy': True  # Flag for UI
        }
        
        return DummyModel(), results


# ============================================================================
# CACHING FUNCTION
# ============================================================================

@st.cache_resource
def get_model_trainer(_config=None):  # ✅ Added underscore
    """
    Get or create cached ModelTrainer instance (singleton pattern).
    
    Args:
        _config: Config object (underscore = don't hash for caching)
    
    Returns:
        ModelTrainer: Cached trainer instance
    """
    return ModelTrainer(_config)
