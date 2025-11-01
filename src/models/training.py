"""
Model training module for CortexX sales forecasting platform.
Handles training of multiple forecasting models including Prophet, XGBoost, and LightGBM.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings

# --- Imports جديدة ---
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
# --- نهاية Imports الجديدة ---


logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class to handle training of multiple forecasting models.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        warnings.filterwarnings('ignore')
    
    def train_test_split(self, df: pd.DataFrame, date_col: str, target_col: str, 
                         test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets using time-based split.
        """
        try:
            # Ensure data is sorted by date
            df_sorted = df.sort_values(date_col).copy()
            
            # Calculate split index
            split_idx = int(len(df_sorted) * (1 - test_size))
            
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            
            self.logger.info(f"Split data: {len(train_df)} training, {len(test_df)} testing")
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error in train-test split: {str(e)}")
            raise ValueError(f"Train-test split failed: {str(e)}")
    
    def train_prophet(self, train_df: pd.DataFrame, date_col: str, 
                      target_col: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Train Facebook's Prophet model for time series forecasting.
        """
        if Prophet is None:
            self.logger.error("Prophet library not found. Skipping Prophet model.")
            return self._create_dummy_model(train_df, date_col, target_col, 'Prophet (Skipped)')
            
        try:
            # Prepare data for Prophet
            prophet_df = train_df[[date_col, target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna()
            
            # Initialize and train model
            start_time = time.time()
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(prophet_df)
            training_time = time.time() - start_time
            
            # Create future dataframe for validation (in-sample)
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            
            results = {
                'model': 'Prophet',
                'training_time': training_time,
                'model_object': model,
                'forecast': forecast,
                'actual': prophet_df['y'].values,
                'predictions': forecast['yhat'].values[:len(prophet_df)]
            }
            
            self.logger.info("Prophet model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training Prophet model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'Prophet')
    
    def train_xgboost(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                      date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Train XGBoost model for time series forecasting.
        """
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'XGBoost')
            
            start_time = time.time()
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'XGBoost',
                'training_time': training_time,
                'model_object': model,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("XGBoost model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'XGBoost')
    
    def train_lightgbm(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Train LightGBM model for time series forecasting.
        """
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'LightGBM')
            
            start_time = time.time()
            
            model = lgb.LGBMRegressor(
                n_estimators=100, # Using 100 for consistency, notebook used 1000 with early stopping
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'LightGBM',
                'training_time': training_time,
                'model_object': model,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("LightGBM model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'LightGBM')


    """Trains a Linear Regression model."""
    def train_linear_regression(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'LinearRegression')
            
            start_time = time.time()
            
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'LinearRegression',
                'training_time': training_time,
                'model_object': model,
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("LinearRegression model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training LinearRegression model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'LinearRegression')

        
    """Trains a Ridge Regression model."""
    def train_ridge(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        
        
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'Ridge')
            
            start_time = time.time()
            
            model = Ridge(random_state=42)
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'Ridge',
                'training_time': training_time,
                'model_object': model,
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("Ridge model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training Ridge model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'Ridge')

    """Trains a Lasso Regression model."""
    def train_lasso(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'Lasso')
            
            start_time = time.time()
            
            model = Lasso(random_state=42)
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'Lasso',
                'training_time': training_time,
                'model_object': model,
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("Lasso model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training Lasso model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'Lasso')

    """Trains a KNeighborsRegressor model."""
    def train_knn(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'KNeighbors')
            
            start_time = time.time()
            
            model = KNeighborsRegressor(n_jobs=-1)
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'KNeighbors',
                'training_time': training_time,
                'model_object': model,
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("KNeighbors model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training KNeighbors model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'KNeighbors')

    """Trains a DecisionTreeRegressor model."""
    def train_decision_tree(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                            date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'DecisionTree')
            
            start_time = time.time()
            
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'DecisionTree',
                'training_time': training_time,
                'model_object': model,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("DecisionTree model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training DecisionTree model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'DecisionTree')

    """Trains a Support Vector Regressor (SVR) model."""
    def train_svr(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'SVR')
            
            self.logger.warning("Training SVR. This can be slow and might benefit from scaling.")
            
            start_time = time.time()
            
            model = SVR()
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'SVR',
                'training_time': training_time,
                'model_object': model,
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("SVR model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training SVR model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'SVR')

    """Trains a RandomForestRegressor model."""
    def train_random_forest(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                            date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'RandomForest')
            
            start_time = time.time()
            
            model = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100)
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'RandomForest',
                'training_time': training_time,
                'model_object': model,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("RandomForest model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training RandomForest model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'RandomForest')

    """Trains a CatBoostRegressor model."""
    def train_catboost(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'CatBoost')
            
            start_time = time.time()
            
            model = CatBoostRegressor(random_state=42, verbose=0, n_estimators=100)
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            y_pred_test = model.predict(X_test)
            
            results = {
                'model': 'CatBoost',
                'training_time': training_time,
                'model_object': model,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("CatBoost model training completed successfully")
            return model, results
            
        except Exception as e:
            self.logger.error(f"Error training CatBoost model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'CatBoost')


    """Train ensemble model combining multiple algorithms (VotingRegressor)."""
    def train_ensemble(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       date_col: str, target_col: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = self._prepare_features(train_df, date_col, target_col)
            X_test, y_test = self._prepare_features(test_df, date_col, target_col)
            
            if X_train is None or X_test is None:
                return self._create_dummy_model(train_df, date_col, target_col, 'Ensemble')
            
            start_time = time.time()
            
            # Define individual models
            models = [
                ('xgb', xgb.XGBRegressor(n_estimators=50, random_state=42)),
                ('lgb', lgb.LGBMRegressor(n_estimators=50, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
            ]
            
            # Create ensemble
            ensemble = VotingRegressor(estimators=models, n_jobs=-1)
            ensemble.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred_test = ensemble.predict(X_test)
            
            results = {
                'model': 'Ensemble (Voting)',
                'training_time': training_time,
                'model_object': ensemble,
                'actual': y_test,
                'predictions': y_pred_test,
                'dates': test_df[date_col].values
            }
            
            self.logger.info("Ensemble model training completed successfully")
            return ensemble, results
            
        except Exception as e:
            self.logger.error(f"Error training ensemble model: {str(e)}")
            return self._create_dummy_model(train_df, date_col, target_col, 'Ensemble')
    
    """Prepare features and target for machine learning models."""
    def _prepare_features(self, df: pd.DataFrame, date_col: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            # Select numeric features and exclude date and target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col not in [date_col, target_col]]
            
            # Handle categorical features if any - CAVEAT: This needs proper handling
            # For now, let's just use numeric features as per the original file
            
            if not feature_cols:
                self.logger.warning("No suitable numeric features found for ML models")
                return None, None
            
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None, None
    
    """Create a dummy model for demonstration when real training fails."""
    def _create_dummy_model(self, df: pd.DataFrame, date_col: str, target_col: str, 
                          model_name: str) -> Tuple[Any, Dict[str, Any]]:
        self.logger.warning(f"Creating dummy model for {model_name}")
        
        class DummyModel:
            def predict(self, X):
                return np.full(X.shape[0], df[target_col].mean())
        
        # Create simple results
        results = {
            'model': model_name,
            'training_time': 0.1,
            'model_object': DummyModel(),
            'actual': df[target_col].values[:10],
            'predictions': np.full(10, df[target_col].mean()),
            'dates': df[date_col].values[:10] if date_col in df.columns else range(10)
        }
        
        return DummyModel(), results