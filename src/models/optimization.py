"""
Hyperparameter optimization module for CortexX sales forecasting platform.

Provides automated hyperparameter tuning using Optuna framework with time series-aware validation.
Supports all 11 forecasting algorithms in the platform.

FIXED: Added proper NaN handling and data preprocessing.
"""


import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
import logging
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer  # NEW: For handling NaN
from sklearn.preprocessing import StandardScaler  # NEW: For scaling
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)



class HyperparameterOptimizer:
    """
    A class to handle hyperparameter optimization for forecasting models.

    Uses Optuna framework with Bayesian optimization (TPE) for efficient search.
    Supports time series cross-validation to prevent data leakage.
    Handles NaN values and data preprocessing automatically.
    """


    def __init__(self, n_trials: int = 50, cv_splits: int = 3, random_state: int = 42):
        """
        Initialize optimizer with configuration.

        Args:
            n_trials (int): Number of optimization trials
            cv_splits (int): Number of time series CV splits
            random_state (int): Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.best_params = {}
        self.study = None
        
        # Initialize imputer and scaler
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()


    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series, 
                        fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data by handling NaN values and scaling.
        
        Args:
            X: Feature matrix
            y: Target variable
            fit: Whether to fit the imputer/scaler (True for training data)
            
        Returns:
            Preprocessed X and y as numpy arrays
        """
        # Handle NaN in features
        if fit:
            X_clean = self.imputer.fit_transform(X)
        else:
            X_clean = self.imputer.transform(X)
        
        # Handle NaN in target
        y_clean = y.fillna(y.mean())
        
        return X_clean, y_clean.values


    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                        metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Optuna."""
        try:
            self.logger.info(f"Starting XGBoost optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Preprocess data
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = xgb.XGBRegressor(**params)
                    model.fit(X_tr_clean, y_tr_clean, verbose=False)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['xgboost'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"XGBoost optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in XGBoost optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                         metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters using Optuna."""
        try:
            self.logger.info(f"Starting LightGBM optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'verbose': -1
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Preprocess data
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr_clean, y_tr_clean)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['lightgbm'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"LightGBM optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in LightGBM optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                              metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters using Optuna."""
        try:
            self.logger.info(f"Starting Random Forest optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Preprocess data
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = RandomForestRegressor(**params)
                    model.fit(X_tr_clean, y_tr_clean)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['random_forest'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"Random Forest optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in Random Forest optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                         metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters using Optuna."""
        try:
            self.logger.info(f"Starting CatBoost optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_state': self.random_state,
                    'verbose': 0
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Preprocess data
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = CatBoostRegressor(**params)
                    model.fit(X_tr_clean, y_tr_clean, verbose=False)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['catboost'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"CatBoost optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in CatBoost optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_ridge(self, X_train: pd.DataFrame, y_train: pd.Series,
                      metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize Ridge Regression hyperparameters using Optuna - FIXED NaN handling."""
        try:
            self.logger.info(f"Starting Ridge optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
                    'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
                    'random_state': self.random_state
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # FIXED: Preprocess data to remove NaN
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = Ridge(**params)
                    model.fit(X_tr_clean, y_tr_clean)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['ridge'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"Ridge optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in Ridge optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_lasso(self, X_train: pd.DataFrame, y_train: pd.Series,
                      metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize Lasso Regression hyperparameters using Optuna - FIXED NaN handling."""
        try:
            self.logger.info(f"Starting Lasso optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
                    'max_iter': trial.suggest_int('max_iter', 1000, 10000),
                    'random_state': self.random_state
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # FIXED: Preprocess data
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = Lasso(**params)
                    model.fit(X_tr_clean, y_tr_clean)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['lasso'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"Lasso optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in Lasso optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_decision_tree(self, X_train: pd.DataFrame, y_train: pd.Series,
                              metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize Decision Tree hyperparameters using Optuna."""
        try:
            self.logger.info(f"Starting Decision Tree optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': self.random_state
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Preprocess data
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = DecisionTreeRegressor(**params)
                    model.fit(X_tr_clean, y_tr_clean)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['decision_tree'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"Decision Tree optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in Decision Tree optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_knn(self, X_train: pd.DataFrame, y_train: pd.Series,
                    metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize K-Nearest Neighbors hyperparameters using Optuna."""
        try:
            self.logger.info(f"Starting KNN optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree']),
                    'leaf_size': trial.suggest_int('leaf_size', 10, 50),
                    'p': trial.suggest_int('p', 1, 2),
                    'n_jobs': -1
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Preprocess data
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = KNeighborsRegressor(**params)
                    model.fit(X_tr_clean, y_tr_clean)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['knn'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"KNN optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in KNN optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_svr(self, X_train: pd.DataFrame, y_train: pd.Series,
                    metric: str = 'rmse') -> Dict[str, Any]:
        """Optimize Support Vector Regression hyperparameters using Optuna."""
        try:
            self.logger.info(f"Starting SVR optimization with {self.n_trials} trials...")

            def objective(trial):
                params = {
                    'C': trial.suggest_float('C', 0.1, 100, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
                }

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Preprocess data
                    X_tr_clean, y_tr_clean = self._preprocess_data(X_tr, y_tr, fit=True)
                    X_val_clean, y_val_clean = self._preprocess_data(X_val, y_val, fit=False)

                    model = SVR(**params)
                    model.fit(X_tr_clean, y_tr_clean)
                    y_pred = model.predict(X_val_clean)

                    score = self._calculate_metric(y_val_clean, y_pred, metric)
                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['svr'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"SVR optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in SVR optimization: {str(e)}")
            return {'best_params': {}, 'best_value': np.nan, 'error': str(e)}


    def optimize_model(self, model_type: str, X_train: pd.DataFrame, 
                      y_train: pd.Series, metric: str = 'rmse') -> Dict[str, Any]:
        """
        Optimize hyperparameters for specified model type.

        Args:
            model_type (str): Model type
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            metric (str): Optimization metric

        Returns:
            Dict[str, Any]: Optimization results
        """
        model_type_lower = model_type.lower().replace(' ', '_').replace('-', '_')
        
        # Map model names to optimizer methods
        if model_type_lower == 'xgboost':
            return self.optimize_xgboost(X_train, y_train, metric)
        elif model_type_lower == 'lightgbm':
            return self.optimize_lightgbm(X_train, y_train, metric)
        elif model_type_lower == 'random_forest':
            return self.optimize_random_forest(X_train, y_train, metric)
        elif model_type_lower == 'catboost':
            return self.optimize_catboost(X_train, y_train, metric)
        elif model_type_lower in ['ridge', 'ridge_regression']:
            return self.optimize_ridge(X_train, y_train, metric)
        elif model_type_lower in ['lasso', 'lasso_regression']:
            return self.optimize_lasso(X_train, y_train, metric)
        elif model_type_lower == 'decision_tree':
            return self.optimize_decision_tree(X_train, y_train, metric)
        elif model_type_lower in ['knn', 'k_nearest_neighbors']:
            return self.optimize_knn(X_train, y_train, metric)
        elif model_type_lower in ['svr', 'support_vector_regression']:
            return self.optimize_svr(X_train, y_train, metric)
        elif model_type_lower in ['linear_regression', 'linearregression']:
            self.logger.info("Linear Regression has no hyperparameters to optimize")
            return {
                'best_params': {},
                'best_value': np.nan,
                'n_trials': 0,
                'message': 'Linear Regression has no hyperparameters to optimize'
            }
        elif model_type_lower == 'prophet':
            self.logger.info("Prophet optimization not yet implemented")
            return {
                'best_params': {},
                'best_value': np.nan,
                'n_trials': 0,
                'message': 'Prophet optimization not yet implemented'
            }
        else:
            self.logger.error(f"Unsupported model type: {model_type}")
            return {'error': f'Unsupported model type: {model_type}'}


    def _calculate_metric(self, y_true, y_pred, metric: str) -> float:
        """Calculate specified metric."""
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'r2':
            return -r2_score(y_true, y_pred)  # Negative for minimization
        else:
            return np.sqrt(mean_squared_error(y_true, y_pred))


    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations performed."""
        return {
            'best_parameters': self.best_params,
            'models_optimized': list(self.best_params.keys()),
            'total_models': len(self.best_params)
        }
