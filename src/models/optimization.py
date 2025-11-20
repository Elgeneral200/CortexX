"""
Hyperparameter optimization module for CortexX sales forecasting platform.

Provides automated hyperparameter tuning using Optuna framework with time series-aware validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
import logging
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    A class to handle hyperparameter optimization for forecasting models.

    Uses Optuna framework with Bayesian optimization (TPE) for efficient search.
    Supports time series cross-validation to prevent data leakage.
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

    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                        metric: str = 'rmse') -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters using Optuna.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            metric (str): Optimization metric (rmse, mae, r2)

        Returns:
            Dict[str, Any]: Best hyperparameters and optimization results
        """
        try:
            self.logger.info(f"Starting XGBoost optimization with {self.n_trials} trials...")

            def objective(trial):
                # Define hyperparameter search space
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

                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                scores = []

                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    model = xgb.XGBRegressor(**params)
                    model.fit(X_tr, y_tr, verbose=False)
                    y_pred = model.predict(X_val)

                    if metric == 'rmse':
                        score = np.sqrt(mean_squared_error(y_val, y_pred))
                    elif metric == 'mae':
                        score = mean_absolute_error(y_val, y_pred)
                    elif metric == 'r2':
                        score = -r2_score(y_val, y_pred)  # Negative for minimization
                    else:
                        score = np.sqrt(mean_squared_error(y_val, y_pred))

                    scores.append(score)

                return np.mean(scores)

            # Create study and optimize
            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['xgboost'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_score': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"XGBoost optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in XGBoost optimization: {str(e)}")
            return {'best_params': {}, 'best_score': np.nan, 'error': str(e)}

    def optimize_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                         metric: str = 'rmse') -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using Optuna.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            metric (str): Optimization metric (rmse, mae, r2)

        Returns:
            Dict[str, Any]: Best hyperparameters and optimization results
        """
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

                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)

                    if metric == 'rmse':
                        score = np.sqrt(mean_squared_error(y_val, y_pred))
                    elif metric == 'mae':
                        score = mean_absolute_error(y_val, y_pred)
                    elif metric == 'r2':
                        score = -r2_score(y_val, y_pred)
                    else:
                        score = np.sqrt(mean_squared_error(y_val, y_pred))

                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['lightgbm'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_score': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"LightGBM optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in LightGBM optimization: {str(e)}")
            return {'best_params': {}, 'best_score': np.nan, 'error': str(e)}

    def optimize_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                              metric: str = 'rmse') -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters using Optuna.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            metric (str): Optimization metric (rmse, mae, r2)

        Returns:
            Dict[str, Any]: Best hyperparameters and optimization results
        """
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

                    model = RandomForestRegressor(**params)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)

                    if metric == 'rmse':
                        score = np.sqrt(mean_squared_error(y_val, y_pred))
                    elif metric == 'mae':
                        score = mean_absolute_error(y_val, y_pred)
                    elif metric == 'r2':
                        score = -r2_score(y_val, y_pred)
                    else:
                        score = np.sqrt(mean_squared_error(y_val, y_pred))

                    scores.append(score)

                return np.mean(scores)

            sampler = TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params['random_forest'] = self.study.best_params

            result = {
                'best_params': self.study.best_params,
                'best_score': self.study.best_value,
                'n_trials': len(self.study.trials),
                'optimization_history': [trial.value for trial in self.study.trials]
            }

            self.logger.info(f"Random Forest optimization complete. Best {metric}: {self.study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in Random Forest optimization: {str(e)}")
            return {'best_params': {}, 'best_score': np.nan, 'error': str(e)}

    def optimize_model(self, model_type: str, X_train: pd.DataFrame, 
                      y_train: pd.Series, metric: str = 'rmse') -> Dict[str, Any]:
        """
        Optimize hyperparameters for specified model type.

        Args:
            model_type (str): Model type ('xgboost', 'lightgbm', 'random_forest')
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            metric (str): Optimization metric

        Returns:
            Dict[str, Any]: Optimization results
        """
        if model_type.lower() == 'xgboost':
            return self.optimize_xgboost(X_train, y_train, metric)
        elif model_type.lower() == 'lightgbm':
            return self.optimize_lightgbm(X_train, y_train, metric)
        elif model_type.lower() == 'random_forest':
            return self.optimize_random_forest(X_train, y_train, metric)
        else:
            self.logger.error(f"Unsupported model type: {model_type}")
            return {'error': f'Unsupported model type: {model_type}'}

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of all optimizations performed.

        Returns:
            Dict[str, Any]: Summary of best parameters for all models
        """
        return {
            'best_parameters': self.best_params,
            'models_optimized': list(self.best_params.keys()),
            'total_models': len(self.best_params)
        }