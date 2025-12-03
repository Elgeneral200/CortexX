"""
Forecasting UI Components for CortexX Forecasting Platform
PHASE 3 - SESSION 10: Interactive Forecast Configuration and Display
✅ UPDATED: All 9 ML Models from your project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ForecastConfigurator:
    """
    Handle forecast configuration and parameter selection.
    
    ✅ NEW: Phase 3 - Session 10
    ✅ UPDATED: All 9 models from CortexX project
    """
    
    MODELS = {
        'XGBoost': 'xgboost',
        'LightGBM': 'lightgbm',
        'CatBoost': 'catboost',
        'Random Forest': 'random_forest',
        'Lasso': 'lasso',
        'Ridge': 'ridge',
        'Linear Regression': 'linear',
        'Decision Tree': 'decision_tree',
        'KNN': 'knn'
    }
    
    @staticmethod
    def get_model_parameters(model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for each model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of default parameters
        """
        if model_name == 'xgboost':
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif model_name == 'lightgbm':
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'min_child_samples': 20
            }
        elif model_name == 'catboost':
            return {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3
            }
        elif model_name == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }
        elif model_name == 'lasso':
            return {
                'alpha': 1.0,
                'max_iter': 1000
            }
        elif model_name == 'ridge':
            return {
                'alpha': 1.0,
                'max_iter': 1000
            }
        elif model_name == 'linear':
            return {
                'fit_intercept': True,
                'normalize': False
            }
        elif model_name == 'decision_tree':
            return {
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': None
            }
        elif model_name == 'knn':
            return {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            }
        return {}
    
    @staticmethod
    def render_model_parameters(model_name: str) -> Dict[str, Any]:
        """
        Render interactive parameter controls for selected model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of user-selected parameters
        """
        params = {}
        
        st.markdown("#### 🎛️ Model Parameters")
        
        # ========================================================================
        # XGBOOST
        # ========================================================================
        if model_name == 'xgboost':
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider(
                    "Number of Trees",
                    min_value=10, max_value=500, value=100, step=10,
                    help="More trees = better fit but slower"
                )
                params['learning_rate'] = st.slider(
                    "Learning Rate",
                    min_value=0.01, max_value=0.3, value=0.1, step=0.01,
                    help="Lower = more robust but slower"
                )
                params['subsample'] = st.slider(
                    "Subsample Ratio",
                    min_value=0.5, max_value=1.0, value=0.8, step=0.1,
                    help="Fraction of samples for training"
                )
            with col2:
                params['max_depth'] = st.slider(
                    "Max Depth",
                    min_value=3, max_value=15, value=5,
                    help="Maximum tree depth"
                )
                params['min_child_weight'] = st.slider(
                    "Min Child Weight",
                    min_value=1, max_value=10, value=1,
                    help="Minimum sum of weights in child"
                )
                params['colsample_bytree'] = st.slider(
                    "Column Sample Ratio",
                    min_value=0.5, max_value=1.0, value=0.8, step=0.1,
                    help="Fraction of features for training"
                )
        
        # ========================================================================
        # LIGHTGBM
        # ========================================================================
        elif model_name == 'lightgbm':
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider(
                    "Number of Trees",
                    min_value=10, max_value=500, value=100, step=10
                )
                params['learning_rate'] = st.slider(
                    "Learning Rate",
                    min_value=0.01, max_value=0.3, value=0.1, step=0.01
                )
                params['num_leaves'] = st.slider(
                    "Number of Leaves",
                    min_value=10, max_value=100, value=31,
                    help="Max number of leaves in one tree"
                )
            with col2:
                params['max_depth'] = st.slider(
                    "Max Depth",
                    min_value=3, max_value=15, value=5
                )
                params['min_child_samples'] = st.slider(
                    "Min Child Samples",
                    min_value=5, max_value=100, value=20,
                    help="Minimum number of data needed in a child"
                )
        
        # ========================================================================
        # CATBOOST
        # ========================================================================
        elif model_name == 'catboost':
            col1, col2 = st.columns(2)
            with col1:
                params['iterations'] = st.slider(
                    "Iterations",
                    min_value=10, max_value=500, value=100, step=10
                )
                params['learning_rate'] = st.slider(
                    "Learning Rate",
                    min_value=0.01, max_value=0.3, value=0.1, step=0.01
                )
            with col2:
                params['depth'] = st.slider(
                    "Depth",
                    min_value=3, max_value=12, value=6
                )
                params['l2_leaf_reg'] = st.slider(
                    "L2 Regularization",
                    min_value=1, max_value=10, value=3,
                    help="L2 regularization coefficient"
                )
        
        # ========================================================================
        # RANDOM FOREST
        # ========================================================================
        elif model_name == 'random_forest':
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider(
                    "Number of Trees",
                    min_value=10, max_value=500, value=100, step=10
                )
                params['max_depth'] = st.slider(
                    "Max Depth",
                    min_value=3, max_value=20, value=10,
                    help="Maximum depth of the tree"
                )
                params['min_samples_split'] = st.slider(
                    "Min Samples Split",
                    min_value=2, max_value=20, value=2,
                    help="Minimum samples required to split"
                )
            with col2:
                params['min_samples_leaf'] = st.slider(
                    "Min Samples Leaf",
                    min_value=1, max_value=20, value=1,
                    help="Minimum samples required at leaf"
                )
                params['max_features'] = st.selectbox(
                    "Max Features",
                    options=['sqrt', 'log2', None],
                    index=0,
                    help="Number of features to consider"
                )
        
        # ========================================================================
        # LASSO
        # ========================================================================
        elif model_name == 'lasso':
            params['alpha'] = st.slider(
                "Alpha (Regularization)",
                min_value=0.0001, max_value=10.0, value=1.0, step=0.1,
                help="Regularization strength. Higher = more regularization"
            )
            params['max_iter'] = st.slider(
                "Max Iterations",
                min_value=100, max_value=5000, value=1000, step=100,
                help="Maximum number of iterations"
            )
        
        # ========================================================================
        # RIDGE
        # ========================================================================
        elif model_name == 'ridge':
            params['alpha'] = st.slider(
                "Alpha (Regularization)",
                min_value=0.0001, max_value=10.0, value=1.0, step=0.1,
                help="Regularization strength. Higher = more regularization"
            )
            params['max_iter'] = st.slider(
                "Max Iterations",
                min_value=100, max_value=5000, value=1000, step=100
            )
        
        # ========================================================================
        # LINEAR REGRESSION
        # ========================================================================
        elif model_name == 'linear':
            params['fit_intercept'] = st.checkbox(
                "Fit Intercept",
                value=True,
                help="Calculate intercept for the model"
            )
            params['normalize'] = st.checkbox(
                "Normalize Features",
                value=False,
                help="Normalize features before regression"
            )
        
        # ========================================================================
        # DECISION TREE
        # ========================================================================
        elif model_name == 'decision_tree':
            col1, col2 = st.columns(2)
            with col1:
                params['max_depth'] = st.slider(
                    "Max Depth",
                    min_value=1, max_value=30, value=10,
                    help="Maximum depth of the tree"
                )
                params['min_samples_split'] = st.slider(
                    "Min Samples Split",
                    min_value=2, max_value=20, value=2
                )
            with col2:
                params['min_samples_leaf'] = st.slider(
                    "Min Samples Leaf",
                    min_value=1, max_value=20, value=1
                )
                params['max_features'] = st.selectbox(
                    "Max Features",
                    options=['sqrt', 'log2', None],
                    index=2
                )
        
        # ========================================================================
        # KNN
        # ========================================================================
        elif model_name == 'knn':
            col1, col2 = st.columns(2)
            with col1:
                params['n_neighbors'] = st.slider(
                    "Number of Neighbors",
                    min_value=1, max_value=20, value=5,
                    help="Number of neighbors to consider"
                )
                params['weights'] = st.selectbox(
                    "Weights",
                    options=['uniform', 'distance'],
                    help="Weight function used in prediction"
                )
            with col2:
                params['algorithm'] = st.selectbox(
                    "Algorithm",
                    options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                    help="Algorithm to compute nearest neighbors"
                )
        
        return params


class ForecastVisualizer:
    """
    Visualize forecast results with interactive charts.
    
    ✅ NEW: Phase 3 - Session 10
    """
    
    @staticmethod
    def create_forecast_plot(
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        date_col: str,
        value_col: str,
        forecast_col: str = 'forecast',
        lower_bound_col: Optional[str] = None,
        upper_bound_col: Optional[str] = None,
        title: str = "Sales Forecast"
    ) -> go.Figure:
        """
        Create interactive forecast visualization.
        
        Args:
            historical_df: Historical data
            forecast_df: Forecast data
            date_col: Date column name
            value_col: Actual values column
            forecast_col: Forecast column name
            lower_bound_col: Lower confidence bound
            upper_bound_col: Upper confidence bound
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Historical actual values
        fig.add_trace(go.Scatter(
            x=historical_df[date_col],
            y=historical_df[value_col],
            mode='lines',
            name='Historical',
            line=dict(color='#667eea', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Actual:</b> %{y:,.2f}<extra></extra>'
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df[forecast_col],
            mode='lines',
            name='Forecast',
            line=dict(color='#f093fb', width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.2f}<extra></extra>'
        ))
        
        # Confidence interval
        if lower_bound_col and upper_bound_col:
            if lower_bound_col in forecast_df.columns and upper_bound_col in forecast_df.columns:
                # Upper bound
                fig.add_trace(go.Scatter(
                    x=forecast_df[date_col],
                    y=forecast_df[upper_bound_col],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Lower bound with fill
                fig.add_trace(go.Scatter(
                    x=forecast_df[date_col],
                    y=forecast_df[lower_bound_col],
                    mode='lines',
                    name='Confidence Interval',
                    fill='tonexty',
                    fillcolor='rgba(240, 147, 251, 0.2)',
                    line=dict(width=0),
                    hovertemplate='<b>Range:</b> %{y:,.2f} - Upper<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def create_residuals_plot(
        actual: pd.Series,
        predicted: pd.Series,
        dates: pd.Series
    ) -> go.Figure:
        """
        Create residuals (errors) plot.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            dates: Date series
            
        Returns:
            Plotly figure
        """
        residuals = actual - predicted
        
        fig = go.Figure()
        
        # Residuals scatter
        fig.add_trace(go.Scatter(
            x=dates,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='#667eea', size=6, opacity=0.6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Error:</b> %{y:,.2f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title="Forecast Residuals (Errors)",
            xaxis_title="Date",
            yaxis_title="Error",
            template='plotly_dark',
            height=350
        )
        
        return fig
    
    @staticmethod
    def create_performance_metrics_chart(metrics: Dict[str, float]) -> go.Figure:
        """
        Create bar chart of performance metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        bar_colors = [colors[i % len(colors)] for i in range(len(metric_names))]
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=bar_colors,
            text=[f'{v:.4f}' for v in metric_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            template='plotly_dark',
            height=350,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_model_comparison_chart(comparison_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create comparison chart for multiple models.
        
        Args:
            comparison_data: List of model results with metrics
            
        Returns:
            Plotly figure
        """
        models = [d['model'] for d in comparison_data]
        rmse = [d['metrics']['RMSE'] for d in comparison_data]
        mae = [d['metrics']['MAE'] for d in comparison_data]
        r2 = [d['metrics']['R²'] for d in comparison_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='RMSE',
            x=models,
            y=rmse,
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Bar(
            name='MAE',
            x=models,
            y=mae,
            marker_color='#764ba2'
        ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Error Value",
            template='plotly_dark',
            height=400,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig


def calculate_forecast_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate forecast performance metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Remove NaN values
    mask = ~(actual.isna() | predicted.isna())
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {
            'RMSE': 0.0,
            'MAE': 0.0,
            'MAPE': 0.0,
            'R²': 0.0
        }
    
    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    mae = mean_absolute_error(actual_clean, predicted_clean)
    r2 = r2_score(actual_clean, predicted_clean)
    
    # MAPE (avoiding division by zero)
    mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean.replace(0, np.nan))) * 100
    mape = mape if not np.isnan(mape) else 0.0
    
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R²': float(r2)
    }
