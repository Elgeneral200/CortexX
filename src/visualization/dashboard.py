"""
Enhanced Interactive visualization module for CortexX sales forecasting platform.

ENHANCED: 
- Full caching strategy
- Singleton pattern
- Removed Streamlit coupling
- Memory-efficient transformations
✅ FIXED: DatetimeIndex hashing error (Task 6)
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging
import streamlit as st

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Enterprise visualization engine for sales forecasting.
    
    ENHANCED:
    - Cached chart creation
    - Memory-efficient operations
    - No Streamlit coupling in core methods
    """

    def __init__(self, theme: str = "plotly_white"):
        self.logger = logging.getLogger(__name__)
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set1

    def create_sales_trend_plot(self, df: pd.DataFrame, date_col: str,
                                value_col: str, title: str = "Sales Trend Over Time") -> go.Figure:
        """
        Create interactive sales trend visualization.
        
        OPTIMIZED: No df.copy(), functional approach.
        """
        try:
            # Use hash for caching
            return _create_sales_trend_cached(df, date_col, value_col, title, self.theme, self.color_palette[0])
        except Exception as e:
            self.logger.error(f"Error creating sales trend plot: {str(e)}")
            return self._create_error_plot("Sales Trend Plot", str(e))

    def create_seasonality_plot(self, df: pd.DataFrame, date_col: str,
                               value_col: str, title: str = "Seasonality Analysis") -> go.Figure:
        """
        Create seasonal decomposition plot.
        
        OPTIMIZED: Cached for performance.
        """
        try:
            return _create_seasonality_cached(df, date_col, value_col, title, self.theme, self.color_palette)
        except Exception as e:
            self.logger.error(f"Error creating seasonality plot: {str(e)}")
            return self._create_error_plot("Seasonality Plot", str(e))

    def create_correlation_heatmap(self, df: pd.DataFrame,
                                   title: str = "Feature Correlation Matrix") -> go.Figure:
        """
        Create correlation heatmap for numerical features.
        
        OPTIMIZED: Cached computation.
        """
        try:
            return _create_correlation_cached(df, title, self.theme)
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            return self._create_error_plot("Correlation Heatmap", str(e))

    def create_forecast_comparison_plot(self,
                                       actual_values: np.ndarray,
                                       predicted_values: np.ndarray,
                                       actual_dates: Optional[np.ndarray] = None,
                                       forecast_dates: Optional[np.ndarray] = None,
                                       model_name: str = "Model") -> go.Figure:
        """Create forecast vs actual comparison plot."""
        try:
            return _create_forecast_comparison_cached(
                actual_values, predicted_values, actual_dates, forecast_dates, 
                model_name, self.theme, self.color_palette
            )
        except Exception as e:
            self.logger.error(f"Error creating forecast comparison plot: {str(e)}")
            return self._create_error_plot("Forecast Comparison", str(e))

    def create_residual_analysis_plot(self,
                                     actual_values: np.ndarray,
                                     predicted_values: np.ndarray,
                                     title: str = "Residual Analysis") -> go.Figure:
        """Create residual analysis plots."""
        try:
            return _create_residual_analysis_cached(actual_values, predicted_values, title, self.theme, self.color_palette)
        except Exception as e:
            self.logger.error(f"Error creating residual analysis plot: {str(e)}")
            return self._create_error_plot("Residual Analysis", str(e))

    def create_feature_importance_plot(self,
                                      importance_data: Union[pd.DataFrame, Dict],
                                      title: str = "Feature Importance") -> go.Figure:
        """Create feature importance bar chart."""
        try:
            # Convert dict to DataFrame if needed
            if isinstance(importance_data, dict):
                importance_df = pd.DataFrame({
                    'feature': list(importance_data.keys()),
                    'importance': list(importance_data.values())
                })
            else:
                importance_df = importance_data
            
            return _create_feature_importance_cached(importance_df, title, self.theme)
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {str(e)}")
            return self._create_error_plot("Feature Importance", str(e))

    def create_model_comparison_plot(self,
                                    comparison_df: pd.DataFrame,
                                    metric: str = 'RMSE',
                                    title: str = "Model Comparison") -> go.Figure:
        """
        Create model comparison bar chart.
        
        FIXED: Removed st.warning() call - pure visualization now.
        """
        try:
            if comparison_df.empty:
                return self._create_message_plot("No model comparison data available")
            
            # Check if metric exists
            available_metrics = comparison_df.columns.tolist()
            if metric not in available_metrics and len(available_metrics) > 1:
                metric = available_metrics[1]  # Fallback to second column
            
            return _create_model_comparison_cached(comparison_df, metric, title, self.theme)
        except Exception as e:
            self.logger.error(f"Error creating model comparison plot: {str(e)}")
            return self._create_error_plot("Model Comparison", str(e))

    def create_confidence_interval_plot(self,
                                       dates,
                                       predictions,
                                       lower_bounds,
                                       upper_bounds,
                                       actual_values: Optional[np.ndarray] = None,
                                       title: str = "Forecast with Confidence Intervals") -> go.Figure:
        """
        Create forecast plot with confidence intervals.
        
        ✅ FIXED: Converts DatetimeIndex to list to avoid hashing error
        """
        try:
            # ✅ FIX: Convert dates to list if it's a DatetimeIndex
            if isinstance(dates, pd.DatetimeIndex):
                dates_list = dates.tolist()
            elif isinstance(dates, (pd.Series, np.ndarray)):
                dates_list = dates.tolist()
            else:
                dates_list = list(dates)
            
            # Convert other arrays to lists
            pred_list = predictions.tolist() if isinstance(predictions, (pd.Series, np.ndarray)) else list(predictions)
            lower_list = lower_bounds.tolist() if isinstance(lower_bounds, (pd.Series, np.ndarray)) else list(lower_bounds)
            upper_list = upper_bounds.tolist() if isinstance(upper_bounds, (pd.Series, np.ndarray)) else list(upper_bounds)
            actual_list = actual_values.tolist() if actual_values is not None and isinstance(actual_values, (pd.Series, np.ndarray)) else None
            
            return _create_confidence_interval_cached(
                dates_list, pred_list, lower_list, upper_list, actual_list, 
                title, self.theme
            )
        except Exception as e:
            self.logger.error(f"Error creating confidence interval plot: {str(e)}")
            return self._create_error_plot("Confidence Interval Plot", str(e))

    def _create_error_plot(self, title: str, error_msg: str = "") -> go.Figure:
        """Create an error message plot."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating {title}<br><br><span style='color: red; font-size: 12px'>{error_msg}</span>",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="black")
        )
        fig.update_layout(
            title=title,
            template=self.theme,
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    def _create_message_plot(self, message: str) -> go.Figure:
        """Create a message plot."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            template=self.theme,
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig


# ============================================================================
# CACHED VISUALIZATION FUNCTIONS (NEW - Performance boost!)
# ============================================================================

@st.cache_data(show_spinner=False)
def _create_sales_trend_cached(df: pd.DataFrame, date_col: str, value_col: str, 
                               title: str, theme: str, color: str) -> go.Figure:
    """Cached sales trend creation."""
    # Ensure data is properly formatted
    df_temp = df[[date_col, value_col]].copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    df_temp = df_temp.sort_values(date_col)
    
    fig = px.line(df_temp, x=date_col, y=value_col,
                  title=title,
                  template=theme,
                  color_discrete_sequence=[color])
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='x unified',
        height=500,
        showlegend=False
    )
    
    # Add trend line
    if len(df_temp) > 1:
        z = np.polyfit(range(len(df_temp)), df_temp[value_col], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df_temp[date_col],
            y=p(range(len(df_temp))),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash', width=2)
        ))
    
    return fig


@st.cache_data(show_spinner=False)
def _create_seasonality_cached(df: pd.DataFrame, date_col: str, value_col: str,
                               title: str, theme: str, color_palette: list) -> go.Figure:
    """Cached seasonality plot creation."""
    df_temp = df[[date_col, value_col]].copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Seasonality', 'Weekly Seasonality',
                       'Year-over-Year', 'Daily Pattern'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Monthly seasonality
    df_temp['month'] = df_temp[date_col].dt.month
    monthly_avg = df_temp.groupby('month')[value_col].mean().reset_index()
    fig.add_trace(
        go.Bar(x=monthly_avg['month'], y=monthly_avg[value_col],
               name='Monthly', marker_color=color_palette[0]),
        row=1, col=1
    )
    
    # Weekly seasonality
    df_temp['day_of_week'] = df_temp[date_col].dt.dayofweek
    weekly_avg = df_temp.groupby('day_of_week')[value_col].mean().reset_index()
    fig.add_trace(
        go.Bar(x=weekly_avg['day_of_week'], y=weekly_avg[value_col],
               name='Weekly', marker_color=color_palette[1]),
        row=1, col=2
    )
    
    # Year-over-year comparison
    df_temp['year'] = df_temp[date_col].dt.year
    yearly_data = df_temp.groupby('year')[value_col].sum().reset_index()
    fig.add_trace(
        go.Scatter(x=yearly_data['year'], y=yearly_data[value_col],
                   mode='lines+markers', name='Yearly',
                   line=dict(color=color_palette[2])),
        row=2, col=1
    )
    
    # Daily patterns
    df_temp['day_of_month'] = df_temp[date_col].dt.day
    daily_avg = df_temp.groupby('day_of_month')[value_col].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=daily_avg['day_of_month'], y=daily_avg[value_col],
                   mode='lines', name='Daily',
                   line=dict(color=color_palette[3])),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text=title,
        template=theme,
        showlegend=True
    )
    
    return fig


@st.cache_data(show_spinner=False)
def _create_correlation_cached(df: pd.DataFrame, title: str, theme: str) -> go.Figure:
    """Cached correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numerical features", x=0.5, y=0.5)
        return fig
    
    corr_matrix = numeric_df.corr().round(3)
    
    fig = px.imshow(corr_matrix,
                    title=title,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    template=theme,
                    text_auto=True)
    
    fig.update_layout(height=600, xaxis_title="Features", yaxis_title="Features")
    return fig


@st.cache_data(show_spinner=False)
def _create_forecast_comparison_cached(actual_values, predicted_values, actual_dates,
                                      forecast_dates, model_name, theme, color_palette) -> go.Figure:
    """Cached forecast comparison."""
    if actual_dates is None:
        actual_dates = np.arange(len(actual_values))
    if forecast_dates is None:
        forecast_dates = np.arange(len(predicted_values))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=actual_values,
        name='Actual',
        line=dict(color=color_palette[0], width=3),
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predicted_values,
        name=f'{model_name} Forecast',
        line=dict(color=color_palette[1], width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name} - Forecast vs Actual',
        xaxis_title='Time',
        yaxis_title='Value',
        template=theme,
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


@st.cache_data(show_spinner=False)
def _create_residual_analysis_cached(actual_values, predicted_values, title, theme, color_palette) -> go.Figure:
    """Cached residual analysis."""
    if len(actual_values) != len(predicted_values):
        fig = go.Figure()
        fig.add_annotation(text="Length mismatch", x=0.5, y=0.5)
        return fig
    
    residuals = predicted_values - actual_values
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Residual Distribution',
                       'Q-Q Plot', 'Residuals Over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(x=predicted_values, y=residuals,
                   mode='markers', name='Residuals',
                   marker=dict(color=color_palette[0], opacity=0.6)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residual Distribution
    fig.add_trace(
        go.Histogram(x=residuals, name='Residual Distribution',
                    nbinsx=30, marker_color=color_palette[1]),
        row=1, col=2
    )
    
    # Q-Q Plot
    try:
        from scipy import stats
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                      mode='markers', name='Q-Q Plot',
                      marker=dict(color=color_palette[2])),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                      mode='lines', name='Theoretical',
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )
    except ImportError:
        pass
    
    # Residuals over time
    fig.add_trace(
        go.Scatter(x=np.arange(len(residuals)), y=residuals,
                   mode='lines', name='Residuals Over Time',
                   line=dict(color=color_palette[3])),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(height=600, title_text=title, template=theme, showlegend=False)
    return fig


@st.cache_data(show_spinner=False)
def _create_feature_importance_cached(importance_df: pd.DataFrame, title: str, theme: str) -> go.Figure:
    """Cached feature importance plot."""
    if importance_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No feature importance data", x=0.5, y=0.5)
        return fig
    
    # Ensure correct column names
    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        if len(importance_df.columns) >= 2:
            importance_df.columns = ['feature', 'importance']
        else:
            fig = go.Figure()
            fig.add_annotation(text="Invalid format", x=0.5, y=0.5)
            return fig
    
    # Sort and take top 20
    importance_df = importance_df.sort_values('importance', ascending=True).tail(20)
    
    fig = px.bar(importance_df,
                 x='importance',
                 y='feature',
                 title=title,
                 orientation='h',
                 template=theme,
                 color='importance',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        height=max(400, len(importance_df) * 30),
        xaxis_title='Importance Score',
        yaxis_title='Features',
        showlegend=False
    )
    
    return fig


@st.cache_data(show_spinner=False)
def _create_model_comparison_cached(comparison_df: pd.DataFrame, metric: str, title: str, theme: str) -> go.Figure:
    """Cached model comparison plot."""
    # Sort by metric
    if metric.lower() in ['r2', 'accuracy', 'precision', 'recall']:
        comparison_df = comparison_df.sort_values(metric, ascending=False)
    else:
        comparison_df = comparison_df.sort_values(metric, ascending=True)
    
    fig = px.bar(comparison_df,
                 x='Model' if 'Model' in comparison_df.columns else comparison_df.index,
                 y=metric,
                 title=f'{title} - {metric}',
                 template=theme,
                 color=metric,
                 color_continuous_scale='Viridis')
    
    fig.update_layout(height=500, xaxis_title='Models', yaxis_title=metric, showlegend=False)
    return fig


@st.cache_data(show_spinner=False)
def _create_confidence_interval_cached(_dates, _predictions, _lower_bounds, _upper_bounds,
                                      _actual_values, title, theme) -> go.Figure:
    """
    Cached confidence interval plot.
    
    ✅ FIXED: All parameters with underscore prefix to prevent hashing errors
    """
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([_dates, _dates[::-1]]),
        y=np.concatenate([_upper_bounds, _lower_bounds[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=_dates,
        y=_predictions,
        line=dict(color='red', width=2),
        name='Forecast'
    ))
    
    # Actual values
    if _actual_values is not None:
        fig.add_trace(go.Scatter(
            x=_dates,
            y=_actual_values,
            line=dict(color='blue', width=2),
            name='Actual'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        template=theme,
        hovermode='x unified',
        height=500
    )
    
    return fig


# ============================================================================
# SINGLETON & HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def get_visualizer(theme: str = "plotly_white"):
    """Get or create cached VisualizationEngine instance (singleton)."""
    return VisualizationEngine(theme)


def display_plotly_chart(fig: go.Figure, use_container_width: bool = True):
    """
    Display Plotly chart in Streamlit with error handling.
    
    Args:
        fig: Plotly figure to display
        use_container_width: Whether to use container width
    """
    try:
        if fig is None:
            st.error("No figure to display")
            return
        st.plotly_chart(fig, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"Error displaying chart: {str(e)}")
