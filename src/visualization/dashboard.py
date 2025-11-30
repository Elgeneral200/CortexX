"""
Enhanced Interactive visualization module for CortexX sales forecasting platform.
FIXED: All Plotly charts now properly render in Streamlit.
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
    A class to create interactive visualizations for sales forecasting.
    ENHANCED: All methods now return Streamlit-compatible Plotly figures.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        self.logger = logging.getLogger(__name__)
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set1
        
    def create_sales_trend_plot(self, df: pd.DataFrame, date_col: str, 
                               value_col: str, title: str = "Sales Trend Over Time") -> go.Figure:
        """
        Create interactive sales trend visualization.
        FIXED: Enhanced error handling and Streamlit compatibility.
        """
        try:
            # Ensure data is properly formatted
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp = df_temp.sort_values(date_col)
            
            fig = px.line(df_temp, x=date_col, y=value_col, 
                         title=title,
                         template=self.theme,
                         color_discrete_sequence=[self.color_palette[0]])
            
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
            
        except Exception as e:
            self.logger.error(f"Error creating sales trend plot: {str(e)}")
            return self._create_error_plot("Sales Trend Plot", str(e))
    
    def create_seasonality_plot(self, df: pd.DataFrame, date_col: str,
                               value_col: str, title: str = "Seasonality Analysis") -> go.Figure:
        """
        Create seasonal decomposition plot.
        FIXED: Better data validation and error handling.
        """
        try:
            df_temp = df.copy()
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
                      name='Monthly', marker_color=self.color_palette[0]),
                row=1, col=1
            )
            
            # Weekly seasonality
            df_temp['day_of_week'] = df_temp[date_col].dt.dayofweek
            weekly_avg = df_temp.groupby('day_of_week')[value_col].mean().reset_index()
            fig.add_trace(
                go.Bar(x=weekly_avg['day_of_week'], y=weekly_avg[value_col],
                      name='Weekly', marker_color=self.color_palette[1]),
                row=1, col=2
            )
            
            # Year-over-year comparison
            df_temp['year'] = df_temp[date_col].dt.year
            yearly_data = df_temp.groupby('year')[value_col].sum().reset_index()
            fig.add_trace(
                go.Scatter(x=yearly_data['year'], y=yearly_data[value_col],
                          mode='lines+markers', name='Yearly',
                          line=dict(color=self.color_palette[2])),
                row=2, col=1
            )
            
            # Daily patterns (day of month)
            df_temp['day_of_month'] = df_temp[date_col].dt.day
            daily_avg = df_temp.groupby('day_of_month')[value_col].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=daily_avg['day_of_month'], y=daily_avg[value_col],
                          mode='lines', name='Daily',
                          line=dict(color=self.color_palette[3])),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                title_text=title,
                template=self.theme,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating seasonality plot: {str(e)}")
            return self._create_error_plot("Seasonality Plot", str(e))
    
    def create_correlation_heatmap(self, df: pd.DataFrame, 
                                  title: str = "Feature Correlation Matrix") -> go.Figure:
        """
        Create correlation heatmap for numerical features.
        FIXED: Better handling of empty dataframes.
        """
        try:
            # Select only numerical columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty or len(numeric_df.columns) < 2:
                return self._create_message_plot("Not enough numerical features for correlation analysis")
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr().round(3)
            
            # Create heatmap
            fig = px.imshow(corr_matrix, 
                          title=title,
                          color_continuous_scale='RdBu_r',
                          aspect="auto",
                          template=self.theme,
                          text_auto=True)
            
            fig.update_layout(
                height=600,
                xaxis_title="Features",
                yaxis_title="Features"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            return self._create_error_plot("Correlation Heatmap", str(e))
    
    def create_forecast_comparison_plot(self, 
                                       actual_values: np.ndarray, 
                                       predicted_values: np.ndarray,
                                       actual_dates: Optional[np.ndarray] = None,
                                       forecast_dates: Optional[np.ndarray] = None,
                                       model_name: str = "Model") -> go.Figure:
        """
        Create forecast vs actual comparison plot.
        FIXED: More flexible parameter handling for Streamlit integration.
        """
        try:
            # Handle different input formats
            if actual_dates is None:
                actual_dates = np.arange(len(actual_values))
            if forecast_dates is None:
                forecast_dates = np.arange(len(predicted_values))
            
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=actual_dates,
                y=actual_values,
                name='Actual',
                line=dict(color=self.color_palette[0], width=3),
                opacity=0.8
            ))
            
            # Add forecast values
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=predicted_values,
                name=f'{model_name} Forecast',
                line=dict(color=self.color_palette[1], width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'{model_name} - Forecast vs Actual',
                xaxis_title='Time',
                yaxis_title='Value',
                template=self.theme,
                hovermode='x unified',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating forecast comparison plot: {str(e)}")
            return self._create_error_plot("Forecast Comparison", str(e))
    
    def create_residual_analysis_plot(self, 
                                     actual_values: np.ndarray, 
                                     predicted_values: np.ndarray,
                                     title: str = "Residual Analysis") -> go.Figure:
        """
        Create residual analysis plots.
        FIXED: Better error handling and data validation.
        """
        try:
            if len(actual_values) != len(predicted_values):
                return self._create_message_plot("Actual and predicted values must have same length")
            
            residuals = predicted_values - actual_values
            
            # Create subplots
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
                          marker=dict(color=self.color_palette[0], opacity=0.6)),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Residual Distribution
            fig.add_trace(
                go.Histogram(x=residuals, name='Residual Distribution',
                           nbinsx=30, marker_color=self.color_palette[1]),
                row=1, col=2
            )
            
            # Q-Q Plot
            try:
                from scipy import stats
                sorted_residuals = np.sort(residuals)
                theoretical_quantiles = stats.norm.ppf(
                    np.linspace(0.01, 0.99, len(sorted_residuals))
                )
                
                fig.add_trace(
                    go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                              mode='markers', name='Q-Q Plot',
                              marker=dict(color=self.color_palette[2])),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                              mode='lines', name='Theoretical', 
                              line=dict(color='red', dash='dash')),
                    row=2, col=1
                )
            except ImportError:
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode='markers',
                              name='Q-Q Plot (scipy required)'),
                    row=2, col=1
                )
            
            # Residuals over time
            fig.add_trace(
                go.Scatter(x=np.arange(len(residuals)), y=residuals,
                          mode='lines', name='Residuals Over Time',
                          line=dict(color=self.color_palette[3])),
                row=2, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
            
            fig.update_layout(
                height=600,
                title_text=title,
                template=self.theme,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating residual analysis plot: {str(e)}")
            return self._create_error_plot("Residual Analysis", str(e))
    
    def create_feature_importance_plot(self, 
                                      importance_data: Union[pd.DataFrame, Dict],
                                      title: str = "Feature Importance") -> go.Figure:
        """
        Create feature importance bar chart.
        FIXED: Flexible input handling for different data formats.
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(importance_data, dict):
                importance_df = pd.DataFrame({
                    'feature': list(importance_data.keys()),
                    'importance': list(importance_data.values())
                })
            else:
                importance_df = importance_data.copy()
            
            if importance_df.empty:
                return self._create_message_plot("No feature importance data available")
            
            # Ensure correct column names
            if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
                if len(importance_df.columns) >= 2:
                    importance_df.columns = ['feature', 'importance'][:len(importance_df.columns)]
                else:
                    return self._create_message_plot("Invalid feature importance data format")
            
            # Sort by importance and take top 20
            importance_df = importance_df.sort_values('importance', ascending=True).tail(20)
            
            fig = px.bar(importance_df, 
                        x='importance', 
                        y='feature',
                        title=title,
                        orientation='h',
                        template=self.theme,
                        color='importance',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(
                height=max(400, len(importance_df) * 30),
                xaxis_title='Importance Score',
                yaxis_title='Features',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {str(e)}")
            return self._create_error_plot("Feature Importance", str(e))
    
    def create_model_comparison_plot(self, 
                                    comparison_df: pd.DataFrame, 
                                    metric: str = 'RMSE',
                                    title: str = "Model Comparison") -> go.Figure:
        """
        Create model comparison bar chart.
        FIXED: Better metric handling and error checking.
        """
        try:
            if comparison_df.empty:
                return self._create_message_plot("No model comparison data available")
            
            # Check if metric exists in dataframe
            available_metrics = comparison_df.columns.tolist()
            if metric not in available_metrics:
                st.warning(f"Metric '{metric}' not found. Available: {available_metrics}")
                metric = available_metrics[1] if len(available_metrics) > 1 else available_metrics[0]
            
            # Sort by metric (ascending for error metrics, descending for accuracy metrics)
            if metric.lower() in ['r2', 'accuracy', 'precision', 'recall']:
                comparison_df = comparison_df.sort_values(metric, ascending=False)
            else:
                comparison_df = comparison_df.sort_values(metric, ascending=True)
            
            fig = px.bar(comparison_df,
                        x='Model' if 'Model' in comparison_df.columns else comparison_df.index,
                        y=metric,
                        title=f'{title} - {metric}',
                        template=self.theme,
                        color=metric,
                        color_continuous_scale='Viridis')
            
            fig.update_layout(
                height=500,
                xaxis_title='Models',
                yaxis_title=metric,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating model comparison plot: {str(e)}")
            return self._create_error_plot("Model Comparison", str(e))
    
    def create_confidence_interval_plot(self,
                                       dates: np.ndarray,
                                       predictions: np.ndarray,
                                       lower_bounds: np.ndarray,
                                       upper_bounds: np.ndarray,
                                       actual_values: Optional[np.ndarray] = None,
                                       title: str = "Forecast with Confidence Intervals") -> go.Figure:
        """
        Create forecast plot with confidence intervals.
        NEW: Added for prediction intervals visualization.
        """
        try:
            fig = go.Figure()
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([upper_bounds, lower_bounds[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                line=dict(color='red', width=2),
                name='Forecast'
            ))
            
            # Actual values (if provided)
            if actual_values is not None:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=actual_values,
                    line=dict(color='blue', width=2),
                    name='Actual'
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Value',
                template=self.theme,
                hovermode='x unified',
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating confidence interval plot: {str(e)}")
            return self._create_error_plot("Confidence Interval Plot", str(e))

    def _create_error_plot(self, title: str, error_msg: str = "") -> go.Figure:
        """Create an error message plot."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating {title}<br><br><span style='color: red; font-size: 12px;'>{error_msg}</span>",
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


# Streamlit-specific helper functions
def display_plotly_chart(fig: go.Figure, use_container_width: bool = True):
    """
    Display Plotly chart in Streamlit with error handling.
    
    Args:
        fig (go.Figure): Plotly figure to display
        use_container_width (bool): Whether to use container width
    """
    try:
        if fig is None:
            st.error("No figure to display")
            return
            
        st.plotly_chart(fig, use_container_width=use_container_width)
        
    except Exception as e:
        st.error(f"Error displaying chart: {str(e)}")


def create_quick_plot(x_data, y_data, plot_type: str = 'line', **kwargs):
    """
    Quick plotting function for Streamlit.
    
    Args:
        x_data: X-axis data
        y_data: Y-axis data
        plot_type (str): Type of plot ('line', 'bar', 'scatter')
        **kwargs: Additional arguments
        
    Returns:
        go.Figure: Plotly figure
    """
    visualizer = VisualizationEngine()
    
    if plot_type == 'line':
        fig = px.line(x=x_data, y=y_data, **kwargs)
    elif plot_type == 'bar':
        fig = px.bar(x=x_data, y=y_data, **kwargs)
    elif plot_type == 'scatter':
        fig = px.scatter(x=x_data, y=y_data, **kwargs)
    else:
        fig = px.line(x=x_data, y=y_data, **kwargs)
    
    fig.update_layout(template='plotly_white')
    return fig