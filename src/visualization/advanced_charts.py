"""
Advanced Visualization Module for CortexX Forecasting Platform
PHASE 3 - SESSION 5: Interactive Charts and Advanced Visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AdvancedChartBuilder:
    """
    Build advanced interactive charts with Plotly.
    
    ✅ NEW: Phase 3 - Session 5
    """
    
    def __init__(self, theme: str = 'plotly_dark'):
        """
        Initialize chart builder.
        
        Args:
            theme: Plotly theme ('plotly_dark', 'plotly', 'simple_white')
        """
        self.theme = theme
        self.default_colors = px.colors.qualitative.Set2
    
    def create_distribution_plot(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int = 30,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create histogram/distribution plot.
        
        Args:
            df: DataFrame
            column: Column to plot
            bins: Number of bins
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if title is None:
            title = f"Distribution of {column}"
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df[column].dropna(),
            nbinsx=bins,
            name=column,
            marker_color='#667eea',
            opacity=0.75,
            hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_val = df[column].mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:,.2f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title=column,
            yaxis_title="Frequency",
            template=self.theme,
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
    
    def create_box_plot(
        self,
        df: pd.DataFrame,
        y_column: str,
        x_column: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create box plot for outlier detection.
        
        Args:
            df: DataFrame
            y_column: Numeric column for box plot
            x_column: Optional categorical column for grouping
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if title is None:
            title = f"Box Plot: {y_column}"
            if x_column:
                title += f" by {x_column}"
        
        if x_column and x_column in df.columns:
            fig = px.box(
                df,
                x=x_column,
                y=y_column,
                title=title,
                color=x_column,
                template=self.theme
            )
        else:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df[y_column].dropna(),
                name=y_column,
                marker_color='#667eea',
                boxmean='sd'  # Show mean and standard deviation
            ))
            fig.update_layout(
                title=title,
                yaxis_title=y_column,
                template=self.theme
            )
        
        fig.update_traces(
            hovertemplate='<b>%{y:,.2f}</b><extra></extra>'
        )
        
        return fig
    
    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Heatmap"
    ) -> go.Figure:
        """
        Create correlation heatmap for numeric columns.
        
        Args:
            df: DataFrame
            columns: Specific columns to include (defaults to all numeric)
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Get numeric columns
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            width=700
        )
        
        return fig
    
    def create_pie_chart(
        self,
        df: pd.DataFrame,
        column: str,
        values_column: Optional[str] = None,
        top_n: int = 10,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create pie chart for categorical breakdown.
        
        Args:
            df: DataFrame
            column: Categorical column
            values_column: Optional numeric column for values (uses count if None)
            top_n: Show only top N categories
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if title is None:
            title = f"Distribution by {column}"
        
        # Aggregate data
        if values_column:
            data = df.groupby(column)[values_column].sum().sort_values(ascending=False)
        else:
            data = df[column].value_counts()
        
        # Limit to top N
        if len(data) > top_n:
            top_data = data.head(top_n)
            others_sum = data.iloc[top_n:].sum()
            if others_sum > 0:
                top_data['Others'] = others_sum
            data = top_data
        
        fig = go.Figure(data=[go.Pie(
            labels=data.index,
            values=data.values,
            hole=0.3,  # Donut chart
            marker=dict(colors=px.colors.qualitative.Set2),
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create scatter plot with optional color and size encoding.
        
        Args:
            df: DataFrame
            x_column: X-axis column
            y_column: Y-axis column
            color_column: Optional column for color encoding
            size_column: Optional column for size encoding
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if title is None:
            title = f"{y_column} vs {x_column}"
        
        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            size=size_column,
            title=title,
            template=self.theme,
            hover_data=df.columns[:5]  # Show first 5 columns in hover
        )
        
        fig.update_traces(
            marker=dict(line=dict(width=0.5, color='DarkSlateGrey'))
        )
        
        fig.update_layout(
            hovermode='closest'
        )
        
        return fig
    
    def create_multi_line_chart(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        title: str = "Multi-Line Time Series"
    ) -> go.Figure:
        """
        Create multi-line chart for comparing multiple metrics over time.
        
        Args:
            df: DataFrame
            date_column: Date column
            value_columns: List of columns to plot
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for idx, col in enumerate(value_columns):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_column],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title=date_column,
            yaxis_title="Values",
            template=self.theme,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_area_chart(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        title: str = "Stacked Area Chart",
        stacked: bool = True
    ) -> go.Figure:
        """
        Create area chart (stacked or overlapping).
        
        Args:
            df: DataFrame
            date_column: Date column
            value_columns: List of columns to plot
            title: Chart title
            stacked: Whether to stack areas
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Pastel
        
        for idx, col in enumerate(value_columns):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_column],
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(width=0.5, color=colors[idx % len(colors)]),
                    stackgroup='one' if stacked else None,
                    fillcolor=colors[idx % len(colors)],
                    hovertemplate=f'<b>{col}</b><br>%{{y:,.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title=date_column,
            yaxis_title="Values",
            template=self.theme,
            hovermode='x unified'
        )
        
        return fig


class ChartExporter:
    """
    Export charts to various formats.
    
    ✅ NEW: Phase 3 - Session 5
    """
    
    @staticmethod
    def to_html(fig: go.Figure, filename: str = "chart.html") -> str:
        """
        Export chart to standalone HTML file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            
        Returns:
            HTML string
        """
        html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
        return html_str
    
    @staticmethod
    def to_image_bytes(fig: go.Figure, format: str = 'png', width: int = 1200, height: int = 800) -> bytes:
        """
        Export chart to image bytes.
        
        Args:
            fig: Plotly figure
            format: Image format ('png', 'jpeg', 'svg')
            width: Image width
            height: Image height
            
        Returns:
            Image bytes
        """
        try:
            img_bytes = fig.to_image(format=format, width=width, height=height)
            return img_bytes
        except Exception as e:
            logger.error(f"Error exporting chart to image: {e}")
            raise
class QualityVisualizer:
    """
    Visualizations for data quality analysis.
    
    ✅ NEW: Phase 3 - Session 6
    """
    
    @staticmethod
    def create_quality_gauge(score: float, title: str = "Overall Data Quality") -> go.Figure:
        """
        Create gauge chart for quality score.
        
        Args:
            score: Quality score (0-100)
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Determine color based on score
        if score >= 90:
            color = "#28a745"  # Green
        elif score >= 80:
            color = "#ffc107"  # Yellow
        elif score >= 70:
            color = "#fd7e14"  # Orange
        else:
            color = "#dc3545"  # Red
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 24}},
            delta={'reference': 100, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': '#ffebee'},
                    {'range': [60, 80], 'color': '#fff3e0'},
                    {'range': [80, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Arial"}
        )
        
        return fig
    
    @staticmethod
    def create_missing_values_heatmap(df: pd.DataFrame) -> go.Figure:
        """
        Create heatmap showing missing values pattern.
        
        Args:
            df: DataFrame to visualize
            
        Returns:
            Plotly figure
        """
        # Create binary matrix: 1 for missing, 0 for present
        missing_matrix = df.isnull().astype(int)
        
        # Limit to columns with missing values
        cols_with_missing = missing_matrix.sum()[missing_matrix.sum() > 0].index
        
        if len(cols_with_missing) == 0:
            # No missing values - create placeholder
            fig = go.Figure()
            fig.add_annotation(
                text="🎉 No Missing Values Detected!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="green")
            )
            fig.update_layout(
                height=300,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        missing_subset = missing_matrix[cols_with_missing]
        
        # Sample if too many rows
        if len(missing_subset) > 100:
            missing_subset = missing_subset.sample(100)
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_subset.T.values,
            x=missing_subset.index,
            y=cols_with_missing,
            colorscale=[[0, '#2ecc71'], [1, '#e74c3c']],
            showscale=False,
            hovertemplate='Row: %{x}<br>Column: %{y}<br>Missing: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Missing Values Pattern (Red = Missing, Green = Present)",
            xaxis_title="Row Index (sampled)",
            yaxis_title="Columns",
            height=max(300, len(cols_with_missing) * 30),
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_missing_values_bar(missing_summary: pd.DataFrame) -> go.Figure:
        """
        Create bar chart of missing values by column.
        
        Args:
            missing_summary: DataFrame with missing value statistics
            
        Returns:
            Plotly figure
        """
        if missing_summary.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="🎉 No Missing Values!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="green")
            )
            fig.update_layout(height=300)
            return fig
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=missing_summary['Missing %'],
            y=missing_summary['Column'],
            orientation='h',
            marker=dict(
                color=missing_summary['Missing %'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Missing %")
            ),
            text=missing_summary['Missing %'].round(1).astype(str) + '%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Columns",
            height=max(300, len(missing_summary) * 40),
            template='plotly_dark',
            showlegend=False
        )
        
        return fig


def get_quality_visualizer() -> QualityVisualizer:
    """Get quality visualizer instance."""
    return QualityVisualizer()



def get_chart_builder(theme: str = 'plotly_dark') -> AdvancedChartBuilder:
    """Get chart builder instance."""
    return AdvancedChartBuilder(theme=theme)
