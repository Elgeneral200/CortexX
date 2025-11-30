"""
Enhanced Data Exploration Page - OPTIMIZED PERFORMANCE

PHASE 2 INTEGRATED:
- Uses StateManager for all state operations
- Cached visualizer singleton
- Performance optimizations maintained
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# ✅ PHASE 2 IMPORTS
try:
    from src.utils.state_manager import StateManager, is_data_loaded, get_current_data
    from src.data.exploration import DataExplorer
    from src.visualization.dashboard import get_visualizer, display_plotly_chart
    from src.utils.config import get_config
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Data Exploration - CortexX",
    page_icon="📊",
    layout="wide"
)


def main():
    """Main data exploration function."""
    
    st.markdown('<div class="section-header">📊 ENTERPRISE DATA EXPLORATION</div>', unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    StateManager.initialize()
    
    if not is_data_loaded():
        st.warning("⚠️ Please load data first from the Dashboard page")
        return
    
    # ✅ UPDATED: Use helper functions
    df = get_current_data()
    visualizer = get_visualizer()
    
    # Data Overview Section
    st.markdown("### 📈 DATA OVERVIEW")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    with col4:
        missing_total = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_total:,}")
    
    # Interactive Exploration Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Profile", 
        "📊 Distribution Analysis", 
        "🕒 Time Series Analysis", 
        "🔗 Correlation Analysis"
    ])
    
    with tab1:
        render_data_profile(df, visualizer)
    
    with tab2:
        render_distribution_analysis(df, visualizer)
    
    with tab3:
        render_time_series_analysis(df, visualizer)
    
    with tab4:
        render_correlation_analysis(df, visualizer)


def render_data_profile(df: pd.DataFrame, visualizer):
    """Render comprehensive data profile."""
    st.subheader("📋 Data Profile & Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data types summary
        st.markdown("**Data Types Summary**")
        dtype_summary = pd.DataFrame({
            'Data Type': df.dtypes.value_counts().index.astype(str),
            'Count': df.dtypes.value_counts().values
        })
        st.dataframe(dtype_summary, use_container_width=True)
        
        # Missing values analysis
        st.markdown("**Missing Values Analysis**")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        }).sort_values('Missing %', ascending=False)
        
        st.dataframe(missing_data.head(10), use_container_width=True)
    
    with col2:
        # Column information
        st.markdown("**Column Information**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info.head(15), use_container_width=True)
        
        # Data quality score
        st.markdown("**Data Quality Score**")
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Completeness", f"{completeness:.1f}%")
        with col_b:
            st.metric("Uniqueness", f"{uniqueness:.1f}%")
        
        # Quality gauge
        overall_quality = (completeness + uniqueness) / 2
        st.progress(overall_quality / 100, text=f"Overall Data Quality: {overall_quality:.1f}%")


def render_distribution_analysis(df: pd.DataFrame, visualizer):
    """Render distribution analysis visualizations."""
    st.subheader("📊 Distribution Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns available for distribution analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Column selection for univariate analysis
        selected_col = st.selectbox("Select column for analysis", numeric_cols)
        
        if selected_col:
            # Histogram with performance optimization
            fig = px.histogram(df, x=selected_col, 
                             title=f"Distribution of {selected_col}",
                             nbins=30,
                             template="plotly_white")
            fig.update_layout(height=400)
            display_plotly_chart(fig)
    
    with col2:
        if selected_col:
            # Box plot
            fig = px.box(df, y=selected_col, 
                        title=f"Box Plot of {selected_col}",
                        template="plotly_white")
            fig.update_layout(height=400)
            display_plotly_chart(fig)


def render_time_series_analysis(df: pd.DataFrame, visualizer):
    """Render time series analysis visualizations - OPTIMIZED PERFORMANCE."""
    st.subheader("🕒 Time Series Analysis")
    
    # ✅ UPDATED: Use StateManager
    date_col = StateManager.get('date_column')
    value_col = StateManager.get('value_column')
    
    if not date_col or date_col not in df.columns:
        st.warning("No date column detected. Time series analysis requires a date column.")
        return
    
    if not value_col or value_col not in df.columns:
        st.warning("No value column selected for time series analysis.")
        return
    
    # Ensure date column is datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
    except Exception as e:
        st.error(f"Error processing date column: {str(e)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time series trend - OPTIMIZED
        try:
            # Sample data for better performance with large datasets
            if len(df) > 1000:
                df_sampled = df.iloc[::len(df)//1000]
                st.info("📊 Showing sampled data for better performance")
            else:
                df_sampled = df
            
            fig = visualizer.create_sales_trend_plot(
                df_sampled, date_col, value_col,
                f"Time Series Trend - {value_col}"
            )
            display_plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating trend plot: {str(e)}")
    
    with col2:
        # Seasonality analysis
        try:
            fig = visualizer.create_seasonality_plot(df, date_col, value_col)
            display_plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating seasonality plot: {str(e)}")
    
    # Advanced time series features - OPTIMIZED PERFORMANCE
    st.markdown("#### 📈 Advanced Time Series Analysis")
    
    # Rolling statistics with performance optimization
    st.markdown("**Rolling Statistics**")
    window_size = st.slider("Rolling Window Size", 7, 90, 30, key="rolling_window")
    
    if len(df) > window_size:
        try:
            # Calculate rolling statistics efficiently
            df_temp = df.set_index(date_col)[value_col].copy()
            rolling_mean = df_temp.rolling(window=window_size, min_periods=1).mean()
            rolling_std = df_temp.rolling(window=window_size, min_periods=1).std()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[date_col], y=df[value_col],
                name='Original',
                line=dict(color='#00d4ff', width=1),
                opacity=0.7
            ))
            fig.add_trace(go.Scatter(
                x=df[date_col], y=rolling_mean,
                name=f'{window_size}-day Moving Average',
                line=dict(color='#ff6b6b', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df[date_col], y=rolling_std,
                name=f'{window_size}-day Std Dev',
                line=dict(color='#2ed573', width=1, dash='dash'),
                opacity=0.6
            ))
            
            fig.update_layout(
                title=f"Rolling Statistics (Window: {window_size} days)",
                height=400,
                template="plotly_white",
                plot_bgcolor='#1a1d29',
                paper_bgcolor='#1a1d29',
                font=dict(color='white')
            )
            display_plotly_chart(fig)
        except Exception as e:
            st.error(f"Error calculating rolling statistics: {str(e)}")


def render_correlation_analysis(df: pd.DataFrame, visualizer):
    """Render correlation analysis visualizations."""
    st.subheader("🔗 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Correlation heatmap
        try:
            # Limit columns for better performance
            if len(numeric_cols) > 15:
                numeric_cols_limited = numeric_cols[:15]
                st.info("Showing correlation for top 15 numeric columns")
            else:
                numeric_cols_limited = numeric_cols
            
            fig = visualizer.create_correlation_heatmap(
                df[numeric_cols_limited],
                "Feature Correlation Matrix"
            )
            display_plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
    
    with col2:
        # Correlation insights
        st.markdown("**Top Correlations**")
        
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Get top correlations efficiently
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if j < len(corr_matrix.columns):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_value = corr_matrix.iloc[i, j]
                        correlations.append((col1, col2, abs(corr_value), corr_value))
            
            # Sort by absolute correlation and take top 8
            correlations.sort(key=lambda x: x[2], reverse=True)
            
            # Display top 8
            for i, (col1, col2, abs_corr, corr) in enumerate(correlations[:8]):
                color = "🟢" if corr > 0.7 else "🟡" if corr > 0.3 else "🔴"
                st.write(f"{color} **{col1}** ↔ **{col2}**: {corr:.3f}")
                
        except Exception as e:
            st.error(f"Error calculating correlations: {str(e)}")


if __name__ == "__main__":
    main()
