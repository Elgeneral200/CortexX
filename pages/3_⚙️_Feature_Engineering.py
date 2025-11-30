"""
Professional Feature Engineering Page - FIXED Rolling Features Display

PHASE 2 INTEGRATED:
- Uses StateManager for all state operations
- Cached singletons where applicable
- All optimizations preserved
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# ✅ PHASE 2 IMPORTS
try:
    from src.utils.state_manager import StateManager, is_data_loaded, get_current_data
    from src.features.engineering import FeatureEngineer
    from src.visualization.dashboard import get_visualizer, display_plotly_chart
    from src.utils.config import get_config
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Feature engineering modules not available: {e}")
    MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Feature Engineering - CortexX",
    page_icon="⚙️",
    layout="wide"
)


def main():
    """Main feature engineering function."""
    
    st.markdown('<div class="section-header">⚙️ ENTERPRISE FEATURE ENGINEERING</div>', unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    StateManager.initialize()
    
    if not is_data_loaded():
        st.warning("⚠️ Please load data first from the Dashboard page")
        return
    
    if not MODULES_AVAILABLE:
        st.error("❌ Feature engineering modules not available.")
        return
    
    # ✅ UPDATED: Use helper function
    df = get_current_data().copy()
    
    # Feature Engineering Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Time Features", 
        "🔄 Lag Features", 
        "📊 Rolling Features", 
        "🎯 Feature Summary"
    ])
    
    with tab1:
        render_time_features(df)
    
    with tab2:
        render_lag_features(df)
    
    with tab3:
        render_rolling_features(df)
    
    with tab4:
        render_feature_summary(df)


def render_time_features(df: pd.DataFrame):
    """Render time-based feature engineering."""
    
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">🕒 TIME-BASED FEATURES</div>
        <div class="card-description">Extract meaningful time components from your date column to capture seasonal patterns, trends, and cyclical behavior in your time series data</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    date_col = StateManager.get('date_column')
    
    if not date_col or date_col not in df.columns:
        st.warning("No date column detected. Time features require a date column.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🛠️ AVAILABLE TIME FEATURES**")
        
        time_features = {
            'Year': 'Extract year component from dates',
            'Month': 'Extract month (1-12) from dates',
            'Quarter': 'Extract quarter (1-4) from dates',
            'Week': 'Extract week number from dates',
            'Day of Week': 'Day of week (0-6) from dates',
            'Day of Month': 'Day of month (1-31) from dates',
            'Day of Year': 'Day of year (1-365) from dates',
            'Is Weekend': 'Boolean indicator for weekend days',
            'Is Month Start': 'Boolean for first day of month',
            'Is Month End': 'Boolean for last day of month'
        }
        
        selected_time_features = []
        for feature, description in time_features.items():
            if st.checkbox(f"{feature}", value=True, help=description):
                selected_time_features.append(feature.lower().replace(' ', '_'))
    
    with col2:
        st.markdown("**🚀 FEATURE GENERATION**")
        
        if st.button("🛠️ CREATE TIME FEATURES", type="primary", use_container_width=True):
            with st.spinner("Creating advanced time features..."):
                try:
                    engineer = FeatureEngineer()
                    df_enhanced = engineer.create_time_features(df, date_col)
                    
                    # Show new features
                    new_columns = [col for col in df_enhanced.columns if col not in df.columns]
                    
                    if new_columns:
                        # ✅ UPDATED: Use StateManager
                        StateManager.set('current_data', df_enhanced)
                        st.success(f"✅ Created {len(new_columns)} time features!")
                        
                        # Display new features
                        st.markdown("**📋 NEW FEATURES CREATED**")
                        feature_grid = st.columns(3)
                        for i, col in enumerate(new_columns[:9]):
                            with feature_grid[i % 3]:
                                st.info(f"`{col}`")
                        
                    else:
                        st.warning("No new time features were created")
                        
                except Exception as e:
                    st.error(f"❌ Error creating time features: {str(e)}")


def render_lag_features(df: pd.DataFrame):
    """Render lag feature engineering."""
    
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">🔄 LAG FEATURES</div>
        <div class="card-description">Create lagged versions of your target variable to capture temporal dependencies and autocorrelation patterns in time series data for improved forecasting accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    value_col = StateManager.get('value_column')
    
    if not value_col or value_col not in df.columns:
        st.warning("No value column selected. Please set a target value column.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚙️ LAG CONFIGURATION**")
        
        # Common lag periods
        lag_options = {
            '1 day (short-term)': 1,
            '7 days (weekly pattern)': 7,
            '14 days (bi-weekly)': 14,
            '30 days (monthly trend)': 30,
            '90 days (quarterly)': 90,
            '180 days (half-year)': 180,
            '365 days (yearly seasonality)': 365
        }
        
        selected_lags = []
        for label, lag_value in lag_options.items():
            if st.checkbox(f"{label}", value=lag_value in [1, 7, 30], help=f"Create lag_{lag_value} feature"):
                selected_lags.append(lag_value)
    
    with col2:
        st.markdown("**📊 FEATURE ANALYSIS**")
        
        if st.button("🔄 CREATE LAG FEATURES", type="primary", use_container_width=True):
            if not selected_lags:
                st.warning("Please select at least one lag period")
                return
                
            with st.spinner("Creating advanced lag features..."):
                try:
                    engineer = FeatureEngineer()
                    df_lagged = engineer.create_lag_features(df, value_col, lags=selected_lags)
                    
                    # Show new features
                    new_columns = [col for col in df_lagged.columns if col not in df.columns]
                    
                    if new_columns:
                        # ✅ UPDATED: Use StateManager
                        StateManager.set('current_data', df_lagged)
                        st.success(f"✅ Created {len(new_columns)} lag features!")
                        
                        # Display correlation with target
                        st.markdown("**📈 LAG CORRELATION ANALYSIS**")
                        lag_correlations = []
                        for lag_col in new_columns:
                            if lag_col in df_lagged.columns and value_col in df_lagged.columns:
                                corr = df_lagged[lag_col].corr(df_lagged[value_col])
                                lag_correlations.append((lag_col, corr))
                        
                        # Sort by absolute correlation
                        lag_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        # Show correlation chart
                        if lag_correlations:
                            corr_df = pd.DataFrame(lag_correlations[:10], columns=['Feature', 'Correlation'])
                            fig = px.bar(corr_df, x='Correlation', y='Feature', 
                                       title='Top 10 Lag Features Correlation with Target',
                                       orientation='h', color='Correlation',
                                       color_continuous_scale='Viridis')
                            fig.update_layout(plot_bgcolor='#1a1d29', paper_bgcolor='#1a1d29',
                                            font=dict(color='white'))
                            display_plotly_chart(fig)
                        
                    else:
                        st.warning("No lag features were created")
                        
                except Exception as e:
                    st.error(f"❌ Error creating lag features: {str(e)}")


def render_rolling_features(df: pd.DataFrame):
    """Render rolling feature engineering - FIXED DISPLAY."""
    
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">📊 ROLLING STATISTICS FEATURES</div>
        <div class="card-description">Create rolling window statistics to capture trends, volatility, and smoothing effects in your time series data over different time horizons for enhanced model performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    value_col = StateManager.get('value_column')
    date_col = StateManager.get('date_column')
    
    if not value_col or value_col not in df.columns:
        st.warning("No value column selected. Please set a target value column.")
        return
    
    if not date_col or date_col not in df.columns:
        st.warning("Rolling features require a date column for proper time-based calculations.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚙️ ROLLING WINDOW CONFIGURATION**")
        
        # Window sizes
        window_options = {
            '7 days (weekly pattern)': 7,
            '14 days (bi-weekly trend)': 14,
            '30 days (monthly analysis)': 30,
            '60 days (bi-monthly)': 60,
            '90 days (quarterly trend)': 90
        }
        
        selected_windows = []
        for label, window_value in window_options.items():
            if st.checkbox(f"{label}", value=window_value in [7, 30], help=f"{window_value}-day rolling window"):
                selected_windows.append(window_value)
    
    with col2:
        st.markdown("**🚀 FEATURE GENERATION**")
        
        if st.button("📈 CREATE ROLLING FEATURES", type="primary", use_container_width=True):
            if not selected_windows:
                st.warning("Please select at least one window size")
                return
                
            with st.spinner("Creating advanced rolling features..."):
                try:
                    engineer = FeatureEngineer()
                    
                    # Create rolling features
                    df_rolling = engineer.create_rolling_features(df, value_col, windows=selected_windows)
                    
                    # Show new features - FIXED: Properly detect new columns
                    original_columns = set(df.columns)
                    new_columns_set = set(df_rolling.columns) - original_columns
                    new_columns = list(new_columns_set)
                    
                    if new_columns:
                        # ✅ UPDATED: Use StateManager
                        StateManager.set('current_data', df_rolling)
                        st.success(f"✅ Created {len(new_columns)} rolling features!")
                        
                        # Show feature overview - FIXED DISPLAY
                        st.markdown("**📋 NEW FEATURES OVERVIEW**")
                        
                        # Categorize features by window size
                        window_features = {}
                        for col in new_columns:
                            for window in selected_windows:
                                if f"rolling_{window}" in col:
                                    if window not in window_features:
                                        window_features[window] = []
                                    window_features[window].append(col)
                        
                        # Display features in expanders
                        for window, features in window_features.items():
                            with st.expander(f"📊 {window}-Day Window Features ({len(features)} features)"):
                                # Display in a grid
                                cols = st.columns(2)
                                for i, feature in enumerate(features):
                                    with cols[i % 2]:
                                        st.code(feature, language='text')
                        
                        # Show sample of new data
                        st.markdown("**👀 FEATURE PREVIEW**")
                        preview_cols = new_columns[:5]  # Show first 5 new columns
                        st.dataframe(df_rolling[preview_cols].head(8), use_container_width=True)
                        
                    else:
                        st.warning("No rolling features were created. Check if your data is properly sorted by date.")
                        
                except Exception as e:
                    st.error(f"❌ Error creating rolling features: {str(e)}")
                    st.info("💡 Tip: Ensure your data is properly sorted by the date column")


def render_feature_summary(df: pd.DataFrame):
    """Render feature engineering summary."""
    
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">🎯 FEATURE ENGINEERING SUMMARY</div>
        <div class="card-description">Review your enhanced dataset with all engineered features and prepare for advanced model training with comprehensive feature analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Features", len(df.columns))
    
    with col2:
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Features", numeric_features)
    
    with col3:
        categorical_features = len(df.select_dtypes(include=['object', 'category']).columns)
        st.metric("Categorical Features", categorical_features)
    
    # Feature categories
    st.markdown("**📊 FEATURE CATEGORIES**")
    
    # Categorize features professionally
    feature_categories = {
        '🎯 Original Features': [col for col in df.columns if not any(x in col for x in ['lag_', 'rolling_', 'year', 'month', 'day', 'quarter', 'week'])],
        '🕒 Time Features': [col for col in df.columns if any(x in col for x in ['year', 'month', 'quarter', 'week', 'day_of_', 'is_'])],
        '📈 Lag Features': [col for col in df.columns if 'lag_' in col],
        '📊 Rolling Features': [col for col in df.columns if 'rolling_' in col]
    }
    
    for category, features in feature_categories.items():
        if features:
            with st.expander(f"{category} ({len(features)} features)"):
                # Display in a grid
                cols = st.columns(2)
                for i, feature in enumerate(features):
                    with cols[i % 2]:
                        st.code(feature, language='text')
    
    # Data preview
    st.markdown("**👀 ENHANCED DATA PREVIEW**")
    st.dataframe(df.head(12), use_container_width=True)


if __name__ == "__main__":
    main()
