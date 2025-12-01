"""
Enterprise Dashboard Page for CortexX Forecasting Platform
PHASE 3 - Sessions 1-3 Complete
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import logging

# Setup
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

logger = logging.getLogger(__name__)

# Imports
try:
    from src.utils.state_manager import StateManager, is_data_loaded, get_current_data
    from src.data.collection import get_data_collector, generate_sample_data_cached
    from src.visualization.dashboard import get_visualizer, display_plotly_chart
    from src.utils.config import get_config
    from src.utils.validators import DataValidator, validate_upload_file
    from src.utils.filters import (DateFilterPresets, DataFilter, ComparisonPeriodCalculator,
                                   get_unique_values, format_filter_summary, apply_filters_from_state)
    from src.utils.export_manager import ExportManager, generate_filename
    from src.analytics.comparison import ComparisonAnalytics, format_comparison_insight
    MODULES_AVAILABLE = True
    VALIDATORS_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    MODULES_AVAILABLE = False
    VALIDATORS_AVAILABLE = False

st.set_page_config(page_title="Dashboard - CortexX", page_icon="🏠", layout="wide")

def main():
    StateManager.initialize()
    
    # CSS
    st.markdown("""
    <style>
    .dashboard-header {font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1rem; padding: 1rem 0;}
    .filter-section {background: linear-gradient(135deg, #1a1d29 0%, #252a3a 100%); padding: 1.5rem; border-radius: 10px;
        border: 1px solid #2d3746; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .filter-active {border-left: 4px solid #667eea; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);}
    .metric-card {background: linear-gradient(135deg, #1a1d29 0%, #252a3a 100%); padding: 1.5rem; border-radius: 10px;
        border: 1px solid #2d3746; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); transition: transform 0.2s, box-shadow 0.2s;}
    .metric-card:hover {transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);}
    [data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: 700; color: #ffffff;}
    [data-testid="stMetricDelta"] {font-size: 1rem; font-weight: 600;}
    .stSuccess {background-color: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745; padding: 1rem; border-radius: 8px;}
    .stInfo {background-color: rgba(102, 126, 234, 0.1); border-left: 4px solid #667eea; padding: 1rem; border-radius: 8px;}
    .stButton > button {border-radius: 8px; font-weight: 600; transition: all 0.3s;}
    .stButton > button:hover {transform: translateY(-1px); box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);}
    .stDownloadButton > button {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
        border: none; border-radius: 8px; font-weight: 600;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-header">🏠 Enterprise Dashboard</div>', unsafe_allow_html=True)
    
    if is_data_loaded():
        render_filter_section()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📂 Data Management")
        uploaded_file = st.file_uploader("Upload your sales data (CSV)", type=['csv'])
        if uploaded_file:
            handle_file_upload(uploaded_file)
    with col2:
        render_quick_actions_sidebar()
    
    if is_data_loaded():
        display_dashboard_analytics()

# FILTERS
def render_filter_section():
    filter_summary = StateManager.get_filter_summary()
    is_filtered = filter_summary.get('has_filters', False)
    filter_class = "filter-section filter-active" if is_filtered else "filter-section"
    
    with st.container():
        st.markdown(f'<div class="{filter_class}">', unsafe_allow_html=True)
        st.markdown("### 🔍 Filters & Analysis")
        tab1, tab2, tab3 = st.tabs(["📅 Date Range", "🏷️ Products/Categories", "⚖️ Comparison"])
        with tab1:
            render_date_filter()
        with tab2:
            render_product_category_filter()
        with tab3:
            render_comparison_options()
        st.markdown('</div>', unsafe_allow_html=True)
        if is_filtered:
            st.info(f"📌 **Active Filters:** {format_filter_summary(filter_summary)}")

def render_date_filter():
    df = get_current_data()
    date_col = StateManager.get('date_column')
    if not df is None and date_col:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        min_date, max_date = df[date_col].min().date(), df[date_col].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Quick Filters:**")
            cols = st.columns(3)
            for idx, preset in enumerate(DateFilterPresets.get_all_presets()):
                with cols[idx % 3]:
                    if st.button(preset['label'], key=f"preset_{preset['value']}", use_container_width=True):
                        StateManager.set_quick_filter(preset['value'])
                        st.rerun()
            if StateManager.get('filter_enabled'):
                if st.button("🗑️ Clear Filters", type="secondary", use_container_width=True):
                    StateManager.clear_filters()
                    st.rerun()
        
        with col2:
            st.markdown("**Custom Date Range:**")
            current_start = StateManager.get('filter_start_date')
            current_end = StateManager.get('filter_end_date')
            start_default = current_start.date() if current_start and hasattr(current_start, 'date') else min_date
            if start_default < min_date or start_default > max_date:
                start_default = min_date
            end_default = current_end.date() if current_end and hasattr(current_end, 'date') else max_date
            if end_default < min_date or end_default > max_date:
                end_default = max_date
            start_date = st.date_input("Start Date", value=start_default, min_value=min_date, max_value=max_date, key="filter_start")
            end_date = st.date_input("End Date", value=end_default, min_value=min_date, max_value=max_date, key="filter_end")
            if st.button("Apply Custom Range", type="primary", use_container_width=True):
                StateManager.set_date_filter(datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()))
                st.rerun()
    else:
        st.warning("⚠️ Load data first")

def render_product_category_filter():
    df = get_current_data()
    if df is None:
        st.warning("⚠️ Load data first")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Product Filter**")
        product_col = 'Product ID' if 'Product ID' in df.columns else None
        if product_col:
            products = get_unique_values(df, product_col)
            selected = st.multiselect("Select Products", options=products, default=StateManager.get('filter_products', []), key="product_filter")
            if st.button("Apply Product Filter", use_container_width=True):
                StateManager.set_product_filter(selected)
                st.success(f"✅ Filtered to {len(selected)} products")
                st.rerun()
        else:
            st.info("ℹ️ No product column detected")
    with col2:
        st.markdown("**Category Filter**")
        category_col = 'Category' if 'Category' in df.columns else None
        if category_col:
            categories = get_unique_values(df, category_col)
            selected = st.multiselect("Select Categories", options=categories, default=StateManager.get('filter_categories', []), key="category_filter")
            if st.button("Apply Category Filter", use_container_width=True):
                StateManager.set_category_filter(selected)
                st.success(f"✅ Filtered to {len(selected)} categories")
                st.rerun()
        else:
            st.info("ℹ️ No category column detected")

def render_comparison_options():
    st.markdown("**Period Comparison**")
    comparison_enabled = StateManager.get('comparison_enabled', False)
    enable = st.checkbox("Enable Comparison Mode", value=comparison_enabled)
    if enable != comparison_enabled:
        StateManager.toggle_comparison(enable)
        st.rerun()
    if enable:
        comp_type = st.radio("Compare with:", options=['previous', 'previous_year'],
                            format_func=lambda x: "Previous Period" if x == 'previous' else "Same Period Last Year", horizontal=True)
        StateManager.set('comparison_period', comp_type)
        if StateManager.get('filter_enabled'):
            start, end = StateManager.get('filter_start_date'), StateManager.get('filter_end_date')
            if start and end:
                if comp_type == 'previous':
                    cs, ce = ComparisonPeriodCalculator.get_previous_period(start, end)
                else:
                    cs, ce = ComparisonPeriodCalculator.get_previous_year_period(start, end)
                st.info(f"**Comparing:**\n- **Current:** {start.date()} to {end.date()}\n- **Comparison:** {cs.date()} to {ce.date()}")

# SIDEBAR
def render_quick_actions_sidebar():
    st.subheader("🚀 Quick Actions")
    if st.button("🎲 Generate Sample Data", use_container_width=True, type="primary"):
        generate_sample_data()
    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()
    
    if is_data_loaded():
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        df = get_current_data()
        if StateManager.is_filtered():
            export_df = apply_filters_from_state(df, StateManager.get('date_column'), StateManager)
            label = "Filtered Data"
        else:
            export_df = df
            label = "All Data"
        st.caption(f"📌 Ready: {len(export_df):,} of {len(df):,} records")
        
        if st.button(f"📊 Export Excel ({label})", use_container_width=True, type="primary"):
            try:
                with st.spinner("📊 Creating Excel..."):
                    excel_data = ExportManager.export_to_excel(export_df, stats_df=export_df.describe(),
                        metadata=ExportManager.create_export_metadata(StateManager.get_filter_summary(),
                        {'total_records': len(df), 'filtered_records': len(export_df)}))
                    st.download_button("⬇️ Download Excel", excel_data, generate_filename("cortexx_data", "xlsx"),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, type="primary")
                    st.success(f"✅ Excel ready! {len(export_df):,} records")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        if st.button(f"📄 Export CSV ({label})", use_container_width=True):
            try:
                with st.spinner("📄 Creating CSV..."):
                    st.download_button("⬇️ Download CSV", ExportManager.export_to_csv(export_df),
                        generate_filename("cortexx_data", "csv"), "text/csv", use_container_width=True)
                    st.success(f"✅ CSV ready! {len(export_df):,} records")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    st.markdown("**System Status**")
    st.metric("Data Loaded", "✅" if is_data_loaded() else "❌")
    if is_data_loaded():
        df = get_current_data()
        if StateManager.is_filtered():
            filtered = apply_filters_from_state(df, StateManager.get('date_column'), StateManager)
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Filtered Records", f"{len(filtered):,}")
        else:
            st.metric("Records", f"{len(df):,}")
        st.metric("Features", len(df.columns))

# DATA LOADING
def handle_file_upload(uploaded_file):
    try:
        if VALIDATORS_AVAILABLE:
            is_valid, message, df = validate_upload_file(uploaded_file)
            if is_valid and df is not None:
                validator = DataValidator()
                result = validator.validate_dataframe(df)
                StateManager.update({'current_data': df, 'data_loaded': True, 'validation_result': result})
                auto_detect_columns(df)
                st.success(f"✅ {message}")
        else:
            df = pd.read_csv(uploaded_file)
            StateManager.update({'current_data': df, 'data_loaded': True})
            auto_detect_columns(df)
            st.success(f"✅ Data loaded! {len(df):,} records")
    except Exception as e:
        st.error(f"❌ Error: {e}")

def auto_detect_columns(df):
    for col in df.columns:
        if any(p in col.lower() for p in ['date', 'time']):
            try:
                df[col] = pd.to_datetime(df[col])
                StateManager.set('date_column', col)
                st.info(f"📅 Date column: **{col}**")
                break
            except:
                pass
    numeric = df.select_dtypes(include=[np.number]).columns
    if len(numeric) > 0:
        StateManager.set('value_column', numeric[0])
        st.info(f"💰 Value column: **{numeric[0]}**")

def generate_sample_data():
    with st.spinner("🎲 Generating..."):
        try:
            df = generate_sample_data_cached(periods=3652, products=5)
            StateManager.update({'current_data': df, 'data_loaded': True, 'date_column': 'Date', 'value_column': 'Units Sold'})
            st.success("✅ Sample data generated!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {e}")

def reset_session():
    StateManager.clear_all()
    st.rerun()

# ANALYTICS DISPLAY
def display_dashboard_analytics():
    df = get_current_data()
    date_col, value_col = StateManager.get('date_column'), StateManager.get('value_column')
    visualizer = get_visualizer()
    
    display_df = apply_filters_from_state(df, date_col, StateManager) if StateManager.is_filtered() else df
    
    st.markdown("### 📈 Business Overview")
    if StateManager.get('comparison_enabled') and StateManager.get('filter_enabled'):
        display_kpis_with_comparison(display_df, df)
    else:
        display_kpis_standard(display_df)
    
    st.markdown("### 📊 Data Visualizations")
    if date_col and value_col:
        col1, col2 = st.columns(2)
        with col1:
            try:
                if StateManager.get('comparison_enabled') and StateManager.get('filter_enabled'):
                    start, end = StateManager.get('filter_start_date'), StateManager.get('filter_end_date')
                    comp_type = StateManager.get('comparison_period', 'previous')
                    cs, ce = (ComparisonPeriodCalculator.get_previous_period(start, end) if comp_type == 'previous'
                             else ComparisonPeriodCalculator.get_previous_year_period(start, end))
                    comp_df = DataFilter(df, date_col).apply_date_filter(cs, ce)
                    fig = visualizer.create_comparison_trend_plot(display_df, comp_df, date_col, value_col,
                        "Current", "Previous", "Sales Trend") if len(comp_df) > 0 else visualizer.create_sales_trend_plot(display_df, date_col, value_col)
                else:
                    fig = visualizer.create_sales_trend_plot(display_df, date_col, value_col)
                display_plotly_chart(fig)
            except Exception as e:
                st.error(f"Error: {e}")
        with col2:
            try:
                fig = visualizer.create_seasonality_plot(display_df, date_col, value_col)
                display_plotly_chart(fig)
            except Exception as e:
                st.error(f"Error: {e}")
    
    # SESSION 3: Comparison Analytics
    if StateManager.get('comparison_enabled') and StateManager.get('filter_enabled'):
        st.markdown("---")
        display_comparison_analytics()
    
    st.markdown("### 📋 Data Details")
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Column Info", "Statistics"])
    with tab1:
        st.dataframe(display_df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(pd.DataFrame({'Column': display_df.columns, 'Type': display_df.dtypes,
            'Non-Null': display_df.count(), 'Null': display_df.isnull().sum()}), use_container_width=True)
    with tab3:
        if display_df.select_dtypes(include=[np.number]).shape[1] > 0:
            st.dataframe(display_df.describe(), use_container_width=True)

# SESSION 3: COMPARISON ANALYTICS
def display_comparison_analytics():
    df, date_col, value_col = get_current_data(), StateManager.get('date_column'), StateManager.get('value_column')
    start, end = StateManager.get('filter_start_date'), StateManager.get('filter_end_date')
    comp_type = StateManager.get('comparison_period', 'previous')
    cs, ce = (ComparisonPeriodCalculator.get_previous_period(start, end) if comp_type == 'previous'
             else ComparisonPeriodCalculator.get_previous_year_period(start, end))
    
    filter_obj = DataFilter(df, date_col)
    current_df = filter_obj.apply_date_filter(start, end)
    comp_df = filter_obj.apply_date_filter(cs, ce)
    
    if len(comp_df) == 0:
        st.warning("⚠️ No comparison data")
        return
    
    analytics = ComparisonAnalytics(current_df, comp_df, value_col, date_col)
    
    st.markdown("### 📊 Comparison Overview")
    variance = analytics.calculate_variance_breakdown()
    growth = analytics.calculate_growth_rates()
    insight = format_comparison_insight(variance, growth)
    
    if variance.get('category') == 'Positive':
        st.success(insight)
    elif variance.get('category') == 'Negative':
        st.error(insight)
    else:
        st.info(insight)
    
    st.markdown("### 📋 Detailed Comparison")
    summary = analytics.generate_summary_table()
    if not summary.empty:
        st.dataframe(summary[['Metric', 'Current Period', 'Previous Period', 'Change']], use_container_width=True, height=250)
    
    st.markdown("### 📈 Growth Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        sg = growth.get('simple_growth', 0)
        st.metric("Overall Growth", f"{sg:+.1f}%", delta=f"{sg:.1f}% vs previous")
    with col2:
        dg = growth.get('daily_growth', 0)
        st.metric("Daily Avg Growth", f"{dg:+.1f}%", delta=f"${growth.get('current_daily_avg', 0):,.0f}/day")
    with col3:
        cagr = growth.get('cagr')
        st.metric("CAGR" if cagr else "Period", f"{cagr:+.1f}%" if cagr else f"{growth.get('current_days', 0)} days")

# KPIs
def display_kpis_standard(df):
    col1, col2, col3, col4, col5 = st.columns(5)
    value_col = StateManager.get('value_column')
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Value", f"${df[value_col].sum():,.0f}" if value_col else "-")
    with col3:
        st.metric("Average Value", f"${df[value_col].mean():,.0f}" if value_col else "-")
    with col4:
        st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
    with col5:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Quality", f"{completeness:.1f}%")

def display_kpis_with_comparison(current_df, original_df):
    value_col, date_col = StateManager.get('value_column'), StateManager.get('date_column')
    start, end = StateManager.get('filter_start_date'), StateManager.get('filter_end_date')
    comp_type = StateManager.get('comparison_period', 'previous')
    cs, ce = (ComparisonPeriodCalculator.get_previous_period(start, end) if comp_type == 'previous'
             else ComparisonPeriodCalculator.get_previous_year_period(start, end))
    comp_df = DataFilter(original_df, date_col).apply_date_filter(cs, ce)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        delta = len(current_df) - len(comp_df)
        st.metric("Total Records", f"{len(current_df):,}", delta=f"{delta:+,}")
    with col2:
        if value_col:
            cv, pv = current_df[value_col].sum(), comp_df[value_col].sum()
            change = ComparisonPeriodCalculator.calculate_period_change(cv, pv)
            st.metric("Total Value", f"${cv:,.0f}", delta=f"{change['percent_change']:+.1f}%")
    with col3:
        if value_col:
            ca, pa = current_df[value_col].mean(), comp_df[value_col].mean()
            change = ComparisonPeriodCalculator.calculate_period_change(ca, pa)
            st.metric("Average Value", f"${ca:,.0f}", delta=f"{change['percent_change']:+.1f}%")
    with col4:
        st.metric("Numeric Features", len(current_df.select_dtypes(include=[np.number]).columns))
    with col5:
        completeness = (1 - current_df.isnull().sum().sum() / (len(current_df) * len(current_df.columns))) * 100
        st.metric("Data Quality", f"{completeness:.1f}%")

if __name__ == "__main__":
    main()
