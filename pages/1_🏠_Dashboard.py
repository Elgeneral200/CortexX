"""
Enterprise Dashboard Page for CortexX Forecasting Platform
PHASE 3 - Sessions 1-6 Complete
✅ Added: Data Quality Dashboard with comprehensive analytics
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
import warnings

# ✅ FIX 1: Suppress Arrow serialization warnings
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')

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
    from src.reports.pdf_report import PDFReportGenerator, generate_filename_pdf
    MODULES_AVAILABLE = True
    VALIDATORS_AVAILABLE = True
    PDF_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    MODULES_AVAILABLE = False
    VALIDATORS_AVAILABLE = False
    PDF_AVAILABLE = False

st.set_page_config(page_title="Dashboard - CortexX", page_icon="🏠", layout="wide")


def safe_float(value, default=0.0):
    """Safely convert value to float with None protection."""
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_comparison_data(variance, growth):
    """
    ✅ CRITICAL FIX: Safely extract comparison data with complete None protection.
    """
    try:
        return {
            'category': str(variance.get('category', 'Neutral') or 'Neutral'),
            'emoji': str(variance.get('emoji', '📊') or '📊'),
            'variance_pct': safe_float(variance.get('variance_pct'), 0.0),
            'daily_growth': safe_float(growth.get('daily_growth'), 0.0)
        }
    except Exception as e:
        logger.warning(f"Error creating comparison data: {e}")
        return {
            'category': 'Neutral',
            'emoji': '📊',
            'variance_pct': 0.0,
            'daily_growth': 0.0
        }

# ============================================================================
# SESSION 7: CUSTOM METRICS BUILDER
# ============================================================================

def render_custom_metrics_builder():
    """
    ✅ NEW: Custom Metrics Builder UI
    PHASE 3 - SESSION 7
    """
    st.markdown("### 🧮 Custom Metrics Builder")
    
    df = get_current_data()
    date_col = StateManager.get('date_column')
    
    # Apply filters
    display_df = apply_filters_from_state(df, date_col, StateManager) if StateManager.is_filtered() else df
    
    if display_df.empty:
        st.warning("⚠️ No data available for custom metrics")
        return
    
    # Import metrics manager
    try:
        from src.analytics.custom_metrics import (
            get_metrics_manager, CustomMetric, PREDEFINED_METRICS
        )
    except ImportError as e:
        st.error(f"❌ Error loading custom metrics module: {e}")
        st.info("💡 Make sure you created `src/analytics/custom_metrics.py`")
        return
    
    # Initialize metrics manager in session state
    if 'metrics_manager' not in st.session_state:
        st.session_state.metrics_manager = get_metrics_manager()
        
        # Load predefined metrics
        for metric in PREDEFINED_METRICS:
            st.session_state.metrics_manager.add_metric(metric)
    
    manager = st.session_state.metrics_manager
    
    # ============================================================================
    # SECTION 1: CREATE NEW METRIC
    # ============================================================================
    
    with st.expander("➕ Create New Metric", expanded=True):
        st.markdown("#### Build Your Custom Metric")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            metric_name = st.text_input(
                "Metric Name",
                placeholder="e.g., Profit Margin %",
                key="new_metric_name"
            )
            
            metric_formula = st.text_area(
                "Formula",
                placeholder="e.g., (Revenue - Cost) / Revenue * 100",
                help="Use column names and operators: +, -, *, /, %",
                key="new_metric_formula",
                height=100
            )
            
            metric_description = st.text_input(
                "Description (Optional)",
                placeholder="e.g., Calculate profit margin as percentage",
                key="new_metric_desc"
            )
        
        with col2:
            st.markdown("##### Available Columns")
            st.caption("Use these exact names in your formula:")
            for col in display_df.columns[:10]:  # Show first 10 columns
                st.code(col, language=None)
            if len(display_df.columns) > 10:
                st.caption(f"...and {len(display_df.columns) - 10} more")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("✅ Create Metric", type="primary", use_container_width=True):
                if not metric_name or not metric_formula:
                    st.error("❌ Name and formula are required!")
                else:
                    try:
                        # Create and add metric
                        new_metric = CustomMetric(
                            name=metric_name,
                            formula=metric_formula,
                            description=metric_description
                        )
                        
                        # Test the metric
                        result = manager.apply_metric(display_df, new_metric)
                        
                        # If successful, add to manager
                        manager.add_metric(new_metric)
                        st.success(f"✅ Metric '{metric_name}' created successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error creating metric: {str(e)}")
        
        with col_b:
            if st.button("🧪 Test Formula", use_container_width=True):
                if not metric_formula:
                    st.error("❌ Formula is required!")
                else:
                    try:
                        test_metric = CustomMetric(
                            name="Test",
                            formula=metric_formula,
                            description=""
                        )
                        result = manager.apply_metric(display_df.head(5), test_metric)
                        st.success("✅ Formula is valid!")
                        with st.expander("View Test Results (First 5 rows)"):
                            st.dataframe(pd.DataFrame({
                                'Result': result.values
                            }))
                    except Exception as e:
                        st.error(f"❌ Formula error: {str(e)}")
    
    # ============================================================================
    # SECTION 2: SAVED METRICS
    # ============================================================================
    
    st.markdown("#### 📋 Saved Metrics")
    
    metrics_list = manager.list_metrics()
    
    if not metrics_list:
        st.info("ℹ️ No custom metrics yet. Create your first metric above!")
    else:
        for idx, metric_data in enumerate(metrics_list):
            col1, col2, col3 = st.columns([3, 3, 2])
            
            with col1:
                st.markdown(f"**{metric_data['name']}**")
                if metric_data['description']:
                    st.caption(metric_data['description'])
            
            with col2:
                st.code(metric_data['formula'], language=None)
            
            with col3:
                delete_key = f"delete_{idx}_{metric_data['name']}"
                if st.button("🗑️", key=delete_key):
                    manager.remove_metric(metric_data['name'])
                    st.success(f"✅ Deleted")
                    st.rerun()
            
            st.markdown("---")


# ============================================================================
# SESSION 6: DATA QUALITY DASHBOARD
# ============================================================================

def render_data_quality_dashboard():
    """
    ✅ NEW: Data Quality Dashboard
    PHASE 3 - SESSION 6
    """
    st.markdown("### 🔍 Data Quality Dashboard")
    
    df = get_current_data()
    date_col = StateManager.get('date_column')
    
    # Apply filters
    display_df = apply_filters_from_state(df, date_col, StateManager) if StateManager.is_filtered() else df
    
    if display_df.empty:
        st.warning("⚠️ No data available for quality analysis")
        return
    
    # Import quality analyzer
    try:
        from src.analytics.data_quality import get_quality_analyzer
        from src.visualization.advanced_charts import get_quality_visualizer
        
        analyzer = get_quality_analyzer(display_df)
        visualizer = get_quality_visualizer()
    except ImportError as e:
        st.error(f"❌ Error loading quality module: {e}")
        st.info("💡 Make sure you created `src/analytics/data_quality.py` and updated `src/visualization/advanced_charts.py`")
        return
    
    # Calculate quality scores
    quality_scores = analyzer.calculate_overall_quality_score()
    
    # ============================================================================
    # SECTION 1: OVERALL QUALITY SCORE
    # ============================================================================
    st.markdown("#### 📊 Overall Data Quality Score")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Quality gauge
        try:
            fig = visualizer.create_quality_gauge(quality_scores['overall_score'])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating gauge: {e}")
    
    with col2:
        st.markdown("##### Component Scores")
        st.metric("Completeness", f"{quality_scores['completeness']:.1f}%")
        st.metric("Uniqueness", f"{quality_scores['uniqueness']:.1f}%")
    
    with col3:
        st.markdown("##### Quality Metrics")
        st.metric("Validity", f"{quality_scores['validity']:.1f}%")
        st.metric("Consistency", f"{quality_scores['consistency']:.1f}%")
    
    # Overall grade display
    grade = quality_scores['grade']
    status = quality_scores['status']
    
    if quality_scores['overall_score'] >= 90:
        st.success(f"🎉 **Grade: {grade}** - {status} Quality! Your data is in excellent condition.")
    elif quality_scores['overall_score'] >= 80:
        st.success(f"✅ **Grade: {grade}** - {status} Quality! Your data is well-maintained.")
    elif quality_scores['overall_score'] >= 70:
        st.warning(f"⚠️ **Grade: {grade}** - {status} Quality. Consider addressing quality issues.")
    else:
        st.error(f"❌ **Grade: {grade}** - {status} Quality! Immediate action required.")
    
    st.markdown("---")
    
    # ============================================================================
    # SECTION 2: MISSING VALUES ANALYSIS
    # ============================================================================
    st.markdown("#### 🔥 Missing Values Analysis")
    
    missing_summary = analyzer.get_missing_values_summary()
    
    if not missing_summary.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Missing Values Heatmap")
            try:
                fig = visualizer.create_missing_values_heatmap(display_df)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating heatmap: {e}")
        
        with col2:
            st.markdown("##### Missing by Column")
            try:
                fig = visualizer.create_missing_values_bar(missing_summary)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating bar chart: {e}")
        
        # Detailed table
        st.markdown("##### Detailed Missing Values Report")
        st.dataframe(
            missing_summary.style.background_gradient(subset=['Missing %'], cmap='Reds'),
            use_container_width=True,
            height=300
        )
    else:
        st.success("🎉 **Perfect!** No missing values detected in your dataset!")
    
    st.markdown("---")
    
    # ============================================================================
    # SECTION 3: DUPLICATE DETECTION
    # ============================================================================
    st.markdown("#### 🔄 Duplicate Detection")
    
    dup_analysis = analyzer.get_duplicate_analysis()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Duplicates",
            f"{dup_analysis['total_duplicates']:,}",
            delta=f"{dup_analysis['duplicate_percentage']:.1f}% of data"
        )
    
    with col2:
        st.metric(
            "Unique Rows",
            f"{dup_analysis['unique_rows']:,}"
        )
    
    with col3:
        if dup_analysis['has_duplicates']:
            st.error("⚠️ Duplicates Found")
        else:
            st.success("✅ No Duplicates")
    
    if dup_analysis['has_duplicates'] and dup_analysis['duplicate_rows'] is not None:
        with st.expander("🔍 View Duplicate Rows (Sample)"):
            st.dataframe(
                dup_analysis['duplicate_rows'].head(20),
                use_container_width=True
            )
            st.caption(f"Showing first 20 of {len(dup_analysis['duplicate_rows'])} duplicate rows")
    
    st.markdown("---")
    
    # ============================================================================
    # SECTION 4: COLUMN STATISTICS
    # ============================================================================
    st.markdown("#### 📋 Column-Level Statistics")
    
    col_stats = analyzer.get_column_statistics()
    
    # Display comprehensive stats
    st.dataframe(
        col_stats.style.background_gradient(subset=['Unique %'], cmap='Greens'),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # ============================================================================
    # SECTION 5: OUTLIER DETECTION
    # ============================================================================
    st.markdown("#### 📊 Outlier Detection Summary")
    
    outliers = analyzer.detect_outliers_summary()
    
    if outliers:
        outlier_data = []
        for col, stats in outliers.items():
            outlier_data.append({
                'Column': col,
                'Outlier Count': stats['count'],
                'Outlier %': stats['percentage'],
                'Lower Bound': stats['lower_bound'],
                'Upper Bound': stats['upper_bound']
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        
        st.dataframe(
            outlier_df.style.background_gradient(subset=['Outlier %'], cmap='Oranges'),
            use_container_width=True
        )
        
        st.info("💡 **Note**: Outliers detected using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)")
    else:
        st.success("✅ No significant outliers detected in numeric columns!")
    
    st.markdown("---")
    
    # ============================================================================
    # SECTION 6: DATA FRESHNESS
    # ============================================================================
    if date_col:
        st.markdown("#### 📅 Data Freshness Analysis")
        
        freshness = analyzer.get_data_freshness(date_col)
        
        if freshness:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Most Recent Date", freshness['most_recent_date'])
            
            with col2:
                st.metric("Oldest Date", freshness['oldest_date'])
            
            with col3:
                st.metric("Days Since Update", f"{freshness['days_since_update']} days")
            
            with col4:
                st.metric(
                    "Freshness Status",
                    freshness['freshness_status'],
                    delta=f"{freshness['freshness_emoji']}"
                )
            
            # Visual freshness indicator
            if freshness['days_since_update'] <= 7:
                st.success(f"🟢 Data is {freshness['freshness_status']}! Updated within the last week.")
            elif freshness['days_since_update'] <= 30:
                st.info(f"🟡 Data freshness is {freshness['freshness_status']}. Updated within the last month.")
            else:
                st.warning(f"🟠 Data may be {freshness['freshness_status']}. Last updated {freshness['days_since_update']} days ago.")
        else:
            st.info("ℹ️ Unable to analyze data freshness")


# ============================================================================
# SESSION 5: ADVANCED VISUALIZATIONS
# ============================================================================

def render_advanced_visualizations():
    """
    ✅ FIXED: Advanced visualization section with chart builder.
    PHASE 3 - SESSION 5
    """
    st.markdown("### 🎨 Advanced Visualizations")
    
    df = get_current_data()
    date_col = StateManager.get('date_column')
    value_col = StateManager.get('value_column')
    
    # Apply filters
    display_df = apply_filters_from_state(df, date_col, StateManager) if StateManager.is_filtered() else df
    
    if display_df.empty:
        st.warning("⚠️ No data available for visualization")
        return
    
    # Import chart builder
    try:
        from src.visualization.advanced_charts import get_chart_builder, ChartExporter
        chart_builder = get_chart_builder()
    except ImportError as e:
        st.error(f"Error loading visualization module: {e}")
        return
    
    # Chart type selector
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution", "📦 Box Plot", "🔥 Correlation", "🥧 Categories"
    ])
    
    # TAB 1: DISTRIBUTION PLOT
    with tab1:
        st.markdown("#### Distribution Analysis")
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                selected_col = st.selectbox("Select Column", options=numeric_cols, key="dist_col")
                bins = st.slider("Number of Bins", 10, 100, 30, key="dist_bins")
                generate_dist = st.button("📊 Generate Distribution", key="btn_dist", use_container_width=True)
            
            with col1:
                if generate_dist or 'last_dist_chart' in st.session_state:
                    try:
                        fig = chart_builder.create_distribution_plot(display_df, selected_col, bins=bins)
                        st.session_state['last_dist_chart'] = fig
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button("💾 Download HTML", ChartExporter.to_html(fig),
                                f"distribution_{selected_col}.html", "text/html", use_container_width=True)
                        with col_b:
                            st.success("✅ Chart generated!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.info("👈 Click 'Generate Distribution' to create chart")
        else:
            st.info("ℹ️ No numeric columns available")
    
    # TAB 2: BOX PLOT
    with tab2:
        st.markdown("#### Box Plot - Outlier Detection")
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = display_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                y_col = st.selectbox("Y-axis (Numeric)", numeric_cols, key="box_y")
                x_col = st.selectbox("Group by (Optional)", options=[None] + categorical_cols, key="box_x")
                generate_box = st.button("📦 Generate Box Plot", key="btn_box", use_container_width=True)
            
            with col1:
                if generate_box or 'last_box_chart' in st.session_state:
                    try:
                        fig = chart_builder.create_box_plot(display_df, y_col, x_col)
                        st.session_state['last_box_chart'] = fig
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button("💾 Download HTML", ChartExporter.to_html(fig),
                                f"boxplot_{y_col}.html", "text/html", use_container_width=True)
                        with col_b:
                            st.success("✅ Chart generated!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.info("👈 Click 'Generate Box Plot' to create chart")
        else:
            st.info("ℹ️ No numeric columns available")
    
    # TAB 3: CORRELATION HEATMAP
    with tab3:
        st.markdown("#### Correlation Matrix")
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                selected_cols = st.multiselect("Select Columns", options=numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols, key="corr_cols")
                generate_corr = st.button("🔥 Generate Heatmap", key="btn_corr", use_container_width=True)
            
            with col1:
                if generate_corr or 'last_corr_chart' in st.session_state:
                    if len(selected_cols) >= 2:
                        try:
                            fig = chart_builder.create_correlation_heatmap(display_df, selected_cols)
                            st.session_state['last_corr_chart'] = fig
                            st.plotly_chart(fig, use_container_width=True)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.download_button("💾 Download HTML", ChartExporter.to_html(fig),
                                    "correlation_heatmap.html", "text/html", use_container_width=True)
                            with col_b:
                                st.success("✅ Chart generated!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.warning("⚠️ Select at least 2 columns")
                else:
                    st.info("👈 Select columns and click 'Generate Heatmap'")
        else:
            st.info("ℹ️ Need at least 2 numeric columns")
    
    # TAB 4: PIE CHART
    with tab4:
        st.markdown("#### Category Breakdown")
        categorical_cols = display_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                cat_col = st.selectbox("Category Column", categorical_cols, key="pie_cat")
                val_col = st.selectbox("Value Column (Optional)", options=[None] + numeric_cols, key="pie_val")
                top_n = st.slider("Top N Categories", 5, 20, 10, key="pie_top")
                generate_pie = st.button("🥧 Generate Pie Chart", key="btn_pie", use_container_width=True)
            
            with col1:
                if generate_pie or 'last_pie_chart' in st.session_state:
                    try:
                        fig = chart_builder.create_pie_chart(display_df, cat_col, val_col, top_n=top_n)
                        st.session_state['last_pie_chart'] = fig
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button("💾 Download HTML", ChartExporter.to_html(fig),
                                f"pie_chart_{cat_col}.html", "text/html", use_container_width=True)
                        with col_b:
                            st.success("✅ Chart generated!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.info("👈 Click 'Generate Pie Chart' to create chart")
        else:
            st.info("ℹ️ No categorical columns available")



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
        
        # ✅ SESSION 7: Custom Metrics Builder
        st.markdown("---")
        render_custom_metrics_builder()
        
        # ✅ SESSION 6: Data Quality Dashboard
        st.markdown("---")
        render_data_quality_dashboard()
        
        # ✅ SESSION 5: Advanced Visualizations
        st.markdown("---")
        render_advanced_visualizations()


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
        
        # Excel Export
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
        
        # CSV Export
        if st.button(f"📄 Export CSV ({label})", use_container_width=True):
            try:
                with st.spinner("📄 Creating CSV..."):
                    st.download_button("⬇️ Download CSV", ExportManager.export_to_csv(export_df),
                        generate_filename("cortexx_data", "csv"), "text/csv", use_container_width=True)
                    st.success(f"✅ CSV ready! {len(export_df):,} records")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        # PDF Export
        if PDF_AVAILABLE:
            if st.button(f"📑 Export PDF Report ({label})", use_container_width=True):
                try:
                    with st.spinner("📑 Generating PDF report..."):
                        comparison_data = None
                        
                        if StateManager.get('comparison_enabled') and StateManager.get('filter_enabled'):
                            date_col = StateManager.get('date_column')
                            value_col = StateManager.get('value_column')
                            start, end = StateManager.get('filter_start_date'), StateManager.get('filter_end_date')
                            comp_type = StateManager.get('comparison_period', 'previous')
                            
                            if comp_type == 'previous':
                                cs, ce = ComparisonPeriodCalculator.get_previous_period(start, end)
                            else:
                                cs, ce = ComparisonPeriodCalculator.get_previous_year_period(start, end)
                            
                            comp_df = DataFilter(df, date_col).apply_date_filter(cs, ce)
                            
                            if len(comp_df) > 0:
                                analytics = ComparisonAnalytics(export_df, comp_df, value_col, date_col)
                                variance = analytics.calculate_variance_breakdown()
                                growth = analytics.calculate_growth_rates()
                                comparison_data = safe_comparison_data(variance, growth)
                        
                        pdf_bytes = PDFReportGenerator.generate_dashboard_report(
                            export_df, df, StateManager.get_filter_summary(), comparison_data
                        )
                        
                        st.download_button("⬇️ Download PDF Report", pdf_bytes, generate_filename_pdf("cortexx_report"),
                            "application/pdf", use_container_width=True)
                        st.success(f"✅ PDF report ready! {len(export_df):,} records")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    with st.expander("🔍 Show Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
    
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
    """Auto-detect columns with proper datetime conversion"""
    for col in df.columns:
        if any(p in col.lower() for p in ['date', 'time']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if pd.api.types.is_datetime64_any_dtype(df[col]):
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
    """Generate sample data with proper datetime handling"""
    with st.spinner("🎲 Generating..."):
        try:
            df = generate_sample_data_cached(periods=3652, products=5)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            StateManager.update({
                'current_data': df,
                'data_loaded': True,
                'date_column': 'Date',
                'value_column': 'Units Sold'
            })
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
    
    if StateManager.get('comparison_enabled') and StateManager.get('filter_enabled'):
        st.markdown("---")
        display_comparison_analytics()
    
    st.markdown("### 📋 Data Details")
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Column Info", "Statistics"])
    
    with tab1:
        try:
            preview_df = display_df.head(20).copy()
            for col in preview_df.columns:
                if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                    preview_df[col] = preview_df[col].astype(str)
            st.dataframe(preview_df, use_container_width=True)
        except Exception as e:
            st.error(f"Preview error: {e}")
    
    with tab2:
        try:
            col_info = pd.DataFrame({
                'Column': display_df.columns,
                'Type': display_df.dtypes.astype(str),
                'Non-Null': display_df.count(),
                'Null': display_df.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
        except Exception as e:
            st.error(f"Column info error: {e}")
    
    with tab3:
        try:
            numeric_df = display_df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 0:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns")
        except Exception as e:
            st.error(f"Statistics error: {e}")


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
        sg = safe_float(growth.get('simple_growth', 0))
        st.metric("Overall Growth", f"{sg:+.1f}%", delta=f"{sg:.1f}% vs previous")
    with col2:
        dg = safe_float(growth.get('daily_growth', 0))
        st.metric("Daily Avg Growth", f"{dg:+.1f}%", delta=f"${safe_float(growth.get('current_daily_avg', 0)):,.0f}/day")
    with col3:
        cagr = growth.get('cagr')
        st.metric("CAGR" if cagr else "Period", f"{safe_float(cagr):+.1f}%" if cagr else f"{int(safe_float(growth.get('current_days', 0)))} days")


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
