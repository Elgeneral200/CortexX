# filename: visualization.py
"""
Data Visualization Module - Fixed Professional Edition

Clean, reliable data visualization functions with:
- Professional chart styling and themes
- Interactive business intelligence dashboards
- Statistical analysis visualizations
- Data quality visual assessments
- Export capabilities
- Streamlit integration

Author: CortexX Team
Version: 1.2.0 - Fixed Professional Edition
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Check for Streamlit availability
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create dummy streamlit module for non-Streamlit environments
    class DummyStreamlit:
        def __getattr__(self, name):
            def dummy_function(*args, **kwargs):
                print(f"Streamlit not available: {name} called with args={args}, kwargs={kwargs}")
                return None
            return dummy_function
    st = DummyStreamlit()

# ============================
# THEME SYSTEM
# ============================

class VisualizationTheme:
    """Simple theme configuration for professional visualizations."""

    def __init__(self, name: str, mode: str = "dark"):
        self.name = name
        self.mode = mode

        if mode == "dark":
            self.bg_primary = "#0f172a"
            self.bg_secondary = "#1e293b"
            self.text_primary = "#f8fafc"
            self.text_secondary = "#e2e8f0"
            self.accent_primary = "#3b82f6"
            self.accent_secondary = "#10b981"
            self.warning_color = "#f59e0b"
            self.error_color = "#ef4444"
            self.success_color = "#10b981"
            self.grid_color = "#374151"
            self.colorway = [
                "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
                "#06b6d4", "#84cc16", "#f97316", "#ec4899", "#6b7280"
            ]
        else:  # light mode
            self.bg_primary = "#ffffff"
            self.bg_secondary = "#f8fafc"
            self.text_primary = "#1e293b"
            self.text_secondary = "#475569"
            self.accent_primary = "#2563eb"
            self.accent_secondary = "#059669"
            self.warning_color = "#d97706"
            self.error_color = "#dc2626"
            self.success_color = "#059669"
            self.grid_color = "#e2e8f0"
            self.colorway = [
                "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed",
                "#0891b2", "#65a30d", "#ea580c", "#db2777", "#6b7280"
            ]

# Global theme
current_theme = VisualizationTheme("Professional Dark", "dark")

def set_theme(theme_mode: str = "dark") -> None:
    """Set the global visualization theme."""
    global current_theme
    current_theme = VisualizationTheme(f"Professional {theme_mode.title()}", theme_mode)

    # Update Plotly defaults
    template = "plotly_dark" if theme_mode == "dark" else "plotly_white"
    px.defaults.template = template
    px.defaults.color_discrete_sequence = current_theme.colorway

# Initialize with dark theme
set_theme("dark")

# ============================
# UTILITY FUNCTIONS
# ============================

def get_unique_key(prefix: str = "viz") -> str:
    """Generate a unique key for Streamlit widgets."""
    return f"{prefix}_{int(time.time() * 1000)}"

def apply_professional_layout(
    fig: go.Figure, 
    title: Optional[str] = None,
    height: Optional[int] = None,
    showlegend: bool = True
) -> None:
    """Apply professional styling to any Plotly figure."""

    theme = current_theme
    height = height or 400

    layout_config = {
        "template": "plotly_dark" if theme.mode == "dark" else "plotly_white",
        "paper_bgcolor": theme.bg_secondary,
        "plot_bgcolor": theme.bg_primary,
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            "size": 12,
            "color": theme.text_primary
        },
        "title": {
            "text": title,
            "font": {"size": 16, "color": theme.text_primary},
            "x": 0.5,
            "xanchor": "center"
        },
        "margin": {"l": 60, "r": 60, "t": 80, "b": 60},
        "height": height,
        "hovermode": "x unified",
        "showlegend": showlegend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11, "color": theme.text_secondary}
        }
    }

    fig.update_layout(**layout_config)

    # Update axes styling
    fig.update_xaxes(
        gridcolor=theme.grid_color,
        tickfont={"color": theme.text_secondary, "size": 11},
        titlefont={"color": theme.text_primary, "size": 12}
    )

    fig.update_yaxes(
        gridcolor=theme.grid_color,
        tickfont={"color": theme.text_secondary, "size": 11},
        titlefont={"color": theme.text_primary, "size": 12}
    )

def sample_large_dataset(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
    """Sample large datasets for better visualization performance."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)

# ============================
# DASHBOARD COMPONENTS
# ============================

def render_header(title: str, subtitle: Optional[str] = None) -> None:
    """Render professional dashboard header."""
    if not STREAMLIT_AVAILABLE:
        print(f"Header: {title}" + (f" - {subtitle}" if subtitle else ""))
        return

    theme = current_theme
    header_html = f"""
    <div style="
        background: linear-gradient(135deg, {theme.accent_primary}, {theme.accent_secondary});
        padding: 2rem 3rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
        text-align: center;
    ">
        <h1 style="
            font-size: 2.5rem;
            font-weight: 700;
            color: white;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        ">📊 {title}</h1>
        {f'<p style="font-size: 1.2rem; color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_kpi_metrics(metrics: Dict[str, Dict[str, Any]]) -> None:
    """Render KPI metrics in a professional layout."""
    if not STREAMLIT_AVAILABLE:
        print("KPI Metrics:")
        for name, data in metrics.items():
            print(f"  {name}: {data.get('value', 'N/A')}")
        return

    num_metrics = len(metrics)
    cols = st.columns(min(num_metrics, 5))

    for i, (metric_name, metric_data) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            value = metric_data.get("value", "N/A")
            delta = metric_data.get("delta")
            icon = metric_data.get("icon", "📊")
            description = metric_data.get("description", "")

            # Format large numbers
            if isinstance(value, (int, float)) and value >= 1000:
                if value >= 1_000_000:
                    formatted_value = f"{value/1_000_000:.1f}M"
                elif value >= 1_000:
                    formatted_value = f"{value/1_000:.1f}K"
                else:
                    formatted_value = f"{value:,.0f}"
            else:
                formatted_value = str(value)

            # Render metric card
            st.metric(
                label=f"{icon} {metric_name}",
                value=formatted_value,
                delta=delta,
                help=description
            )

# ============================
# MAIN VISUALIZATION FUNCTIONS
# ============================

def create_enhanced_data_overview_dashboard(
    df: pd.DataFrame, 
    column_types: Dict[str, str],
    quality_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Create an enhanced executive-level data overview dashboard."""

    if df.empty:
        if STREAMLIT_AVAILABLE:
            st.warning("Cannot create dashboard: DataFrame is empty")
        else:
            print("Warning: Cannot create dashboard - DataFrame is empty")
        return

    render_header(
        "Sales & Demand Data Overview",
        "Comprehensive analysis of your dataset quality and characteristics"
    )

    # Calculate metrics
    total_cells = len(df) * len(df.columns) if len(df.columns) > 0 else 0
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0

    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df) * 100) if len(df) > 0 else 0

    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    quality_score = quality_metrics.get("overall_score", 0) if quality_metrics else 0

    # KPI Metrics
    kpi_metrics = {
        "Total Records": {
            "value": len(df),
            "icon": "📊",
            "description": "Total number of rows in the dataset"
        },
        "Data Quality": {
            "value": f"{quality_score:.1f}/10" if quality_score > 0 else "N/A",
            "delta": "Good" if quality_score >= 8 else "Needs Improvement" if quality_score >= 6 else "Poor",
            "icon": "⭐",
            "description": "Overall data quality assessment"
        },
        "Completeness": {
            "value": f"{100-missing_pct:.1f}%",
            "delta": f"{missing_count:,} missing",
            "icon": "✅",
            "description": "Percentage of complete data"
        },
        "Uniqueness": {
            "value": f"{100-duplicate_pct:.1f}%",
            "delta": f"{duplicate_count:,} duplicates",
            "icon": "🔄",
            "description": "Percentage of unique records"
        },
        "Memory Usage": {
            "value": f"{memory_mb:.1f}MB",
            "icon": "💾",
            "description": "Total memory usage of the dataset"
        }
    }

    render_kpi_metrics(kpi_metrics)

    if not STREAMLIT_AVAILABLE:
        return

    # Data profiling section
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Column type distribution
        type_counts = pd.Series([column_types.get(col, 'unknown') for col in df.columns]).value_counts()
        if not type_counts.empty:
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Column Types Distribution",
                color_discrete_sequence=current_theme.colorway
            )
            apply_professional_layout(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_key("type_dist"))

    with col2:
        # Missing values by column
        missing_by_col = df.isnull().sum().sort_values(ascending=False)
        top_missing = missing_by_col[missing_by_col > 0].head(10)

        if not top_missing.empty:
            fig = px.bar(
                x=top_missing.values,
                y=top_missing.index,
                orientation='h',
                title="Missing Values by Column",
                color=top_missing.values,
                color_continuous_scale="Reds"
            )
            apply_professional_layout(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_key("missing_by_col"))
        else:
            st.success("✅ No missing values detected!")

    with col3:
        # Data type memory usage
        memory_by_type = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            col_memory = df[col].memory_usage(deep=True) / 1024 / 1024  # MB
            if dtype in memory_by_type:
                memory_by_type[dtype] += col_memory
            else:
                memory_by_type[dtype] = col_memory

        if memory_by_type:
            fig = px.pie(
                values=list(memory_by_type.values()),
                names=list(memory_by_type.keys()),
                title="Memory Usage by Data Type",
                color_discrete_sequence=current_theme.colorway
            )
            apply_professional_layout(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_key("memory_by_type"))

def create_advanced_missing_value_analysis(df: pd.DataFrame) -> None:
    """Create comprehensive missing value analysis with business insights."""

    if df.empty:
        if STREAMLIT_AVAILABLE:
            st.warning("⚠️ Cannot analyze missing values: DataFrame is empty")
        else:
            print("Warning: Cannot analyze missing values - DataFrame is empty")
        return

    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()

    if total_missing == 0:
        if STREAMLIT_AVAILABLE:
            st.success("🎉 Excellent! No missing values detected in your dataset!")
        else:
            print("✅ No missing values detected in the dataset!")
        return

    if STREAMLIT_AVAILABLE:
        st.markdown("### 🔍 Missing Value Analysis")

        # Missing value summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Missing", f"{total_missing:,}")
        with col2:
            missing_pct = (total_missing / df.size) * 100
            st.metric("Missing %", f"{missing_pct:.2f}%")
        with col3:
            affected_cols = (missing_data > 0).sum()
            st.metric("Affected Columns", f"{affected_cols}")
        with col4:
            affected_rows = df.isnull().any(axis=1).sum()
            st.metric("Affected Rows", f"{affected_rows:,}")

        # Detailed analysis
        col1, col2 = st.columns(2)

        with col1:
            # Missing values by column
            missing_cols = missing_data[missing_data > 0].sort_values(ascending=True)
            missing_pct_cols = (missing_cols / len(df) * 100).round(2)

            if not missing_cols.empty:
                fig = px.bar(
                    x=missing_pct_cols.values,
                    y=missing_pct_cols.index,
                    orientation='h',
                    title="Missing Values by Column (%)",
                    color=missing_pct_cols.values,
                    color_continuous_scale="Reds",
                    text=missing_pct_cols.values
                )

                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                apply_professional_layout(fig, height=400)
                st.plotly_chart(fig, use_container_width=True, key=get_unique_key("missing_by_col"))

        with col2:
            # Missing value pattern heatmap
            sample_size = min(1000, len(df))
            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df

            missing_matrix = sample_df.isnull().astype(int)

            if not missing_matrix.empty:
                fig = px.imshow(
                    missing_matrix.T.values,
                    x=list(range(len(sample_df))),
                    y=sample_df.columns,
                    color_continuous_scale=[[0, current_theme.bg_primary], [1, current_theme.warning_color]],
                    title=f"Missing Value Pattern {'(Sampled)' if len(df) > sample_size else ''}",
                    aspect="auto"
                )

                apply_professional_layout(fig, height=400)
                fig.update_xaxes(title="Row Index")
                fig.update_yaxes(title="Columns")
                st.plotly_chart(fig, use_container_width=True, key=get_unique_key("missing_pattern"))

        # Business impact assessment
        st.markdown("#### 💼 Business Impact Assessment")

        impact_analysis = []
        missing_cols = missing_data[missing_data > 0]
        for col in missing_cols.index:
            missing_count = missing_cols[col]
            missing_pct = (missing_count / len(df)) * 100

            if missing_pct > 50:
                impact = "🔴 High Risk - Consider column removal"
            elif missing_pct > 20:
                impact = "🟡 Medium Risk - Requires attention"
            elif missing_pct > 5:
                impact = "🟢 Low Risk - Monitor"
            else:
                impact = "✅ Minimal Risk"

            impact_analysis.append({
                'Column': col,
                'Missing Count': missing_count,
                'Missing %': f"{missing_pct:.1f}%",
                'Business Impact': impact
            })

        if impact_analysis:
            impact_df = pd.DataFrame(impact_analysis)
            st.dataframe(impact_df, use_container_width=True)

def create_statistical_distribution_analysis(df: pd.DataFrame, num_columns: List[str]) -> None:
    """Create statistical distribution analysis."""

    if not num_columns:
        if STREAMLIT_AVAILABLE:
            st.info("📊 No numerical columns available for statistical analysis.")
        else:
            print("No numerical columns available for statistical analysis.")
        return

    if STREAMLIT_AVAILABLE:
        st.markdown("### 📈 Statistical Distribution Analysis")

        # Column selection
        selected_cols = st.multiselect(
            "Select columns for distribution analysis:",
            num_columns,
            default=num_columns[:min(3, len(num_columns))],
            key=get_unique_key("dist_cols")
        )

        if not selected_cols:
            st.warning("Please select at least one column for analysis.")
            return
    else:
        selected_cols = num_columns[:3]  # Use first 3 columns in non-Streamlit mode

    for col in selected_cols:
        if STREAMLIT_AVAILABLE:
            with st.expander(f"📊 Analysis: {col}", expanded=len(selected_cols) == 1):
                _render_column_distribution_analysis(df, col)
        else:
            print(f"\nDistribution Analysis for {col}:")
            _render_column_distribution_analysis(df, col)

def _render_column_distribution_analysis(df: pd.DataFrame, col: str) -> None:
    """Render distribution analysis for a single column."""
    data = df[col].dropna()
    if len(data) < 10:
        if STREAMLIT_AVAILABLE:
            st.warning(f"Not enough data points in {col} for meaningful analysis.")
        else:
            print(f"Warning: Not enough data points in {col} for meaningful analysis.")
        return

    if STREAMLIT_AVAILABLE:
        # Create distribution visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig = px.histogram(
                x=data,
                nbins=30,
                title=f"Distribution of {col}",
                color_discrete_sequence=[current_theme.accent_primary]
            )

            # Add statistical lines
            mean_val = data.mean()
            median_val = data.median()

            fig.add_vline(x=mean_val, line_dash="dash", 
                         line_color=current_theme.warning_color, 
                         annotation_text="Mean")
            fig.add_vline(x=median_val, line_dash="dot", 
                         line_color=current_theme.success_color,
                         annotation_text="Median")

            apply_professional_layout(fig, height=400)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_key(f"hist_{col}"))

        with col2:
            # Box plot
            fig = px.box(
                y=data,
                title=f"Box Plot of {col}",
                color_discrete_sequence=[current_theme.accent_secondary]
            )

            apply_professional_layout(fig, height=400)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_key(f"box_{col}"))

        # Statistics table
        stats_data = {
            'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Skewness'],
            'Value': [
                f"{len(data):,}",
                f"{data.mean():.2f}",
                f"{data.std():.2f}",
                f"{data.min():.2f}",
                f"{data.quantile(0.25):.2f}",
                f"{data.median():.2f}",
                f"{data.quantile(0.75):.2f}",
                f"{data.max():.2f}",
                f"{data.skew():.2f}"
            ]
        }

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    else:
        # Print basic statistics in non-Streamlit mode
        print(f"  Count: {len(data):,}")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Std: {data.std():.2f}")
        print(f"  Min: {data.min():.2f}")
        print(f"  Max: {data.max():.2f}")

def create_categorical_visualizations(df: pd.DataFrame, cat_columns: List[str]) -> None:
    """Enhanced categorical visualizations."""

    if not cat_columns:
        if STREAMLIT_AVAILABLE:
            st.info("🏷️ No categorical columns available for visualization.")
        else:
            print("No categorical columns available for visualization.")
        return

    if STREAMLIT_AVAILABLE:
        st.markdown("### 🏷️ Categorical Analysis")

        # Column selection
        selected_cols = st.multiselect(
            "Select categorical columns:",
            cat_columns,
            default=cat_columns[:min(3, len(cat_columns))],
            key=get_unique_key("cat_select")
        )

        max_categories = st.slider("Max Categories to Display", 5, 50, 15, key=get_unique_key("max_cat"))

        if not selected_cols:
            st.warning("Please select at least one column.")
            return
    else:
        selected_cols = cat_columns[:3]
        max_categories = 15

    for col in selected_cols:
        if STREAMLIT_AVAILABLE:
            with st.expander(f"📊 Analysis: {col}", expanded=len(selected_cols) == 1):
                _render_categorical_analysis(df, col, max_categories)
        else:
            print(f"\nCategorical Analysis for {col}:")
            _render_categorical_analysis(df, col, max_categories)

def _render_categorical_analysis(df: pd.DataFrame, col: str, max_categories: int) -> None:
    """Render categorical analysis for a single column."""
    value_counts = df[col].value_counts().head(max_categories)

    if value_counts.empty:
        if STREAMLIT_AVAILABLE:
            st.warning(f"No data found in column {col}")
        else:
            print(f"Warning: No data found in column {col}")
        return

    if STREAMLIT_AVAILABLE:
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution: {col}",
                color_discrete_sequence=current_theme.colorway,
                hole=0.4
            )

            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=10
            )

            apply_professional_layout(fig, height=400)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_key(f"pie_{col}"))

        with col2:
            # Bar chart
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f"Top Categories: {col}",
                color=value_counts.values,
                color_continuous_scale=current_theme.colorway[0],
                text=value_counts.values
            )

            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            apply_professional_layout(fig, height=max(300, len(value_counts) * 25))
            st.plotly_chart(fig, use_container_width=True, key=get_unique_key(f"bar_{col}"))

        # Category statistics
        unique_count = df[col].nunique()
        total_count = df[col].count()
        missing_count = df[col].isnull().sum()

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Unique Values", f"{unique_count:,}")
        with metric_col2:
            st.metric("Total Records", f"{total_count:,}")
        with metric_col3:
            st.metric("Missing Values", f"{missing_count:,}")
    else:
        print(f"  Top categories: {value_counts.head().to_dict()}")
        print(f"  Unique values: {df[col].nunique()}")

def create_business_insights_dashboard(
    df: pd.DataFrame, 
    column_types: Dict[str, str]
) -> None:
    """Create business intelligence dashboard with actionable insights."""

    if STREAMLIT_AVAILABLE:
        st.markdown("### 💼 Business Intelligence Dashboard")
    else:
        print("Business Intelligence Dashboard")

    numeric_cols = [c for c, t in column_types.items() if t in ["num", "numeric", "float", "int"] and c in df.columns]
    categorical_cols = [c for c, t in column_types.items() if t in ["cat", "categorical", "object", "string"] and c in df.columns]

    if STREAMLIT_AVAILABLE:
        # Business context setup
        col1, col2, col3 = st.columns(3)
        with col1:
            revenue_col = st.selectbox("Revenue/Value Column:", ["None"] + numeric_cols, key=get_unique_key("revenue"))
        with col2:
            date_col = st.selectbox("Date Column:", ["None"] + list(df.columns), key=get_unique_key("date"))
        with col3:
            category_col = st.selectbox("Category Column:", ["None"] + categorical_cols, key=get_unique_key("category"))

        # Convert "None" to None
        revenue_col = None if revenue_col == "None" else revenue_col
        date_col = None if date_col == "None" else date_col
        category_col = None if category_col == "None" else category_col
    else:
        revenue_col = numeric_cols[0] if numeric_cols else None
        date_col = None
        category_col = categorical_cols[0] if categorical_cols else None

    # Revenue analysis
    if revenue_col and revenue_col in df.columns:
        _create_revenue_analysis(df, revenue_col)

    # Category performance analysis
    if category_col and category_col in df.columns:
        _create_category_performance_analysis(df, category_col, revenue_col)

def _create_revenue_analysis(df: pd.DataFrame, revenue_col: str) -> None:
    """Create revenue analysis section."""
    if STREAMLIT_AVAILABLE:
        st.markdown("#### 💰 Revenue Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_revenue = df[revenue_col].sum()
            st.metric("Total Revenue", f"${total_revenue:,.0f}")

        with col2:
            avg_revenue = df[revenue_col].mean()
            st.metric("Average Revenue", f"${avg_revenue:,.0f}")

        with col3:
            median_revenue = df[revenue_col].median()
            st.metric("Median Revenue", f"${median_revenue:,.0f}")

        with col4:
            revenue_std = df[revenue_col].std()
            cv = (revenue_std / avg_revenue) * 100 if avg_revenue != 0 else 0
            st.metric("Variability (CV)", f"{cv:.1f}%")

        # Revenue distribution
        fig = px.histogram(
            df,
            x=revenue_col,
            nbins=30,
            title="Revenue Distribution",
            color_discrete_sequence=[current_theme.accent_primary]
        )

        avg_revenue = df[revenue_col].mean()
        median_revenue = df[revenue_col].median()

        fig.add_vline(x=avg_revenue, line_dash="dash", line_color=current_theme.warning_color, 
                     annotation_text="Mean")
        fig.add_vline(x=median_revenue, line_dash="dot", line_color=current_theme.success_color,
                     annotation_text="Median")

        apply_professional_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True, key=get_unique_key("revenue_dist"))
    else:
        total_revenue = df[revenue_col].sum()
        avg_revenue = df[revenue_col].mean()
        print(f"Revenue Analysis:")
        print(f"  Total: ${total_revenue:,.0f}")
        print(f"  Average: ${avg_revenue:,.0f}")

def _create_category_performance_analysis(df: pd.DataFrame, category_col: str, revenue_col: Optional[str] = None) -> None:
    """Create category performance analysis."""
    if STREAMLIT_AVAILABLE:
        st.markdown("#### 📊 Category Performance")
    else:
        print("Category Performance Analysis:")

    if revenue_col and revenue_col in df.columns:
        # Revenue by category
        category_revenue = df.groupby(category_col)[revenue_col].agg(['sum', 'mean', 'count']).reset_index()
        category_revenue.columns = [category_col, 'Total Revenue', 'Avg Revenue', 'Count']
        category_revenue = category_revenue.sort_values('Total Revenue', ascending=False)

        if STREAMLIT_AVAILABLE:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    category_revenue.head(10),
                    x='Total Revenue',
                    y=category_col,
                    orientation='h',
                    title="Total Revenue by Category (Top 10)",
                    color='Total Revenue',
                    color_continuous_scale="Blues"
                )
                apply_professional_layout(fig, height=400)
                st.plotly_chart(fig, use_container_width=True, key=get_unique_key("category_revenue"))

            with col2:
                fig = px.bar(
                    category_revenue.head(10),
                    x='Avg Revenue',
                    y=category_col,
                    orientation='h',
                    title="Average Revenue by Category (Top 10)",
                    color='Avg Revenue',
                    color_continuous_scale="Greens"
                )
                apply_professional_layout(fig, height=400)
                st.plotly_chart(fig, use_container_width=True, key=get_unique_key("category_avg"))

            st.dataframe(category_revenue.head(15), use_container_width=True)
        else:
            print(f"  Top categories by revenue: {category_revenue.head(3).to_dict('records')}")

def create_advanced_analysis(df: pd.DataFrame, column_types: Dict[str, str]) -> None:
    """Enhanced advanced analysis with multiple analytical approaches."""

    if STREAMLIT_AVAILABLE:
        st.markdown("### 🔬 Advanced Data Analysis")

        analysis_tabs = st.tabs([
            "🎯 Outlier Detection", 
            "🔗 Correlation Analysis", 
            "💼 Business Intelligence",
            "📊 Data Quality Patterns"
        ])

        numeric_cols = [c for c, t in column_types.items() if t in ["num", "numeric", "float", "int"] and c in df.columns]

        with analysis_tabs[0]:
            if numeric_cols:
                _create_outlier_analysis(df, numeric_cols)
            else:
                st.info("No numeric columns available for outlier analysis.")

        with analysis_tabs[1]:
            if len(numeric_cols) >= 2:
                _create_correlation_analysis(df, numeric_cols)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")

        with analysis_tabs[2]:
            create_business_insights_dashboard(df, column_types)

        with analysis_tabs[3]:
            _create_data_quality_patterns(df, column_types)
    else:
        print("Advanced Data Analysis:")
        numeric_cols = [c for c, t in column_types.items() if t in ["num", "numeric", "float", "int"] and c in df.columns]
        if numeric_cols:
            _create_outlier_analysis(df, numeric_cols)
        if len(numeric_cols) >= 2:
            _create_correlation_analysis(df, numeric_cols)

def _create_outlier_analysis(df: pd.DataFrame, columns: List[str]) -> None:
    """Create comprehensive outlier analysis."""

    if STREAMLIT_AVAILABLE:
        st.markdown("#### 🎯 Outlier Detection")

        method = st.selectbox(
            "Detection Method:",
            ["IQR Method", "Z-Score", "Modified Z-Score"],
            key=get_unique_key("outlier_method")
        )
    else:
        method = "IQR Method"
        print("Outlier Detection (IQR Method):")

    outlier_results = []

    for col in columns[:5]:  # Limit to 5 columns for performance
        data = df[col].dropna()

        if len(data) < 10:
            continue

        outliers = _detect_outliers(data, method)
        outlier_count = outliers.sum()
        outlier_pct = (outlier_count / len(data)) * 100

        outlier_results.append({
            'Column': col,
            'Total Values': len(data),
            'Outliers': outlier_count,
            'Outlier %': f"{outlier_pct:.2f}%",
            'Min Outlier': data[outliers].min() if outlier_count > 0 else 'N/A',
            'Max Outlier': data[outliers].max() if outlier_count > 0 else 'N/A'
        })

    if outlier_results:
        results_df = pd.DataFrame(outlier_results)
        if STREAMLIT_AVAILABLE:
            st.dataframe(results_df, use_container_width=True)
        else:
            print(f"  Outlier summary: {results_df.to_dict('records')}")

def _detect_outliers(data: pd.Series, method: str) -> pd.Series:
    """Detect outliers using various methods."""

    if method == "IQR Method":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    elif method == "Z-Score":
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > 3

    elif method == "Modified Z-Score":
        median = data.median()
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return pd.Series([False] * len(data), index=data.index)
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > 3.5

    return pd.Series([False] * len(data), index=data.index)

def _create_correlation_analysis(df: pd.DataFrame, columns: List[str]) -> None:
    """Create correlation analysis."""

    if STREAMLIT_AVAILABLE:
        st.markdown("#### 🔗 Correlation Analysis")
    else:
        print("Correlation Analysis:")

    if len(columns) < 2:
        if STREAMLIT_AVAILABLE:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
        else:
            print("  Warning: Need at least 2 numeric columns")
        return

    # Calculate correlation matrix
    corr_matrix = df[columns].corr()

    if STREAMLIT_AVAILABLE:
        # Correlation heatmap
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            text_auto=True
        )
        apply_professional_layout(fig, height=500)
        st.plotly_chart(fig, use_container_width=True, key=get_unique_key("correlation_heatmap"))

        # Strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': round(corr_val, 3)
                    })

        if strong_corr:
            st.markdown("**🔗 Strong Correlations (|r| > 0.7):**")
            corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df, use_container_width=True)
    else:
        print(f"  Correlation matrix computed for {len(columns)} columns")

def _create_data_quality_patterns(df: pd.DataFrame, column_types: Dict[str, str]) -> None:
    """Create data quality pattern analysis."""

    if STREAMLIT_AVAILABLE:
        st.markdown("#### 📊 Data Quality Patterns")
    else:
        print("Data Quality Patterns:")

    categorical_cols = [c for c, t in column_types.items() if t in ["cat", "categorical", "object", "string"] and c in df.columns]

    quality_patterns = {
        'Missing Data Points': int(df.isnull().sum().sum()),
        'Duplicate Records': int(df.duplicated().sum()),
        'Constant Columns': len([col for col in df.columns if df[col].nunique() <= 1]),
        'High Cardinality Columns': len([col for col in categorical_cols if df[col].nunique() > len(df) * 0.8]),
        'Empty Columns': len([col for col in df.columns if df[col].isnull().all()])
    }

    if STREAMLIT_AVAILABLE:
        quality_df = pd.DataFrame(list(quality_patterns.items()), columns=['Pattern', 'Count'])

        fig = px.bar(
            quality_df,
            x='Count',
            y='Pattern',
            orientation='h',
            title="Data Quality Patterns Detected",
            color='Count',
            color_continuous_scale="Reds"
        )

        apply_professional_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True, key=get_unique_key("quality_patterns"))

        # Quality recommendations
        recommendations = []

        if quality_patterns['Missing Data Points'] > len(df) * 0.1:
            recommendations.append("🔧 High number of missing values - implement data validation at source")

        if quality_patterns['Duplicate Records'] > 0:
            recommendations.append("🔄 Remove duplicate records to improve data quality")

        if quality_patterns['Constant Columns'] > 0:
            recommendations.append("📊 Remove constant columns as they provide no analytical value")

        if quality_patterns['High Cardinality Columns'] > 0:
            recommendations.append("🏷️ High cardinality columns may need grouping or encoding")

        if not recommendations:
            recommendations.append("✅ Data quality is good! Continue current practices")

        st.markdown("**💡 Quality Recommendations:**")
        for rec in recommendations:
            st.markdown(f"• {rec}")
    else:
        print(f"  Quality patterns: {quality_patterns}")

# ============================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================

def create_data_overview_dashboard(df: pd.DataFrame, column_types: Dict[str, str]) -> None:
    """Backward compatible data overview dashboard."""
    create_enhanced_data_overview_dashboard(df, column_types)

def create_compact_missing_overview(df: pd.DataFrame) -> None:
    """Backward compatible missing value overview."""
    create_advanced_missing_value_analysis(df)

def create_numerical_visualizations(df: pd.DataFrame, num_columns: List[str]) -> None:
    """Backward compatible numerical visualizations."""
    create_statistical_distribution_analysis(df, num_columns)

# ============================
# EXPORTS
# ============================

__all__ = [
    'create_enhanced_data_overview_dashboard',
    'create_advanced_missing_value_analysis',
    'create_statistical_distribution_analysis',
    'create_categorical_visualizations',
    'create_business_insights_dashboard',
    'create_advanced_analysis',
    'set_theme',
    'apply_professional_layout',
    'render_header',
    'render_kpi_metrics',
    # Backward compatibility
    'create_data_overview_dashboard',
    'create_compact_missing_overview',
    'create_numerical_visualizations'
]
