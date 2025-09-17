"""
Data Visualization Module (Dark Theme)

Provides professional data visualization functions for data analysis including:
- Dataset overview dashboard (metrics + summaries)
- Missing value analysis
- Numerical and categorical visualizations
- Advanced analysis (outliers, scatter matrix, distribution by category)
with a cohesive, modern dark theme.

This version fixes widget key stability and unifies color themes. It also adds
sampling safeguards for large datasets to keep the app smooth and responsive.

Author: CortexX Team
Version: 1.0.0
"""

from __future__ import annotations
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

# Dark theme constants
DARK_BG_PRIMARY = "#0b1220"  # App background
DARK_BG_PANEL = "#0f172a"    # Plot area background
DARK_BG_SURFACE = "#111827"  # Paper background
TEXT_COLOR = "#e5e7eb"
GRID_COLOR = "#334155"

# Unified colorway for discrete data
COLORWAY = [
    "#60a5fa", "#f59e0b", "#22c55e", "#f472b6", "#a78bfa",
    "#06b6d4", "#84cc16", "#f97316", "#ef4444", "#94a3b8",
]

# Unified continuous scale
CONTINUOUS_SCALE = "Cividis"

# Plotly defaults
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = COLORWAY
px.defaults.color_continuous_scale = CONTINUOUS_SCALE

# Accent colors
PROFESSIONAL_COLORS = {
    "primary": "#60a5fa",
    "secondary": "#f59e0b",
    "success": "#22c55e",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#06b6d4",
    "purple": "#a78bfa",
    "pink": "#f472b6",
    "gray": "#94a3b8",
    "lime": "#84cc16",
}


def _apply_dark_layout(fig, height: Optional[int] = None, hovermode: Optional[str] = "x unified") -> None:
    """Apply cohesive dark theme layout styling to the Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG_SURFACE,
        plot_bgcolor=DARK_BG_PANEL,
        font=dict(color=TEXT_COLOR, size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode=hovermode,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TEXT_COLOR),
        ),
        bargap=0.05,
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR))
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR))


def _enhance_bars(fig) -> None:
    """Apply enhanced bar and histogram styling to the Plotly figure."""
    fig.update_traces(
        selector=dict(type="bar"),
        marker_line_color=GRID_COLOR,
        marker_line_width=1.2,
        opacity=0.95,
        textfont=dict(color=TEXT_COLOR),
    )
    fig.update_traces(
        selector=dict(type="histogram"),
        marker_line_color=GRID_COLOR,
        marker_line_width=1.2,
        opacity=0.95,
    )


def _uniq(prefix: str) -> str:
    """Generate a unique key with a given prefix for Streamlit widgets."""
    return f"{prefix}_{uuid4().hex}"


def _maybe_sample(df: pd.DataFrame, max_rows: int = 100_000) -> pd.DataFrame:
    """
    Sample the DataFrame to max_rows for heavy plotting operations.
    Deterministic sampling for reproducibility.
    """
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df


def create_data_overview_dashboard(df: pd.DataFrame, column_types: Dict[str, str]) -> None:
    """Display comprehensive dataset overview metrics and column summaries."""
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #0ea5e9 0%, #1d4ed8 100%);
        padding: 18px; border-radius: 12px; margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.12);">
        <h3 style="color: #e5e7eb; margin: 0;">📊 Dataset Overview</h3>
        <p style="color: #e5e7eb; opacity: 0.85; margin: 6px 0 0 0;">Comprehensive analysis of your data</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("📏 Total Rows", f"{len(df):,}")
    with c2:
        st.metric("📊 Total Columns", f"{len(df.columns):,}")
    with c3:
        num_cols = sum(1 for t in column_types.values() if t == "num")
        st.metric("🔢 Numerical", f"{num_cols}")
    with c4:
        cat_cols = sum(1 for t in column_types.values() if t == "cat")
        st.metric("🏷️ Categorical", f"{cat_cols}")
    with c5:
        missing_total = int(df.isna().sum().sum())
        total_cells = int(len(df) * max(1, len(df.columns)))
        missing_pct = (missing_total / total_cells) * 100 if total_cells else 0.0
        st.metric("❓ Missing", f"{missing_total:,}", delta=f"{missing_pct:.1f}%")
    st.markdown("---")
    _display_column_type_summary(df, column_types)


def _display_column_type_summary(df: pd.DataFrame, column_types: Dict[str, str]) -> None:
    """Display summaries for numerical and categorical columns."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        <div style="background: #0f172a; padding: 16px; border-radius: 10px; 
                    border-left: 4px solid #60a5fa; border: 1px solid #334155;">
            <h4 style="color: #60a5fa; margin-top: 0;">🔢 Numerical Columns</h4>
        """,
            unsafe_allow_html=True,
        )
        num_columns = [c for c, t in column_types.items() if t == "num" and c in df.columns]
        if num_columns:
            for col in num_columns[:8]:
                missing_pct = (df[col].isna().sum() / len(df) * 100) if len(df) else 0.0
                unique_pct = (df[col].nunique() / len(df) * 100) if len(df) else 0.0
                st.write(f"• {col} — {missing_pct:.1f}% missing, {unique_pct:.1f}% unique")
            if len(num_columns) > 8:
                st.write(f"... and {len(num_columns) - 8} more")
        else:
            st.write("No numerical columns detected")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
        <div style="background: #0f172a; padding: 16px; border-radius: 10px; 
                    border-left: 4px solid #f59e0b; border: 1px solid #334155;">
            <h4 style="color: #f59e0b; margin-top: 0;">🏷️ Categorical Columns</h4>
        """,
            unsafe_allow_html=True,
        )
        cat_columns = [c for c, t in column_types.items() if t == "cat" and c in df.columns]
        if cat_columns:
            for col in cat_columns[:8]:
                unique_count = df[col].nunique(dropna=True)
                most_common = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                st.write(f"• {col} — {unique_count} unique, top: {str(most_common)[:15]}")
            if len(cat_columns) > 8:
                st.write(f"... and {len(cat_columns) - 8} more")
        else:
            st.write("No categorical columns detected")
        st.markdown("</div>", unsafe_allow_html=True)


def create_compact_missing_overview(df: pd.DataFrame, key_prefix: Optional[str] = None) -> None:
    """Display compact overview of missing values."""
    if df.empty:
        st.warning("⚠️ Cannot analyze missing values: DataFrame is empty")
        return
    kp = key_prefix or uuid4().hex
    missing_data = df.isna().sum()
    missing_percent = (missing_data / len(df)) * 100 if len(df) else 0.0
    if missing_data.sum() == 0:
        st.success("✅ No missing values detected in your dataset!")
        return
    col1, col2 = st.columns(2)
    with col1:
        total_missing = int(missing_data.sum())
        coverage = float((total_missing / (len(df) * len(df.columns))) * 100) if len(df.columns) else 0.0
        st.metric(
            label="🔍 Total Missing Values",
            value=f"{total_missing:,}",
            delta=f"{coverage:.1f}% of dataset",
        )
        missing_cols = missing_percent[missing_percent > 0].sort_values(ascending=True)
        if len(missing_cols) > 0:
            fig = px.bar(
                x=missing_cols.values,
                y=missing_cols.index,
                orientation="h",
                title="Missing Values by Column (%)",
                color=missing_cols.values,
                color_continuous_scale=CONTINUOUS_SCALE,
            )
            _enhance_bars(fig)
            _apply_dark_layout(fig, height=360)
            st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_missing_bar"))
    with col2:
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        missing_matrix = sample_df.isna().astype(int)
        missing_scale = [(0.0, DARK_BG_PANEL), (1.0, PROFESSIONAL_COLORS["warning"])]
        fig = px.imshow(missing_matrix.T, aspect="auto", color_continuous_scale=missing_scale, title="Missing Values Pattern")
        _apply_dark_layout(fig, height=360)
        fig.update_xaxes(title=f"Rows ({'sampled' if len(df) > sample_size else 'all'})")
        fig.update_yaxes(title="Columns")
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_missing_heatmap"))


def create_numerical_visualizations(
    df: pd.DataFrame, num_columns: List[str], key_prefix: Optional[str] = None
) -> None:
    """Create visualizations for numerical columns."""
    if not num_columns:
        st.info("📊 No numerical columns available for visualization.")
        return

    kp = key_prefix or "num"
    col_select, viz_select = st.columns([2, 1])

    with col_select:
        selected_cols = st.multiselect(
            "📈 Select numerical columns to visualize:",
            num_columns,
            default=num_columns[: min(4, len(num_columns))],
            help="Choose up to 6 columns for optimal display",
            key=f"{kp}_num_select",
        )
    with viz_select:
        viz_type = st.selectbox(
            "📊 Visualization Type:",
            ["Distribution", "Box Plot", "Correlation Matrix", "Summary Stats"],
            help="Choose the type of analysis",
            key=f"{kp}_num_viztype",
        )

    if not selected_cols:
        st.warning("Please select at least one column to visualize.")
        return

    _create_numerical_plots(df, selected_cols, viz_type, kp)


def _create_numerical_plots(df: pd.DataFrame, selected_cols: List[str], viz_type: str, kp: str) -> None:
    if viz_type == "Distribution":
        _create_distribution_plots(df, selected_cols, kp)
    elif viz_type == "Box Plot":
        _create_box_plots(df, selected_cols, kp)
    elif viz_type == "Correlation Matrix":
        _create_correlation_matrix(df, selected_cols, kp)
    elif viz_type == "Summary Stats":
        _create_summary_stats(df, selected_cols)


def _create_distribution_plots(df: pd.DataFrame, selected_cols: List[str], kp: str) -> None:
    dfp = _maybe_sample(df)
    num_plots = len(selected_cols)
    if num_plots == 1:
        fig = px.histogram(
            dfp,
            x=selected_cols[0],
            nbins=30,
            title=f"Distribution of {selected_cols[0]}",
            color_discrete_sequence=[PROFESSIONAL_COLORS["primary"]],
            opacity=0.9,
            marginal="rug",
        )
        _enhance_bars(fig)
        _apply_dark_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_dist_{selected_cols[0]}"))
    elif num_plots == 2:
        col1, col2 = st.columns(2)
        for i, col in enumerate(selected_cols):
            with col1 if i == 0 else col2:
                fig = px.histogram(
                    dfp,
                    x=col,
                    nbins=25,
                    title=f"{col}",
                    color_discrete_sequence=[
                        PROFESSIONAL_COLORS["primary"] if i == 0 else PROFESSIONAL_COLORS["secondary"]
                    ],
                    opacity=0.9,
                )
                _enhance_bars(fig)
                _apply_dark_layout(fig, height=320)
                st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_dist_{col}_{i}"))
    else:
        cols = st.columns(2)
        colors = [
            PROFESSIONAL_COLORS["primary"],
            PROFESSIONAL_COLORS["secondary"],
            PROFESSIONAL_COLORS["success"],
            PROFESSIONAL_COLORS["warning"],
            PROFESSIONAL_COLORS["purple"],
            PROFESSIONAL_COLORS["pink"],
        ]
        for i, col in enumerate(selected_cols):
            with cols[i % 2]:
                fig = px.histogram(
                    dfp,
                    x=col,
                    nbins=20,
                    title=f"{col}",
                    color_discrete_sequence=[colors[i % len(colors)]],
                    opacity=0.9,
                )
                _enhance_bars(fig)
                _apply_dark_layout(fig, height=300)
                st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_dist_multi_{col}_{i}"))


def _create_box_plots(df: pd.DataFrame, selected_cols: List[str], kp: str) -> None:
    dfp = _maybe_sample(df)
    num_plots = len(selected_cols)
    if num_plots == 1:
        fig = px.box(
            dfp,
            y=selected_cols[0],
            title=f"Box Plot: {selected_cols[0]}",
            color_discrete_sequence=[PROFESSIONAL_COLORS["success"]],
            points="outliers",
        )
        _apply_dark_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_box_{selected_cols[0]}"))
    elif num_plots == 2:
        col1, col2 = st.columns(2)
        for i, col in enumerate(selected_cols):
            with col1 if i == 0 else col2:
                fig = px.box(
                    dfp,
                    y=col,
                    title=f"{col}",
                    color_discrete_sequence=[
                        PROFESSIONAL_COLORS["success"] if i == 0 else PROFESSIONAL_COLORS["warning"]
                    ],
                    points="outliers",
                )
                _apply_dark_layout(fig, height=320)
                st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_box_{col}_{i}"))
    else:
        df_melted = _maybe_sample(df[selected_cols]).melt(var_name="Column", value_name="Value")
        fig = px.box(
            df_melted,
            x="Column",
            y="Value",
            title="Box Plots Comparison",
            color="Column",
            color_discrete_sequence=COLORWAY,
            points=False,
        )
        _apply_dark_layout(fig, height=420)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_box_multi"))


def _create_correlation_matrix(df: pd.DataFrame, selected_cols: List[str], kp: str) -> None:
    if len(selected_cols) > 1:
        dfp = df[selected_cols]
        corr_matrix = dfp.corr(numeric_only=True)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=CONTINUOUS_SCALE,
            zmin=corr_matrix.min().min(),
            zmax=corr_matrix.max().max(),
            title="Correlation Matrix",
        )
        _apply_dark_layout(fig, height=460, hovermode=None)
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_corr"))
        _display_correlation_insights(corr_matrix)
    else:
        st.warning("⚠️ Select at least 2 columns for correlation analysis.")


def _display_correlation_insights(corr_matrix: pd.DataFrame) -> None:
    st.markdown("#### 🔍 Correlation Insights")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    if corr_pairs:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strong Positive Correlations (>0.5):**")
            for a, b, v in sorted(corr_pairs, key=lambda x: x[2], reverse=True):
                if v > 0.5:
                    st.write(f"• {a} ↔ {b}: {v:.3f}")
        with col2:
            st.markdown("**Strong Negative Correlations (<-0.5):**")
            for a, b, v in sorted(corr_pairs, key=lambda x: x[2]):
                if v < -0.5:
                    st.write(f"• {a} ↔ {b}: {v:.3f}")
    else:
        st.info("No strong correlations (>0.5 or <-0.5) found between selected columns.")


def _create_summary_stats(df: pd.DataFrame, selected_cols: List[str]) -> None:
    st.markdown("#### 📊 Statistical Summary")
    stats_df = df[selected_cols].describe()
    for col in selected_cols:
        with st.expander(f"📈 {col} Statistics", expanded=len(selected_cols) <= 2):
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Mean", f"{stats_df.loc['mean', col]:.2f}")
            with m2:
                st.metric("Std Dev", f"{stats_df.loc['std', col]:.2f}")
            with m3:
                st.metric("Min", f"{stats_df.loc['min', col]:.2f}")
            with m4:
                st.metric("Max", f"{stats_df.loc['max', col]:.2f}")


def create_categorical_visualizations(
    df: pd.DataFrame, cat_columns: List[str], key_prefix: Optional[str] = None
) -> None:
    """Create visualizations for categorical columns."""
    if not cat_columns:
        st.info("🏷️ No categorical columns available for visualization.")
        return

    kp = key_prefix or "cat"
    col_select, viz_select = st.columns([2, 1])

    with col_select:
        selected_cols = st.multiselect(
            "🏷️ Select categorical columns to visualize:",
            cat_columns,
            default=cat_columns[: min(4, len(cat_columns))],
            help="Choose columns to analyze",
            key=f"{kp}_cat_select",
        )
    with viz_select:
        viz_type = st.selectbox(
            "📊 Visualization Type:",
            ["Value Counts", "Top Categories", "Category Distribution"],
            help="Choose analysis type",
            key=f"{kp}_cat_viztype",
        )

    if not selected_cols:
        st.warning("Please select at least one column to visualize.")
        return

    _create_categorical_plots(df, selected_cols, viz_type, kp)


def _create_categorical_plots(df: pd.DataFrame, selected_cols: List[str], viz_type: str, kp: str) -> None:
    if viz_type == "Value Counts":
        _create_value_count_plots(df, selected_cols, kp)
    elif viz_type == "Top Categories":
        _create_top_categories_display(df, selected_cols)
    elif viz_type == "Category Distribution":
        _create_category_distribution_plots(df, selected_cols, kp)


def _create_value_count_plots(df: pd.DataFrame, selected_cols: List[str], kp: str) -> None:
    num_plots = len(selected_cols)
    if num_plots == 1:
        col = selected_cols[0]
        top_values = df[col].value_counts().head(15)
        fig = px.bar(
            x=top_values.values,
            y=top_values.index,
            orientation="h",
            title=f"Value Counts: {col}",
            color=top_values.index,
            color_discrete_sequence=COLORWAY,
        )
        fig.update_traces(text=top_values.values, textposition="outside")
        _enhance_bars(fig)
        _apply_dark_layout(fig, height=440)
        fig.update_xaxes(title="Count")
        fig.update_yaxes(title="Categories", categoryorder="total ascending")
        fig.update_layout(margin=dict(r=70))
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_cat_counts_{col}"))
    elif num_plots == 2:
        col1, col2 = st.columns(2)
        for i, col in enumerate(selected_cols):
            with col1 if i == 0 else col2:
                top_values = df[col].value_counts().head(10)
                fig = px.bar(
                    x=top_values.values,
                    y=top_values.index,
                    orientation="h",
                    title=f"{col}",
                    color=top_values.index,
                    color_discrete_sequence=COLORWAY,
                )
                fig.update_traces(text=top_values.values, textposition="outside")
                _enhance_bars(fig)
                _apply_dark_layout(fig, height=360)
                fig.update_xaxes(title="Count")
                fig.update_yaxes(title="", categoryorder="total ascending")
                fig.update_layout(margin=dict(r=60))
                st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_cat_counts_{col}_{i}"))
    else:
        cols = st.columns(2)
        for i, col in enumerate(selected_cols):
            with cols[i % 2]:
                top_values = df[col].value_counts().head(8)
                fig = px.bar(
                    x=top_values.values,
                    y=top_values.index,
                    orientation="h",
                    title=f"{col}",
                    color=top_values.index,
                    color_discrete_sequence=COLORWAY,
                )
                fig.update_traces(text=top_values.values, textposition="outside")
                _enhance_bars(fig)
                _apply_dark_layout(fig, height=320)
                fig.update_xaxes(title="Count")
                fig.update_yaxes(title="", categoryorder="total ascending")
                fig.update_layout(margin=dict(r=50))
                st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_cat_counts_{col}_{i}"))


def _create_top_categories_display(df: pd.DataFrame, selected_cols: List[str]) -> None:
    num_cols = len(selected_cols)
    cols = st.columns(num_cols if num_cols <= 3 else 3)
    for i, col in enumerate(selected_cols):
        with cols[i % len(cols)]:
            if not df[col].empty:
                counts = df[col].value_counts()
                if counts.empty:
                    continue
                top_value = counts.index[0]
                top_count = int(counts.iloc[0])
                total = int(len(df[col].dropna()))
                percentage = (top_count / total * 100) if total else 0
                st.metric(
                    label=f"🏆 Top in {col}",
                    value=str(top_value)[:20] + "..." if len(str(top_value)) > 20 else str(top_value),
                    delta=f"{top_count:,} ({percentage:.1f}%)",
                )
                st.caption(f"Total unique: {counts.index.nunique():,}")


def _create_category_distribution_plots(df: pd.DataFrame, selected_cols: List[str], kp: str) -> None:
    num_plots = len(selected_cols)
    if num_plots == 1:
        col = selected_cols[0]
        top_values = df[col].value_counts().head(8)
        fig = px.pie(
            values=top_values.values,
            names=top_values.index,
            title=f"Distribution: {col}",
            color_discrete_sequence=COLORWAY,
            hole=0.35,
        )
        _apply_dark_layout(fig, height=420)
        fig.update_traces(textinfo="percent+label", textfont=dict(color=TEXT_COLOR))
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_cat_pie_{col}"))
    elif num_plots == 2:
        col1, col2 = st.columns(2)
        for i, col in enumerate(selected_cols):
            with col1 if i == 0 else col2:
                top_values = df[col].value_counts().head(6)
                fig = px.pie(
                    values=top_values.values,
                    names=top_values.index,
                    title=f"{col}",
                    color_discrete_sequence=COLORWAY,
                    hole=0.35,
                )
                _apply_dark_layout(fig, height=360)
                fig.update_traces(textinfo="percent+label", textfont=dict(color=TEXT_COLOR))
                st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_cat_pie_{col}_{i}"))
    else:
        cols = st.columns(2)
        for i, col in enumerate(selected_cols):
            with cols[i % 2]:
                top_values = df[col].value_counts().head(5)
                total = int(df[col].count())
                percentages = (top_values / total * 100).round(1) if total else top_values
                fig = px.bar(
                    x=percentages.values,
                    y=percentages.index,
                    orientation="h",
                    title=f"{col} (%)",
                    color_discrete_sequence=[PROFESSIONAL_COLORS["primary"]],
                )
                fig.update_traces(text=percentages.values, textposition="outside")
                _enhance_bars(fig)
                _apply_dark_layout(fig, height=260)
                fig.update_xaxes(title="Percentage (%)")
                st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_cat_bar_{col}_{i}"))


def create_advanced_analysis(df: pd.DataFrame, column_types: Dict[str, str], key_prefix: Optional[str] = None) -> None:
    """Create advanced analysis including outlier rates, scatter matrix, and distribution by category."""
    kp = key_prefix or "adv"
    num_cols = [c for c, t in column_types.items() if t == "num" and c in df.columns]
    cat_cols = [c for c, t in column_types.items() if t == "cat" and c in df.columns]

    # 1) Outlier analysis
    st.markdown("### 📌 Outlier Analysis (IQR method)")
    if num_cols:
        summary = []
        for c in num_cols:
            series = df[c].dropna()
            if series.empty:
                summary.append((c, 0.0, 0))
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((series < lower) | (series > upper)).sum()
            pct = (outliers / len(series) * 100) if len(series) else 0.0
            summary.append((c, pct, outliers))
        out_df = pd.DataFrame(summary, columns=["Column", "Outlier %", "Outlier Count"]).sort_values(
            "Outlier %", ascending=False
        )
        top_out = out_df.head(8)
        if not top_out.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    top_out,
                    x="Outlier %",
                    y="Column",
                    orientation="h",
                    color="Outlier %",
                    color_continuous_scale=CONTINUOUS_SCALE,
                    title="Top Outlier Rates by Column",
                )
                fig.update_traces(text=top_out["Outlier %"].round(1), textposition="outside")
                _enhance_bars(fig)
                _apply_dark_layout(fig, height=360)
                fig.update_layout(margin=dict(r=70))
                st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_outliers"))
            with col2:
                st.dataframe(top_out, use_container_width=True, height=360)
        else:
            st.info("No numerical data to analyze outliers.")
    else:
        st.info("No numerical columns available.")

    st.markdown("---")

    # 2) Scatter matrix (sampled)
    st.markdown("### 🔬 Scatter Matrix")
    if len(num_cols) >= 2:
        sm_cols = num_cols[:5]
        sample_n = min(1000, len(df))
        sample_df = df.sample(sample_n, random_state=42) if len(df) > sample_n else df
        fig = px.scatter_matrix(sample_df, dimensions=sm_cols, color=None, title="Scatter Matrix (sampled)")
        _apply_dark_layout(fig, height=520, hovermode=None)
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_scatter_matrix"))
    else:
        st.info("Need at least 2 numerical columns for a scatter matrix.")

    st.markdown("---")

    # 3) Distribution by category
    st.markdown("### 🧩 Distribution by Category")
    if num_cols and cat_cols:
        c1, c2, c3 = st.columns([1.5, 1.5, 1])
        with c1:
            num_col = st.selectbox("Numerical Column", num_cols, key=f"{kp}_adv_num_col")
        with c2:
            cat_col = st.selectbox("Category Column", cat_cols, key=f"{kp}_adv_cat_col")
        with c3:
            plot_mode = st.selectbox("Plot", ["Box", "Violin"], key=f"{kp}_adv_plot_mode")
        dfp = _maybe_sample(df)
        if plot_mode == "Box":
            fig = px.box(
                dfp,
                x=cat_col,
                y=num_col,
                color=cat_col,
                color_discrete_sequence=COLORWAY,
                title=f"{num_col} by {cat_col}",
                points="outliers",
            )
        else:
            fig = px.violin(
                dfp,
                x=cat_col,
                y=num_col,
                color=cat_col,
                color_discrete_sequence=COLORWAY,
                title=f"{num_col} by {cat_col}",
                box=True,
                points=False,
            )
        _apply_dark_layout(fig, height=460)
        st.plotly_chart(fig, use_container_width=True, key=_uniq(f"{kp}_dist_by_cat"))
    else:
        st.info("Select at least one numerical and one categorical column to compare distributions.")