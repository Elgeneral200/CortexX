# filename: visualization.py
"""
Data Visualization Module - Enhanced Professional Edition v2.4 - Phase 1+ (Generalized Safe Labels)

Advanced visualization system with:
- Automated Exploratory Data Analysis (EDA)
- Professional business intelligence dashboards
- Interactive correlation and distribution analysis
- Advanced outlier detection visualizations
- Memory-optimized rendering for large datasets
- FIXED: Plotly titleside error, Streamlit duplicate chart IDs
- Phase 1: Robust label placement for mean/median within subplot grids
- Phase 1+: Generalized safe annotations (including Missing Data), ECDF & QQ-Plot, improved heatmap options

Author: CortexX Team
Version: 2.4.2 - Stable (Backwards Compatible + ECDF validation + Stable EDA selectbox)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime
import time
import math
import uuid

warnings.filterwarnings('ignore')

# Optional dependencies
try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not available - some advanced statistical features will be limited")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available - some ML-based features will be limited")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("⚠️ Streamlit not available - dashboard features will be limited")

# ============================
# ENHANCED THEME SYSTEM
# ============================

class EnhancedThemeManager:
    """Professional theme management with business intelligence styling."""
    def __init__(self):
        self.themes = {
            "professional_dark": {
                "background_color": "#0f172a",
                "paper_color": "#1e293b",
                "text_color": "#f8fafc",
                "grid_color": "#374151",
                "accent_color": "#3b82f6",
                "success_color": "#10b981",
                "warning_color": "#f59e0b",
                "error_color": "#ef4444",
                "color_palette": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#f97316"]
            },
            "professional_light": {
                "background_color": "#ffffff",
                "paper_color": "#f8fafc",
                "text_color": "#1f2937",
                "grid_color": "#e5e7eb",
                "accent_color": "#2563eb",
                "success_color": "#059669",
                "warning_color": "#d97706",
                "error_color": "#dc2626",
                "color_palette": ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#65a30d", "#ea580c"]
            },
            "business_intelligence": {
                "background_color": "#1a1d29",
                "paper_color": "#252a41",
                "text_color": "#ffffff",
                "grid_color": "#3d4465",
                "accent_color": "#00d4ff",
                "success_color": "#00ff88",
                "warning_color": "#ffaa00",
                "error_color": "#ff4757",
                "color_palette": ["#00d4ff", "#00ff88", "#ffaa00", "#ff4757", "#a55eea", "#26de81", "#fed330", "#ff3838"]
            }
        }
        self.current_theme = "professional_dark"

    def get_theme(self, theme_name: str = None) -> Dict[str, str]:
        theme_name = theme_name or self.current_theme
        return self.themes.get(theme_name, self.themes["professional_dark"])

    def apply_theme_to_figure(self, fig: go.Figure, theme_name: str = None) -> go.Figure:
        theme = self.get_theme(theme_name)
        fig.update_layout(
            plot_bgcolor=theme["background_color"],
            paper_bgcolor=theme["paper_color"],
            font_color=theme["text_color"],
            font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            font_size=12,
            title_font_size=16,
            title_font_color=theme["text_color"],
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=theme["grid_color"], borderwidth=1),
            margin=dict(l=50, r=50, t=70, b=50)
        )
        fig.update_xaxes(gridcolor=theme["grid_color"], linecolor=theme["grid_color"], tickcolor=theme["text_color"], title_font_color=theme["text_color"])
        fig.update_yaxes(gridcolor=theme["grid_color"], linecolor=theme["grid_color"], tickcolor=theme["text_color"], title_font_color=theme["text_color"])
        return fig

# Initialize theme manager
theme_manager = EnhancedThemeManager()

# ============================
# AUTOMATED EDA FUNCTIONS
# ============================

def generate_automated_eda_report(df: pd.DataFrame, sample_size: int = 10000) -> Dict[str, Any]:
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        sampled = True
    else:
        df_sample = df.copy()
        sampled = False

    report = {
        "dataset_info": get_dataset_overview(df),
        "column_analysis": analyze_columns_detailed(df_sample),
        "correlation_analysis": calculate_correlations(df_sample),
        "missing_data_analysis": analyze_missing_patterns(df),
        "outlier_analysis": detect_outliers_comprehensive(df_sample),
        "distribution_analysis": analyze_distributions(df_sample),
        "recommendations": generate_eda_recommendations(df),
        "sampled": sampled,
        "sample_size": len(df_sample) if sampled else len(df)
    }
    return report

def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = []
    for col in categorical_cols:
        sample = df[col].dropna().head(100)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, errors='raise')
                datetime_cols.append(col)
            except:
                pass
    return {
        "shape": df.shape,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "datetime_columns": len(datetime_cols),
        "missing_values": df.isnull().sum().sum(),
        "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100,
        "column_types": {"numeric": numeric_cols, "categorical": categorical_cols, "datetime": datetime_cols}
    }

def analyze_columns_detailed(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    analysis = {}
    for col in df.columns:
        col_analysis = {
            "dtype": str(df[col].dtype),
            "non_null_count": df[col].count(),
            "null_count": df[col].isnull().sum(),
            "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
            "unique_count": df[col].nunique(),
            "unique_percentage": (df[col].nunique() / len(df)) * 100,
            "memory_usage": df[col].memory_usage(deep=True)
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                col_analysis.update({
                    "min": float(non_null_data.min()),
                    "max": float(non_null_data.max()),
                    "mean": float(non_null_data.mean()),
                    "median": float(non_null_data.median()),
                    "std": float(non_null_data.std()),
                    "skewness": float(non_null_data.skew()),
                    "kurtosis": float(non_null_data.kurtosis()),
                    "q25": float(non_null_data.quantile(0.25)),
                    "q75": float(non_null_data.quantile(0.75)),
                    "iqr": float(non_null_data.quantile(0.75) - non_null_data.quantile(0.25))
                })
        elif pd.api.types.is_object_dtype(df[col]):
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                value_counts = non_null_data.value_counts()
                col_analysis.update({
                    "most_frequent": str(value_counts.index[0]),
                    "most_frequent_count": int(value_counts.iloc[0]),
                    "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else "N/A",
                    "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    "average_length": float(non_null_data.astype(str).str.len().mean()),
                    "max_length": int(non_null_data.astype(str).str.len().max()),
                    "min_length": int(non_null_data.astype(str).str.len().min())
                })
        analysis[col] = col_analysis
    return analysis

def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return {"correlation_matrix": None, "strong_correlations": [], "warning": "Less than 2 numeric columns"}
    corr_matrix = numeric_df.corr()

    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                strong_correlations.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": float(corr_value),
                    "strength": "Very Strong" if abs(corr_value) > 0.9 else "Strong"
                })
    return {"correlation_matrix": corr_matrix, "strong_correlations": strong_correlations, "numeric_columns": len(numeric_df.columns)}

def analyze_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    missing_data = df.isnull()
    missing_by_column = missing_data.sum().sort_values(ascending=False)
    missing_by_column = missing_by_column[missing_by_column > 0]
    if len(missing_by_column) > 0:
        return {"missing_by_column": missing_by_column.to_dict(), "total_missing": missing_data.sum().sum(), "columns_with_missing": len(missing_by_column), "missing_patterns": []}
    else:
        return {"missing_by_column": {}, "total_missing": 0, "columns_with_missing": 0, "missing_patterns": []}

def detect_outliers_comprehensive(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {"outliers_by_method": {}, "outlier_summary": {}}
    outlier_results = {}
    for col in numeric_cols[:5]:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        col_outliers = {}
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((series < lower_bound) | (series > upper_bound))
        col_outliers["iqr"] = {"count": int(iqr_outliers.sum()), "percentage": float((iqr_outliers.sum() / len(series)) * 100), "bounds": [float(lower_bound), float(upper_bound)]}
        if SCIPY_AVAILABLE:
            try:
                z_scores = np.abs(stats.zscore(series))
                zscore_outliers = z_scores > 3
                col_outliers["zscore"] = {"count": int(zscore_outliers.sum()), "percentage": float((zscore_outliers.sum() / len(series)) * 100), "threshold": 3.0}
            except:
                pass
        if SKLEARN_AVAILABLE and len(series) > 100:
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                iso_outliers = outlier_labels == -1
                col_outliers["isolation_forest"] = {"count": int(iso_outliers.sum()), "percentage": float((iso_outliers.sum() / len(series)) * 100), "contamination": 0.1}
            except:
                pass
        outlier_results[col] = col_outliers
    summary = {}
    for method in ["iqr", "zscore", "isolation_forest"]:
        method_counts = [col_data.get(method, {}).get("count", 0) for col_data in outlier_results.values()]
        if method_counts:
            summary[method] = {"total_outliers": sum(method_counts), "avg_per_column": np.mean(method_counts), "columns_with_outliers": sum(1 for count in method_counts if count > 0)}
    return {"outliers_by_method": outlier_results, "outlier_summary": summary}

def analyze_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {"distributions": {}, "summary": "No numeric columns found"}
    distributions = {}
    for col in numeric_cols[:10]:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        dist_analysis = {"skewness": float(series.skew()), "kurtosis": float(series.kurtosis()), "normality_test": None}
        skew = dist_analysis["skewness"]
        if skew > 0.5:
            dist_analysis["distribution_shape"] = "Right Skewed"
        elif skew < -0.5:
            dist_analysis["distribution_shape"] = "Left Skewed"
        else:
            dist_analysis["distribution_shape"] = "Approximately Symmetric"
        if SCIPY_AVAILABLE and len(series) >= 8:
            try:
                statistic, p_value = stats.normaltest(series)
                dist_analysis["normality_test"] = {"statistic": float(statistic), "p_value": float(p_value), "is_normal": p_value > 0.05}
            except:
                pass
        distributions[col] = dist_analysis
    return {"distributions": distributions}

def generate_eda_recommendations(df: pd.DataFrame) -> List[str]:
    recommendations = []
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    if missing_pct > 10:
        recommendations.append(f"High missing data ({missing_pct:.1f}%) - Consider imputation strategies or investigate data collection issues")
    duplicate_pct = (df.duplicated().sum() / len(df)) * 100
    if duplicate_pct > 1:
        recommendations.append(f"Found {duplicate_pct:.1f}% duplicate rows - Review data collection process and remove duplicates")
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if memory_mb > 100:
        recommendations.append(f"Large dataset ({memory_mb:.1f}MB) - Consider data type optimization and sampling for analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(numeric_cols) > 10:
        recommendations.append("Many numeric columns - Consider dimensionality reduction or feature selection")
    if len(categorical_cols) > len(numeric_cols) * 2:
        recommendations.append("Many categorical columns - Consider encoding strategies for machine learning")
    for col in categorical_cols[:5]:
        unique_pct = (df[col].nunique() / len(df)) * 100
        if unique_pct > 50:
            recommendations.append(f"Column '{col}' has high cardinality ({unique_pct:.1f}%) - Consider grouping rare categories")
    if not recommendations:
        recommendations.append("Data quality looks good! Ready for advanced analysis and modeling")
    return recommendations

# ============================
# PHASE 1 HELPERS (GENERALIZED SAFE LABEL PLACEMENT)
# ============================

def _axis_index(row: int, col: int, n_cols: int) -> int:
    return (row - 1) * n_cols + col

def _axis_name(axis_index: int, axis: str) -> str:
    return axis if axis_index == 1 else f"{axis}{axis_index}"

def _get_domains(fig: go.Figure, axis_index: int) -> Tuple[List[float], List[float]]:
    xaxis_key = _axis_name(axis_index, "xaxis")
    yaxis_key = _axis_name(axis_index, "yaxis")
    xdom = list(getattr(fig.layout, xaxis_key).domain) if getattr(fig.layout, xaxis_key, None) else [0.0, 1.0]
    ydom = list(getattr(fig.layout, yaxis_key).domain) if getattr(fig.layout, yaxis_key, None) else [0.0, 1.0]
    return xdom, ydom

def _add_stat_line_with_label(fig: go.Figure, *, n_cols: int, row: int, col: int, x_value: float, label_text: str, color: str, line_dash: str, y_level: Optional[float] = None, xshift: int = 0) -> None:
    fig.add_vline(x=x_value, line_dash=line_dash, line_color=color, row=row, col=col)
    axis_idx = _axis_index(row, col, n_cols)
    _, ydom = _get_domains(fig, axis_idx)
    y_pos = y_level if y_level is not None else (ydom[1] - 0.035)
    fig.add_annotation(
        x=x_value,
        y=y_pos,
        xref=_axis_name(axis_idx, "x"),
        yref="paper",
        text=label_text,
        showarrow=False,
        font=dict(size=10, color="white"),
        align="center",
        bgcolor="rgba(0,0,0,0.35)",
        bordercolor="rgba(255,255,255,0.12)",
        borderwidth=1,
        xanchor="center",
        yanchor="bottom",
        xshift=xshift
    )

def _add_top_band_labels_for_x(fig: go.Figure, texts: List[str], x_positions: List[Any], angle: int = 0, top_pad: float = 0.04) -> None:
    y = 1.0 + top_pad
    for x, t in zip(x_positions, texts):
        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="paper",
            text=t,
            showarrow=False,
            font=dict(size=10),
            textangle=angle,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0.25)"
        )

# ============================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================

def create_correlation_heatmap(df: pd.DataFrame, theme: str = "professional_dark", annotate_threshold: Optional[float] = None, mask_upper: bool = False) -> go.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 numeric columns for correlation analysis", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
        return theme_manager.apply_theme_to_figure(fig, theme)

    corr_matrix = numeric_df.corr().copy()
    corr_vals = corr_matrix.mask(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)) if mask_upper else corr_matrix

    fig = go.Figure(data=go.Heatmap(
        z=corr_vals.values,
        x=corr_vals.columns,
        y=corr_vals.index,
        colorscale='RdBu',
        zmid=0,
        hoverongaps=False,
        colorbar=dict(title="Correlation")
    ))

    if annotate_threshold is not None:
        for r, row_name in enumerate(corr_vals.index):
            for c, col_name in enumerate(corr_vals.columns):
                val = corr_vals.iloc[r, c]
                if pd.notna(val) and abs(val) >= annotate_threshold:
                    fig.add_annotation(x=col_name, y=row_name, text=f"{val:.2f}", showarrow=False, font=dict(size=10), xref="x", yref="y")

    fig.update_layout(title="📊 Correlation Matrix - Feature Relationships", width=800, height=600, xaxis_tickangle=-45)
    return theme_manager.apply_theme_to_figure(fig, theme)

def create_distribution_grid(df: pd.DataFrame, theme: str = "professional_dark", max_cols: int = 4, show_stat_labels: bool = True, label_mode: str = "auto") -> go.Figure:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns found for distribution analysis", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
        return theme_manager.apply_theme_to_figure(fig, theme)

    numeric_cols = numeric_cols[:12]
    n_cols = min(max_cols, len(numeric_cols))
    n_rows = math.ceil(len(numeric_cols) / n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{col}" for col in numeric_cols],
        vertical_spacing=0.16,
        horizontal_spacing=0.10
    )

    theme_colors = theme_manager.get_theme(theme)["color_palette"]

    for i, col in enumerate(numeric_cols):
        row = (i // n_cols) + 1
        col_pos = (i % n_cols) + 1
        series = df[col].dropna()
        if len(series) == 0:
            continue

        fig.add_trace(
            go.Histogram(x=series, name=col, showlegend=False, marker_color=theme_colors[i % len(theme_colors)], opacity=0.7, nbinsx=30),
            row=row, col=col_pos
        )

        if show_stat_labels and label_mode in ("auto", "inside"):
            try:
                mean_val = float(series.mean())
                median_val = float(series.median())
                x_min = float(series.min())
                x_max = float(series.max())
                x_range = max(x_max - x_min, 1e-9)
                close = abs(mean_val - median_val) / x_range < 0.05

                if not np.isnan(mean_val):
                    _add_stat_line_with_label(fig, n_cols=n_cols, row=row, col=col_pos, x_value=mean_val, label_text=f"μ={mean_val:.2f}", color="yellow", line_dash="dash", xshift=-10 if close else 0)
                if not np.isnan(median_val):
                    axis_idx = _axis_index(row, col_pos, n_cols)
                    _, ydom = _get_domains(fig, axis_idx)
                    y_alt = (ydom[1] - 0.085) if close else None
                    _add_stat_line_with_label(fig, n_cols=n_cols, row=row, col=col_pos, x_value=median_val, label_text=f"M={median_val:.2f}", color="orange", line_dash="dot", y_level=y_alt, xshift=10 if close else 0)
            except Exception:
                pass

    fig.update_layout(title="📈 Distribution Analysis - Statistical Overview", height=300 * n_rows, showlegend=False)
    return theme_manager.apply_theme_to_figure(fig, theme)

def create_missing_data_heatmap(df: pd.DataFrame, theme: str = "professional_dark") -> go.Figure:
    missing_data = df.isnull()
    if missing_data.sum().sum() == 0:
        fig = go.Figure()
        fig.add_annotation(text="🎉 No missing data detected! Dataset is complete.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16, font_color="green")
        return theme_manager.apply_theme_to_figure(fig, theme)

    missing_sample = missing_data.sample(1000, random_state=42) if len(df) > 1000 else missing_data

    fig = go.Figure(data=go.Heatmap(
        z=missing_sample.values.astype(int),
        x=missing_sample.columns,
        y=list(range(len(missing_sample))),
        colorscale=[[0, '#10b981'], [1, '#ef4444']],
        showscale=False,
        hovertemplate='Column: %{x}<br>Row: %{y}<br>Missing: %{z}<extra></extra>'
    ))

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(1)

    cols_with_missing = [c for c in df.columns if missing_counts[c] > 0]
    if cols_with_missing:
        texts = [f"{missing_pct[c]}%" for c in cols_with_missing]
        _add_top_band_labels_for_x(
            fig,
            texts=texts,
            x_positions=cols_with_missing,
            angle=-90 if len(cols_with_missing) > 20 else 0,
            top_pad=0.06
        )

    fig.update_layout(
        title="❓ Missing Data Patterns - Data Completeness Analysis",
        xaxis_title="Columns",
        yaxis_title="Sample Rows" if len(df) > 1000 else "Rows",
        height=650,
        xaxis_tickangle=-45,
        margin=dict(t=90, l=50, r=50, b=60)
    )
    return theme_manager.apply_theme_to_figure(fig, theme)

def create_outlier_detection_plot(df: pd.DataFrame, column: str, theme: str = "professional_dark") -> go.Figure:
    if column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{column}' not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
        return theme_manager.apply_theme_to_figure(fig, theme)

    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{column}' is not numeric", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
        return theme_manager.apply_theme_to_figure(fig, theme)
    if len(series) < 4:
        fig = go.Figure()
        fig.add_annotation(text=f"Insufficient data in column '{column}'", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
        return theme_manager.apply_theme_to_figure(fig, theme)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Box Plot', 'Histogram with Outliers', 'Q-Q Plot', 'Outlier Summary'),
        specs=[[{"type": "box"}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "table"}]],
        vertical_spacing=0.18
    )

    fig.add_trace(go.Box(y=series, name=column, boxpoints='outliers'), row=1, col=1)
    fig.add_trace(go.Histogram(x=series, name='Distribution', opacity=0.7), row=1, col=2)

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", row=1, col=2)

    if SCIPY_AVAILABLE:
        try:
            qq_data = stats.probplot(series, dist="norm")
            fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q Plot', marker=dict(size=4)), row=2, col=1)
            fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0], mode='lines', name='Reference Line', line=dict(color='red', dash='dash')), row=2, col=1)
        except Exception as e:
            fig.add_annotation(text=f"Q-Q plot unavailable: {str(e)[:50]}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=10, row=2, col=1)
    else:
        fig.add_annotation(text="Q-Q plot requires SciPy", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=12, row=2, col=1)

    iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    summary_data = [["Method", "Count", "Percentage"], ["IQR (1.5×IQR)", f"{iqr_outliers}", f"{(iqr_outliers/len(series)*100):.1f}%"]]
    if SCIPY_AVAILABLE:
        try:
            z_outliers = (np.abs(stats.zscore(series)) > 3).sum()
            summary_data.append(["Z-Score (>3)", f"{z_outliers}", f"{(z_outliers/len(series)*100):.1f}%"])
        except:
            pass
    fig.add_trace(go.Table(header=dict(values=summary_data[0]), cells=dict(values=list(zip(*summary_data[1:])))), row=2, col=2)
    fig.update_layout(title=f"🎯 Outlier Analysis - {column}", height=820, showlegend=False, margin=dict(t=90, l=50, r=50, b=50))
    return theme_manager.apply_theme_to_figure(fig, theme)

# ============================
# NEW: ECDF & QQ-PLOT (STANDALONE)
# ============================

def create_ecdf_plot(df: pd.DataFrame, column: str, theme: str = "professional_dark", ecdfnorm: Optional[str] = "percent", color: Optional[str] = None) -> go.Figure:
    """Empirical Cumulative Distribution Function using Plotly Express with validated normalization."""
    if column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{column}' not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
        return theme_manager.apply_theme_to_figure(fig, theme)
    allowed = {None, "percent", "probability"}
    norm = ecdfnorm if ecdfnorm in allowed else "percent"
    fig = px.ecdf(df.dropna(subset=[column]), x=column, color=color, ecdfnorm=norm)
    fig.update_layout(title=f"📈 ECDF - {column}", height=500)
    return theme_manager.apply_theme_to_figure(fig, theme)

def create_qq_plot(df: pd.DataFrame, column: str, theme: str = "professional_dark") -> go.Figure:
    if column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{column}' not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
        return theme_manager.apply_theme_to_figure(fig, theme)
    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series) or len(series) < 4:
        fig = go.Figure()
        fig.add_annotation(text=f"QQ-Plot requires numeric data (≥4) for '{column}'", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=14)
        return theme_manager.apply_theme_to_figure(fig, theme)
    if not SCIPY_AVAILABLE:
        fig = go.Figure()
        fig.add_annotation(text="SciPy is required for QQ-Plot (stats.probplot)", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=14)
        return theme_manager.apply_theme_to_figure(fig, theme)

    qq = stats.probplot(series, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0] * qq[0][0], mode='lines', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f"📐 QQ-Plot - {column}", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", height=520, showlegend=False)
    return theme_manager.apply_theme_to_figure(fig, theme)

# ============================
# STREAMLIT INTEGRATION (unchanged layout; stable keys & values)
# ============================

def render_automated_eda_dashboard(df: pd.DataFrame, theme: str = "professional_dark"):
    """Comprehensive automated EDA dashboard in Streamlit - fixed unique chart keys (behavior unchanged)."""
    if not STREAMLIT_AVAILABLE:
        print("⚠️ Streamlit not available - cannot render dashboard")
        return None

    st.markdown("## 🤖 Automated Exploratory Data Analysis")

    # Stable chart session prefix across reruns to preserve widget values
    if "eda_chart_session" not in st.session_state:
        st.session_state["eda_chart_session"] = f"eda_{uuid.uuid4().hex[:8]}"
    chart_session = st.session_state["eda_chart_session"]

    with st.spinner("Generating comprehensive EDA report..."):
        try:
            eda_report = generate_automated_eda_report(df)
        except Exception as e:
            st.error(f"❌ Error generating EDA report: {e}")
            return None

    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        shape = eda_report['dataset_info']['shape']
        st.metric("📏 Shape", f"{shape[0]:,} × {shape[1]}")
    with col2:
        memory_mb = eda_report['dataset_info']['memory_usage_mb']
        st.metric("💾 Memory", f"{memory_mb:.1f} MB")
    with col3:
        numeric_cols = eda_report['dataset_info']['numeric_columns']
        st.metric("🔢 Numeric Cols", f"{numeric_cols}")
    with col4:
        categorical_cols = eda_report['dataset_info']['categorical_columns']
        st.metric("📝 Categorical Cols", f"{categorical_cols}")

    if eda_report['missing_data_analysis']['total_missing'] > 0:
        st.markdown("### ❓ Missing Data Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                missing_fig = create_missing_data_heatmap(df, theme)
                st.plotly_chart(missing_fig, use_container_width=True, key=f"{chart_session}_missing_data")
            except Exception as e:
                st.error(f"❌ Visualization Error: {e}")
        with col2:
            st.markdown("**Missing by Column:**")
            missing_data = eda_report['missing_data_analysis']['missing_by_column']
            for col, count in list(missing_data.items())[:10]:
                pct = (count / len(df)) * 100
                st.write(f"• **{col}**: {count:,} ({pct:.1f}%)")

    if eda_report['correlation_analysis']['correlation_matrix'] is not None:
        st.markdown("### 🔗 Correlation Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                corr_fig = create_correlation_heatmap(df, theme)
                st.plotly_chart(corr_fig, use_container_width=True, key=f"{chart_session}_correlation")
            except Exception as e:
                st.error(f"❌ Visualization Error: {e}")
        with col2:
            strong_corrs = eda_report['correlation_analysis']['strong_correlations']
            if strong_corrs:
                st.markdown("**Strong Correlations:**")
                for corr in strong_corrs[:5]:
                    st.write(f"• **{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f}")
            else:
                st.info("No strong correlations found (|r| > 0.7)")

    st.markdown("### 📈 Distribution Analysis")
    try:
        dist_fig = create_distribution_grid(df, theme)
        st.plotly_chart(dist_fig, use_container_width=True, key=f"{chart_session}_distribution")
    except Exception as e:
        st.error(f"❌ Visualization Error: {e}")

    # Outlier Detection (final fix: initialize state, then call selectbox with key only)
    numeric_cols_full = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols_full:
        st.markdown("### 🎯 Outlier Detection")
        outlier_key = f"{chart_session}_outlier_select"
        if (outlier_key not in st.session_state) or (st.session_state[outlier_key] not in numeric_cols_full):
            st.session_state[outlier_key] = numeric_cols_full[0]
        selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols_full, key=outlier_key)
        if selected_col:
            try:
                outlier_fig = create_outlier_detection_plot(df, selected_col, theme)
                st.plotly_chart(outlier_fig, use_container_width=True, key=f"{chart_session}_outlier_{selected_col}")
            except Exception as e:
                st.error(f"❌ Visualization Error: {e}")

    st.markdown("### 💡 Analysis Recommendations")
    recommendations = eda_report['recommendations']
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

    return eda_report

# ============================
# EXPORTS
# ============================

__all__ = [
    'EnhancedThemeManager',
    'theme_manager',
    'generate_automated_eda_report',
    'create_correlation_heatmap',
    'create_distribution_grid',
    'create_missing_data_heatmap',
    'create_outlier_detection_plot',
    'create_ecdf_plot',
    'create_qq_plot',
    'render_automated_eda_dashboard'
]

print("✅ Enhanced Visualization Module v2.4.2 Loaded (ECDF validated + Stable EDA selectbox)")
print(f"   📊 SciPy Available: {SCIPY_AVAILABLE}")
print(f"   🤖 Scikit-learn Available: {SKLEARN_AVAILABLE}")
print(f"   🎨 Streamlit Available: {STREAMLIT_AVAILABLE}")
print("   🔧 FIXED: titleside error and duplicate chart IDs resolved!")
print("   ✨ NEW: Persistent outlier selectbox with session state.")
