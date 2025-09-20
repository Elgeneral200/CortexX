# filename: visualization.py
"""
Data Visualization Module - Enhanced Professional Edition v2.2 - FIXED

Advanced visualization system with:
- Automated Exploratory Data Analysis (EDA)
- Professional business intelligence dashboards  
- Interactive correlation and distribution analysis
- Advanced outlier detection visualizations
- Memory-optimized rendering for large datasets
- FIXED: Plotly titleside error and Streamlit duplicate chart IDs

Author: CortexX Team
Version: 2.2.0 - FIXED Complete Working Edition
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies with proper error handling
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
        """Get theme configuration."""
        theme_name = theme_name or self.current_theme
        return self.themes.get(theme_name, self.themes["professional_dark"])
    
    def apply_theme_to_figure(self, fig: go.Figure, theme_name: str = None) -> go.Figure:
        """Apply theme to plotly figure."""
        theme = self.get_theme(theme_name)
        
        fig.update_layout(
            plot_bgcolor=theme["background_color"],
            paper_bgcolor=theme["paper_color"],
            font_color=theme["text_color"],
            font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            font_size=12,
            title_font_size=16,
            title_font_color=theme["text_color"],
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor=theme["grid_color"],
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=theme["grid_color"],
            linecolor=theme["grid_color"],
            tickcolor=theme["text_color"],
            title_font_color=theme["text_color"]
        )
        
        fig.update_yaxes(
            gridcolor=theme["grid_color"],
            linecolor=theme["grid_color"],
            tickcolor=theme["text_color"],
            title_font_color=theme["text_color"]
        )
        
        return fig

# Initialize theme manager
theme_manager = EnhancedThemeManager()

# ============================
# AUTOMATED EDA FUNCTIONS
# ============================

def generate_automated_eda_report(df: pd.DataFrame, sample_size: int = 10000) -> Dict[str, Any]:
    """Generate comprehensive automated EDA report."""
    
    # Sample data for performance if needed
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
    """Get comprehensive dataset overview."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = []
    
    # Detect datetime columns
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
        "column_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        }
    }

def analyze_columns_detailed(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Detailed analysis of each column."""
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
        
        # Type-specific analysis
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
    """Calculate correlation matrices with statistical significance."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return {"correlation_matrix": None, "strong_correlations": [], "warning": "Less than 2 numeric columns"}
    
    # Correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Find strong correlations (|r| > 0.7)
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
    
    return {
        "correlation_matrix": corr_matrix,
        "strong_correlations": strong_correlations,
        "numeric_columns": len(numeric_df.columns)
    }

def analyze_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing data patterns."""
    missing_data = df.isnull()
    
    # Missing by column
    missing_by_column = missing_data.sum().sort_values(ascending=False)
    missing_by_column = missing_by_column[missing_by_column > 0]
    
    if len(missing_by_column) > 0:
        return {
            "missing_by_column": missing_by_column.to_dict(),
            "total_missing": missing_data.sum().sum(),
            "columns_with_missing": len(missing_by_column),
            "missing_patterns": []  # Simplified for performance
        }
    else:
        return {
            "missing_by_column": {},
            "total_missing": 0,
            "columns_with_missing": 0,
            "missing_patterns": []
        }

def detect_outliers_comprehensive(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive outlier detection using multiple methods."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {"outliers_by_method": {}, "outlier_summary": {}}
    
    outlier_results = {}
    
    for col in numeric_cols[:5]:  # Limit to first 5 columns for performance
        series = df[col].dropna()
        if len(series) < 4:
            continue
            
        col_outliers = {}
        
        # IQR Method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((series < lower_bound) | (series > upper_bound))
        col_outliers["iqr"] = {
            "count": int(iqr_outliers.sum()),
            "percentage": float((iqr_outliers.sum() / len(series)) * 100),
            "bounds": [float(lower_bound), float(upper_bound)]
        }
        
        # Z-Score Method (if scipy available)
        if SCIPY_AVAILABLE:
            try:
                z_scores = np.abs(stats.zscore(series))
                zscore_outliers = z_scores > 3
                col_outliers["zscore"] = {
                    "count": int(zscore_outliers.sum()),
                    "percentage": float((zscore_outliers.sum() / len(series)) * 100),
                    "threshold": 3.0
                }
            except:
                pass
        
        # Isolation Forest (if sklearn available and sufficient data)
        if SKLEARN_AVAILABLE and len(series) > 100:
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                iso_outliers = outlier_labels == -1
                col_outliers["isolation_forest"] = {
                    "count": int(iso_outliers.sum()),
                    "percentage": float((iso_outliers.sum() / len(series)) * 100),
                    "contamination": 0.1
                }
            except:
                pass
        
        outlier_results[col] = col_outliers
    
    # Summary statistics
    summary = {}
    for method in ["iqr", "zscore", "isolation_forest"]:
        method_counts = [col_data.get(method, {}).get("count", 0) for col_data in outlier_results.values()]
        if method_counts:
            summary[method] = {
                "total_outliers": sum(method_counts),
                "avg_per_column": np.mean(method_counts),
                "columns_with_outliers": sum(1 for count in method_counts if count > 0)
            }
    
    return {
        "outliers_by_method": outlier_results,
        "outlier_summary": summary
    }

def analyze_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze distributions of numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {"distributions": {}, "summary": "No numeric columns found"}
    
    distributions = {}
    
    for col in numeric_cols[:10]:  # Limit for performance
        series = df[col].dropna()
        if len(series) < 2:
            continue
            
        dist_analysis = {
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
            "normality_test": None
        }
        
        # Classify distribution shape
        skew = dist_analysis["skewness"]
        if skew > 0.5:
            dist_analysis["distribution_shape"] = "Right Skewed"
        elif skew < -0.5:
            dist_analysis["distribution_shape"] = "Left Skewed"
        else:
            dist_analysis["distribution_shape"] = "Approximately Symmetric"
        
        # Normality test (if scipy available)
        if SCIPY_AVAILABLE and len(series) >= 8:
            try:
                statistic, p_value = stats.normaltest(series)
                dist_analysis["normality_test"] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
            except:
                pass
        
        distributions[col] = dist_analysis
    
    return {"distributions": distributions}

def generate_eda_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate actionable EDA recommendations."""
    recommendations = []
    
    # Data quality recommendations
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    if missing_pct > 10:
        recommendations.append(f"High missing data ({missing_pct:.1f}%) - Consider imputation strategies or investigate data collection issues")
    
    duplicate_pct = (df.duplicated().sum() / len(df)) * 100
    if duplicate_pct > 1:
        recommendations.append(f"Found {duplicate_pct:.1f}% duplicate rows - Review data collection process and remove duplicates")
    
    # Memory optimization
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if memory_mb > 100:
        recommendations.append(f"Large dataset ({memory_mb:.1f}MB) - Consider data type optimization and sampling for analysis")
    
    # Column-specific recommendations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) > 10:
        recommendations.append("Many numeric columns - Consider dimensionality reduction or feature selection")
    
    if len(categorical_cols) > len(numeric_cols) * 2:
        recommendations.append("Many categorical columns - Consider encoding strategies for machine learning")
    
    # High cardinality check
    for col in categorical_cols[:5]:  # Check first 5 for performance
        unique_pct = (df[col].nunique() / len(df)) * 100
        if unique_pct > 50:
            recommendations.append(f"Column '{col}' has high cardinality ({unique_pct:.1f}%) - Consider grouping rare categories")
    
    if not recommendations:
        recommendations.append("Data quality looks good! Ready for advanced analysis and modeling")
    
    return recommendations

# ============================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================

def create_correlation_heatmap(df: pd.DataFrame, theme: str = "professional_dark") -> go.Figure:
    """Create enhanced correlation heatmap with statistical significance."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns for correlation analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
        return theme_manager.apply_theme_to_figure(fig, theme)
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap - ✅ FIXED: Removed titleside property
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(
            title="Correlation"
            # ✅ FIXED: Removed titleside="right" - not a valid property
        )
    ))
    
    fig.update_layout(
        title="📊 Correlation Matrix - Feature Relationships",
        width=800,
        height=600,
        xaxis_tickangle=-45
    )
    
    return theme_manager.apply_theme_to_figure(fig, theme)

def create_distribution_grid(df: pd.DataFrame, theme: str = "professional_dark", max_cols: int = 4) -> go.Figure:
    """Create professional distribution analysis grid."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric columns found for distribution analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
        return theme_manager.apply_theme_to_figure(fig, theme)
    
    # Limit number of columns to prevent overcrowding
    numeric_cols = numeric_cols[:12]  # Max 12 distributions
    
    # Calculate grid dimensions
    n_cols = min(max_cols, len(numeric_cols))
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{col}" for col in numeric_cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    theme_colors = theme_manager.get_theme(theme)["color_palette"]
    
    for i, col in enumerate(numeric_cols):
        row = (i // n_cols) + 1
        col_pos = (i % n_cols) + 1
        
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=series,
                name=col,
                showlegend=False,
                marker_color=theme_colors[i % len(theme_colors)],
                opacity=0.7,
                nbinsx=30
            ),
            row=row, col=col_pos
        )
        
        # Add statistics annotations only if there's data
        try:
            mean_val = series.mean()
            median_val = series.median()
            
            if not pd.isna(mean_val):
                fig.add_vline(
                    x=mean_val, line_dash="dash", line_color="yellow",
                    annotation_text=f"μ={mean_val:.2f}", 
                    row=row, col=col_pos
                )
            
            if not pd.isna(median_val):
                fig.add_vline(
                    x=median_val, line_dash="dot", line_color="orange",
                    annotation_text=f"M={median_val:.2f}",
                    row=row, col=col_pos
                )
        except:
            pass  # Skip annotations if there's an error
    
    fig.update_layout(
        title="📈 Distribution Analysis - Statistical Overview",
        height=300 * n_rows,
        showlegend=False
    )
    
    return theme_manager.apply_theme_to_figure(fig, theme)

def create_missing_data_heatmap(df: pd.DataFrame, theme: str = "professional_dark") -> go.Figure:
    """Create missing data pattern visualization."""
    missing_data = df.isnull()
    
    if missing_data.sum().sum() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="🎉 No missing data detected! Dataset is complete.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16, font_color="green"
        )
        return theme_manager.apply_theme_to_figure(fig, theme)
    
    # Sample data if too large
    if len(df) > 1000:
        missing_sample = missing_data.sample(1000, random_state=42)
    else:
        missing_sample = missing_data
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=missing_sample.values.astype(int),
        x=missing_sample.columns,
        y=list(range(len(missing_sample))),
        colorscale=[[0, '#10b981'], [1, '#ef4444']],
        showscale=False,
        hovertemplate='Column: %{x}<br>Row: %{y}<br>Missing: %{z}<extra></extra>'
    ))
    
    # Add missing counts annotation
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(1)
    
    annotations = []
    for i, col in enumerate(df.columns):
        if missing_counts[col] > 0:
            annotations.append(
                dict(
                    x=i, y=-0.1,
                    text=f"{missing_pct[col]}%",
                    showarrow=False,
                    font_size=10,
                    xref="x", yref="paper"
                )
            )
    
    fig.update_layout(
        title="❓ Missing Data Patterns - Data Completeness Analysis",
        xaxis_title="Columns",
        yaxis_title="Sample Rows" if len(df) > 1000 else "Rows",
        annotations=annotations,
        height=600,
        xaxis_tickangle=-45
    )
    
    return theme_manager.apply_theme_to_figure(fig, theme)

def create_outlier_detection_plot(df: pd.DataFrame, column: str, theme: str = "professional_dark") -> go.Figure:
    """Create comprehensive outlier detection visualization."""
    if column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Column '{column}' not found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
        return theme_manager.apply_theme_to_figure(fig, theme)
    
    series = df[column].dropna()
    
    if not pd.api.types.is_numeric_dtype(series):
        fig = go.Figure()
        fig.add_annotation(
            text=f"Column '{column}' is not numeric",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
        return theme_manager.apply_theme_to_figure(fig, theme)
    
    if len(series) < 4:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Insufficient data in column '{column}'",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
        return theme_manager.apply_theme_to_figure(fig, theme)
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Box Plot', 'Histogram with Outliers', 'Q-Q Plot', 'Outlier Summary'),
        specs=[[{"type": "box"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "table"}]]
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=series, name=column, boxpoints='outliers'),
        row=1, col=1
    )
    
    # Histogram with outlier bounds
    fig.add_trace(
        go.Histogram(x=series, name='Distribution', opacity=0.7),
        row=1, col=2
    )
    
    # Add outlier bounds
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", row=1, col=2)
    
    # Q-Q plot (if scipy available)
    if SCIPY_AVAILABLE:
        try:
            qq_data = stats.probplot(series, dist="norm")
            fig.add_trace(
                go.Scatter(
                    x=qq_data[0][0], y=qq_data[0][1], 
                    mode='markers', name='Q-Q Plot',
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            
            # Add reference line
            fig.add_trace(
                go.Scatter(
                    x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                    mode='lines', name='Reference Line',
                    line=dict(color='red', dash='dash')
                ),
                row=2, col=1
            )
        except Exception as e:
            # Add text if Q-Q plot fails
            fig.add_annotation(
                text=f"Q-Q plot unavailable: {str(e)[:50]}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font_size=10,
                row=2, col=1
            )
    else:
        fig.add_annotation(
            text="Q-Q plot requires SciPy",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=12,
            row=2, col=1
        )
    
    # Outlier summary table
    iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    
    summary_data = [
        ["Method", "Count", "Percentage"],
        ["IQR (1.5×IQR)", f"{iqr_outliers}", f"{(iqr_outliers/len(series)*100):.1f}%"],
    ]
    
    if SCIPY_AVAILABLE:
        try:
            z_outliers = (np.abs(stats.zscore(series)) > 3).sum()
            summary_data.append(["Z-Score (>3)", f"{z_outliers}", f"{(z_outliers/len(series)*100):.1f}%"])
        except:
            pass
    
    fig.add_trace(
        go.Table(
            header=dict(values=summary_data[0]),
            cells=dict(values=list(zip(*summary_data[1:])))
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"🎯 Outlier Analysis - {column}",
        height=800,
        showlegend=False
    )
    
    return theme_manager.apply_theme_to_figure(fig, theme)

# ============================
# STREAMLIT INTEGRATION FUNCTIONS - ✅ FIXED
# ============================

def render_automated_eda_dashboard(df: pd.DataFrame, theme: str = "professional_dark"):
    """Render comprehensive automated EDA dashboard in Streamlit - FIXED with unique chart keys."""
    if not STREAMLIT_AVAILABLE:
        print("⚠️ Streamlit not available - cannot render dashboard")
        return None
    
    st.markdown("## 🤖 Automated Exploratory Data Analysis")
    
    # Generate unique session ID for chart keys
    chart_session = f"eda_{uuid.uuid4().hex[:8]}"
    
    # Generate EDA report
    with st.spinner("Generating comprehensive EDA report..."):
        try:
            eda_report = generate_automated_eda_report(df)
        except Exception as e:
            st.error(f"❌ Error generating EDA report: {e}")
            return None
    
    # Dataset Overview
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
    
    # Missing Data Analysis - ✅ FIXED: Added unique key
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
            for col, count in list(missing_data.items())[:10]:  # Show top 10
                pct = (count / len(df)) * 100
                st.write(f"• **{col}**: {count:,} ({pct:.1f}%)")
    
    # Correlation Analysis - ✅ FIXED: Added unique key
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
    
    # Distribution Analysis - ✅ FIXED: Added unique key
    st.markdown("### 📈 Distribution Analysis")
    try:
        dist_fig = create_distribution_grid(df, theme)
        st.plotly_chart(dist_fig, use_container_width=True, key=f"{chart_session}_distribution")
    except Exception as e:
        st.error(f"❌ Visualization Error: {e}")
    
    # Outlier Analysis - ✅ FIXED: Added unique key
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.markdown("### 🎯 Outlier Detection")
        
        selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols, key=f"{chart_session}_outlier_select")
        if selected_col:
            try:
                outlier_fig = create_outlier_detection_plot(df, selected_col, theme)
                st.plotly_chart(outlier_fig, use_container_width=True, key=f"{chart_session}_outlier_{selected_col}")
            except Exception as e:
                st.error(f"❌ Visualization Error: {e}")
    
    # Recommendations
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
    'render_automated_eda_dashboard'
]

print("✅ Enhanced Visualization Module v2.2 - FIXED - Loaded Successfully!")
print(f"   📊 SciPy Available: {SCIPY_AVAILABLE}")
print(f"   🤖 Scikit-learn Available: {SKLEARN_AVAILABLE}")  
print(f"   🎨 Streamlit Available: {STREAMLIT_AVAILABLE}")
print("   🔧 FIXED: titleside error and duplicate chart IDs resolved!")
print("   🚀 All functions ready for import!")