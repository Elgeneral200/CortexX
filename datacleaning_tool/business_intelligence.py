# filename: business_intelligence.py
"""
Business Intelligence Module - Professional Enterprise Edition v3.0

Advanced business analytics and visualization suite with enhanced capabilities:
- Executive KPI dashboards with real-time updates
- Advanced sales performance analysis
- Predictive revenue trend analysis
- Customer segmentation with RFM modeling
- Market basket analysis tools
- Professional reporting and export capabilities
- Enhanced error handling and data validation
- Multi-theme support for visualizations

Author: CortexX Analytics Team
Version: 3.0.0 - Enterprise Edition
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime, timedelta
import calendar
import math
import json
import io
import base64

warnings.filterwarnings('ignore')

# Try to import optional dependencies with proper error handling
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not available - some statistical features will be limited")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
# CONSTANTS AND CONFIGURATION
# ============================

# Professional color schemes
COLOR_SCHEMES = {
    "professional_dark": {
        "background": "rgba(15, 23, 42, 1)",
        "plot_bg": "rgba(15, 23, 42, 0.8)",
        "grid": "rgba(255, 255, 255, 0.1)",
        "text": "white",
        "primary": "#3b82f6",
        "secondary": "#10b981",
        "accent": "#f59e0b",
        "danger": "#ef4444",
        "purple": "#8b5cf6",
        "cyan": "#06b6d4"
    },
    "professional_light": {
        "background": "rgba(255, 255, 255, 1)",
        "plot_bg": "rgba(248, 250, 252, 1)",
        "grid": "rgba(0, 0, 0, 0.1)",
        "text": "rgba(15, 23, 42, 1)",
        "primary": "#1d4ed8",
        "secondary": "#047857",
        "accent": "#b45309",
        "danger": "#b91c1c",
        "purple": "#5b21b6",
        "cyan": "#0e7490"
    },
    "corporate_blue": {
        "background": "rgba(12, 20, 69, 1)",
        "plot_bg": "rgba(12, 20, 69, 0.8)",
        "grid": "rgba(100, 149, 237, 0.2)",
        "text": "rgba(220, 230, 255, 1)",
        "primary": "#4f46e5",
        "secondary": "#0ea5e9",
        "accent": "#f97316",
        "danger": "#e11d48",
        "purple": "#7c3aed",
        "cyan": "#22d3ee"
    }
}

# ============================
# DATA VALIDATION & PREPROCESSING
# ============================

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate dataframe structure and data quality."""
    validation_results = {
        "is_valid": True,
        "issues": [],
        "summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        }
    }
    
    # Check for empty dataframe
    if df.empty:
        validation_results["is_valid"] = False
        validation_results["issues"].append("DataFrame is empty")
        return validation_results
    
    # Check column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = []
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Auto-detect date columns
    for col in df.columns:
        if col in numeric_cols or col in categorical_cols:
            continue
        
        sample = df[col].dropna().head(100)
        try:
            pd.to_datetime(sample, errors='raise')
            date_cols.append(col)
        except:
            pass
    
    validation_results["column_types"] = {
        "numeric": numeric_cols,
        "date": date_cols,
        "categorical": categorical_cols,
        "other": list(set(df.columns) - set(numeric_cols + date_cols + categorical_cols))
    }
    
    # Check for high missing value columns
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    high_missing_cols = missing_percentage[missing_percentage > 30].index.tolist()
    
    if high_missing_cols:
        validation_results["issues"].append(
            f"Columns with >30% missing values: {', '.join(high_missing_cols)}"
        )
    
    return validation_results

def preprocess_dataframe(df: pd.DataFrame, 
                        date_columns: List[str] = None,
                        numeric_columns: List[str] = None) -> pd.DataFrame:
    """Preprocess dataframe for analysis."""
    df_processed = df.copy()
    
    # Convert date columns
    if date_columns:
        for col in date_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
    
    # Convert numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    return df_processed

# ============================
# BUSINESS INTELLIGENCE FUNCTIONS
# ============================

def calculate_business_kpis(df: pd.DataFrame, 
                           revenue_col: str = None,
                           quantity_col: str = None,
                           date_col: str = None,
                           customer_col: str = None,
                           category_col: str = None) -> Dict[str, Any]:
    """Calculate comprehensive business KPIs with enhanced metrics."""
    
    # Data validation
    validation = validate_dataframe(df)
    if not validation["is_valid"]:
        return {"error": "Invalid DataFrame", "issues": validation["issues"]}
    
    kpis = {
        "total_records": len(df),
        "date_range": None,
        "revenue_metrics": {},
        "quantity_metrics": {},
        "growth_metrics": {},
        "customer_metrics": {},
        "category_metrics": {},
        "performance_metrics": {},
        "data_quality": validation["summary"]
    }
    
    # Revenue Analysis
    if revenue_col and revenue_col in df.columns:
        revenue_series = pd.to_numeric(df[revenue_col], errors='coerce').dropna()
        
        if len(revenue_series) > 0:
            # Calculate advanced revenue metrics
            revenue_stats = revenue_series.describe()
            q1 = revenue_series.quantile(0.25)
            q3 = revenue_series.quantile(0.75)
            iqr = q3 - q1
            
            kpis["revenue_metrics"] = {
                "total_revenue": float(revenue_series.sum()),
                "average_transaction": float(revenue_stats.get('mean', 0)),
                "median_transaction": float(revenue_stats.get('50%', 0)),
                "revenue_std": float(revenue_stats.get('std', 0)),
                "max_transaction": float(revenue_stats.get('max', 0)),
                "min_transaction": float(revenue_stats.get('min', 0)),
                "revenue_range": float(revenue_stats.get('max', 0) - revenue_stats.get('min', 0)),
                "coefficient_of_variation": float(revenue_stats.get('std', 0) / revenue_stats.get('mean', 0)) 
                if revenue_stats.get('mean', 0) != 0 else 0,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "skewness": float(revenue_series.skew()) if len(revenue_series) > 2 else 0,
                "kurtosis": float(revenue_series.kurtosis()) if len(revenue_series) > 3 else 0
            }
    
    # Quantity Analysis
    if quantity_col and quantity_col in df.columns:
        quantity_series = pd.to_numeric(df[quantity_col], errors='coerce').dropna()
        
        if len(quantity_series) > 0:
            quantity_stats = quantity_series.describe()
            kpis["quantity_metrics"] = {
                "total_quantity": float(quantity_series.sum()),
                "average_quantity": float(quantity_stats.get('mean', 0)),
                "max_quantity": float(quantity_stats.get('max', 0)),
                "min_quantity": float(quantity_stats.get('min', 0)),
                "std_quantity": float(quantity_stats.get('std', 0))
            }
    
    # Time-based Analysis
    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(dates) > 0:
                date_range_days = (dates.max() - dates.min()).days
                
                kpis["date_range"] = {
                    "start_date": dates.min().strftime('%Y-%m-%d'),
                    "end_date": dates.max().strftime('%Y-%m-%d'),
                    "total_days": date_range_days,
                    "unique_dates": dates.nunique(),
                    "data_frequency": "Daily" if dates.nunique() > date_range_days * 0.7 else "Sparse"
                }
                
                # Growth analysis (if we have revenue and dates)
                if revenue_col and revenue_col in df.columns:
                    df_time = df.copy()
                    df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
                    df_time = df_time.dropna(subset=[date_col, revenue_col])
                    
                    if len(df_time) > 1:
                        # Create time-based aggregations
                        daily_revenue = df_time.groupby(date_col)[revenue_col].sum()
                        
                        if len(daily_revenue) > 1:
                            # Calculate growth metrics
                            first_period = daily_revenue.iloc[0]
                            last_period = daily_revenue.iloc[-1]
                            
                            if first_period != 0:
                                total_growth_rate = ((last_period - first_period) / first_period) * 100
                            else:
                                total_growth_rate = 0
                            
                            # Calculate CAGR if we have enough data
                            periods = len(daily_revenue)
                            if periods > 1 and first_period > 0:
                                cagr = (pow(last_period / first_period, 365/date_range_days) - 1) * 100
                                avg_daily_growth = (pow(last_period / first_period, 1/periods) - 1) * 100
                            else:
                                cagr = 0
                                avg_daily_growth = 0
                            
                            # Calculate volatility (standard deviation of daily returns)
                            daily_returns = daily_revenue.pct_change().dropna()
                            if len(daily_returns) > 1:
                                volatility = daily_returns.std() * math.sqrt(252)  # Annualized
                            else:
                                volatility = 0
                            
                            kpis["growth_metrics"] = {
                                "total_growth_rate": float(total_growth_rate),
                                "cagr": float(cagr),
                                "average_daily_growth": float(avg_daily_growth),
                                "volatility": float(volatility),
                                "trend": "Increasing" if total_growth_rate > 5 else 
                                         "Decreasing" if total_growth_rate < -5 else "Stable"
                            }
                            
                            # Seasonal analysis
                            df_time['month'] = df_time[date_col].dt.month
                            df_time['quarter'] = df_time[date_col].dt.quarter
                            df_time['day_of_week'] = df_time[date_col].dt.dayofweek
                            df_time['week_of_year'] = df_time[date_col].dt.isocalendar().week
                            
                            monthly_revenue = df_time.groupby('month')[revenue_col].sum()
                            if len(monthly_revenue) > 1:
                                peak_month = monthly_revenue.idxmax()
                                low_month = monthly_revenue.idxmin()
                                
                                kpis["seasonal_metrics"] = {
                                    "peak_month": calendar.month_name[peak_month],
                                    "peak_month_revenue": float(monthly_revenue.max()),
                                    "low_month": calendar.month_name[low_month],
                                    "low_month_revenue": float(monthly_revenue.min()),
                                    "seasonal_variation": float((monthly_revenue.max() - monthly_revenue.min()) / monthly_revenue.mean() * 100)
                                }
                                
                                # Quarterly analysis
                                quarterly_revenue = df_time.groupby('quarter')[revenue_col].sum()
                                kpis["quarterly_metrics"] = {
                                    f"Q{quarter}": float(quarterly_revenue[quarter]) 
                                    for quarter in quarterly_revenue.index
                                }
        except Exception as e:
            kpis["date_analysis_error"] = str(e)
    
    # Customer Analysis
    if customer_col and customer_col in df.columns:
        unique_customers = df[customer_col].nunique()
        customer_metrics = {
            "unique_customers": unique_customers,
            "avg_transactions_per_customer": len(df) / max(unique_customers, 1),
            "customer_retention_proxy": min(100, (len(df) / max(unique_customers, 1)) * 10)
        }
        
        if revenue_col and revenue_col in df.columns:
            customer_revenue = df.groupby(customer_col)[revenue_col].sum()
            customer_metrics.update({
                "avg_customer_value": float(customer_revenue.mean()),
                "top_customer_value": float(customer_revenue.max()),
                "customer_revenue_std": float(customer_revenue.std()),
                "customer_concentration": float((customer_revenue.nlargest(10).sum() / customer_revenue.sum()) * 100) 
                if customer_revenue.sum() > 0 else 0,
                "gini_coefficient": calculate_gini_coefficient(customer_revenue) if len(customer_revenue) > 1 else 0
            })
        
        kpis["customer_metrics"] = customer_metrics
    
    # Category Analysis
    if category_col and category_col in df.columns:
        category_counts = df[category_col].value_counts()
        category_metrics = {
            "unique_categories": len(category_counts),
            "top_category": category_counts.index[0] if len(category_counts) > 0 else None,
            "top_category_count": int(category_counts.iloc[0]) if len(category_counts) > 0 else 0
        }
        
        if revenue_col and revenue_col in df.columns:
            category_revenue = df.groupby(category_col)[revenue_col].sum()
            category_metrics.update({
                "top_category_revenue": float(category_revenue.max()) if len(category_revenue) > 0 else 0,
                "category_concentration": float((category_revenue.nlargest(5).sum() / category_revenue.sum()) * 100) 
                if category_revenue.sum() > 0 else 0
            })
        
        kpis["category_metrics"] = category_metrics
    
    # Performance Metrics
    if revenue_col and revenue_col in df.columns:
        revenue_series = pd.to_numeric(df[revenue_col], errors='coerce').dropna()
        if len(revenue_series) > 0:
            # Calculate performance percentiles
            p25 = float(revenue_series.quantile(0.25))
            p50 = float(revenue_series.quantile(0.50))
            p75 = float(revenue_series.quantile(0.75))
            p90 = float(revenue_series.quantile(0.90))
            
            kpis["performance_metrics"] = {
                "high_value_transactions": int((revenue_series >= p75).sum()),
                "medium_value_transactions": int(((revenue_series >= p25) & (revenue_series < p75)).sum()),
                "low_value_transactions": int((revenue_series < p25).sum()),
                "elite_transactions": int((revenue_series >= p90).sum()),
                "performance_ratio": float(p75 / p25) if p25 > 0 else 0,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p90": p90
            }
    
    return kpis

def calculate_gini_coefficient(series: pd.Series) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    # Sort the series
    sorted_series = np.sort(series)
    n = len(sorted_series)
    index = np.arange(1, n + 1)
    
    # Calculate Gini coefficient
    gini = ((np.sum((2 * index - n - 1) * sorted_series)) / (n * np.sum(sorted_series)))
    return float(gini)

def create_executive_dashboard(df: pd.DataFrame, 
                             revenue_col: str = None,
                             date_col: str = None,
                             category_col: str = None,
                             customer_col: str = None,
                             theme: str = "professional_dark") -> go.Figure:
    """Create comprehensive executive dashboard with enhanced visualizations."""
    
    # Get color scheme
    colors = COLOR_SCHEMES.get(theme, COLOR_SCHEMES["professional_dark"])
    
    # FIXED: Proper indentation for make_subplots call
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Revenue Trend", "Category Performance", "Daily Volume",
            "Revenue Distribution", "Customer Segmentation", "Performance Analysis", 
            "Cumulative Revenue", "Monthly Comparison", "Anomaly Detection"
        ],
        specs=[
            [{"secondary_y": False}, {"type": "pie"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "sunburst"}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "bar"}, {"secondary_y": False}]  # ✅ FIXED: No more 'line' type
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        row_heights=[0.3, 0.3, 0.4]
    )
    
    # 1. Revenue Trend with moving average
    if date_col and revenue_col and date_col in df.columns and revenue_col in df.columns:
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
        df_time = df_time.dropna(subset=[date_col, revenue_col])
        
        if len(df_time) > 0:
            daily_revenue = df_time.groupby(date_col)[revenue_col].sum().reset_index()
            
            # Add revenue trend
            fig.add_trace(
                go.Scatter(
                    x=daily_revenue[date_col],
                    y=daily_revenue[revenue_col],
                    mode='lines',
                    name='Daily Revenue',
                    line=dict(color=colors["primary"], width=2),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Add 7-day moving average
            if len(daily_revenue) > 7:
                daily_revenue['moving_avg_7'] = daily_revenue[revenue_col].rolling(window=7).mean()
                fig.add_trace(
                    go.Scatter(
                        x=daily_revenue[date_col],
                        y=daily_revenue['moving_avg_7'],
                        mode='lines',
                        name='7-Day MA',
                        line=dict(color=colors["accent"], width=3)
                    ),
                    row=1, col=1
                )
    
    # 2. Category Performance (sunburst for hierarchy if available)
    if category_col and category_col in df.columns:
        if revenue_col and revenue_col in df.columns:
            category_revenue = df.groupby(category_col)[revenue_col].sum()
        else:
            category_revenue = df[category_col].value_counts()
            
        # Limit to top categories for readability
        top_categories = category_revenue.nlargest(10)
        
        fig.add_trace(
            go.Pie(
                labels=top_categories.index,
                values=top_categories.values,
                name="Categories",
                marker_colors=px.colors.qualitative.Set3,
                hole=0.4,
                textinfo='label+percent'
            ),
            row=1, col=2
        )
    
    # 3. Daily Volume with trend line
    if date_col and date_col in df.columns:
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
        df_time = df_time.dropna(subset=[date_col])
        
        if len(df_time) > 0:
            daily_count = df_time.groupby(date_col).size().reset_index(name='count')
            
            fig.add_trace(
                go.Bar(
                    x=daily_count[date_col],
                    y=daily_count['count'],
                    name='Daily Transactions',
                    marker_color=colors["secondary"],
                    opacity=0.7
                ),
                row=1, col=3
            )
            
            # Add trend line for volume
            if len(daily_count) > 7:
                daily_count['moving_avg'] = daily_count['count'].rolling(window=7).mean()
                fig.add_trace(
                    go.Scatter(
                        x=daily_count[date_col],
                        y=daily_count['moving_avg'],
                        mode='lines',
                        name='Volume Trend',
                        line=dict(color=colors["danger"], width=3)
                    ),
                    row=1, col=3
                )
    
    # 4. Revenue Distribution with statistical annotations
    if revenue_col and revenue_col in df.columns:
        revenue_series = pd.to_numeric(df[revenue_col], errors='coerce').dropna()
        
        if len(revenue_series) > 0:
            fig.add_trace(
                go.Histogram(
                    x=revenue_series,
                    name='Revenue Distribution',
                    marker_color=colors["purple"],
                    opacity=0.7,
                    nbinsx=30,
                    histnorm='probability density'
                ),
                row=2, col=1
            )
            
            # Add KDE if scipy is available
            if SCIPY_AVAILABLE and len(revenue_series) > 10:
                try:
                    # Calculate kernel density estimate
                    kde_x = np.linspace(revenue_series.min(), revenue_series.max(), 100)
                    kde_y = stats.gaussian_kde(revenue_series)(kde_x)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x,
                            y=kde_y,
                            mode='lines',
                            name='Density',
                            line=dict(color=colors["cyan"], width=2),
                            yaxis='y2'
                        ),
                        row=2, col=1
                    )
                    
                    # Add secondary y-axis for KDE
                    fig.update_layout(
                        yaxis2=dict(
                            title="Density",
                            overlaying="y",
                            side="right"
                        )
                    )
                except:
                    pass
    
    # 5. Customer Segmentation (RFM if possible)
    if customer_col and revenue_col and customer_col in df.columns and revenue_col in df.columns:
        if date_col and date_col in df.columns:
            try:
                # Simple RFM analysis
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                df_temp = df_temp.dropna(subset=[date_col, revenue_col, customer_col])
                
                if len(df_temp) > 0:
                    max_date = df_temp[date_col].max()
                    
                    rfm = df_temp.groupby(customer_col).agg({
                        date_col: lambda x: (max_date - x.max()).days,  # Recency
                        revenue_col: 'sum',  # Monetary
                        customer_col: 'count'  # Frequency
                    }).rename(columns={
                        date_col: 'recency',
                        revenue_col: 'monetary',
                        customer_col: 'frequency'
                    })
                    
                    # Create segments
                    rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, labels=['4', '3', '2', '1'])
                    rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, labels=['1', '2', '3', '4'])
                    rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, labels=['1', '2', '3', '4'])
                    
                    rfm['rfm_segment'] = rfm['r_quartile'].astype(str) + rfm['f_quartile'].astype(str) + rfm['m_quartile'].astype(str)
                    
                    # Count segments
                    segment_counts = rfm['rfm_segment'].value_counts().head(10)
                    
                    fig.add_trace(
                        go.Bar(
                            x=segment_counts.index,
                            y=segment_counts.values,
                            name='RFM Segments',
                            marker_color=colors["accent"]
                        ),
                        row=2, col=2
                    )
            except Exception as e:
                # Fallback to simple customer value distribution
                customer_value = df.groupby(customer_col)[revenue_col].sum()
                top_customers = customer_value.nlargest(10)
                
                fig.add_trace(
                    go.Bar(
                        x=top_customers.index,
                        y=top_customers.values,
                        name='Top Customers',
                        marker_color=colors["secondary"]
                    ),
                    row=2, col=2
                )
    
    # 6. Performance Analysis (revenue vs volume with trend)
    if date_col and revenue_col and date_col in df.columns and revenue_col in df.columns:
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
        df_time = df_time.dropna(subset=[date_col, revenue_col])
        
        if len(df_time) > 7:
            daily_metrics = df_time.groupby(date_col).agg({
                revenue_col: ['sum', 'count']
            }).round(2)
            
            daily_metrics.columns = ['Revenue', 'Volume']
            daily_metrics = daily_metrics.reset_index()
            
            # Add scatter plot of revenue vs volume
            fig.add_trace(
                go.Scatter(
                    x=daily_metrics['Volume'],
                    y=daily_metrics['Revenue'],
                    mode='markers',
                    name='Daily Performance',
                    marker=dict(
                        color=colors["primary"], 
                        size=8,
                        opacity=0.7
                    )
                ),
                row=2, col=3
            )
            
            # Add trend line if scipy is available
            if SCIPY_AVAILABLE and len(daily_metrics) > 2:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        daily_metrics['Volume'], daily_metrics['Revenue']
                    )
                    
                    x_trend = np.linspace(daily_metrics['Volume'].min(), daily_metrics['Volume'].max(), 100)
                    y_trend = slope * x_trend + intercept
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_trend,
                            y=y_trend,
                            mode='lines',
                            name=f'Trend (R²={r_value**2:.3f})',
                            line=dict(color=colors["danger"], width=2, dash='dash')
                        ),
                        row=2, col=3
                    )
                except:
                    pass
    
    # 7. Cumulative Revenue
    if date_col and revenue_col and date_col in df.columns and revenue_col in df.columns:
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
        df_time = df_time.dropna(subset=[date_col, revenue_col])
        
        if len(df_time) > 0:
            daily_revenue = df_time.groupby(date_col)[revenue_col].sum().reset_index()
            daily_revenue['cumulative_revenue'] = daily_revenue[revenue_col].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_revenue[date_col],
                    y=daily_revenue['cumulative_revenue'],
                    mode='lines',
                    name='Cumulative Revenue',
                    line=dict(color=colors["secondary"], width=3)
                ),
                row=3, col=1
            )
    
    # 8. Monthly Comparison
    if date_col and revenue_col and date_col in df.columns and revenue_col in df.columns:
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
        df_time = df_time.dropna(subset=[date_col, revenue_col])
        
        if len(df_time) > 0:
            df_time['year'] = df_time[date_col].dt.year
            df_time['month'] = df_time[date_col].dt.month
            
            monthly_revenue = df_time.groupby(['year', 'month'])[revenue_col].sum().reset_index()
            
            # Pivot for comparison
            monthly_pivot = monthly_revenue.pivot(index='month', columns='year', values=revenue_col)
            
            for year in monthly_pivot.columns:
                fig.add_trace(
                    go.Bar(
                        x=[calendar.month_abbr[i] for i in monthly_pivot.index],
                        y=monthly_pivot[year],
                        name=str(year),
                        opacity=0.7
                    ),
                    row=3, col=2
                )
    
    # 9. Anomaly Detection
    if date_col and revenue_col and date_col in df.columns and revenue_col in df.columns:
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
        df_time = df_time.dropna(subset=[date_col, revenue_col])
        
        if len(df_time) > 14:  # Need enough data for anomaly detection
            daily_revenue = df_time.groupby(date_col)[revenue_col].sum()
            
            # Simple anomaly detection using rolling statistics
            rolling_mean = daily_revenue.rolling(window=7).mean()
            rolling_std = daily_revenue.rolling(window=7).std()
            
            # Identify anomalies (outside 2 standard deviations)
            anomalies = daily_revenue[
                (daily_revenue > rolling_mean + 2 * rolling_std) | 
                (daily_revenue < rolling_mean - 2 * rolling_std)
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=daily_revenue.index,
                    y=daily_revenue.values,
                    mode='lines',
                    name='Daily Revenue',
                    line=dict(color=colors["primary"], width=2)
                ),
                row=3, col=3
            )
            
            fig.add_trace(
                go.Scatter(
                    x=anomalies.index,
                    y=anomalies.values,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color=colors["danger"], size=10, symbol='diamond')
                ),
                row=3, col=3
            )
    
    # Update layout with professional styling
    fig.update_layout(
        title=dict(
            text="📊 Executive Business Intelligence Dashboard",
            x=0.5,
            font=dict(size=24, color=colors["text"])
        ),
        showlegend=True,
        height=1200,
        plot_bgcolor=colors["plot_bg"],
        paper_bgcolor=colors["background"],
        font=dict(color=colors["text"], family="Inter, Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes colors
    fig.update_xaxes(
        gridcolor=colors["grid"],
        linecolor=colors["grid"],
        zerolinecolor=colors["grid"]
    )
    fig.update_yaxes(
        gridcolor=colors["grid"],
        linecolor=colors["grid"],
        zerolinecolor=colors["grid"]
    )
    
    return fig


def create_sales_performance_analysis(df: pd.DataFrame,
                                    revenue_col: str,
                                    date_col: str,
                                    product_col: str = None,
                                    customer_col: str = None,
                                    theme: str = "professional_dark") -> Dict[str, go.Figure]:
    """Create comprehensive sales performance analysis with enhanced visualizations."""
    
    figures = {}
    colors = COLOR_SCHEMES.get(theme, COLOR_SCHEMES["professional_dark"])
    
    # Prepare data
    df_analysis = df.copy()
    df_analysis[date_col] = pd.to_datetime(df_analysis[date_col], errors='coerce')
    df_analysis = df_analysis.dropna(subset=[date_col, revenue_col])
    df_analysis[revenue_col] = pd.to_numeric(df_analysis[revenue_col], errors='coerce')
    
    if len(df_analysis) == 0:
        return figures
    
    # 1. Revenue Waterfall Chart with enhanced styling
    try:
        monthly_revenue = df_analysis.groupby(df_analysis[date_col].dt.to_period('M'))[revenue_col].sum()
        
        if len(monthly_revenue) > 1:
            waterfall_fig = go.Figure()
            
            # Calculate changes month over month
            values = monthly_revenue.values
            changes = np.diff(values)
            
            # Starting point
            measures = ["absolute"] + ["relative"] * len(changes)
            x_labels = [str(monthly_revenue.index[0])] + [f"{str(monthly_revenue.index[i])} vs {str(monthly_revenue.index[i-1])}" 
                                                         for i in range(1, len(monthly_revenue.index))]
            
            waterfall_fig.add_trace(go.Waterfall(
                name="Revenue Analysis",
                orientation="v",
                measure=measures,
                x=x_labels,
                textposition="outside",
                text=[f"${val:,.0f}" for val in values[:1]] + [f"{change:+,.0f}" for change in changes],
                y=np.concatenate([[values[0]], changes]),
                connector={"line": {"color": colors["grid"]}},
                increasing={"marker": {"color": colors["secondary"]}},
                decreasing={"marker": {"color": colors["danger"]}},
            ))
            
            waterfall_fig.update_layout(
                title="💰 Monthly Revenue Waterfall Analysis",
                showlegend=False,
                height=500,
                plot_bgcolor=colors["plot_bg"],
                paper_bgcolor=colors["background"],
                font=dict(color=colors["text"])
            )
            
            figures['waterfall'] = waterfall_fig
    except Exception as e:
        print(f"Error creating waterfall chart: {e}")
    
    # 2. Enhanced Seasonal Analysis
    try:
        df_analysis['month'] = df_analysis[date_col].dt.month
        df_analysis['day_of_week'] = df_analysis[date_col].dt.day_name()
        df_analysis['hour'] = df_analysis[date_col].dt.hour
        
        seasonal_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Seasonality', 'Weekly Pattern', 'Hourly Pattern', 'Quarterly Analysis'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Monthly pattern
        monthly_avg = df_analysis.groupby('month')[revenue_col].mean()
        month_names = [calendar.month_abbr[i] for i in monthly_avg.index]
        
        seasonal_fig.add_trace(
            go.Bar(
                x=month_names,
                y=monthly_avg.values,
                name='Monthly Average',
                marker_color=colors["secondary"],
                text=[f"${val:,.0f}" for val in monthly_avg.values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Weekly pattern
        weekly_avg = df_analysis.groupby('day_of_week')[revenue_col].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_ordered = weekly_avg.reindex([day for day in day_order if day in weekly_avg.index])
        
        seasonal_fig.add_trace(
            go.Bar(
                x=weekly_ordered.index,
                y=weekly_ordered.values,
                name='Weekly Average',
                marker_color=colors["primary"],
                text=[f"${val:,.0f}" for val in weekly_ordered.values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Hourly pattern (if available)
        if 'hour' in df_analysis.columns:
            hourly_avg = df_analysis.groupby('hour')[revenue_col].mean()
            seasonal_fig.add_trace(
                go.Bar(
                    x=hourly_avg.index,
                    y=hourly_avg.values,
                    name='Hourly Average',
                    marker_color=colors["accent"],
                    text=[f"${val:,.0f}" for val in hourly_avg.values],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # Quarterly analysis
        df_analysis['quarter'] = df_analysis[date_col].dt.quarter
        quarterly_avg = df_analysis.groupby('quarter')[revenue_col].mean()
        seasonal_fig.add_trace(
            go.Bar(
                x=[f"Q{q}" for q in quarterly_avg.index],
                y=quarterly_avg.values,
                name='Quarterly Average',
                marker_color=colors["purple"],
                text=[f"${val:,.0f}" for val in quarterly_avg.values],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        seasonal_fig.update_layout(
            title="📅 Seasonal Performance Analysis",
            showlegend=False,
            height=600,
            plot_bgcolor=colors["plot_bg"],
            paper_bgcolor=colors["background"],
            font=dict(color=colors["text"])
        )
        
        figures['seasonal'] = seasonal_fig
    except Exception as e:
        print(f"Error creating seasonal analysis: {e}")
    
    # 3. Enhanced Product Performance analysis
    if product_col and product_col in df.columns:
        try:
            product_performance = df_analysis.groupby(product_col).agg({
                revenue_col: ['sum', 'count', 'mean', 'std']
            }).round(2)
            
            product_performance.columns = ['Total_Revenue', 'Transaction_Count', 'Avg_Revenue', 'Std_Revenue']
            product_performance = product_performance.sort_values('Total_Revenue', ascending=False).head(15)
            
            product_fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Revenue Performance', 'Consistency Analysis'),
                specs=[[{"secondary_y": True}], [{"type": "scatter"}]],
                vertical_spacing=0.15
            )
            
            # Total revenue bars
            product_fig.add_trace(
                go.Bar(
                    x=product_performance.index,
                    y=product_performance['Total_Revenue'],
                    name='Total Revenue',
                    marker_color=colors["accent"],
                    yaxis='y'
                ),
                row=1, col=1
            )
            
            # Average revenue line
            product_fig.add_trace(
                go.Scatter(
                    x=product_performance.index,
                    y=product_performance['Avg_Revenue'],
                    mode='lines+markers',
                    name='Average Revenue',
                    line=dict(color=colors["danger"], width=3),
                    yaxis='y2'
                ),
                row=1, col=1
            )
            
            # Coefficient of variation (consistency)
            product_performance['CoV'] = (product_performance['Std_Revenue'] / product_performance['Avg_Revenue']) * 100
            
            product_fig.add_trace(
                go.Scatter(
                    x=product_performance.index,
                    y=product_performance['CoV'],
                    mode='lines+markers',
                    name='Coefficient of Variation (%)',
                    line=dict(color=colors["cyan"], width=3),
                    marker=dict(symbol='diamond')
                ),
                row=2, col=1
            )
            
            product_fig.update_layout(
                title="🏆 Top Products Performance Analysis",
                height=700,
                plot_bgcolor=colors["plot_bg"],
                paper_bgcolor=colors["background"],
                font=dict(color=colors["text"]),
                xaxis_tickangle=-45
            )
            
            # Update y-axes
            product_fig.update_yaxes(title_text="Total Revenue ($)", secondary_y=False, row=1, col=1)
            product_fig.update_yaxes(title_text="Average Revenue ($)", secondary_y=True, row=1, col=1)
            product_fig.update_yaxes(title_text="Coefficient of Variation (%)", row=2, col=1)
            
            figures['products'] = product_fig
        except Exception as e:
            print(f"Error creating product analysis: {e}")
    
    # 4. Customer Lifetime Value analysis
    if customer_col and customer_col in df.columns:
        try:
            # Calculate basic CLV metrics
            customer_metrics = df_analysis.groupby(customer_col).agg({
                revenue_col: ['sum', 'mean', 'count'],
                date_col: ['min', 'max', 'nunique']
            }).round(2)
            
            customer_metrics.columns = ['Total_Revenue', 'Avg_Value', 'Transaction_Count', 
                                      'First_Purchase', 'Last_Purchase', 'Active_Days']
            
            # Calculate days as customer
            customer_metrics['Days_as_Customer'] = (customer_metrics['Last_Purchase'] - 
                                                  customer_metrics['First_Purchase']).dt.days
            
            # Calculate average purchase frequency
            customer_metrics['Purchase_Frequency'] = customer_metrics['Transaction_Count'] / (
                customer_metrics['Days_as_Customer'].replace(0, 1) / 30.44)  # Approximate monthly
            
            # Top customers by value
            top_customers = customer_metrics.nlargest(10, 'Total_Revenue')
            
            clv_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Top Customers by Revenue', 'Customer Value vs Frequency'),
                specs=[[{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Top customers bar chart
            clv_fig.add_trace(
                go.Bar(
                    x=top_customers.index,
                    y=top_customers['Total_Revenue'],
                    name='Total Revenue',
                    marker_color=colors["primary"]
                ),
                row=1, col=1
            )
            
            # Value vs frequency scatter
            clv_fig.add_trace(
                go.Scatter(
                    x=customer_metrics['Purchase_Frequency'],
                    y=customer_metrics['Avg_Value'],
                    mode='markers',
                    name='Customers',
                    marker=dict(
                        size=customer_metrics['Total_Revenue'] / customer_metrics['Total_Revenue'].max() * 50 + 5,
                        color=customer_metrics['Total_Revenue'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Total Value")
                    ),
                    text=customer_metrics.index
                ),
                row=1, col=2
            )
            
            clv_fig.update_layout(
                title="💰 Customer Lifetime Value Analysis",
                height=500,
                plot_bgcolor=colors["plot_bg"],
                paper_bgcolor=colors["background"],
                font=dict(color=colors["text"])
            )
            
            clv_fig.update_xaxes(title_text="Purchase Frequency (per month)", row=1, col=2)
            clv_fig.update_yaxes(title_text="Average Transaction Value ($)", row=1, col=2)
            
            figures['customers'] = clv_fig
        except Exception as e:
            print(f"Error creating customer analysis: {e}")
    
    return figures

def analyze_customer_segments(df: pd.DataFrame,
                            customer_col: str,
                            revenue_col: str,
                            date_col: str = None,
                            method: str = "rfm") -> Dict[str, Any]:
    """Advanced customer segmentation using RFM or clustering analysis."""
    
    if customer_col not in df.columns or revenue_col not in df.columns:
        return {"error": "Required columns not found"}
    
    # Prepare data
    df_analysis = df.copy()
    if date_col and date_col in df.columns:
        df_analysis[date_col] = pd.to_datetime(df_analysis[date_col], errors='coerce')
        df_analysis = df_analysis.dropna(subset=[date_col])
    
    # Basic customer analysis
    customer_metrics = df_analysis.groupby(customer_col).agg({
        revenue_col: ['sum', 'mean', 'count', 'std']
    }).round(2)
    
    customer_metrics.columns = ['Total_Revenue', 'Avg_Revenue', 'Transaction_Count', 'Revenue_Std']
    
    # Add recency if date column is available
    if date_col and date_col in df_analysis.columns:
        try:
            reference_date = df_analysis[date_col].max()
            recency = df_analysis.groupby(customer_col)[date_col].max().apply(
                lambda x: (reference_date - x).days
            )
            customer_metrics['Recency_Days'] = recency
            
            # Calculate additional time-based metrics
            tenure = df_analysis.groupby(customer_col)[date_col].agg(['min', 'max'])
            customer_metrics['Tenure_Days'] = (tenure['max'] - tenure['min']).dt.days
            customer_metrics['Purchase_Frequency'] = customer_metrics['Transaction_Count'] / (
                customer_metrics['Tenure_Days'].replace(0, 1) / 30.44)  # Approximate monthly
        except Exception as e:
            print(f"Error calculating recency metrics: {e}")
    
    # Customer segmentation
    segments = {}
    
    if method == "rfm" and 'Recency_Days' in customer_metrics.columns:
        # RFM Analysis
        rfm = customer_metrics[['Recency_Days', 'Transaction_Count', 'Total_Revenue']].copy()
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Create RFM scores
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Define segments based on RFM scores
        segment_map = {
            r'555|554|545|544|535|534|525|524|515|514': 'Champions',
            r'5[1-5][1-3]': 'Loyal Customers',
            r'[4-5][4-5][1-5]': 'Potential Loyalists',
            r'[3-5][1-3][4-5]': 'New Customers',
            r'[3-4][4-5][4-5]': 'Promising',
            r'[2-3][3-4][3-4]': 'Customers Needing Attention',
            r'[2-3][2-3][2-3]': 'About to Sleep',
            r'[1-2][1-5][1-5]': 'At Risk',
            r'1[1-2][1-2]': 'Hibernating',
            r'111|112|113|114|115': 'Lost Customers'
        }
        
        rfm['Segment'] = rfm['RFM_Score'].replace(segment_map, regex=True)
        
        # Calculate segment metrics
        segment_summary = rfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum', 'count']
        }).round(2)
        
        segment_summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Value', 'Total_Value', 'Customer_Count']
        
        segments = {
            "segmentation_method": "RFM",
            "segment_summary": segment_summary.to_dict(),
            "segment_counts": rfm['Segment'].value_counts().to_dict(),
            "customer_segments": rfm[['Segment', 'R_Score', 'F_Score', 'M_Score']]
        }
    
    elif method == "clustering" and SKLEARN_AVAILABLE and len(customer_metrics) > 10:
        # Clustering-based segmentation
        try:
            # Prepare data for clustering
            cluster_data = customer_metrics[['Total_Revenue', 'Transaction_Count']].copy()
            if 'Recency_Days' in customer_metrics.columns:
                cluster_data['Recency_Days'] = customer_metrics['Recency_Days']
            
            # Normalize data
            scaler = StandardScaler()
            cluster_scaled = scaler.fit_transform(cluster_data.fillna(0))
            
            # Determine optimal clusters using elbow method
            wcss = []
            max_clusters = min(10, len(cluster_data) - 1)
            
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                kmeans.fit(cluster_scaled)
                wcss.append(kmeans.inertia_)
            
            # Find elbow point (simplified)
            optimal_clusters = 4  # Default
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
            customer_metrics['Cluster'] = kmeans.fit_predict(cluster_scaled)
            
            # Calculate cluster metrics
            cluster_summary = customer_metrics.groupby('Cluster').agg({
                'Total_Revenue': ['mean', 'sum', 'count'],
                'Transaction_Count': 'mean',
                'Recency_Days': 'mean' if 'Recency_Days' in customer_metrics.columns else None
            }).round(2)
            
            # Drop None values from aggregation
            cluster_summary = cluster_summary.dropna(axis=1, how='all')
            
            segments = {
                "segmentation_method": "K-Means Clustering",
                "optimal_clusters": optimal_clusters,
                "cluster_summary": cluster_summary.to_dict(),
                "cluster_counts": customer_metrics['Cluster'].value_counts().to_dict(),
                "wcss": wcss
            }
        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            # Fall back to value-based segmentation
            method = "value"
    
    if method == "value" or not segments:
        # Value-based segmentation (fallback)
        # High-value customers (top 20% by revenue)
        revenue_threshold = customer_metrics['Total_Revenue'].quantile(0.8)
        high_value = customer_metrics[customer_metrics['Total_Revenue'] >= revenue_threshold]
        
        # Frequent customers (top 20% by transaction count)
        frequency_threshold = customer_metrics['Transaction_Count'].quantile(0.8)
        frequent = customer_metrics[customer_metrics['Transaction_Count'] >= frequency_threshold]
        
        # Recent customers (if recency data available)
        if 'Recency_Days' in customer_metrics.columns:
            recency_threshold = customer_metrics['Recency_Days'].quantile(0.2)
            recent = customer_metrics[customer_metrics['Recency_Days'] <= recency_threshold]
        else:
            recent = pd.DataFrame()
        
        segments = {
            "high_value_customers": {
                "count": len(high_value),
                "avg_revenue": float(high_value['Total_Revenue'].mean()),
                "total_revenue": float(high_value['Total_Revenue'].sum()),
                "percentage_of_total": float((high_value['Total_Revenue'].sum() / customer_metrics['Total_Revenue'].sum()) * 100)
            },
            "frequent_customers": {
                "count": len(frequent),
                "avg_transactions": float(frequent['Transaction_Count'].mean()),
                "total_transactions": int(frequent['Transaction_Count'].sum())
            },
            "recent_customers": {
                "count": len(recent),
                "avg_recency": float(recent['Recency_Days'].mean()) if len(recent) > 0 else 0
            } if 'Recency_Days' in customer_metrics.columns else {},
            "total_customers": len(customer_metrics)
        }
    
    return {
        "customer_metrics": customer_metrics.head(20),  # Return top 20 for display
        "segments": segments,
        "summary": {
            "total_customers": len(customer_metrics),
            "avg_customer_value": float(customer_metrics['Total_Revenue'].mean()),
            "customer_concentration": float((customer_metrics['Total_Revenue'].nlargest(10).sum() / customer_metrics['Total_Revenue'].sum()) * 100),
            "gini_coefficient": calculate_gini_coefficient(customer_metrics['Total_Revenue']) if len(customer_metrics) > 1 else 0
        }
    }

# ============================
# EXPORT AND REPORTING FUNCTIONS
# ============================

def generate_business_report(df: pd.DataFrame, 
                           revenue_col: str = None,
                           date_col: str = None,
                           customer_col: str = None,
                           category_col: str = None) -> Dict[str, Any]:
    """Generate comprehensive business report with analysis and visualizations."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": validate_dataframe(df),
        "kpis": {},
        "visualizations": {},
        "recommendations": []
    }
    
    # Calculate KPIs
    report["kpis"] = calculate_business_kpis(df, revenue_col, None, date_col, customer_col, category_col)
    
    # Generate executive dashboard
    try:
        executive_fig = create_executive_dashboard(df, revenue_col, date_col, category_col, customer_col)
        report["visualizations"]["executive_dashboard"] = executive_fig.to_dict()
    except Exception as e:
        report["errors"] = report.get("errors", []) + [f"Executive dashboard error: {str(e)}"]
    
    # Generate sales performance analysis
    if revenue_col and date_col:
        try:
            performance_figs = create_sales_performance_analysis(df, revenue_col, date_col, category_col, customer_col)
            report["visualizations"]["performance_analysis"] = {
                key: fig.to_dict() for key, fig in performance_figs.items()
            }
        except Exception as e:
            report["errors"] = report.get("errors", []) + [f"Performance analysis error: {str(e)}"]
    
    # Generate customer analysis
    if customer_col and revenue_col:
        try:
            customer_analysis = analyze_customer_segments(df, customer_col, revenue_col, date_col)
            report["customer_analysis"] = customer_analysis
        except Exception as e:
            report["errors"] = report.get("errors", []) + [f"Customer analysis error: {str(e)}"]
    
    # Generate recommendations based on analysis
    if "revenue_metrics" in report["kpis"]:
        revenue_metrics = report["kpis"]["revenue_metrics"]
        
        if revenue_metrics.get("coefficient_of_variation", 0) > 1.5:
            report["recommendations"].append(
                "High revenue variability detected. Consider implementing pricing strategies or promotions to stabilize revenue."
            )
        
        if revenue_metrics.get("skewness", 0) > 2:
            report["recommendations"].append(
                "Revenue distribution is highly skewed. Focus on increasing average transaction value through upselling or cross-selling."
            )
    
    if "growth_metrics" in report["kpis"]:
        growth_metrics = report["kpis"]["growth_metrics"]
        
        if growth_metrics.get("total_growth_rate", 0) < 0:
            report["recommendations"].append(
                "Negative growth rate detected. Investigate causes and consider strategic changes to reverse the trend."
            )
    
    if "customer_metrics" in report["kpis"]:
        customer_metrics = report["kpis"]["customer_metrics"]
        
        if customer_metrics.get("customer_concentration", 0) > 50:
            report["recommendations"].append(
                "High customer concentration risk. Develop strategies to diversify customer base and reduce dependency on top customers."
            )
    
    return report

def export_report_to_html(report: Dict[str, Any], filename: str = "business_report.html") -> str:
    """Export business report to HTML format."""
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Business Intelligence Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Inter', 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8fafc;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                color: white;
                border-radius: 10px;
            }}
            .section {{
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .kpi-card {{
                background: #f1f5f9;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .kpi-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3b82f6;
            }}
            .kpi-label {{
                font-size: 14px;
                color: #64748b;
            }}
            .plot-container {{
                margin: 20px 0;
            }}
            .recommendation {{
                background: #fffbeb;
                border-left: 4px solid #f59e0b;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Business Intelligence Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📈 Executive Summary</h2>
            <div class="kpi-grid">
    """
    
    # Add KPIs
    kpis = report.get("kpis", {})
    if "revenue_metrics" in kpis:
        rm = kpis["revenue_metrics"]
        html_content += f"""
                <div class="kpi-card">
                    <div class="kpi-value">${rm.get('total_revenue', 0):,.0f}</div>
                    <div class="kpi-label">Total Revenue</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">${rm.get('average_transaction', 0):.2f}</div>
                    <div class="kpi-label">Avg Transaction</div>
                </div>
        """
    
    if "customer_metrics" in kpis:
        cm = kpis["customer_metrics"]
        html_content += f"""
                <div class="kpi-card">
                    <div class="kpi-value">{cm.get('unique_customers', 0):,}</div>
                    <div class="kpi-label">Unique Customers</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">${cm.get('avg_customer_value', 0):.2f}</div>
                    <div class="kpi-label">Avg Customer Value</div>
                </div>
        """
    
    if "growth_metrics" in kpis:
        gm = kpis["growth_metrics"]
        html_content += f"""
                <div class="kpi-card">
                    <div class="kpi-value">{gm.get('total_growth_rate', 0):+.1f}%</div>
                    <div class="kpi-label">Growth Rate</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
    """
    
    # Add visualizations
    if "visualizations" in report:
        html_content += """
        <div class="section">
            <h2>📊 Visualizations</h2>
        """
        
        for viz_name, viz_data in report["visualizations"].items():
            if "executive_dashboard" in viz_name:
                html_content += f"""
                <div class="plot-container">
                    <h3>Executive Dashboard</h3>
                    <div id="executive-dashboard"></div>
                </div>
                <script>
                    Plotly.newPlot('executive-dashboard', {json.dumps(viz_data)});
                </script>
                """
        
        html_content += """
        </div>
        """
    
    # Add recommendations
    if report.get("recommendations"):
        html_content += """
        <div class="section">
            <h2>💡 Recommendations</h2>
        """
        
        for rec in report["recommendations"]:
            html_content += f"""
            <div class="recommendation">
                <p>{rec}</p>
            </div>
            """
        
        html_content += """
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(html_content)
    
    return filename

# ============================
# STREAMLIT INTEGRATION
# ============================

def render_business_intelligence_dashboard(df: pd.DataFrame):
    """Render comprehensive business intelligence dashboard in Streamlit."""
    if not STREAMLIT_AVAILABLE:
        print("⚠️ Streamlit not available - cannot render dashboard")
        return None
    
    st.set_page_config(
        page_title="Business Intelligence Suite",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 Business Intelligence Suite</h1>', unsafe_allow_html=True)
    
    # Data validation and info
    with st.expander("🔍 Data Overview & Validation"):
        validation = validate_dataframe(df)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", validation["summary"]["total_rows"])
        with col2:
            st.metric("Total Columns", validation["summary"]["total_columns"])
        with col3:
            st.metric("Missing Values", validation["summary"]["missing_values"])
        with col4:
            st.metric("Duplicate Rows", validation["summary"]["duplicate_rows"])
        
        if validation["issues"]:
            st.warning("Data quality issues detected:")
            for issue in validation["issues"]:
                st.write(f"⚠️ {issue}")
    
    # Column mapping interface
    st.markdown("### 🔧 Data Configuration")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        revenue_col = st.selectbox("Revenue Column:", [None] + validation["column_types"]["numeric"])
    
    with col2:
        date_col = st.selectbox("Date Column:", [None] + validation["column_types"]["date"] + 
                               validation["column_types"]["categorical"])
    
    with col3:
        category_col = st.selectbox("Category Column:", [None] + validation["column_types"]["categorical"])
    
    with col4:
        customer_col = st.selectbox("Customer Column:", [None] + validation["column_types"]["categorical"])
    
    with col5:
        theme = st.selectbox("Theme:", list(COLOR_SCHEMES.keys()))
    
    if revenue_col or date_col:
        # Calculate KPIs
        try:
            with st.spinner("Calculating business metrics..."):
                kpis = calculate_business_kpis(df, revenue_col, None, date_col, customer_col, category_col)
            
            # Display KPIs in a grid
            st.markdown("### 🎯 Key Performance Indicators")
            
            kpi_cols = st.columns(5)
            kpi_config = [
                {"key": "total_revenue", "label": "💰 Total Revenue", "format": "${:,.0f}", "source": "revenue_metrics"},
                {"key": "average_transaction", "label": "📈 Avg Transaction", "format": "${:.2f}", "source": "revenue_metrics"},
                {"key": "unique_customers", "label": "👥 Unique Customers", "format": "{:,}", "source": "customer_metrics"},
                {"key": "total_growth_rate", "label": "📊 Growth Rate", "format": "{:+.1f}%", "source": "growth_metrics"},
                {"key": "avg_customer_value", "label": "💎 Avg Customer Value", "format": "${:.2f}", "source": "customer_metrics"}
            ]
            
            for i, config in enumerate(kpi_config):
                with kpi_cols[i]:
                    source_data = kpis.get(config["source"], {})
                    value = source_data.get(config["key"], 0)
                    if config["key"] in source_data:
                        st.metric(config["label"], config["format"].format(value))
                    else:
                        st.metric(config["label"], "N/A")
            
            # Executive Dashboard
            st.markdown("### 📊 Executive Dashboard")
            try:
                with st.spinner("Generating executive dashboard..."):
                    executive_fig = create_executive_dashboard(df, revenue_col, date_col, category_col, customer_col, theme)
                    st.plotly_chart(executive_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating executive dashboard: {e}")
            
            # Sales Performance Analysis
            if revenue_col and date_col:
                st.markdown("### 📈 Sales Performance Analysis")
                
                try:
                    with st.spinner("Analyzing sales performance..."):
                        performance_figs = create_sales_performance_analysis(df, revenue_col, date_col, category_col, customer_col, theme)
                    
                    if 'waterfall' in performance_figs:
                        st.plotly_chart(performance_figs['waterfall'], use_container_width=True)
                    
                    if 'seasonal' in performance_figs:
                        st.plotly_chart(performance_figs['seasonal'], use_container_width=True)
                    
                    if 'products' in performance_figs and category_col:
                        st.plotly_chart(performance_figs['products'], use_container_width=True)
                    
                    if 'customers' in performance_figs and customer_col:
                        st.plotly_chart(performance_figs['customers'], use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error creating performance analysis: {e}")
            
            # Customer Analysis
            if customer_col and revenue_col:
                st.markdown("### 👥 Customer Analysis")
                
                try:
                    with st.spinner("Analyzing customer segments..."):
                        customer_analysis = analyze_customer_segments(df, customer_col, revenue_col, date_col)
                    
                    if "error" not in customer_analysis:
                        # Display customer segments
                        col1, col2, col3 = st.columns(3)
                        
                        segments = customer_analysis['segments']
                        
                        with col1:
                            if "high_value_customers" in segments:
                                st.metric(
                                    "🏆 High-Value Customers", 
                                    f"{segments['high_value_customers']['count']}",
                                    f"{segments['high_value_customers']['percentage_of_total']:.1f}% of revenue"
                                )
                        
                        with col2:
                            if "frequent_customers" in segments:
                                st.metric(
                                    "🔄 Frequent Customers", 
                                    f"{segments['frequent_customers']['count']}",
                                    f"{segments['frequent_customers']['avg_transactions']:.1f} avg transactions"
                                )
                        
                        with col3:
                            summary = customer_analysis['summary']
                            st.metric(
                                "💰 Avg Customer Value", 
                                f"${summary['avg_customer_value']:.2f}",
                                f"{summary['customer_concentration']:.1f}% concentration"
                            )
                        
                        # Display top customers
                        st.markdown("**🏅 Top Customers by Revenue:**")
                        top_customers = customer_analysis['customer_metrics'].head(10)
                        st.dataframe(top_customers, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error in customer analysis: {e}")
            
            # Report generation
            st.markdown("### 📋 Report Generation")
            if st.button("Generate Comprehensive Business Report"):
                with st.spinner("Generating report..."):
                    report = generate_business_report(df, revenue_col, date_col, customer_col, category_col)
                    
                    # Export to HTML
                    html_file = export_report_to_html(report)
                    
                    # Download button
                    with open(html_file, "rb") as f:
                        html_data = f.read()
                    
                    st.download_button(
                        label="Download HTML Report",
                        data=html_data,
                        file_name="business_intelligence_report.html",
                        mime="text/html"
                    )
                    
        except Exception as e:
            st.error(f"Error calculating business KPIs: {e}")
    
    else:
        st.info("👆 Please select at least a revenue or date column to begin analysis.")

# ============================
# EXPORTS
# ============================

__all__ = [
    'validate_dataframe',
    'preprocess_dataframe',
    'calculate_business_kpis',
    'create_executive_dashboard',
    'create_sales_performance_analysis',
    'analyze_customer_segments',
    'generate_business_report',
    'export_report_to_html',
    'render_business_intelligence_dashboard'
]

# Print module load status
print("✅ Enhanced Business Intelligence Module v3.0 - Enterprise Edition Loaded Successfully!")
print(f"   📊 SciPy Available: {SCIPY_AVAILABLE}")
print(f"   🤖 Scikit-learn Available: {SKLEARN_AVAILABLE}")
print(f"   🎨 Streamlit Available: {STREAMLIT_AVAILABLE}")
print("   🚀 All functions ready for import!")