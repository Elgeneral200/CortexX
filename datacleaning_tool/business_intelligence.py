# filename: business_intelligence.py
"""
Business Intelligence Module - Professional Edition v2.2

Advanced business analytics and visualization suite:
- Executive KPI dashboards
- Sales performance analysis
- Revenue trend analysis
- Customer segmentation insights
- Market analysis tools
- Professional reporting capabilities
- Fixed imports and error handling

Author: CortexX Team
Version: 2.2.0 - Complete Working Edition
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime, timedelta
import calendar
import math

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
    from sklearn.preprocessing import StandardScaler
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
# BUSINESS INTELLIGENCE FUNCTIONS
# ============================

def calculate_business_kpis(df: pd.DataFrame, 
                           revenue_col: str = None,
                           quantity_col: str = None,
                           date_col: str = None,
                           customer_col: str = None) -> Dict[str, Any]:
    """Calculate comprehensive business KPIs."""
    
    kpis = {
        "total_records": len(df),
        "date_range": None,
        "revenue_metrics": {},
        "growth_metrics": {},
        "customer_metrics": {},
        "performance_metrics": {}
    }
    
    # Revenue Analysis
    if revenue_col and revenue_col in df.columns:
        revenue_series = pd.to_numeric(df[revenue_col], errors='coerce').dropna()
        
        if len(revenue_series) > 0:
            kpis["revenue_metrics"] = {
                "total_revenue": float(revenue_series.sum()),
                "average_transaction": float(revenue_series.mean()),
                "median_transaction": float(revenue_series.median()),
                "revenue_std": float(revenue_series.std()),
                "max_transaction": float(revenue_series.max()),
                "min_transaction": float(revenue_series.min()),
                "revenue_range": float(revenue_series.max() - revenue_series.min()),
                "coefficient_of_variation": float(revenue_series.std() / revenue_series.mean()) if revenue_series.mean() != 0 else 0
            }
    
    # Quantity Analysis
    if quantity_col and quantity_col in df.columns:
        quantity_series = pd.to_numeric(df[quantity_col], errors='coerce').dropna()
        
        if len(quantity_series) > 0:
            kpis["quantity_metrics"] = {
                "total_quantity": float(quantity_series.sum()),
                "average_quantity": float(quantity_series.mean()),
                "max_quantity": float(quantity_series.max()),
                "min_quantity": float(quantity_series.min())
            }
    
    # Time-based Analysis
    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(dates) > 0:
                kpis["date_range"] = {
                    "start_date": dates.min().strftime('%Y-%m-%d'),
                    "end_date": dates.max().strftime('%Y-%m-%d'),
                    "total_days": (dates.max() - dates.min()).days,
                    "unique_dates": dates.nunique(),
                    "data_frequency": "Daily" if dates.nunique() > (dates.max() - dates.min()).days * 0.7 else "Sparse"
                }
                
                # Growth analysis (if we have revenue and dates)
                if revenue_col and revenue_col in df.columns:
                    df_time = df.copy()
                    df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
                    df_time = df_time.dropna(subset=[date_col, revenue_col])
                    
                    if len(df_time) > 1:
                        daily_revenue = df_time.groupby(date_col)[revenue_col].sum()
                        if len(daily_revenue) > 1:
                            # Calculate growth metrics
                            first_period = daily_revenue.iloc[0]
                            last_period = daily_revenue.iloc[-1]
                            
                            if first_period != 0:
                                total_growth_rate = ((last_period - first_period) / first_period) * 100
                            else:
                                total_growth_rate = 0
                            
                            # Calculate average daily growth
                            periods = len(daily_revenue)
                            if periods > 1 and first_period > 0:
                                avg_daily_growth = (pow(last_period / first_period, 1/periods) - 1) * 100
                            else:
                                avg_daily_growth = 0
                            
                            kpis["growth_metrics"] = {
                                "total_growth_rate": float(total_growth_rate),
                                "average_daily_growth": float(avg_daily_growth),
                                "trend": "Increasing" if total_growth_rate > 0 else "Decreasing" if total_growth_rate < 0 else "Stable"
                            }
                            
                            # Seasonal analysis
                            df_time['month'] = df_time[date_col].dt.month
                            df_time['day_of_week'] = df_time[date_col].dt.dayofweek
                            
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
        except Exception:
            pass
    
    # Customer Analysis
    if customer_col and customer_col in df.columns:
        unique_customers = df[customer_col].nunique()
        
        kpis["customer_metrics"] = {
            "unique_customers": unique_customers,
            "avg_transactions_per_customer": len(df) / max(unique_customers, 1),
            "customer_retention_proxy": min(100, (len(df) / max(unique_customers, 1)) * 10)  # Proxy metric
        }
        
        if revenue_col and revenue_col in df.columns:
            customer_revenue = df.groupby(customer_col)[revenue_col].sum()
            kpis["customer_metrics"].update({
                "avg_customer_value": float(customer_revenue.mean()),
                "top_customer_value": float(customer_revenue.max()),
                "customer_revenue_std": float(customer_revenue.std()),
                "customer_concentration": float((customer_revenue.nlargest(10).sum() / customer_revenue.sum()) * 100) if customer_revenue.sum() > 0 else 0
            })
    
    # Performance Metrics
    if revenue_col and revenue_col in df.columns:
        revenue_series = pd.to_numeric(df[revenue_col], errors='coerce').dropna()
        if len(revenue_series) > 0:
            # Calculate performance percentiles
            p25 = float(revenue_series.quantile(0.25))
            p75 = float(revenue_series.quantile(0.75))
            
            kpis["performance_metrics"] = {
                "high_value_transactions": int((revenue_series >= p75).sum()),
                "low_value_transactions": int((revenue_series <= p25).sum()),
                "medium_value_transactions": int(((revenue_series > p25) & (revenue_series < p75)).sum()),
                "performance_ratio": float(p75 / p25) if p25 > 0 else 0
            }
    
    return kpis

def create_executive_dashboard(df: pd.DataFrame, 
                             revenue_col: str = None,
                             date_col: str = None,
                             category_col: str = None,
                             theme: str = "professional_dark") -> go.Figure:
    """Create comprehensive executive dashboard."""
    
    # Create subplot structure
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Revenue Trend', 'Category Performance', 'Daily Volume',
                       'Revenue Distribution', 'Top Categories', 'Performance Analysis'),
        specs=[[{"secondary_y": False}, {"type": "pie"}, {"type": "bar"}],
              [{"type": "histogram"}, {"type": "bar"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
    
    # 1. Revenue Trend (if date and revenue columns available)
    if date_col and revenue_col and date_col in df.columns and revenue_col in df.columns:
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
        df_time = df_time.dropna(subset=[date_col, revenue_col])
        
        if len(df_time) > 0:
            # Aggregate by date to handle multiple transactions per day
            daily_revenue = df_time.groupby(date_col)[revenue_col].sum().reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_revenue[date_col],
                    y=daily_revenue[revenue_col],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color=colors[0], width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
    
    # 2. Category Performance (pie chart)
    if category_col and category_col in df.columns:
        if revenue_col and revenue_col in df.columns:
            category_revenue = df.groupby(category_col)[revenue_col].sum()
        else:
            category_revenue = df[category_col].value_counts()
            
        # Limit to top 8 categories for readability
        top_categories = category_revenue.nlargest(8)
        
        fig.add_trace(
            go.Pie(
                labels=top_categories.index,
                values=top_categories.values,
                name="Categories",
                marker_colors=colors[:len(top_categories)]
            ),
            row=1, col=2
        )
    
    # 3. Daily Volume
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
                    marker_color=colors[1]
                ),
                row=1, col=3
            )
    
    # 4. Revenue Distribution (histogram)
    if revenue_col and revenue_col in df.columns:
        revenue_series = pd.to_numeric(df[revenue_col], errors='coerce').dropna()
        
        if len(revenue_series) > 0:
            fig.add_trace(
                go.Histogram(
                    x=revenue_series,
                    name='Revenue Distribution',
                    marker_color=colors[2],
                    opacity=0.7,
                    nbinsx=30
                ),
                row=2, col=1
            )
    
    # 5. Top Categories (horizontal bar chart)
    if category_col and category_col in df.columns:
        top_categories = df[category_col].value_counts().head(8)
        
        fig.add_trace(
            go.Bar(
                x=top_categories.values,
                y=top_categories.index,
                orientation='h',
                name='Top Categories',
                marker_color=colors[3]
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
                        color=colors[4], 
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
                            line=dict(color=colors[5], width=2, dash='dash')
                        ),
                        row=2, col=3
                    )
                except:
                    pass
    
    # Update layout with professional styling
    fig.update_layout(
        title=dict(
            text="📊 Executive Business Intelligence Dashboard",
            x=0.5,
            font=dict(size=24, color='white')
        ),
        showlegend=True,
        height=800,
        plot_bgcolor='rgba(15, 23, 42, 0.8)',
        paper_bgcolor='rgba(15, 23, 42, 1)',
        font=dict(color='white', family="Inter"),
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
        gridcolor='rgba(255,255,255,0.1)',
        linecolor='rgba(255,255,255,0.2)'
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.1)',
        linecolor='rgba(255,255,255,0.2)'
    )
    
    return fig

def create_sales_performance_analysis(df: pd.DataFrame,
                                    revenue_col: str,
                                    date_col: str,
                                    product_col: str = None,
                                    theme: str = "professional_dark") -> Dict[str, go.Figure]:
    """Create comprehensive sales performance analysis."""
    
    figures = {}
    
    # Prepare data
    df_analysis = df.copy()
    df_analysis[date_col] = pd.to_datetime(df_analysis[date_col], errors='coerce')
    df_analysis = df_analysis.dropna(subset=[date_col, revenue_col])
    df_analysis[revenue_col] = pd.to_numeric(df_analysis[revenue_col], errors='coerce')
    
    if len(df_analysis) == 0:
        return figures
    
    # 1. Revenue Waterfall Chart
    try:
        monthly_revenue = df_analysis.groupby(df_analysis[date_col].dt.to_period('M'))[revenue_col].sum()
        
        if len(monthly_revenue) > 1:
            waterfall_fig = go.Figure()
            
            # Calculate changes month over month
            values = monthly_revenue.values
            changes = np.diff(values)
            
            # Starting point
            waterfall_fig.add_trace(go.Waterfall(
                name="Revenue Analysis",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(changes),
                x=[str(monthly_revenue.index[0])] + [f"Change {str(period)}" for period in monthly_revenue.index[1:]],
                textposition="outside",
                text=[f"${values[0]:,.0f}"] + [f"${change:+,.0f}" for change in changes],
                y=np.concatenate([[values[0]], changes]),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            waterfall_fig.update_layout(
                title="💰 Monthly Revenue Waterfall Analysis",
                showlegend=False,
                height=500,
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                paper_bgcolor='rgba(15, 23, 42, 1)',
                font=dict(color='white')
            )
            
            figures['waterfall'] = waterfall_fig
    except Exception as e:
        print(f"Error creating waterfall chart: {e}")
    
    # 2. Seasonal Analysis
    try:
        df_analysis['month'] = df_analysis[date_col].dt.month
        df_analysis['day_of_week'] = df_analysis[date_col].dt.day_name()
        
        seasonal_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Monthly Seasonality', 'Weekly Pattern'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Monthly pattern
        monthly_avg = df_analysis.groupby('month')[revenue_col].mean()
        month_names = [calendar.month_abbr[i] for i in monthly_avg.index]
        
        seasonal_fig.add_trace(
            go.Bar(
                x=month_names,
                y=monthly_avg.values,
                name='Monthly Average',
                marker_color='#10b981',
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
                marker_color='#3b82f6',
                text=[f"${val:,.0f}" for val in weekly_ordered.values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        seasonal_fig.update_layout(
            title="📅 Seasonal Performance Analysis",
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            paper_bgcolor='rgba(15, 23, 42, 1)',
            font=dict(color='white')
        )
        
        figures['seasonal'] = seasonal_fig
    except Exception as e:
        print(f"Error creating seasonal analysis: {e}")
    
    # 3. Product Performance (if product column available)
    if product_col and product_col in df.columns:
        try:
            product_performance = df_analysis.groupby(product_col).agg({
                revenue_col: ['sum', 'count', 'mean']
            }).round(2)
            
            product_performance.columns = ['Total_Revenue', 'Transaction_Count', 'Avg_Revenue']
            product_performance = product_performance.sort_values('Total_Revenue', ascending=False).head(15)
            
            product_fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Total revenue bars
            product_fig.add_trace(
                go.Bar(
                    x=product_performance.index,
                    y=product_performance['Total_Revenue'],
                    name='Total Revenue',
                    marker_color='#f59e0b',
                    yaxis='y'
                )
            )
            
            # Average revenue line
            product_fig.add_trace(
                go.Scatter(
                    x=product_performance.index,
                    y=product_performance['Avg_Revenue'],
                    mode='lines+markers',
                    name='Average Revenue',
                    line=dict(color='#ef4444', width=3),
                    yaxis='y2'
                )
            )
            
            product_fig.update_layout(
                title="🏆 Top Products Performance Analysis",
                xaxis_title="Products",
                height=500,
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                paper_bgcolor='rgba(15, 23, 42, 1)',
                font=dict(color='white'),
                xaxis_tickangle=-45
            )
            
            # Update y-axes
            product_fig.update_yaxes(title_text="Total Revenue ($)", secondary_y=False)
            product_fig.update_yaxes(title_text="Average Revenue ($)", secondary_y=True)
            
            figures['products'] = product_fig
        except Exception as e:
            print(f"Error creating product analysis: {e}")
    
    return figures

def analyze_customer_segments(df: pd.DataFrame,
                            customer_col: str,
                            revenue_col: str,
                            date_col: str = None) -> Dict[str, Any]:
    """Analyze customer segments using RFM analysis if possible."""
    
    if customer_col not in df.columns or revenue_col not in df.columns:
        return {"error": "Required columns not found"}
    
    # Basic customer analysis
    customer_metrics = df.groupby(customer_col).agg({
        revenue_col: ['sum', 'mean', 'count']
    }).round(2)
    
    customer_metrics.columns = ['Total_Revenue', 'Avg_Revenue', 'Transaction_Count']
    
    # Add recency if date column is available
    if date_col and date_col in df.columns:
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[date_col])
            
            if len(df_temp) > 0:
                reference_date = df_temp[date_col].max()
                recency = df_temp.groupby(customer_col)[date_col].max().apply(
                    lambda x: (reference_date - x).days
                )
                customer_metrics['Recency_Days'] = recency
        except:
            pass
    
    # Customer segmentation
    segments = {}
    
    # High-value customers (top 20% by revenue)
    revenue_threshold = customer_metrics['Total_Revenue'].quantile(0.8)
    high_value = customer_metrics[customer_metrics['Total_Revenue'] >= revenue_threshold]
    
    # Frequent customers (top 20% by transaction count)
    frequency_threshold = customer_metrics['Transaction_Count'].quantile(0.8)
    frequent = customer_metrics[customer_metrics['Transaction_Count'] >= frequency_threshold]
    
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
        "total_customers": len(customer_metrics)
    }
    
    return {
        "customer_metrics": customer_metrics.head(20),  # Return top 20 for display
        "segments": segments,
        "summary": {
            "total_customers": len(customer_metrics),
            "avg_customer_value": float(customer_metrics['Total_Revenue'].mean()),
            "customer_concentration": float((customer_metrics['Total_Revenue'].nlargest(10).sum() / customer_metrics['Total_Revenue'].sum()) * 100)
        }
    }

# ============================
# STREAMLIT INTEGRATION
# ============================

def render_business_intelligence_dashboard(df: pd.DataFrame):
    """Render comprehensive business intelligence dashboard in Streamlit."""
    if not STREAMLIT_AVAILABLE:
        print("⚠️ Streamlit not available - cannot render dashboard")
        return None
    
    st.markdown("## 📊 Business Intelligence Suite")
    
    # Column mapping interface
    st.markdown("### 🔧 Data Configuration")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = []
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Auto-detect date columns
    for col in categorical_cols:
        sample = df[col].dropna().head(100)
        try:
            pd.to_datetime(sample, errors='raise')
            date_cols.append(col)
        except:
            continue
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue_col = st.selectbox("Revenue Column:", [None] + numeric_cols)
    
    with col2:
        date_col = st.selectbox("Date Column:", [None] + date_cols + categorical_cols)
    
    with col3:
        category_col = st.selectbox("Category Column:", [None] + categorical_cols)
    
    with col4:
        customer_col = st.selectbox("Customer Column:", [None] + categorical_cols)
    
    if revenue_col or date_col:
        # Calculate KPIs
        try:
            kpis = calculate_business_kpis(df, revenue_col, None, date_col, customer_col)
            
            # Display KPIs
            st.markdown("### 🎯 Key Performance Indicators")
            
            kpi_cols = st.columns(5)
            
            with kpi_cols[0]:
                if 'total_revenue' in kpis.get('revenue_metrics', {}):
                    total_revenue = kpis['revenue_metrics']['total_revenue']
                    st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
                else:
                    st.metric("📊 Total Records", f"{kpis['total_records']:,}")
            
            with kpi_cols[1]:
                if 'average_transaction' in kpis.get('revenue_metrics', {}):
                    avg_transaction = kpis['revenue_metrics']['average_transaction']
                    st.metric("📈 Avg Transaction", f"${avg_transaction:.2f}")
                else:
                    date_range = kpis.get('date_range', {})
                    if date_range:
                        st.metric("📅 Date Range", f"{date_range.get('total_days', 'N/A')} days")
                    else:
                        st.metric("📅 Date Range", "N/A")
            
            with kpi_cols[2]:
                if 'unique_customers' in kpis.get('customer_metrics', {}):
                    unique_customers = kpis['customer_metrics']['unique_customers']
                    st.metric("👥 Unique Customers", f"{unique_customers:,}")
                else:
                    st.metric("📋 Data Points", f"{len(df):,}")
            
            with kpi_cols[3]:
                if 'total_growth_rate' in kpis.get('growth_metrics', {}):
                    growth_rate = kpis['growth_metrics']['total_growth_rate']
                    st.metric("📊 Growth Rate", f"{growth_rate:+.1f}%")
                else:
                    if revenue_col and revenue_col in df.columns:
                        max_revenue = df[revenue_col].max()
                        st.metric("📈 Max Transaction", f"${max_revenue:,.2f}")
                    else:
                        st.metric("🔢 Numeric Cols", f"{len(numeric_cols)}")
            
            with kpi_cols[4]:
                if 'avg_customer_value' in kpis.get('customer_metrics', {}):
                    avg_customer_value = kpis['customer_metrics']['avg_customer_value']
                    st.metric("💎 Avg Customer Value", f"${avg_customer_value:.2f}")
                else:
                    st.metric("📝 Categorical Cols", f"{len(categorical_cols)}")
            
            # Executive Dashboard
            st.markdown("### 📊 Executive Dashboard")
            try:
                executive_fig = create_executive_dashboard(df, revenue_col, date_col, category_col)
                st.plotly_chart(executive_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating executive dashboard: {e}")
            
            # Sales Performance Analysis
            if revenue_col and date_col:
                st.markdown("### 📈 Sales Performance Analysis")
                
                try:
                    performance_figs = create_sales_performance_analysis(df, revenue_col, date_col, category_col)
                    
                    if 'waterfall' in performance_figs:
                        st.plotly_chart(performance_figs['waterfall'], use_container_width=True)
                    
                    if 'seasonal' in performance_figs:
                        st.plotly_chart(performance_figs['seasonal'], use_container_width=True)
                    
                    if 'products' in performance_figs:
                        st.plotly_chart(performance_figs['products'], use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error creating performance analysis: {e}")
            
            # Customer Analysis
            if customer_col and revenue_col:
                st.markdown("### 👥 Customer Analysis")
                
                try:
                    customer_analysis = analyze_customer_segments(df, customer_col, revenue_col, date_col)
                    
                    if "error" not in customer_analysis:
                        # Display customer segments
                        col1, col2, col3 = st.columns(3)
                        
                        segments = customer_analysis['segments']
                        
                        with col1:
                            st.metric(
                                "🏆 High-Value Customers", 
                                f"{segments['high_value_customers']['count']}",
                                f"{segments['high_value_customers']['percentage_of_total']:.1f}% of revenue"
                            )
                        
                        with col2:
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
                    
        except Exception as e:
            st.error(f"Error calculating business KPIs: {e}")
    
    else:
        st.info("👆 Please select at least a revenue or date column to begin analysis.")

# ============================
# EXPORTS
# ============================

__all__ = [
    'calculate_business_kpis',
    'create_executive_dashboard',
    'create_sales_performance_analysis',
    'analyze_customer_segments',
    'render_business_intelligence_dashboard'
]

# Print module load status
print("✅ Enhanced Business Intelligence Module v2.2 - Loaded Successfully!")
print(f"   📊 SciPy Available: {SCIPY_AVAILABLE}")
print(f"   🤖 Scikit-learn Available: {SKLEARN_AVAILABLE}")
print(f"   🎨 Streamlit Available: {STREAMLIT_AVAILABLE}")
print("   🚀 All functions ready for import!")
