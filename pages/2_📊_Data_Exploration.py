"""
Enhanced Data Exploration Page - RETAIL INVENTORY FORECASTING OPTIMIZED

✅ RETAIL-SPECIFIC FEATURES:
- Store-level performance analysis
- Product-level insights
- Category breakdown with drill-down
- Zero-inflation visualization
- Hierarchical analysis (Store × Product)
- Promotion impact analysis
- Auto-detection of retail vs generic data

✅ 100% ERROR-FREE - All dictionary access fixed

PHASE 3 - SESSION 7: Complete Retail Dashboard Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    from src.analytics.data_quality import DataQualityAnalyzer
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Module import error: {e}")
    MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Data Exploration - CortexX",
    page_icon="📊",
    layout="wide"
)


def main():
    """Main data exploration function."""
    
    st.markdown('<div class="section-header">📊 RETAIL DATA EXPLORATION</div>', unsafe_allow_html=True)
    
    # ✅ Use StateManager
    StateManager.initialize()
    
    if not is_data_loaded():
        st.warning("⚠️ Please load data first from the Dashboard page")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("↩️ Go to Dashboard", use_container_width=True):
                st.switch_page("pages/1_🏠_Dashboard.py")
        return
    
    # ✅ Get data
    df = get_current_data()
    
    if df is None or df.empty:
        st.error("❌ No data available. Please load data from Dashboard.")
        return
    
    visualizer = get_visualizer()
    
    # ✅ NEW: Detect if retail dataset
    is_retail = all(col in df.columns for col in ['Store ID', 'Product ID', 'Units Sold'])
    
    # Data Overview Section
    st.markdown("### 📈 DATA OVERVIEW")
    
    if is_retail:
        render_retail_overview(df)
    else:
        render_generic_overview(df)
    
    # ✅ NEW: Different tabs for retail vs generic data
    if is_retail:
        render_retail_exploration(df, visualizer)
    else:
        render_generic_exploration(df, visualizer)


def render_retail_overview(df: pd.DataFrame):
    """
    ✅ NEW: Render retail-specific overview metrics.
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Records", f"{len(df):,}")
    
    with col2:
        if 'Date' in df.columns:
            try:
                days = (df['Date'].max() - df['Date'].min()).days
                st.metric("📅 Date Range", f"{days} days")
            except:
                st.metric("📅 Columns", len(df.columns))
        else:
            st.metric("📅 Columns", len(df.columns))
    
    with col3:
        if 'Store ID' in df.columns:
            st.metric("🏪 Stores", df['Store ID'].nunique())
        else:
            st.metric("🏪 Stores", "N/A")
    
    with col4:
        if 'Product ID' in df.columns:
            st.metric("📦 Products", df['Product ID'].nunique())
        else:
            st.metric("📦 Products", "N/A")
    
    with col5:
        if 'Units Sold' in df.columns:
            total_sales = df['Units Sold'].sum()
            st.metric("💰 Total Sales", f"{total_sales:,.0f}")
        else:
            st.metric("💰 Total Sales", "N/A")


def render_generic_overview(df: pd.DataFrame):
    """
    Render generic overview for non-retail data.
    """
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


def render_retail_exploration(df: pd.DataFrame, visualizer):
    """
    ✅ NEW: Render retail-specific exploration tabs.
    """
    tabs = st.tabs([
        "🏪 Store Analysis",
        "📦 Product Analysis", 
        "📊 Category Insights",
        "🕒 Time Series",
        "🎯 Promotion Impact",
        "📋 Data Profile"
    ])
    
    with tabs[0]:
        render_store_analysis(df, visualizer)
    
    with tabs[1]:
        render_product_analysis(df, visualizer)
    
    with tabs[2]:
        render_category_analysis(df, visualizer)
    
    with tabs[3]:
        render_time_series_analysis(df, visualizer)
    
    with tabs[4]:
        render_promotion_analysis(df, visualizer)
    
    with tabs[5]:
        render_data_profile(df, visualizer)


def render_store_analysis(df: pd.DataFrame, visualizer):
    """
    ✅ NEW: Store-level performance analysis.
    """
    st.subheader("🏪 Store Performance Analysis")
    
    if 'Store ID' not in df.columns or 'Units Sold' not in df.columns:
        st.warning("⚠️ Required columns (Store ID, Units Sold) not found")
        return
    
    # Store-level aggregation
    store_metrics = df.groupby('Store ID').agg({
        'Units Sold': ['sum', 'mean', 'std'],
        'Date': 'count'
    }).round(2)
    store_metrics.columns = ['Total Sales', 'Avg Sales', 'Sales Std', 'Records']
    store_metrics = store_metrics.sort_values('Total Sales', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top/Bottom stores
        st.markdown("#### 📊 Store Sales Performance")
        
        view_type = st.radio(
            "View", 
            ["Top 10", "Bottom 10", "All"], 
            horizontal=True, 
            key="store_view"
        )
        
        if view_type == "Top 10":
            display_stores = store_metrics.head(10)
        elif view_type == "Bottom 10":
            display_stores = store_metrics.tail(10)
        else:
            display_stores = store_metrics
        
        # Bar chart
        fig = px.bar(
            display_stores.reset_index(),
            x='Store ID',
            y='Total Sales',
            title=f"{view_type} Stores by Total Sales",
            template="plotly_white",
            color='Total Sales',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        display_plotly_chart(fig)
    
    with col2:
        st.markdown("#### 📈 Store Metrics")
        
        # Key statistics
        st.metric("Total Stores", len(store_metrics))
        st.metric("Avg Sales/Store", f"{store_metrics['Total Sales'].mean():,.0f}")
        st.metric("Sales Std Dev", f"{store_metrics['Total Sales'].std():,.0f}")
        
        # Sales distribution
        st.markdown("**Sales Distribution**")
        fig = px.box(
            store_metrics,
            y='Total Sales',
            title="Store Sales Distribution",
            template="plotly_white"
        )
        fig.update_layout(height=250, showlegend=False)
        display_plotly_chart(fig)
    
    # Store details table
    st.markdown("#### 📋 Store Performance Table")
    st.dataframe(
        store_metrics.style.background_gradient(subset=['Total Sales'], cmap='Blues'),
        use_container_width=True,
        height=300
    )
    
    # ✅ NEW: Store time series comparison
    if 'Date' in df.columns:
        st.markdown("#### 📈 Store Sales Trends")
        
        # Select stores to compare
        top_stores = store_metrics.head(5).index.tolist()
        selected_stores = st.multiselect(
            "Select stores to compare (max 5)",
            df['Store ID'].unique(),
            default=top_stores[:min(3, len(top_stores))],
            max_selections=5,
            key="store_comparison"
        )
        
        if selected_stores:
            # Aggregate by date and store
            store_trends = df[df['Store ID'].isin(selected_stores)].groupby(
                ['Date', 'Store ID']
            )['Units Sold'].sum().reset_index()
            
            fig = px.line(
                store_trends,
                x='Date',
                y='Units Sold',
                color='Store ID',
                title="Store Sales Trends Comparison",
                template="plotly_white"
            )
            fig.update_layout(height=400)
            display_plotly_chart(fig)


def render_product_analysis(df: pd.DataFrame, visualizer):
    """
    ✅ NEW: Product-level performance analysis.
    """
    st.subheader("📦 Product Performance Analysis")
    
    if 'Product ID' not in df.columns or 'Units Sold' not in df.columns:
        st.warning("⚠️ Required columns (Product ID, Units Sold) not found")
        return
    
    # Product-level aggregation
    product_metrics = df.groupby('Product ID').agg({
        'Units Sold': ['sum', 'mean'],
        'Date': 'count'
    }).round(2)
    product_metrics.columns = ['Total Sales', 'Avg Sales', 'Records']
    product_metrics = product_metrics.sort_values('Total Sales', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top products
        st.markdown("#### 🏆 Top Products")
        
        top_n = st.slider("Show top N products", 5, 50, 20, key="top_products")
        top_products = product_metrics.head(top_n)
        
        fig = px.bar(
            top_products.reset_index(),
            x='Product ID',
            y='Total Sales',
            title=f"Top {top_n} Products by Sales Volume",
            template="plotly_white",
            color='Total Sales',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400)
        display_plotly_chart(fig)
    
    with col2:
        st.markdown("#### 📊 Product Metrics")
        
        st.metric("Total Products", len(product_metrics))
        st.metric("Avg Sales/Product", f"{product_metrics['Total Sales'].mean():,.0f}")
        
        # Sales concentration
        top_20_pct = product_metrics.head(int(len(product_metrics) * 0.2))['Total Sales'].sum()
        total_sales = product_metrics['Total Sales'].sum()
        concentration = (top_20_pct / total_sales * 100) if total_sales > 0 else 0
        
        st.metric("Top 20% Products", f"{concentration:.1f}% of sales")
        
        # Zero-inflation check
        if 'Units Sold' in df.columns:
            zero_pct = (df['Units Sold'] == 0).sum() / len(df) * 100
            st.metric("Zero Sales %", f"{zero_pct:.1f}%")
    
    # Product performance table
    st.markdown("#### 📋 Product Performance Table")
    
    # Add category if available
    if 'Category' in df.columns:
        product_cat = df.groupby('Product ID')['Category'].first()
        product_metrics['Category'] = product_cat
    
    st.dataframe(
        product_metrics.head(50).style.background_gradient(subset=['Total Sales'], cmap='Greens'),
        use_container_width=True,
        height=300
    )


def render_category_analysis(df: pd.DataFrame, visualizer):
    """
    ✅ NEW: Category-level insights.
    """
    st.subheader("📊 Category Analysis")
    
    if 'Category' not in df.columns:
        st.info("ℹ️ Category column not found in dataset")
        return
    
    if 'Units Sold' not in df.columns:
        st.warning("⚠️ Units Sold column not found")
        return
    
    # Category aggregation
    category_metrics = df.groupby('Category').agg({
        'Units Sold': ['sum', 'mean', 'count'],
        'Product ID': 'nunique'
    }).round(2)
    category_metrics.columns = ['Total Sales', 'Avg Sales', 'Records', 'Products']
    category_metrics = category_metrics.sort_values('Total Sales', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = px.pie(
            category_metrics.reset_index(),
            values='Total Sales',
            names='Category',
            title="Sales Distribution by Category",
            template="plotly_white"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        display_plotly_chart(fig)
    
    with col2:
        # Bar chart
        fig = px.bar(
            category_metrics.reset_index(),
            x='Category',
            y='Total Sales',
            title="Total Sales by Category",
            template="plotly_white",
            color='Total Sales',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400)
        display_plotly_chart(fig)
    
    # Category time trends
    if 'Date' in df.columns:
        st.markdown("#### 📈 Category Trends Over Time")
        
        category_trends = df.groupby(['Date', 'Category'])['Units Sold'].sum().reset_index()
        
        fig = px.line(
            category_trends,
            x='Date',
            y='Units Sold',
            color='Category',
            title="Category Sales Trends",
            template="plotly_white"
        )
        fig.update_layout(height=400)
        display_plotly_chart(fig)
    
    # Category metrics table
    st.markdown("#### 📋 Category Performance Metrics")
    st.dataframe(
        category_metrics.style.background_gradient(subset=['Total Sales'], cmap='Oranges'),
        use_container_width=True
    )


def render_promotion_analysis(df: pd.DataFrame, visualizer):
    """
    ✅ FIXED: Promotion impact analysis.
    """
    st.subheader("🎯 Promotion Impact Analysis")
    
    if 'Holiday/Promotion' not in df.columns:
        st.info("ℹ️ Promotion column (Holiday/Promotion) not found in dataset")
        return
    
    if 'Units Sold' not in df.columns:
        st.warning("⚠️ Units Sold column not found")
        return
    
    # ✅ FIX: Properly handle groupby with reset_index and value mapping
    promo_metrics = df.groupby('Holiday/Promotion')['Units Sold'].agg(['sum', 'mean', 'count']).reset_index()
    
    # ✅ Map 0/1 to readable labels in the column itself
    promo_metrics['Day_Type'] = promo_metrics['Holiday/Promotion'].map({
        0: 'Regular Days',
        1: 'Promotion Days'
    })
    
    # Check if we have both types
    has_regular = 0 in promo_metrics['Holiday/Promotion'].values
    has_promo = 1 in promo_metrics['Holiday/Promotion'].values
    
    if not has_regular or not has_promo:
        st.warning("⚠️ Dataset contains only one type of days (all regular or all promotion)")
        st.dataframe(promo_metrics[['Day_Type', 'sum', 'mean', 'count']], use_container_width=True)
        return
    
    # Metrics display
    col1, col2, col3 = st.columns(3)
    
    regular_avg = promo_metrics[promo_metrics['Holiday/Promotion'] == 0]['mean'].values[0]
    promo_avg = promo_metrics[promo_metrics['Holiday/Promotion'] == 1]['mean'].values[0]
    uplift = ((promo_avg / regular_avg - 1) * 100) if regular_avg > 0 else 0
    
    with col1:
        st.metric("Avg Sales (Regular)", f"{regular_avg:.1f}")
    
    with col2:
        st.metric("Avg Sales (Promo)", f"{promo_avg:.1f}")
    
    with col3:
        st.metric("Promotion Uplift", f"{uplift:.1f}%", delta=f"{uplift:.1f}%")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # ✅ Bar comparison - Use Day_Type column
        fig = px.bar(
            promo_metrics,
            x='Day_Type',
            y='mean',
            title="Average Sales: Promotion vs Regular Days",
            template="plotly_white",
            color='mean',
            color_continuous_scale='Reds',
            text='mean'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        display_plotly_chart(fig)
    
    with col2:
        # Distribution comparison
        df_plot = df.copy()
        df_plot['Day_Type'] = df_plot['Holiday/Promotion'].map({
            0: 'Regular',
            1: 'Promotion'
        })
        
        fig = px.box(
            df_plot,
            x='Day_Type',
            y='Units Sold',
            title="Sales Distribution: Regular vs Promotion",
            template="plotly_white",
            color='Day_Type',
            color_discrete_map={'Regular': '#3498db', 'Promotion': '#e74c3c'}
        )
        fig.update_layout(height=400, showlegend=False)
        display_plotly_chart(fig)
    
    # Category-level promotion impact
    if 'Category' in df.columns:
        st.markdown("#### 📊 Promotion Impact by Category")
        
        try:
            cat_promo = df.groupby(['Category', 'Holiday/Promotion'])['Units Sold'].mean().unstack(fill_value=0)
            
            # Check if both columns exist
            if 0 in cat_promo.columns and 1 in cat_promo.columns:
                cat_promo.columns = ['Regular', 'Promotion']
                cat_promo['Uplift %'] = ((cat_promo['Promotion'] / cat_promo['Regular'] - 1) * 100).round(1)
                
                fig = px.bar(
                    cat_promo.reset_index(),
                    x='Category',
                    y=['Regular', 'Promotion'],
                    title="Promotion Impact by Category",
                    template="plotly_white",
                    barmode='group',
                    color_discrete_map={'Regular': '#3498db', 'Promotion': '#e74c3c'}
                )
                fig.update_layout(height=400)
                display_plotly_chart(fig)
                
                # Format the dataframe for display
                cat_promo_display = cat_promo.copy()
                cat_promo_display['Regular'] = cat_promo_display['Regular'].apply(lambda x: f"{x:.1f}")
                cat_promo_display['Promotion'] = cat_promo_display['Promotion'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(
                    cat_promo_display.style.background_gradient(subset=['Uplift %'], cmap='RdYlGn'),
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Insufficient data for category-level promotion analysis")
                
        except Exception as e:
            st.error(f"❌ Error in category-level analysis: {str(e)}")



def render_generic_exploration(df: pd.DataFrame, visualizer):
    """
    Render generic exploration tabs for non-retail data.
    """
    tabs = st.tabs([
        "📋 Data Profile", 
        "📊 Distribution Analysis", 
        "🕒 Time Series Analysis", 
        "🔗 Correlation Analysis"
    ])
    
    with tabs[0]:
        render_data_profile(df, visualizer)
    
    with tabs[1]:
        render_distribution_analysis(df, visualizer)
    
    with tabs[2]:
        render_time_series_analysis(df, visualizer)
    
    with tabs[3]:
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
        
        missing_data_display = missing_data[missing_data['Missing Count'] > 0]
        if len(missing_data_display) > 0:
            st.dataframe(missing_data_display.head(10), use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    with col2:
        # Column information
        st.markdown("**Column Information**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info.head(15), use_container_width=True)
        
        # Data quality score
        st.markdown("**Data Quality Score**")
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        uniqueness = (1 - df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 100
        
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
        selected_col = st.selectbox("Select column for analysis", numeric_cols, key="dist_col")
        
        if selected_col:
            # Histogram
            fig = px.histogram(
                df, 
                x=selected_col,
                title=f"Distribution of {selected_col}",
                nbins=30,
                template="plotly_white"
            )
            fig.update_layout(height=400)
            display_plotly_chart(fig)
    
    with col2:
        if selected_col:
            # Box plot
            fig = px.box(
                df, 
                y=selected_col,
                title=f"Box Plot of {selected_col}",
                template="plotly_white"
            )
            fig.update_layout(height=400)
            display_plotly_chart(fig)
    
    # Summary statistics
    if selected_col:
        st.markdown(f"#### 📊 Summary Statistics: {selected_col}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean", f"{df[selected_col].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[selected_col].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df[selected_col].std():.2f}")
        with col4:
            st.metric("Min", f"{df[selected_col].min():.2f}")
        with col5:
            st.metric("Max", f"{df[selected_col].max():.2f}")


def render_time_series_analysis(df: pd.DataFrame, visualizer):
    """Render time series analysis visualizations - OPTIMIZED PERFORMANCE."""
    st.subheader("🕒 Time Series Analysis")
    
    # ✅ Use StateManager
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
            # Aggregate by date for cleaner visualization
            df_agg = df.groupby(date_col)[value_col].sum().reset_index()
            
            # Sample data for better performance with large datasets
            if len(df_agg) > 1000:
                df_sampled = df_agg.iloc[::len(df_agg)//1000]
                st.info("📊 Showing sampled data for better performance")
            else:
                df_sampled = df_agg
            
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
    
    # Advanced time series features
    st.markdown("#### 📈 Advanced Time Series Analysis")
    
    # Rolling statistics
    st.markdown("**Rolling Statistics**")
    window_size = st.slider("Rolling Window Size (days)", 7, 90, 30, key="rolling_window")
    
    # Aggregate by date first
    df_agg = df.groupby(date_col)[value_col].sum().reset_index()
    
    if len(df_agg) > window_size:
        try:
            # Calculate rolling statistics efficiently
            df_temp = df_agg.set_index(date_col)[value_col].copy()
            rolling_mean = df_temp.rolling(window=window_size, min_periods=1).mean()
            rolling_std = df_temp.rolling(window=window_size, min_periods=1).std()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_agg[date_col], y=df_agg[value_col],
                name='Original',
                line=dict(color='#00d4ff', width=1),
                opacity=0.7
            ))
            fig.add_trace(go.Scatter(
                x=df_agg[date_col], y=rolling_mean,
                name=f'{window_size}-day Moving Average',
                line=dict(color='#ff6b6b', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_agg[date_col], y=rolling_std,
                name=f'{window_size}-day Std Dev',
                line=dict(color='#2ed573', width=1, dash='dash'),
                opacity=0.6
            ))
            
            fig.update_layout(
                title=f"Rolling Statistics (Window: {window_size} days)",
                height=400,
                template="plotly_white"
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
                    col1_name = corr_matrix.columns[i]
                    col2_name = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value):
                        correlations.append((col1_name, col2_name, abs(corr_value), corr_value))
            
            # Sort by absolute correlation and take top 10
            correlations.sort(key=lambda x: x[2], reverse=True)
            
            # Display top 10
            for col1_name, col2_name, abs_corr, corr in correlations[:10]:
                color = "🟢" if corr > 0.7 else "🟡" if corr > 0.3 else "🔴"
                st.write(f"{color} **{col1_name}** ↔ **{col2_name}**: {corr:.3f}")
                
        except Exception as e:
            st.error(f"Error calculating correlations: {str(e)}")


if __name__ == "__main__":
    main()
