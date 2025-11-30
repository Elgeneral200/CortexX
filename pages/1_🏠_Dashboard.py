"""
Enterprise Dashboard Page for CortexX Forecasting Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from src.data.collection import DataCollector
    from src.visualization.dashboard import VisualizationEngine, display_plotly_chart
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Dashboard - CortexX",
    page_icon="🏠",
    layout="wide"
)

def main():
    """Main dashboard function."""
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    
    # Header
    st.markdown("""
    <style>
    .dashboard-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-header">🏠 Enterprise Dashboard</div>', unsafe_allow_html=True)
    
    # Data management section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your sales data (CSV)",
            type=['csv'],
            help="Upload a CSV file with your sales data for analysis"
        )
        
        if uploaded_file is not None:
            try:
                if MODULES_AVAILABLE:
                    collector = DataCollector()
                    df = collector.load_csv_data(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                if not df.empty:
                    st.session_state.current_data = df
                    st.session_state.data_loaded = True
                    
                    # Auto-detect columns
                    auto_detect_columns(df)
                    
                    st.success(f"✅ Data loaded successfully! {len(df):,} records, {len(df.columns)} columns")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        st.subheader("🚀 Quick Actions")
        
        if st.button("🎲 Generate Sample Data", use_container_width=True, type="primary"):
            generate_sample_data()
        
        if st.button("🔄 Reset Session", use_container_width=True):
            reset_session()
        
        st.markdown("---")
        st.markdown("**System Status**")
        st.metric("Data Loaded", "✅" if st.session_state.data_loaded else "❌")
        
        if st.session_state.data_loaded:
            df = st.session_state.current_data
            st.metric("Records", f"{len(df):,}")
            st.metric("Features", len(df.columns))
    
    # Display data if loaded
    if st.session_state.data_loaded:
        display_dashboard_analytics()

def auto_detect_columns(df: pd.DataFrame):
    """Auto-detect date and value columns."""
    # Date column detection
    date_patterns = ['date', 'time', 'timestamp', 'datetime']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in date_patterns):
            try:
                df[col] = pd.to_datetime(df[col])
                st.session_state.date_column = col
                st.info(f"📅 Auto-detected date column: {col}")
                break
            except:
                continue
    
    # Value column detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.session_state.value_column = numeric_cols[0]
        st.info(f"💰 Auto-detected value column: {numeric_cols[0]}")

def generate_sample_data():
    """Generate sample data for demonstration."""
    with st.spinner("🔄 Generating enterprise sample data..."):
        try:
            if MODULES_AVAILABLE:
                collector = DataCollector()
                df = collector.generate_sample_data(periods=365*2, products=5)
            else:
                # Fallback sample data
                dates = pd.date_range(start='2022-01-01', periods=730, freq='D')
                data = []
                for product_id in range(1, 6):
                    base_sales = 1000 * product_id
                    trend = np.linspace(0, 500, 730)
                    seasonality = 300 * np.sin(2 * np.pi * np.arange(730) / 365)
                    noise = np.random.randn(730) * 100
                    sales = base_sales + trend + seasonality + noise

                    for i, date in enumerate(dates):
                        data.append({
                            'date': date,
                            'product_id': f'Product_{product_id}',
                            'sales': max(0, sales[i]),
                            'price': np.random.uniform(20, 200),
                            'promotion': np.random.choice([0, 1], p=[0.8, 0.2])
                        })
                df = pd.DataFrame(data)
            
            st.session_state.current_data = df
            st.session_state.data_loaded = True
            st.session_state.date_column = 'date'
            st.session_state.value_column = 'sales'
            st.success("✅ Enterprise sample data generated!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating sample data: {str(e)}")

def reset_session():
    """Reset the session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def display_dashboard_analytics():
    """Display dashboard analytics for loaded data."""
    df = st.session_state.current_data
    visualizer = VisualizationEngine()
    
    # KPI Cards
    st.markdown("### 📈 Business Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_records = len(df)
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        if st.session_state.value_column:
            total_value = df[st.session_state.value_column].sum()
            st.metric("Total Value", f"${total_value:,.0f}")
        else:
            st.metric("Total Value", "-")
    
    with col3:
        if st.session_state.value_column:
            avg_value = df[st.session_state.value_column].mean()
            st.metric("Average Value", f"${avg_value:,.0f}")
        else:
            st.metric("Average Value", "-")
    
    with col4:
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Features", numeric_features)
    
    with col5:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Quality", f"{completeness:.1f}%")
    
    # Visualizations
    st.markdown("### 📊 Data Visualizations")
    
    if st.session_state.date_column and st.session_state.value_column:
        col1, col2 = st.columns(2)
        
        with col1:
            # Time series trend
            try:
                fig = visualizer.create_sales_trend_plot(
                    df, st.session_state.date_column, st.session_state.value_column,
                    "Sales Trend Over Time"
                )
                display_plotly_chart(fig)
            except Exception as e:
                st.error(f"Error creating trend plot: {str(e)}")
        
        with col2:
            # Seasonality analysis
            try:
                fig = visualizer.create_seasonality_plot(
                    df, st.session_state.date_column, st.session_state.value_column
                )
                display_plotly_chart(fig)
            except Exception as e:
                st.error(f"Error creating seasonality plot: {str(e)}")
    
    # Data preview and statistics
    st.markdown("### 📋 Data Details")
    
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Column Info", "Statistics"])
    
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
    
    with tab2:
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab3:
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns available for statistics")

if __name__ == "__main__":
    main()