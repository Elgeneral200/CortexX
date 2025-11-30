"""
CortexX - Enterprise Sales Forecasting Platform
PROFESSIONAL ENTERPRISE EDITION - FINAL VERSION
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
from typing import Optional, Dict, Any
import base64
import io

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from src.data.collection import DataCollector
    from src.data.preprocessing import DataPreprocessor
    from src.data.exploration import DataExplorer
    from src.features.engineering import FeatureEngineer
    from src.features.selection import FeatureSelector
    from src.models.training import ModelTrainer
    from src.models.evaluation import ModelEvaluator
    from src.models.optimization import HyperparameterOptimizer
    from src.models.intervals import PredictionIntervals
    from src.models.backtesting import Backtester
    from src.visualization.dashboard import VisualizationEngine, display_plotly_chart
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}. Running in demo mode.")
    MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration - Professional Enterprise Setup
st.set_page_config(
    page_title="CortexX - Enterprise Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ENTERPRISE DARK THEME CSS WITH ANIMATIONS
st.markdown("""
<style>
    /* Main Theme - Dark Professional */
    .main {
        background-color: #0f1116;
        color: #ffffff;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #2d3746;
        animation: fadeIn 1s ease-in;
    }
    
    /* Enterprise KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1d29 0%, #252a3a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2d3746;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: slideUp 0.5s ease-out;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.3);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00d4ff;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #00d4ff;
        font-weight: 700;
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Professional Cards - Consistent Styling */
    .enterprise-card {
        background: linear-gradient(135deg, #1a1d29 0%, #252a3a 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #2d3746;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: slideUp 0.6s ease-out;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .enterprise-card:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.3);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        font-size: 0.95rem;
        color: #8b9bb4;
        line-height: 1.5;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, #1a1d29 0%, #0f1116 100%) !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4) !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #0099ff 100%) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%) !important;
        border: 1px solid #00b894 !important;
        color: white !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #ff7675 0%, #d63031 100%) !important;
        border: 1px solid #ff7675 !important;
        color: white !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Dataframe Styling */
    .dataframe {
        background: #1a1d29 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

class EnterpriseForecastingApp:
    """Professional Enterprise Forecasting Application"""

    def __init__(self):
        self.initialize_session_state()
        self.visualizer = VisualizationEngine()

    def initialize_session_state(self):
        """Initialize enterprise session state with proper error handling."""
        default_states = {
            'data_loaded': False,
            'current_data': None,
            'date_column': None,
            'value_column': None,
            'trained_models': {},
            'model_results': {},
            'best_model_name': None,
            'current_page': 'Dashboard',
            'backtest_results': {}
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_enterprise_header(self):
        """Render professional enterprise header."""
        st.markdown(
            '<div class="main-header">🚀 CORTEXX ENTERPRISE FORECASTING PLATFORM</div>', 
            unsafe_allow_html=True
        )
        
        if st.session_state.data_loaded:
            self.render_kpi_dashboard()
        else:
            self.render_welcome_dashboard()

    def render_kpi_dashboard(self):
        """Render enterprise KPI dashboard with error handling."""
        df = st.session_state.current_data
        
        if df is None or df.empty:
            return
            
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            (f"{len(df):,}", "Total Records"),
            (f"{len(df.columns)}", "Features"),
            (f"{len(st.session_state.trained_models)}", "Trained Models"),
            (f"${df[st.session_state.value_column].mean():.0f}" if st.session_state.value_column else "-", "Avg Value"),
            (f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%", "Data Quality")
        ]
        
        for i, (value, label) in enumerate(metrics):
            with [col1, col2, col3, col4, col5][i]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    def render_welcome_dashboard(self):
        """Render professional welcome dashboard."""
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #1a1d29 0%, #252a3a 100%); 
                    border-radius: 15px; border: 1px solid #2d3746; margin: 2rem 0;'>
            <h2 style='color: #00d4ff; margin-bottom: 1rem;'>🚀 ENTERPRISE FORECASTING PLATFORM</h2>
            <p style='color: #8b9bb4; font-size: 1.2rem; margin-bottom: 2rem;'>Advanced AI-Powered Sales Forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enterprise Features Grid - EQUAL SIZED CARDS
        st.markdown("### 🎯 ENTERPRISE CAPABILITIES")
        
        features = [
            ("📊", "Advanced Analytics", "Comprehensive EDA with statistical insights and interactive visualizations"),
            ("🤖", "11 ML Models", "XGBoost, LightGBM, Prophet, Random Forest and more advanced algorithms"),
            ("🔬", "Hyperparameter Tuning", "Automatic optimization with Optuna for maximum performance"),
            ("📈", "Real-time Forecasting", "Live predictions with confidence intervals and uncertainty quantification"),
            ("🔄", "Backtesting", "Walk-forward validation for robust time series evaluation"),
            ("🚀", "Enterprise Scale", "Handles large datasets with optimized performance and scalability")
        ]
        
        # Create equal columns for consistent sizing
        cols = st.columns(3)
        for idx, (icon, title, desc) in enumerate(features):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="enterprise-card">
                    <div class="card-title">{icon} {title}</div>
                    <div class="card-description">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    def render_enterprise_sidebar(self):
        """Render professional sidebar navigation."""
        with st.sidebar:
            # Enterprise Branding
            st.markdown("""
            <div style='text-align: center; padding: 1.5rem 0; border-bottom: 1px solid #2d3746; margin-bottom: 1rem;'>
                <h2 style='color: #00d4ff; margin: 0; font-size: 1.8rem;'>CORTEXX</h2>
                <p style='color: #8b9bb4; margin: 0; font-size: 0.8rem;'>ENTERPRISE EDITION</p>
            </div>
            """, unsafe_allow_html=True)
            
            # PROFESSIONAL NAVIGATION - NO "🧭 NAVIGATION" HEADER
            page_options = [
                "🏠 Dashboard",
                "📊 Data Exploration", 
                "⚙️ Feature Engineering",
                "🤖 Model Training",
                "📈 Forecasting",
                "📋 Model Evaluation"
            ]
            
            selected_page = st.radio(
                "SELECT PAGE",
                page_options,
                label_visibility="visible"
            )
            
            st.session_state.current_page = selected_page
            
            st.markdown("---")
            
            # System Status
            st.markdown("### 📊 SYSTEM STATUS")
            
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.metric("Data", "✅" if st.session_state.data_loaded else "❌")
            with status_col2:
                st.metric("Models", f"{len(st.session_state.trained_models)}")
            
            # Quick Actions
            st.markdown("### ⚡ QUICK ACTIONS")
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            
            if st.button("📊 Generate Sample Data", use_container_width=True):
                self.generate_sample_data()
            
            st.markdown("---")
            st.markdown("**Version:** 3.0 Enterprise")
            st.markdown("**Status:** 🔴 **LIVE**")

    def generate_sample_data(self):
        """Generate professional sample data."""
        with st.spinner("🔄 Generating enterprise sample data..."):
            try:
                if MODULES_AVAILABLE:
                    collector = DataCollector()
                    df = collector.generate_sample_data(periods=365*2, products=5)
                else:
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

    def run(self):
        """Run the enterprise application."""
        self.render_enterprise_sidebar()
        self.render_enterprise_header()
        
        # Route to appropriate page
        page = st.session_state.current_page
        
        if page == "🏠 Dashboard":
            self.dashboard_page()
        elif page == "📊 Data Exploration":
            st.info("Navigate to Data Exploration page for detailed analysis")
        elif page == "⚙️ Feature Engineering":
            st.info("Navigate to Feature Engineering page for feature creation")
        elif page == "🤖 Model Training":
            st.info("Navigate to Model Training page to train ML models")
        elif page == "📈 Forecasting":
            st.info("Navigate to Forecasting page for predictions")
        elif page == "📋 Model Evaluation":
            st.info("Navigate to Model Evaluation page for analysis")

    def dashboard_page(self):
        """Professional dashboard page."""
        st.markdown('<div class="section-header">🏠 ENTERPRISE DASHBOARD</div>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            self.render_data_upload_section()
        else:
            self.render_dashboard_analytics()

    def render_data_upload_section(self):
        """Render professional data upload section."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="enterprise-card">
                <div class="card-title">📤 UPLOAD YOUR DATA</div>
                <div class="card-description">Upload your sales data in CSV format for enterprise analysis and forecasting</div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose CSV File",
                type=['csv'],
                label_visibility="collapsed"
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
                        self.auto_detect_columns(df)
                        st.success(f"✅ Data loaded successfully! {len(df):,} records")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        with col2:
            st.markdown("""
            <div class="enterprise-card">
                <div class="card-title">🚀 QUICK START</div>
                <div class="card-description">Get started instantly with comprehensive sample enterprise data</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎲 GENERATE SAMPLE DATA", use_container_width=True, type="primary"):
                self.generate_sample_data()

    def auto_detect_columns(self, df: pd.DataFrame):
        """Auto-detect columns with error handling."""
        date_patterns = ['date', 'time', 'timestamp', 'datetime']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col])
                    st.session_state.date_column = col
                    break
                except:
                    continue
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.session_state.value_column = numeric_cols[0]

    def render_dashboard_analytics(self):
        """Render professional dashboard analytics."""
        df = st.session_state.current_data
        
        # Quick Insights
        st.markdown("### 📈 QUICK INSIGHTS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.date_column:
                date_range = f"{df[st.session_state.date_column].min().strftime('%Y-%m-%d')} to {df[st.session_state.date_column].max().strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)
        
        with col2:
            if st.session_state.value_column:
                total_value = df[st.session_state.value_column].sum()
                st.metric("Total Value", f"${total_value:,.0f}")
        
        with col3:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Quality", f"{completeness:.1f}%")
        
        # Data Preview
        st.markdown("### 📋 DATA PREVIEW")
        st.dataframe(df.head(10), use_container_width=True)

def main():
    """Main entry point."""
    try:
        app = EnterpriseForecastingApp()
        app.run()
    except Exception as e:
        st.error(f"🚨 Application error: {str(e)}")

if __name__ == "__main__":
    main()