"""
CortexX - Enterprise Sales Forecasting Platform
PROFESSIONAL ENTERPRISE EDITION - PHASE 2 COMPLETE

ENHANCED:
- Extracted CSS to external file
- Integrated StateManager
- Uses cached singletons
- Clean, focused main app
- Fixed navigation (file-based only)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import base64

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import configuration and utilities
from src.utils.config import get_config
from src.utils.state_manager import StateManager
from src.data.collection import get_data_collector, generate_sample_data_cached
from src.visualization.dashboard import get_visualizer


# Page configuration - Professional Enterprise Setup
st.set_page_config(
    page_title="CortexX - Enterprise Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    """Load external CSS file for styling."""
    css_file = Path(__file__).parent / 'assets' / 'style.css'
    
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline CSS if file doesn't exist
        st.markdown("""
        <style>
        .main { background-color: #0f1116; color: #ffffff; }
        .section-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
            border-bottom: 1px solid #2d3746;
        }
        .enterprise-card {
            background: linear-gradient(135deg, #1a1d29 0%, #252a3a 100%);
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #2d3746;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
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
        .kpi-card {
            background: linear-gradient(135deg, #1a1d29 0%, #252a3a 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #2d3746;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
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
        </style>
        """, unsafe_allow_html=True)


def initialize_app():
    """
    Initialize application configuration and state.
    
    ENHANCED: Uses config system and StateManager.
    """
    # Initialize configuration
    config = get_config()
    
    # Setup logging (explicit call)
    if not hasattr(st.session_state, '_logging_configured'):
        config.setup_logging()
        st.session_state._logging_configured = True
    
    # Validate configuration
    if not config.validate():
        st.error("⚠️ Configuration validation failed. Check logs for details.")
    
    # Initialize session state
    StateManager.initialize()
    
    # Load CSS
    load_css()


class EnterpriseForecastingApp:
    """
    Professional Enterprise Forecasting Application.
    
    ENHANCED: Cleaner, focused on orchestration only.
    """

    def __init__(self):
        self.config = get_config()
        self.collector = get_data_collector()
        self.visualizer = get_visualizer()
    

    def render_welcome_banner(self):
        logo_path = Path(__file__).parent / "assets" / "logo.png"  # adjust if needed
        logo_b64 = ""
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")

        st.markdown(
            f"""
            <div class="cortexx-welcome-container">
                <div class="cortexx-welcome-logo">
                    <img src="data:image/png;base64,{logo_b64}" alt="CortexX Logo" />
                </div>
                <div>
                    <div class="cortexx-welcome-text-title">
                        Welcome to CortexX Sales Forecasting & Demand Planning Platform
                    </div>
                    <div class="cortexx-welcome-text-subtitle">
                        Enterprise-grade sales forecasting, demand planning, and scenario analytics in one workspace.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_enterprise_header(self):
        """Render professional enterprise header."""
        self.render_welcome_banner()
        st.markdown(
            '<div class="main-header">🧠 CORTEXX ENTERPRISE FORECASTING PLATFORM</div>',
            unsafe_allow_html=True
        )
        
        if StateManager.get('data_loaded'):
            self.render_kpi_dashboard()
        else:
            self.render_welcome_dashboard()

    def render_kpi_dashboard(self):
        """Render enterprise KPI dashboard."""
        df = StateManager.get('current_data')
        if df is None or df.empty:
            return
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate metrics
        metrics = [
            (f"{len(df):,}", "Total Records"),
            (f"{len(df.columns)}", "Features"),
            (f"{len(StateManager.get('trained_models', {}))}", "Trained Models"),
            (f"{df[StateManager.get('value_column')].mean():.0f}" if StateManager.get('value_column') else "-", "Avg Value"),
            (f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%", "Data Quality")
        ]
        
        # Display KPI cards
        for i, (value, label) in enumerate(metrics):
            with [col1, col2, col3, col4, col5][i]:
                st.markdown(f'''
                <div class="kpi-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                ''', unsafe_allow_html=True)

    def render_welcome_dashboard(self):
        """Render professional welcome dashboard."""
        st.markdown('''
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #1a1d29 0%, #252a3a 100%); 
             border-radius: 15px; border: 1px solid #2d3746; margin: 2rem 0;">
            <h2 style="color: #00d4ff; margin-bottom: 1rem;">⚡ ENTERPRISE FORECASTING PLATFORM</h2>
            <p style="color: #8b9bb4; font-size: 1.2rem; margin-bottom: 2rem;">
                Advanced AI-Powered Sales Forecasting
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("### 🚀 ENTERPRISE CAPABILITIES")
        
        features = [
            ("📊", "Advanced Analytics", "Comprehensive EDA with statistical insights and interactive visualizations"),
            ("🤖", "9 ML Models", "XGBoost, LightGBM, Random Forest, CatBoost and more advanced algorithms"),
            ("🔬", "Hyperparameter Tuning", "Automatic optimization with Optuna for maximum performance"),
            ("📈", "Real-time Forecasting", "Live predictions with confidence intervals and uncertainty quantification"),
            ("🔄", "Backtesting", "Walk-forward validation for robust time series evaluation"),
            ("⚡", "Enterprise Scale", "Handles large datasets with optimized performance and scalability")
        ]
        
        cols = st.columns(3)
        for idx, (icon, title, desc) in enumerate(features):
            with cols[idx % 3]:
                st.markdown(f'''
                <div class="enterprise-card">
                    <div class="card-title">{icon} {title}</div>
                    <div class="card-description">{desc}</div>
                </div>
                ''', unsafe_allow_html=True)

    def render_sidebar_status(self):
        """
        Render sidebar status section.
        
        TASK 4: Cleaned up - removed fake navigation.
        """
        st.markdown("---")
        st.markdown("### 📊 SYSTEM STATUS")
        
        # System metrics
        col1, col2 = st.columns(2)
        
        with col1:
            data_status = "✅" if StateManager.get('data_loaded') else "❌"
            st.metric("Data Loaded", data_status)
        
        with col2:
            model_count = len(StateManager.get('trained_models', {}))
            st.metric("Models", f"{model_count}")
        
        # Additional status info
        if StateManager.get('data_loaded'):
            df = StateManager.get('current_data')
            if df is not None:
                st.metric("Records", f"{len(df):,}")
                
                if StateManager.get('best_model_name'):
                    st.success(f"🏆 Best: {StateManager.get('best_model_name')}")

    def render_quick_actions(self):
        """Render quick action buttons."""
        st.markdown("---")
        st.markdown("### ⚡ QUICK ACTIONS")
        
        # Generate sample data
        if st.button("📊 Generate Sample Data", use_container_width=True, key="sidebar_sample"):
            self.generate_sample_data()
        
        # Reset session
        if st.button("🔄 Reset Session", use_container_width=True, key="sidebar_reset"):
            StateManager.clear_all()
            st.rerun()
        
        # Download current data
        if StateManager.get('data_loaded'):
            df = StateManager.get('current_data')
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download Data",
                data=csv,
                file_name="cortexx_data.csv",
                mime="text/csv",
                use_container_width=True,
                key="sidebar_download"
            )

    def generate_sample_data(self):
        """Generate professional sample data using cached function."""
        with st.spinner('Generating enterprise sample data...'):
            try:
                # Use cached function
                df = generate_sample_data_cached(periods=3652, products=5)
                
                # Update state
                StateManager.update({
                    'current_data': df,
                    'data_loaded': True,
                    'date_column': 'date',
                    'value_column': 'sales'
                })
                
                st.success('✅ Enterprise sample data generated!')
                st.rerun()
                
            except Exception as e:
                st.error(f'❌ Error generating sample data: {str(e)}')

    def render_data_upload_section(self):
        """Render professional data upload section."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('''
            <div class="enterprise-card">
                <div class="card-title">📁 UPLOAD YOUR DATA</div>
                <div class="card-description">
                    Upload your sales data in CSV format for enterprise analysis and forecasting
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose CSV File",
                type='csv',
                label_visibility='collapsed'
            )
            
            if uploaded_file is not None:
                try:
                    df = self.collector.load_csv_data(uploaded_file)
                    
                    if not df.empty:
                        StateManager.update({
                            'current_data': df,
                            'data_loaded': True
                        })
                        
                        # Auto-detect columns
                        self.auto_detect_columns(df)
                        
                        st.success(f'✅ Data loaded successfully! {len(df):,} records')
                        st.rerun()
                        
                except Exception as e:
                    st.error(f'❌ Error loading file: {str(e)}')
        
        with col2:
            st.markdown('''
            <div class="enterprise-card">
                <div class="card-title">🚀 QUICK START</div>
                <div class="card-description">
                    Get started instantly with comprehensive sample enterprise data
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            if st.button("📊 GENERATE SAMPLE DATA", use_container_width=True, type="primary"):
                self.generate_sample_data()

    def auto_detect_columns(self, df: pd.DataFrame):
        """Auto-detect date and value columns."""
        date_patterns = ['date', 'time', 'timestamp', 'datetime']
        
        # Detect date column
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col])
                    StateManager.set('date_column', col)
                    break
                except:
                    continue
        
        # Detect value column (first numeric column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            StateManager.set('value_column', numeric_cols[0])

    def render_dashboard_analytics(self):
        """Render professional dashboard analytics."""
        df = StateManager.get('current_data')
        
        st.markdown("### 📈 QUICK INSIGHTS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_col = StateManager.get('date_column')
            if date_col and date_col in df.columns:
                date_range = f"{df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)
        
        with col2:
            value_col = StateManager.get('value_column')
            if value_col and value_col in df.columns:
                total_value = df[value_col].sum()
                st.metric("Total Value", f"{total_value:,.0f}")
        
        with col3:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Quality", f"{completeness:.1f}%")
        
        st.markdown("---")
        st.markdown("### 📊 DATA PREVIEW")
        st.dataframe(df.head(10), use_container_width=True)

    def dashboard_page(self):
        """Main dashboard page."""
        st.markdown(
            '<div class="section-header">🏠 ENTERPRISE DASHBOARD</div>',
            unsafe_allow_html=True
        )
        
        if not StateManager.get('data_loaded'):
            self.render_data_upload_section()
        else:
            self.render_dashboard_analytics()

    def run(self):
        """
        Run the enterprise application.
        
        TASK 4: Simplified - removed fake navigation, file-based pages handle routing.
        """
        # Render sidebar components
        with st.sidebar:
            st.markdown('''
            <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid #2d3746; margin-bottom: 1rem;">
                <h2 style="color: #00d4ff; margin: 0; font-size: 1.8rem;">CORTEXX</h2>
                <p style="color: #8b9bb4; margin: 0; font-size: 0.8rem;">ENTERPRISE EDITION</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # AI Assistant Toggle
            st.markdown("### 💬 CortexX AI Assistant")
            is_open = StateManager.get('chatbot_open', False)
            label = "Hide Assistant" if is_open else "Open Assistant"

            if st.button(label, use_container_width=True, key="toggle_chatbot"):
                StateManager.set('chatbot_open', not is_open)
                st.rerun()
            
            st.markdown("---")  # separator after chatbot section
            
            self.render_sidebar_status()
            self.render_quick_actions()
            
            st.markdown("---")
            st.markdown("**Version** 3.0 Enterprise")
            st.markdown("**Status** 🟢 LIVE")
            st.markdown("**Platform** ✅ Complete")
        
        # Render header and main content
        self.render_enterprise_header()
        self.dashboard_page()



def main():
    """Main entry point."""
    try:
        # Initialize app
        initialize_app()
        
        # Run app
        app = EnterpriseForecastingApp()
        app.run()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
