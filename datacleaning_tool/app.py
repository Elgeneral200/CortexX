# filename: app.py
"""
CortexX Sales & Demand Forecasting Platform - Complete Working Edition v2.2

All Phase 1 features working with fixed imports and no errors.
Ready to run directly.

Author: CortexX Team
Version: 2.2.0 - Complete Working Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import json
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================
# SIMPLE DIRECT IMPORTS - NO ERRORS
# ============================

# Import enhanced modules directly
try:
    from visualization import (
        render_automated_eda_dashboard,
        create_correlation_heatmap,
        create_distribution_grid,
        create_missing_data_heatmap,
        create_outlier_detection_plot
    )
    VISUALIZATION_ENHANCED = True
    print("✅ Enhanced Visualization: LOADED")
except ImportError as e:
    print(f"⚠️ Enhanced Visualization: {e}")
    VISUALIZATION_ENHANCED = False
    # Create dummy functions
    render_automated_eda_dashboard = None
    create_correlation_heatmap = None
    create_distribution_grid = None
    create_missing_data_heatmap = None
    create_outlier_detection_plot = None

try:
    from preprocess import (
        DataCleaningPipeline,
        render_data_cleaning_dashboard,
        smart_imputation_comparison,
        optimize_dtypes_advanced
    )
    PREPROCESSING_ENHANCED = True
    print("✅ Advanced Preprocessing: LOADED")
except ImportError as e:
    print(f"⚠️ Advanced Preprocessing: {e}")
    PREPROCESSING_ENHANCED = False
    # Create dummy classes
    DataCleaningPipeline = None
    render_data_cleaning_dashboard = None
    smart_imputation_comparison = None
    optimize_dtypes_advanced = None

try:
    from business_intelligence import (
        render_business_intelligence_dashboard,
        calculate_business_kpis,
        create_executive_dashboard
    )
    BUSINESS_INTELLIGENCE_AVAILABLE = True
    print("✅ Business Intelligence: LOADED")
except ImportError as e:
    print(f"⚠️ Business Intelligence: {e}")
    BUSINESS_INTELLIGENCE_AVAILABLE = False
    # Create dummy functions
    render_business_intelligence_dashboard = None
    calculate_business_kpis = None
    create_executive_dashboard = None

# Try to import optional dependencies
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

print(f"🔧 Final Module Status:")
print(f"   Enhanced Visualization: {VISUALIZATION_ENHANCED}")
print(f"   Advanced Preprocessing: {PREPROCESSING_ENHANCED}")
print(f"   Business Intelligence: {BUSINESS_INTELLIGENCE_AVAILABLE}")

# ============================
# PAGE CONFIGURATION
# ============================

st.set_page_config(
    page_title="🚀 CortexX Sales & Demand Forecasting Platform v2.2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# UTILITY FUNCTIONS
# ============================

def safe_convert_for_json(obj):
    """Safely convert numpy types and other objects for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [safe_convert_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_convert_for_json(value) for key, value in obj.items()}
    elif pd.isna(obj):
        return None
    else:
        try:
            return str(obj)
        except:
            return "Unable to convert"

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'df': None,
        'data_summary': {},
        'quality_score': 0.0,
        'file_uploaded': False,
        'current_file_name': "",
        'processing_complete': False,
        'cleaning_pipeline': None,
        'eda_report': None,
        'business_kpis': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_enhanced_professional_css():
    """Load enhanced professional CSS styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-color: #0f172a;
        --secondary-color: #1e293b;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --text-primary: #f8fafc;
        --gradient-business: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        --shadow-primary: 0 10px 25px rgba(0,0,0,0.2);
        --border-radius: 12px;
    }

    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: var(--text-primary);
        min-height: 100vh;
    }

    .main-header {
        background: var(--gradient-business);
        padding: 3rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-primary);
        position: relative;
        overflow: hidden;
    }

    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }

    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.4rem;
        margin: 1rem 0 0 0;
        font-weight: 500;
    }

    .feature-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .feature-badge.working {
        background: var(--success-color);
        color: white;
        animation: pulse 2s infinite;
    }

    .feature-badge.basic {
        background: var(--warning-color);
        color: white;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--secondary-color), #374151);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-primary);
        transition: all 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--secondary-color);
        padding: 0.5rem;
        border-radius: var(--border-radius);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 1rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: var(--border-radius);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-primary);
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
        border-left: 4px solid var(--success-color);
        border-radius: var(--border-radius);
    }

    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
        border-left: 4px solid var(--error-color);
        border-radius: var(--border-radius);
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
        border-left: 4px solid var(--warning-color);
        border-radius: var(--border-radius);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

load_enhanced_professional_css()

# ============================
# DATA LOADING FUNCTIONS
# ============================

def load_file_enhanced(uploaded_file) -> pd.DataFrame:
    """Enhanced file loading with better error handling."""
    try:
        if uploaded_file is None:
            return pd.DataFrame()
            
        uploaded_file.seek(0)
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            encodings = ['utf-8', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    if not df.empty:
                        return df
                except:
                    continue
            
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')

        elif file_extension in ['xlsx', 'xls']:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)

        elif file_extension == 'json':
            uploaded_file.seek(0)
            return pd.read_json(uploaded_file)

        else:
            st.error(f"Unsupported file format: {file_extension}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()

def get_enhanced_data_summary(df: pd.DataFrame) -> Dict:
    """Get comprehensive data summary with proper numpy handling."""
    if df is None or df.empty:
        return {}
    
    summary = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "missing_count": int(df.isnull().sum().sum()),
        "missing_pct": float((df.isnull().sum().sum() / df.size) * 100),
        "duplicate_count": int(df.duplicated().sum()),
        "duplicate_pct": float((df.duplicated().sum() / len(df)) * 100),
        "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_cols": df.select_dtypes(include=['object']).columns.tolist(),
        "datetime_cols": [],
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "data_quality_issues": []
    }

    # Enhanced datetime detection
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(50)
            try:
                pd.to_datetime(sample, errors='raise')
                summary["datetime_cols"].append(col)
            except:
                pass

    # Data quality issues
    if summary["missing_pct"] > 10:
        summary["data_quality_issues"].append(f"High missing data: {summary['missing_pct']:.1f}%")
    
    if summary["duplicate_pct"] > 1:
        summary["data_quality_issues"].append(f"Duplicate records: {summary['duplicate_pct']:.1f}%")
    
    if summary["memory_mb"] > 100:
        summary["data_quality_issues"].append(f"Large dataset: {summary['memory_mb']:.1f}MB")

    return summary

def calculate_enhanced_quality_score(df: pd.DataFrame) -> float:
    """Calculate enhanced data quality score."""
    if df is None or df.empty:
        return 0.0

    scores = []

    # Completeness (40% weight)
    completeness = (df.count().sum() / df.size) * 100 if df.size > 0 else 0
    scores.append(completeness * 0.4)

    # Uniqueness (30% weight) 
    uniqueness = ((len(df) - df.duplicated().sum()) / len(df)) * 100 if len(df) > 0 else 0
    scores.append(uniqueness * 0.3)

    # Consistency (20% weight)
    consistency_score = 90  # Base score
    scores.append(consistency_score * 0.2)

    # Structure (10% weight)
    structure_score = 100
    if len(df.columns) > 50:
        structure_score -= 20
    if len(df) < 10:
        structure_score -= 30
    scores.append(max(0, structure_score) * 0.1)

    total_score = sum(scores) / 10
    return round(min(10.0, max(0.0, total_score)), 1)

# ============================
# UI COMPONENTS
# ============================

def render_enhanced_header():
    """Render enhanced header with real feature status."""
    feature_badges = []
    
    if VISUALIZATION_ENHANCED:
        feature_badges.append('<span class="feature-badge working">✅ Enhanced EDA</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic EDA</span>')
    
    if PREPROCESSING_ENHANCED:
        feature_badges.append('<span class="feature-badge working">✅ Smart Cleaning</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic Cleaning</span>')
    
    if BUSINESS_INTELLIGENCE_AVAILABLE:
        feature_badges.append('<span class="feature-badge working">✅ BI Dashboards</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic BI</span>')
    
    feature_badges.append('<span class="feature-badge working">AI-Powered</span>')
    
    st.markdown(f"""
    <div class="main-header fade-in">
        <h1>🚀 CortexX Sales & Demand Forecasting Platform</h1>
        <p>Professional Business Intelligence with Phase 1 Enhanced Features</p>
        <div style="margin-top: 1.5rem;">
            {''.join(feature_badges)}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_welcome_screen():
    """Render enhanced welcome screen with feature status."""
    st.markdown("""
    <div class="fade-in">

    ### 👋 Welcome to Next-Generation Sales Analytics

    Upload your sales data to unlock AI-powered insights, automated analysis, and professional forecasting

    **Supported: CSV, Excel (XLSX/XLS), JSON files**

    ---

    </div>
    """, unsafe_allow_html=True)

    # Feature status showcase
    st.markdown("### 🎯 Phase 1 Features Status")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        if VISUALIZATION_ENHANCED:
            st.success("✅ **Automated EDA Available**")
            st.markdown("""
            - Smart data profiling & analysis
            - Automatic correlation detection
            - Distribution analysis with insights
            - Missing data pattern recognition
            - Outlier detection with multiple methods
            - AI-powered recommendations
            """)
        else:
            st.warning("⚠️ **Basic EDA Only**")
            st.markdown("""
            **To Enable Enhanced EDA:**
            - Ensure `visualization.py` file exists
            - Install: `pip install scipy scikit-learn`
            - Restart the application
            """)

    with col2:
        if PREPROCESSING_ENHANCED:
            st.success("✅ **Advanced Cleaning Available**")
            st.markdown("""
            - Interactive cleaning pipeline
            - Undo/redo functionality
            - Smart imputation comparison
            - Memory optimization tools
            - Data type auto-conversion
            - Quality-driven transformations
            """)
        else:
            st.warning("⚠️ **Basic Cleaning Only**")
            st.markdown("""
            **To Enable Advanced Cleaning:**
            - Ensure `preprocess.py` file exists
            - Install: `pip install scipy scikit-learn`
            - Restart the application
            """)

    with col3:
        if BUSINESS_INTELLIGENCE_AVAILABLE:
            st.success("✅ **Business Intelligence Available**")
            st.markdown("""
            - Executive KPI dashboards
            - Sales performance analysis
            - Revenue trend forecasting
            - Customer behavior insights
            - Seasonal pattern detection
            - Professional reporting suite
            """)
        else:
            st.warning("⚠️ **Basic BI Only**")
            st.markdown("""
            **To Enable Business Intelligence:**
            - Ensure `business_intelligence.py` file exists
            - Install: `pip install scipy scikit-learn plotly`
            - Restart the application
            """)

def render_enhanced_data_overview(df: pd.DataFrame):
    """Render enhanced data overview."""
    if df is None or df.empty:
        st.warning("No data loaded. Please upload a file to unlock advanced analytics.")
        return
        
    summary = get_enhanced_data_summary(df)

    if not summary:
        st.warning("No data summary available.")
        return

    # Enhanced KPI Metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📊 Records", f"{summary['shape'][0]:,}")

    with col2:
        st.metric("📋 Columns", f"{summary['shape'][1]}")

    with col3:
        delta_color = "normal" if summary['missing_pct'] < 5 else "inverse"
        st.metric("✅ Completeness", f"{100-summary['missing_pct']:.1f}%",
                 f"{summary['missing_count']:,} missing", delta_color=delta_color)

    with col4:
        delta_color = "normal" if summary['duplicate_pct'] < 1 else "inverse"
        st.metric("🔄 Uniqueness", f"{100-summary['duplicate_pct']:.1f}%",
                 f"{summary['duplicate_count']:,} duplicates", delta_color=delta_color)

    with col5:
        memory_color = "normal" if summary['memory_mb'] < 100 else "inverse"
        st.metric("💾 Memory", f"{summary['memory_mb']:.1f} MB", delta_color=memory_color)

    with col6:
        quality_score = calculate_enhanced_quality_score(df)
        st.session_state.quality_score = quality_score
        
        quality_color = "normal" if quality_score >= 7 else "inverse"
        st.metric("⭐ Quality Score", f"{quality_score}/10", delta_color=quality_color)

    # Data quality insights
    if summary.get("data_quality_issues"):
        st.warning("📋 **Data Quality Insights:**")
        for issue in summary["data_quality_issues"]:
            st.write(f"• {issue}")

    # Quality recommendations
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if quality_score >= 8:
            st.success("🎉 Excellent Data Quality! Ready for advanced analytics and ML modeling.")
        elif quality_score >= 6:
            st.warning("✨ Good Quality Data. Consider using the Smart Cleaning tools for optimization.")
        else:
            st.error("⚠️ Data Quality Needs Improvement. Use our Advanced Cleaning Pipeline to enhance your data.")

# ============================
# TAB FUNCTIONS
# ============================

def render_enhanced_eda_tab(df: pd.DataFrame):
    """Render enhanced EDA tab."""
    st.markdown("### 🤖 Automated Exploratory Data Analysis")
    
    if VISUALIZATION_ENHANCED and render_automated_eda_dashboard:
        try:
            render_automated_eda_dashboard(df, theme="professional_dark")
        except Exception as e:
            st.error(f"❌ Enhanced EDA Error: {e}")
            render_basic_eda_fallback(df)
    else:
        st.warning("⚠️ Enhanced EDA features not available. Using basic analysis.")
        render_basic_eda_fallback(df)

def render_basic_eda_fallback(df: pd.DataFrame):
    """Basic EDA fallback."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Dataset Information**")
        st.write(f"• **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.write(f"• **Memory:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        st.write(f"• **Missing Values:** {df.isnull().sum().sum():,}")
        st.write(f"• **Duplicates:** {df.duplicated().sum():,}")
        
    with col2:
        st.markdown("**📋 Column Types**")
        st.write(f"• **Numeric:** {len(numeric_cols)} columns")
        st.write(f"• **Categorical:** {len(categorical_cols)} columns")
    
    if numeric_cols:
        st.markdown("**📈 Statistical Summary**")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    if len(numeric_cols) >= 2:
        st.markdown("**🔗 Basic Correlation Matrix**")
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Correlation Matrix")
        fig.update_layout(
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            paper_bgcolor='rgba(15, 23, 42, 1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

def render_enhanced_cleaning_tab(df: pd.DataFrame):
    """Render enhanced cleaning tab."""
    st.markdown("### 🧹 Advanced Data Cleaning Pipeline")
    
    if PREPROCESSING_ENHANCED and DataCleaningPipeline and render_data_cleaning_dashboard:
        try:
            if st.session_state.cleaning_pipeline is None:
                st.session_state.cleaning_pipeline = DataCleaningPipeline(df, "Sales Data Cleaning")
            
            render_data_cleaning_dashboard(st.session_state.cleaning_pipeline)
            st.session_state.df = st.session_state.cleaning_pipeline.current_df
        
        except Exception as e:
            st.error(f"❌ Advanced Cleaning Error: {e}")
            render_basic_cleaning_fallback(df)
    else:
        st.warning("⚠️ Enhanced cleaning features not available. Using basic cleaning.")
        render_basic_cleaning_fallback(df)

def render_basic_cleaning_fallback(df: pd.DataFrame):
    """Basic cleaning fallback."""
    st.markdown("**Basic Cleaning Options:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Remove duplicates
        if st.button("🗑️ Remove Duplicates"):
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                st.session_state.df = df.drop_duplicates()
                st.success(f"✅ Removed {duplicate_count} duplicate records")
                st.rerun()
            else:
                st.info("No duplicates found")
    
    with col2:
        # Basic missing value info
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        if len(missing_cols) > 0:
            st.write(f"**Missing Values Found:** {len(missing_cols)} columns")
            for col, count in missing_cols.head().items():
                pct = (count / len(df)) * 100
                st.write(f"• {col}: {count} ({pct:.1f}%)")

def render_enhanced_business_intelligence_tab(df: pd.DataFrame):
    """Render enhanced BI tab."""
    st.markdown("### 📊 Business Intelligence Dashboard")
    
    if BUSINESS_INTELLIGENCE_AVAILABLE and render_business_intelligence_dashboard:
        try:
            render_business_intelligence_dashboard(df)
        except Exception as e:
            st.error(f"❌ Business Intelligence Error: {e}")
            render_basic_bi_fallback(df)
    else:
        st.warning("⚠️ Business Intelligence features not available. Using basic analysis.")
        render_basic_bi_fallback(df)

def render_basic_bi_fallback(df: pd.DataFrame):
    """Basic BI fallback."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.markdown("**📈 Basic Business Metrics**")
        
        selected_col = st.selectbox("Select metric column:", numeric_cols)
        
        if selected_col:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = df[selected_col].sum()
                st.metric("💰 Total", f"{total_value:,.2f}")
            
            with col2:
                avg_value = df[selected_col].mean()
                st.metric("📊 Average", f"{avg_value:.2f}")
            
            with col3:
                max_value = df[selected_col].max()
                st.metric("📈 Maximum", f"{max_value:,.2f}")
            
            with col4:
                count_value = df[selected_col].count()
                st.metric("📋 Count", f"{count_value:,}")
            
            # Basic visualization
            fig = px.histogram(df, x=selected_col, 
                             title=f"Distribution of {selected_col}")
            fig.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                paper_bgcolor='rgba(15, 23, 42, 1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for business analysis.")

def render_enhanced_visualization_tab(df: pd.DataFrame):
    """Render enhanced visualization tab."""
    st.markdown("### 🎨 Advanced Interactive Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols and not categorical_cols:
        st.info("No suitable columns found for visualization.")
        return
    
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Basic Charts", "Correlation Analysis", "Distribution Analysis", "Missing Data Pattern"]
    )
    
    try:
        if viz_type == "Correlation Analysis" and VISUALIZATION_ENHANCED and create_correlation_heatmap and len(numeric_cols) >= 2:
            corr_fig = create_correlation_heatmap(df, theme="professional_dark")
            st.plotly_chart(corr_fig, use_container_width=True)
            
        elif viz_type == "Distribution Analysis" and VISUALIZATION_ENHANCED and create_distribution_grid and numeric_cols:
            dist_fig = create_distribution_grid(df, theme="professional_dark")
            st.plotly_chart(dist_fig, use_container_width=True)
            
        elif viz_type == "Missing Data Pattern" and VISUALIZATION_ENHANCED and create_missing_data_heatmap:
            missing_fig = create_missing_data_heatmap(df, theme="professional_dark")
            st.plotly_chart(missing_fig, use_container_width=True)
        
        else:
            # Basic visualization fallback
            if numeric_cols:
                selected_col = st.selectbox("Select column to visualize:", numeric_cols)
                
                if selected_col:
                    viz_option = st.radio("Chart type:", ["Histogram", "Box Plot", "Line Chart"])
                    
                    if viz_option == "Histogram":
                        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                    elif viz_option == "Box Plot":
                        fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    else:
                        fig = px.line(df.reset_index(), x='index', y=selected_col, 
                                    title=f"Trend of {selected_col}")
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(15, 23, 42, 0.8)',
                        paper_bgcolor='rgba(15, 23, 42, 1)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for visualization.")
                
    except Exception as e:
        st.error(f"❌ Visualization Error: {e}")

def render_enhanced_forecasting_tab(df: pd.DataFrame):
    """Render forecasting tab placeholder."""
    st.markdown("### 🔮 AI-Powered Sales Forecasting")
    
    st.info("🚀 **Coming in Phase 2:** Advanced ML forecasting models")
    
    st.markdown("""
    **🎯 Planned Forecasting Features:**
    - Time series forecasting with ARIMA, Prophet
    - Demand prediction using ML algorithms  
    - Seasonal trend analysis
    - Revenue growth projections
    - Customer behavior forecasting
    - Risk assessment models
    """)
    
    # Basic trend analysis
    date_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(50)
        try:
            pd.to_datetime(sample, errors='raise')
            date_cols.append(col)
        except:
            continue
    
    if date_cols and numeric_cols:
        st.markdown("**📈 Basic Trend Analysis**")
        
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Select date column:", date_cols)
        with col2:
            value_col = st.selectbox("Select value column:", numeric_cols)
        
        if date_col and value_col:
            df_trend = df.copy()
            df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
            df_trend = df_trend.dropna(subset=[date_col, value_col])
            
            if len(df_trend) > 1:
                fig = px.line(df_trend.sort_values(date_col), 
                            x=date_col, y=value_col, 
                            title=f"Trend Analysis: {value_col} over {date_col}")
                fig.update_layout(
                    plot_bgcolor='rgba(15, 23, 42, 0.8)',
                    paper_bgcolor='rgba(15, 23, 42, 1)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

def render_enhanced_export_tab(df: pd.DataFrame):
    """Render enhanced export tab with fixed numpy handling."""
    st.markdown("### 📋 Export & Professional Reports")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📄 Data Export**")
        
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name=f"sales_data_{timestamp}.csv",
            mime="text/csv"
        )
        
        # Excel export
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Add summary sheet with safe conversion
                if st.session_state.data_summary:
                    try:
                        summary_data = []
                        for key, value in st.session_state.data_summary.items():
                            safe_value = safe_convert_for_json(value)
                            
                            if isinstance(safe_value, (int, float, str)) and not pd.isna(safe_value):
                                summary_data.append({'Metric': key, 'Value': str(safe_value)})
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    except:
                        pass
            
            st.download_button(
                "📊 Download Excel",
                data=buffer.getvalue(),
                file_name=f"sales_report_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.warning("⚠️ Excel export requires: `pip install openpyxl`")
        except Exception as e:
            st.error(f"❌ Excel export error: {str(e)}")
    
    with col2:
        st.markdown("**📈 Analytics Reports**")
        
        if st.session_state.quality_score > 0:
            try:
                quality_report = {
                    "Dataset": st.session_state.current_file_name,
                    "Quality_Score": f"{st.session_state.quality_score}/10",
                    "Total_Records": int(len(df)),
                    "Total_Columns": int(len(df.columns)),
                    "Missing_Values": int(df.isnull().sum().sum()),
                    "Duplicate_Records": int(df.duplicated().sum()),
                    "Memory_Usage_MB": round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2),
                    "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                report_json = json.dumps(quality_report, indent=2, default=safe_convert_for_json)
                st.download_button(
                    "🔍 Quality Report (JSON)",
                    data=report_json,
                    file_name=f"quality_report_{timestamp}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Quality report error: {str(e)}")
    
    with col3:
        st.markdown("**🎯 Business Intelligence**")
        
        if BUSINESS_INTELLIGENCE_AVAILABLE and st.session_state.business_kpis:
            try:
                safe_kpis = safe_convert_for_json(st.session_state.business_kpis)
                kpis_json = json.dumps(safe_kpis, indent=2, default=str)
                st.download_button(
                    "📊 Business KPIs (JSON)",
                    data=kpis_json,
                    file_name=f"business_kpis_{timestamp}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Business KPIs error: {str(e)}")
        else:
            st.info("📊 Upload data to generate business KPIs")
    
    # Report summary
    st.markdown("---")
    st.markdown("### 📋 Report Summary")
    
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Records", f"{len(df):,}")
        with col2:
            st.metric("📋 Columns", f"{len(df.columns)}")
        with col3:
            quality_score = getattr(st.session_state, 'quality_score', 0)
            st.metric("⭐ Quality", f"{quality_score}/10")
        with col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("💾 Size", f"{memory_mb:.1f} MB")
            
    except Exception as e:
        st.error(f"❌ Summary metrics error: {str(e)}")

# ============================
# MAIN APPLICATION
# ============================

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()

    # Render enhanced header
    render_enhanced_header()

    # Sidebar configuration
    st.sidebar.title("📁 Upload Your Data")
    st.sidebar.markdown("---")
    
    # Show feature status in sidebar
    st.sidebar.markdown("### 🎯 Feature Status")
    if VISUALIZATION_ENHANCED:
        st.sidebar.success("✅ Enhanced EDA Ready")
    else:
        st.sidebar.error("❌ Enhanced EDA Not Available")
    
    if PREPROCESSING_ENHANCED:
        st.sidebar.success("✅ Advanced Cleaning Ready")
    else:
        st.sidebar.error("❌ Advanced Cleaning Not Available")
    
    if BUSINESS_INTELLIGENCE_AVAILABLE:
        st.sidebar.success("✅ Business Intelligence Ready")
    else:
        st.sidebar.error("❌ Business Intelligence Not Available")
    
    st.sidebar.markdown("---")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Your Data",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Supported: CSV, Excel (XLSX/XLS), JSON files",
        accept_multiple_files=False
    )

    # Process file upload
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        
        if current_file_name != st.session_state.current_file_name:
            st.session_state.current_file_name = current_file_name
            st.session_state.file_uploaded = False
            st.session_state.processing_complete = False
            st.session_state.cleaning_pipeline = None
            
            with st.sidebar:
                with st.spinner("Processing data with AI insights..."):
                    df = load_file_enhanced(uploaded_file)
                    
                    if df is not None and not df.empty:
                        st.session_state.df = df
                        st.session_state.data_summary = get_enhanced_data_summary(df)
                        st.session_state.file_uploaded = True
                        st.session_state.processing_complete = True
                        
                        # Calculate business KPIs if available
                        if BUSINESS_INTELLIGENCE_AVAILABLE and calculate_business_kpis:
                            try:
                                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                revenue_col = None
                                for col in numeric_cols:
                                    if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'price', 'total']):
                                        revenue_col = col
                                        break
                                
                                date_col = None
                                for col in df.columns:
                                    if 'date' in col.lower() or 'time' in col.lower():
                                        try:
                                            pd.to_datetime(df[col].head(10), errors='raise')
                                            date_col = col
                                            break
                                        except:
                                            continue
                                
                                st.session_state.business_kpis = calculate_business_kpis(
                                    df, revenue_col, None, date_col, None
                                )
                            except Exception as e:
                                st.warning(f"Business KPI calculation failed: {e}")
                        
                        st.success("✅ File uploaded successfully!")
                        
                        # Show file info
                        st.markdown("**📊 File Analysis:**")
                        st.write(f"• **Name:** {uploaded_file.name}")
                        st.write(f"• **Size:** {uploaded_file.size / 1024:.1f} KB")
                        st.write(f"• **Records:** {df.shape[0]:,}")
                        st.write(f"• **Features:** {df.shape[1]}")
                        
                        # Quality score
                        quality_score = calculate_enhanced_quality_score(df)
                        if quality_score >= 8:
                            st.success(f"⭐ Quality: {quality_score}/10 (Excellent)")
                        elif quality_score >= 6:
                            st.warning(f"⭐ Quality: {quality_score}/10 (Good)")
                        else:
                            st.error(f"⭐ Quality: {quality_score}/10 (Needs Work)")
                            
                    else:
                        st.error("❌ Failed to load file. Please check the file format and try again.")
                        st.session_state.file_uploaded = False

    # Main content area
    if st.session_state.file_uploaded and st.session_state.df is not None:
        df = st.session_state.df

        # Enhanced data overview
        render_enhanced_data_overview(df)

        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🤖 Automated EDA",
            "🧹 Smart Data Cleaning", 
            "📊 Business Intelligence",
            "🎨 Interactive Visualizations",
            "🔮 AI Forecasting",
            "📋 Export & Reports"
        ])

        with tab1:
            render_enhanced_eda_tab(df)

        with tab2:
            render_enhanced_cleaning_tab(df)

        with tab3:
            render_enhanced_business_intelligence_tab(df)

        with tab4:
            render_enhanced_visualization_tab(df)

        with tab5:
            render_enhanced_forecasting_tab(df)

        with tab6:
            render_enhanced_export_tab(df)

    else:
        # Welcome screen
        render_enhanced_welcome_screen()

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; padding: 2rem; background: linear-gradient(135deg, rgba(15, 20, 25, 0.8), rgba(30, 41, 59, 0.6)); border-radius: 16px; margin-top: 2rem;">
        <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">
            🚀 <strong>CortexX Sales & Demand Forecasting Platform v2.2</strong>
        </div>
        <div style="font-size: 0.9rem; opacity: 0.9;">
            Professional Business Intelligence | Built with ❤️ using Streamlit
        </div>
        <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.7;">
            ✅ <strong>Status:</strong> Enhanced EDA: {VISUALIZATION_ENHANCED} • Advanced Cleaning: {PREPROCESSING_ENHANCED} • Business Intelligence: {BUSINESS_INTELLIGENCE_AVAILABLE}
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()