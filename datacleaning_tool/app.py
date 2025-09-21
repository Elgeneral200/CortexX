# filename: app.py
"""
CortexX Sales & Demand Forecasting Platform - Complete Working Edition v2.4

Phase 1+:
- Safe label/annotation integration across plots (in visualization.py)
- ECDF and QQ-Plot integration into Streamlit (this file)
- Optional improved correlation heatmap controls (mask/threshold)
- Backwards compatible; no data-logic changes

Author: CortexX Team
Version: 2.4.0 - Stable (Baseline + Interactive stability fixes)
"""

import streamlit as st
from PIL import Image
import base64
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
import uuid  # already present in your baseline

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
        create_outlier_detection_plot,
        # NEW: added safely
        create_ecdf_plot,
        create_qq_plot,
        theme_manager
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
    create_ecdf_plot = None
    create_qq_plot = None

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
    page_title="CortexX Sales & Demand Forecasting Platform",
    page_icon=Image.open(r"D:\cortexx-forecasting\datacleaning_tool\assets\logo.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# ENHANCED DARK THEME STYLING
# ============================

def load_enterprise_dark_theme():
    """Load world-class enterprise dark theme CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Dark Theme */
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0f172a;
    }
    
    /* Header Styles */
    .enterprise-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
        border: 1px solid #334155;
    }
    
    .enterprise-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.1;
    }
    
    .enterprise-header h1 {
        color: #e2e8f0;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }
    
    .enterprise-header p {
        color: #94a3b8;
        font-size: 1.2rem;
        margin: 1rem 0 0 0;
        font-weight: 500;
    }
    
    /* Feature Badges */
    .feature-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background: rgba(255,255,255,0.1);
        color: #e2e8f0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
    }
    
    .feature-badge.working {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(16, 185, 129, 0.2));
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    .feature-badge.basic {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.3), rgba(245, 158, 11, 0.2));
        color: #f59e0b;
        border-color: rgba(245, 158, 11, 0.3);
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #1a2438 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-color: #4f6ba8;
    }
    
    [data-testid="metric-container"] .stMetricLabel {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 600;
    }
    
    [data-testid="metric-container"] .stMetricValue {
        font-size: 1.8rem;
        color: #e2e8f0;
        font-weight: 700;
    }
    
    [data-testid="metric-container"] .stMetricDelta {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1e293b;
        border-radius: 8px 8px 0 0;
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: #94a3b8;
        transition: all 0.3s ease;
        border: 1px solid #334155;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #2d3748;
        color: #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #1a2438;
        color: #60a5fa;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.2);
        border-color: #4f6ba8;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Success, Error, Warning */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.1));
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        color: #e2e8f0;
        border-color: #10b981;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.1));
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        color: #e2e8f0;
        border-color: #ef4444;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.1));
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        color: #e2e8f0;
        border-color: #f59e0b;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #e2e8f0;
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
    }
    
    .streamlit-expanderContent {
        background: #1a2438;
        border: 1px solid #334155;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border: 1px solid #334155;
    }
    
    /* Hide elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .fade-in { 
        animation: fadeIn 0.6s ease-in; 
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .float {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e293b 0%, #1a2438 100%);
        color: #e2e8f0;
    }
    
    /* Custom cards */
    .enterprise-card {
        background: linear-gradient(135deg, #1e293b 0%, #1a2438 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
    }
    
    .enterprise-card h3 {
        margin-top: 0;
        color: #e2e8f0;
        font-weight: 700;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.75rem;
    }
    
    /* Footer */
    .enterprise-footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #1e293b 0%, #1a2438 100%);
        border-radius: 12px;
        border: 1px solid #334155;
    }
    
    .enterprise-footer h4 {
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .enterprise-footer p {
        color: #94a3b8;
        margin: 0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #334155;
    }
    
    .status-indicator.available {
        border-left: 4px solid #10b981;
    }
    
    .status-indicator.unavailable {
        border-left: 4px solid #ef4444;
    }
    
    .status-icon {
        margin-right: 0.75rem;
        font-size: 1.2rem;
    }
    
    .status-content {
        flex: 1;
    }
    
    .status-content h4 {
        margin: 0;
        font-size: 0.9rem;
        color: #e2e8f0;
    }
    
    .status-content p {
        margin: 0;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Global company branding */
    .global-watermark {
        position: fixed;
        bottom: 20px;
        right: 20px;
        opacity: 0.1;
        font-size: 5rem;
        font-weight: 900;
        color: #e2e8f0;
        z-index: -1;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Load CSS
load_enterprise_dark_theme()

# Add global watermark
st.markdown('<div class="global-watermark">CORTEXX</div>', unsafe_allow_html=True)

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
    """Initialize session state variables (baseline + minimal persistent UI state)."""
    defaults = {
        'df': None,
        'data_summary': {},
        'quality_score': 0.0,
        'file_uploaded': False,
        'current_file_name': "",
        'processing_complete': False,
        'cleaning_pipeline': None,
        'eda_report': None,
        'business_kpis': {},
        # NEW: persisted selections for stable widgets (non-breaking)
        'viz_session': None,
        'ecdf_column': None,
        'ecdf_norm': 'percent',   # allowed by px.ecdf: 'percent' or 'probability' (or None)
        'qq_column': None,
        # Interactive tab persistent keys/values
        'viz_tab_type': None,
        'basic_col': None,
        'basic_kind': None,
        'corr_mask_upper': False,
        'corr_annotate': False,
        'corr_threshold': 0.8,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state['viz_session'] is None:
        st.session_state['viz_session'] = uuid.uuid4().hex[:8]

def key_of(name: str) -> str:
    """Stable widget key based on a session prefix."""
    return f"{st.session_state['viz_session']}_{name}"

# ============================
# SIDEBAR ENHANCEMENTS
# ============================

def render_enterprise_sidebar():
    """Render enterprise-grade sidebar with enhanced feature status."""
    # Function to encode local image to base64
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except:
            return None

    # Sidebar styling
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1e293b 0%, #1a2438 100%);
            color: #e2e8f0;
        }
        .logo-container {
            text-align: center;
            padding: 1.5rem 0 1rem 0;
            border-bottom: 1px solid #334155;
            margin-bottom: 1.5rem;
        }
        .caption {
            font-size: 1.25rem;
            font-weight: 700;
            color: #60a5fa;
            margin-top: 0.75rem;
            letter-spacing: 0.5px;
        }
        .tagline {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-top: 0.25rem;
            margin-bottom: 0.5rem;
        }
        .about-section {
            background: linear-gradient(135deg, #1e293b 0%, #1a2438 100%);
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            border-left: 4px solid #6366f1;
            margin-top: 1.5rem;
            border: 1px solid #334155;
        }
        .about-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #60a5fa;
            margin-bottom: 0.75rem;
        }
        .about-content {
            font-size: 0.9rem;
            color: #94a3b8;
            line-height: 1.5;
        }
        .sidebar-section {
            margin-bottom: 1.5rem;
        }
        .sidebar-section h3 {
            font-size: 1rem;
            color: #e2e8f0;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Logo display with enhanced styling
    logo_base64 = get_base64_image(r"D:\cortexx-forecasting\datacleaning_tool\assets\logo.png")
    
    if logo_base64:
        st.sidebar.markdown(
            f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_base64}" width="150">
                <div class="caption">CortexX Platform</div>
                <div class="tagline">Enterprise Data Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div class="logo-container">
                <div class="caption">CortexX Platform</div>
                <div class="tagline">Enterprise Data Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # File upload section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Data Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Supported formats: CSV, Excel (XLSX/XLS), JSON",
        label_visibility="collapsed"
    )
    
    # Enhanced Feature Status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Platform Features")
    
    # EDA Status
    if VISUALIZATION_ENHANCED:
        st.sidebar.markdown("""
        <div class="status-indicator available">
            <div class="status-icon">📊</div>
            <div class="status-content">
                <h4>Exploratory Data Analysis</h4>
                <p>Enhanced visualization & analytics</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">📊</div>
            <div class="status-content">
                <h4>Exploratory Data Analysis</h4>
                <p>Basic functionality only</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Cleaning Status
    if PREPROCESSING_ENHANCED:
        st.sidebar.markdown("""
        <div class="status-indicator available">
            <div class="status-icon">🧹</div>
            <div class="status-content">
                <h4>Data Cleaning</h4>
                <p>Advanced preprocessing pipeline</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">🧹</div>
            <div class="status-content">
                <h4>Data Cleaning</h4>
                <p>Basic functionality only</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # BI Status
    if BUSINESS_INTELLIGENCE_AVAILABLE:
        st.sidebar.markdown("""
        <div class="status-indicator available">
            <div class="status-icon">📈</div>
            <div class="status-content">
                <h4>Business Intelligence</h4>
                <p>Advanced analytics & dashboards</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">📈</div>
            <div class="status-content">
                <h4>Business Intelligence</h4>
                <p>Basic functionality only</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="about-section">
            <div class="about-header">Global Enterprise Platform</div>
            <div class="about-content">
                CortexX delivers world-class data intelligence solutions to Fortune 500 companies worldwide. Our AI-powered platform transforms complex data into actionable business insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add a subtle footer with updated year
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; margin-top: 1rem; color: #64748b; font-size: 0.75rem;'>
            © 2025 CortexX Global Inc. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    return uploaded_file

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

def render_enterprise_header():
    """Render enterprise-grade header with real feature status."""
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
    
    feature_badges.append('<span class="feature-badge pulse">🌍 Global Platform</span>')
    
    st.markdown(f"""
    <div class="enterprise-header fade-in">
        <h1>CortexX Enterprise Intelligence Platform</h1>
        <p>World-Class Data Analytics & Forecasting Solutions</p>
        <div style="margin-top: 1.5rem;">
            {''.join(feature_badges)}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_enterprise_welcome_screen():
    """Render enterprise-grade welcome screen with feature status."""
    st.markdown("""
    <div class="fade-in">
    <div class="enterprise-card float">
        <h3>👋 Welcome to CortexX Enterprise Platform</h3>
        <p>Upload your business data to unlock AI-powered insights, predictive analytics, and enterprise-grade forecasting</p>
        <p><strong>Supported: CSV, Excel (XLSX/XLS), JSON files</strong></p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature status showcase
    st.markdown("### 🎯 Enterprise Capabilities")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        card = st.container()
        card.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        if VISUALIZATION_ENHANCED:
            card.success("✅ **Advanced Data Exploration**")
            card.markdown("""
            - Automated data profiling & analysis
            - Intelligent correlation detection
            - Distribution analysis with insights
            - Missing data pattern recognition
            - Multi-method outlier detection
            - AI-powered recommendations
            """)
        else:
            card.warning("⚠️ **Basic Data Exploration**")
            card.markdown("""
            **To Enable Enhanced EDA:**
            - Ensure `visualization.py` file exists
            - Install: `pip install scipy scikit-learn`
            - Restart the application
            """)
        card.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        card = st.container()
        card.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        if PREPROCESSING_ENHANCED:
            card.success("✅ **Intelligent Data Preparation**")
            card.markdown("""
            - Interactive cleaning pipeline
            - Version control with undo/redo
            - Smart imputation comparison
            - Memory optimization tools
            - Data type auto-conversion
            - Quality-driven transformations
            """)
        else:
            card.warning("⚠️ **Basic Data Preparation**")
            card.markdown("""
            **To Enable Advanced Cleaning:**
            - Ensure `preprocess.py` file exists
            - Install: `pip install scipy scikit-learn`
            - Restart the application
            """)
        card.markdown("</div>", unsafe_allow_html=True)

    with col3:
        card = st.container()
        card.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        if BUSINESS_INTELLIGENCE_AVAILABLE:
            card.success("✅ **Business Intelligence Suite**")
            card.markdown("""
            - Executive KPI dashboards
            - Sales performance analysis
            - Revenue trend forecasting
            - Customer behavior insights
            - Seasonal pattern detection
            - Professional reporting suite
            """)
        else:
            card.warning("⚠️ **Basic Business Intelligence**")
            card.markdown("""
            **To Enable Business Intelligence:**
            - Ensure `business_intelligence.py` file exists
            - Install: `pip install scipy scikit-learn plotly`
            - Restart the application
            """)
        card.markdown("</div>", unsafe_allow_html=True)

def render_enterprise_data_overview(df: pd.DataFrame):
    """Render enterprise-grade data overview."""
    if df is None or df.empty:
        st.warning("No data loaded. Please upload a file to unlock advanced analytics.")
        return
        
    summary = get_enhanced_data_summary(df)

    if not summary:
        st.warning("No data summary available.")
        return

    # Enhanced KPI Metrics
    st.markdown("### 📊 Dataset Overview")
    
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

# ============================
# TAB FUNCTIONS (unchanged functionality)
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
    """Render enhanced visualization tab (now with ECDF & QQ-Plot) with persistent widget state."""
    st.markdown("### 🎨 Advanced Interactive Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols and not categorical_cols:
        st.info("No suitable columns found for visualization.")
        return

    # Stable viz_type (init once, then call with key only)
    viz_options = [
        "Basic Charts",
        "Correlation Analysis",
        "Distribution Analysis",
        "Missing Data Pattern",
        "ECDF",
        "QQ-Plot"
    ]
    viz_key = key_of("viz_type")
    if st.session_state.get('viz_tab_type') not in viz_options:
        st.session_state['viz_tab_type'] = viz_options[0]
    viz_type = st.selectbox("Select visualization type:", viz_options, key=viz_key)

    try:
        if viz_type == "Correlation Analysis" and VISUALIZATION_ENHANCED and create_correlation_heatmap and len(numeric_cols) >= 2:
            with st.expander("Options", expanded=False):
                # Stable checkboxes/slider: init once, then call with key only
                mask_key = key_of("corr_mask")
                annot_key = key_of("corr_annot")
                thr_key = key_of("corr_thr")
                if mask_key not in st.session_state:
                    st.session_state[mask_key] = st.session_state.get('corr_mask_upper', False)
                if annot_key not in st.session_state:
                    st.session_state[annot_key] = st.session_state.get('corr_annotate', False)
                if thr_key not in st.session_state:
                    st.session_state[thr_key] = st.session_state.get('corr_threshold', 0.8)

                mask_upper = st.checkbox("Hide upper triangle", key=mask_key)
                annotate = st.checkbox("Annotate strong correlations only", key=annot_key)
                threshold = st.slider("Annotation threshold |r|", 0.5, 0.99, st.session_state[thr_key], 0.01, key=thr_key)

                # mirror to friendly names (non-breaking)
                st.session_state['corr_mask_upper'] = mask_upper
                st.session_state['corr_annotate'] = annotate
                st.session_state['corr_threshold'] = threshold

            corr_fig = create_correlation_heatmap(
                df,
                theme="professional_dark",
                annotate_threshold=st.session_state['corr_threshold'] if st.session_state['corr_annotate'] else None,
                mask_upper=st.session_state['corr_mask_upper']
            )
            st.plotly_chart(corr_fig, use_container_width=True, key=key_of("corr_fig"))
            
        elif viz_type == "Distribution Analysis" and VISUALIZATION_ENHANCED and create_distribution_grid and numeric_cols:
            dist_fig = create_distribution_grid(df, theme="professional_dark")
            st.plotly_chart(dist_fig, use_container_width=True, key=key_of("dist_fig"))
            
        elif viz_type == "Missing Data Pattern" and VISUALIZATION_ENHANCED and create_missing_data_heatmap:
            missing_fig = create_missing_data_heatmap(df, theme="professional_dark")
            st.plotly_chart(missing_fig, use_container_width=True, key=key_of("missing_fig"))
        
        elif viz_type == "ECDF" and VISUALIZATION_ENHANCED and create_ecdf_plot and numeric_cols:
            ecdf_col_key = key_of("ecdf_col")
            ecdf_norm_key = key_of("ecdf_norm")
            # init once
            if ('ecdf_column' not in st.session_state) or (st.session_state['ecdf_column'] not in numeric_cols):
                st.session_state['ecdf_column'] = numeric_cols[0]
            if st.session_state.get('ecdf_norm') not in ["percent", "probability"]:
                st.session_state['ecdf_norm'] = "percent"
            # widgets with key only
            st.selectbox("Select numeric column:", numeric_cols, key=ecdf_col_key)
            st.selectbox("Normalization", ["percent", "probability"], key=ecdf_norm_key)
            # sync friendly names
            st.session_state['ecdf_column'] = st.session_state[ecdf_col_key]
            st.session_state['ecdf_norm'] = st.session_state[ecdf_norm_key]
            ecdf_fig = create_ecdf_plot(df, st.session_state['ecdf_column'], theme="professional_dark", ecdfnorm=st.session_state['ecdf_norm'])
            st.plotly_chart(ecdf_fig, use_container_width=True, key=key_of("ecdf_fig"))
        
        elif viz_type == "QQ-Plot" and VISUALIZATION_ENHANCED and create_qq_plot and numeric_cols:
            qq_col_key = key_of("qq_col")
            if ('qq_column' not in st.session_state) or (st.session_state['qq_column'] not in numeric_cols):
                st.session_state['qq_column'] = numeric_cols[0]
            st.selectbox("Select numeric column:", numeric_cols, key=qq_col_key)
            st.session_state['qq_column'] = st.session_state[qq_col_key]
            qq_fig = create_qq_plot(df, st.session_state['qq_column'], theme="professional_dark")
            st.plotly_chart(qq_fig, use_container_width=True, key=key_of("qq_fig"))
        
        else:
            # Basic visualization fallback
            if numeric_cols:
                basic_col_key = key_of("basic_col")
                basic_kind_key = key_of("basic_kind")
                if (st.session_state.get('basic_col') not in numeric_cols):
                    st.session_state['basic_col'] = numeric_cols[0]
                if st.session_state.get('basic_kind') not in ["Histogram", "Box Plot", "Line Chart"]:
                    st.session_state['basic_kind'] = "Histogram"
                st.selectbox("Select column to visualize:", numeric_cols, key=basic_col_key)
                st.session_state['basic_col'] = st.session_state[basic_col_key]
                st.radio("Chart type:", ["Histogram", "Box Plot", "Line Chart"], key=basic_kind_key)
                st.session_state['basic_kind'] = st.session_state[basic_kind_key]

                if st.session_state['basic_kind'] == "Histogram":
                    fig = px.histogram(df, x=st.session_state['basic_col'], title=f"Distribution of {st.session_state['basic_col']}")
                elif st.session_state['basic_kind'] == "Box Plot":
                    fig = px.box(df, y=st.session_state['basic_col'], title=f"Box Plot of {st.session_state['basic_col']}")
                else:
                    fig = px.line(df.reset_index(), x='index', y=st.session_state['basic_col'], title=f"Trend of {st.session_state['basic_col']}")
                
                fig.update_layout(
                    plot_bgcolor='rgba(15, 23, 42, 0.8)',
                    paper_bgcolor='rgba(15, 23, 42, 1)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True, key=key_of("basic_fig"))
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

    # Render enterprise header
    render_enterprise_header()

    # Enterprise sidebar
    uploaded_file = render_enterprise_sidebar()

    # Process file upload
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        
        if current_file_name != st.session_state.current_file_name:
            st.session_state.current_file_name = current_file_name
            st.session_state.file_uploaded = False
            st.session_state.processing_complete = False
            st.session_state.cleaning_pipeline = None
            
            with st.spinner("🔄 Processing data with AI insights..."):
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

        # Enterprise data overview
        render_enterprise_data_overview(df)

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
        render_enterprise_welcome_screen()

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div class="enterprise-footer">
        <h4>🚀 CortexX Enterprise Intelligence Platform v2.4</h4>
        <p>World-Class Data Analytics | Trusted by Fortune 500 Companies Worldwide</p>
        <p>✅ <strong>Platform Status:</strong> Enhanced EDA: {VISUALIZATION_ENHANCED} • Advanced Cleaning: {PREPROCESSING_ENHANCED} • Business Intelligence: {BUSINESS_INTELLIGENCE_AVAILABLE}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()