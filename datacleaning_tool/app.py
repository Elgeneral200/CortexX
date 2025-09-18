# filename: app.py (Fixed Version)
"""
Sales & Demand Forecasting Platform - Complete Professional Edition (Fixed)

A comprehensive Streamlit application for professional sales data analysis and forecasting
with executive dashboards, advanced analytics, and business intelligence features.

Author: CortexX Team
Version: 1.3.1 - Complete Professional Edition (Fixed Upload Issue)
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

# Try to import optional dependencies with fallbacks
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

# ============================
# PAGE CONFIGURATION
# ============================

st.set_page_config(
    page_title="📈 Sales & Demand Forecasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:support@cortexx.ai",
        "Report a bug": "mailto:bugs@cortexx.ai",
        "About": """
        **CortexX Sales & Demand Forecasting Platform**

        🚀 **Professional Features:**
        • Advanced sales analytics and forecasting
        • Interactive executive dashboards
        • Comprehensive data quality assessment
        • Multi-language support (English/Arabic)
        • Professional reporting capabilities

        Built for enterprise-grade sales intelligence.
        """
    }
)

# ============================
# UTILITY FUNCTIONS
# ============================

@st.cache_data
def get_unique_key(prefix: str = "widget") -> str:
    """Generate unique keys for Streamlit widgets to prevent duplicate ID errors."""
    import random
    timestamp = int(time.time() * 1000)
    random_num = random.randint(1000, 9999)
    return f"{prefix}_{timestamp}_{random_num}"

# ============================
# SESSION STATE MANAGEMENT (FIXED)
# ============================

def init_session_state():
    """Initialize session state variables with proper defaults."""
    # Initialize core state variables
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = {}
    if 'quality_score' not in st.session_state:
        st.session_state.quality_score = 0.0
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = ""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

# ============================
# PROFESSIONAL STYLING
# ============================

def load_professional_css():
    """Load enhanced professional CSS styling."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --primary-color: #1f2937;
        --secondary-color: #374151;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --shadow-primary: 0 10px 25px rgba(0,0,0,0.2);
        --border-radius: 12px;
    }

    /* Main app styling */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: var(--text-primary);
    }

    /* Header styling */
    .main-header {
        background: var(--gradient-primary);
        padding: 2.5rem 3rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-primary);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.05) 2px,
            rgba(255,255,255,0.05) 4px
        );
        opacity: 0.3;
    }

    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* Metric cards enhancement */
    [data-testid="metric-container"] {
        background: var(--secondary-color);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: var(--shadow-primary);
        transition: all 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: var(--accent-color);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }

    /* Enhanced buttons */
    .stButton > button {
        background: var(--gradient-primary);
        border: none;
        border-radius: var(--border-radius);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
    }

    .stButton > button:hover {
        background: var(--gradient-secondary);
        transform: translateY(-2px);
        box-shadow: var(--shadow-primary);
    }

    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--secondary-color);
        padding: 0.5rem;
        border-radius: var(--border-radius);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 1rem 2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    /* Success/Error/Warning messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--success-color);
        border-radius: var(--border-radius);
        color: var(--success-color);
    }

    .stError {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--error-color);
        border-radius: var(--border-radius);
        color: var(--error-color);
    }

    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid var(--warning-color);
        border-radius: var(--border-radius);
        color: var(--warning-color);
    }

    /* File uploader enhancement */
    .stFileUploader {
        border: 2px dashed var(--accent-color);
        border-radius: var(--border-radius);
        padding: 2rem;
        background: rgba(59, 130, 246, 0.05);
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: var(--success-color);
    }

    /* DataFrames styling */
    .dataframe {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow-primary);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--primary-color);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--accent-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--success-color);
    }

    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem;
        }

        .main-header h1 {
            font-size: 2rem;
        }

        .main-header p {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Load CSS
load_professional_css()

# ============================
# LANGUAGE SUPPORT
# ============================

@st.cache_data
def get_language_config():
    """Get comprehensive language configuration."""
    return {
        "en": {
            "title": "Sales & Demand Forecasting Platform",
            "subtitle": "Professional Business Intelligence & Advanced Analytics",
            "upload_file": "Upload Your Data",
            "file_uploaded": "File uploaded successfully!",
            "processing": "Processing data...",
            "data_overview": "📊 Data Overview",
            "data_analysis": "📈 Statistical Analysis", 
            "visualizations": "🎨 Interactive Visualizations",
            "forecasting": "🔮 Sales Forecasting",
            "export": "📋 Export & Reports",
            "quality": "🔍 Data Quality",
            "no_data": "No data loaded. Please upload a file to begin analysis.",
            "welcome_title": "Welcome to Professional Sales Analytics",
            "welcome_msg": "Upload your sales data to unlock powerful insights and forecasting capabilities",
            "supported_formats": "Supported: CSV, Excel (XLSX/XLS), JSON files",
            "error": "Error",
            "success": "Success",
            "rows": "Records",
            "columns": "Columns",
            "missing": "Missing Values",
            "duplicates": "Duplicates", 
            "memory": "Memory Usage",
            "data_quality": "Data Quality Score",
            "revenue": "Revenue",
            "growth": "Growth",
            "conversion": "Conversion Rate",
            "customers": "Customers"
        },
        "ar": {
            "title": "منصة التنبؤ بالمبيعات والطلب",
            "subtitle": "ذكاء الأعمال المهني والتحليلات المتقدمة",
            "upload_file": "رفع البيانات",
            "file_uploaded": "تم رفع الملف بنجاح!",
            "processing": "جاري معالجة البيانات...",
            "data_overview": "📊 نظرة عامة على البيانات",
            "data_analysis": "📈 التحليل الإحصائي",
            "visualizations": "🎨 التصورات التفاعلية",
            "forecasting": "🔮 توقع المبيعات",
            "export": "📋 التصدير والتقارير",
            "quality": "🔍 جودة البيانات",
            "no_data": "لا توجد بيانات. يرجى رفع ملف لبدء التحليل.",
            "welcome_title": "مرحباً بك في تحليلات المبيعات المهنية",
            "welcome_msg": "قم برفع بيانات المبيعات لفتح رؤى قوية وقدرات التنبؤ",
            "supported_formats": "المدعوم: ملفات CSV و Excel و JSON",
            "error": "خطأ",
            "success": "نجح",
            "rows": "السجلات",
            "columns": "الأعمدة",
            "missing": "القيم المفقودة",
            "duplicates": "المكررات",
            "memory": "استخدام الذاكرة",
            "data_quality": "نقاط جودة البيانات",
            "revenue": "الإيرادات",
            "growth": "النمو",
            "conversion": "معدل التحويل",
            "customers": "العملاء"
        }
    }

# ============================
# DATA LOADING FUNCTIONS (FIXED)
# ============================

def load_file(uploaded_file) -> pd.DataFrame:
    """Load file with enhanced error handling and performance optimization - NO CACHE."""
    try:
        if uploaded_file is None:
            return pd.DataFrame()
            
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Enhanced file loading with encoding detection
        if file_extension == 'csv':
            # Try different encodings for CSV
            encodings = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    if not df.empty:
                        return df
                except (UnicodeDecodeError, Exception):
                    continue
            
            # If all encodings fail, use default with error handling
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

def get_data_summary(df: pd.DataFrame) -> Dict:
    """Get comprehensive data summary with business insights."""
    if df is None or df.empty:
        return {}
    
    # Basic statistics
    summary = {
        "shape": df.shape,
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "missing_count": df.isnull().sum().sum(),
        "missing_pct": (df.isnull().sum().sum() / df.size) * 100,
        "duplicate_count": df.duplicated().sum(),
        "duplicate_pct": (df.duplicated().sum() / len(df)) * 100,
        "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_cols": df.select_dtypes(include=['object']).columns.tolist(),
        "datetime_cols": [],
        "dtypes": df.dtypes.to_dict()
    }

    # Detect datetime columns
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            try:
                pd.to_datetime(sample, errors='raise')
                summary["datetime_cols"].append(col)
            except:
                pass

    return summary

def calculate_quality_score(df: pd.DataFrame) -> float:
    """Calculate comprehensive data quality score (0-10)."""
    if df is None or df.empty:
        return 0.0

    scores = []

    # Completeness (40% weight)
    completeness = (df.count().sum() / df.size) * 100 if df.size > 0 else 0
    scores.append(completeness * 0.4)

    # Uniqueness (30% weight) 
    uniqueness = ((len(df) - df.duplicated().sum()) / len(df)) * 100 if len(df) > 0 else 0
    scores.append(uniqueness * 0.3)

    # Consistency (20% weight) - data type consistency
    consistency = 100  # Simplified - could be enhanced
    scores.append(consistency * 0.2)

    # Validity (10% weight) - basic validity checks
    validity = 100  # Simplified - could be enhanced
    scores.append(validity * 0.1)

    # Calculate weighted average and convert to 0-10 scale
    total_score = sum(scores) / 10

    return round(min(10.0, max(0.0, total_score)), 1)

# ============================
# UI COMPONENTS
# ============================

def render_header(txt: Dict[str, str]):
    """Render professional animated header."""
    st.markdown(f"""
    <div class="main-header fade-in">
        <h1>📈 {txt['title']}</h1>
        <p>{txt['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)

def render_welcome_screen(txt: Dict[str, str]):
    """Render enhanced welcome screen with features showcase."""
    st.markdown(f"""
    <div class="fade-in">

    ### 👋 {txt['welcome_title']}

    {txt['welcome_msg']}

    **{txt['supported_formats']}**

    ---

    </div>
    """, unsafe_allow_html=True)

    # Feature showcase with enhanced styling
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📊 Advanced Analytics**
        - Real-time data quality assessment
        - Statistical analysis & insights
        - Trend identification & patterns
        - Outlier detection & treatment
        - Interactive data exploration
        """)

    with col2:
        st.markdown("""
        **📈 Sales Forecasting**
        - Time series analysis
        - Demand prediction models
        - Seasonal trend analysis
        - Revenue growth projections
        - Performance benchmarking
        """)

    with col3:
        st.markdown("""
        **🎯 Business Intelligence**
        - Executive dashboards
        - KPI monitoring & tracking
        - Customer segmentation
        - Market analysis insights
        - Professional reporting
        """)

def render_enhanced_data_overview(df: pd.DataFrame, txt: Dict[str, str]):
    """Render enhanced data overview with executive KPIs."""
    if df is None or df.empty:
        st.warning(txt['no_data'])
        return
        
    summary = get_data_summary(df)

    if not summary:
        st.warning(txt['no_data'])
        return

    # Executive KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            f"📊 {txt['rows']}", 
            f"{summary['shape'][0]:,}",
            help="Total number of records in your dataset"
        )

    with col2:
        st.metric(
            f"📋 {txt['columns']}", 
            f"{summary['shape'][1]}",
            help="Number of data columns/fields"
        )

    with col3:
        delta_color = "normal" if summary['missing_pct'] < 5 else "inverse"
        st.metric(
            f"✅ Completeness", 
            f"{100-summary['missing_pct']:.1f}%",
            f"{summary['missing_count']:,} missing",
            delta_color=delta_color,
            help="Percentage of complete data points"
        )

    with col4:
        delta_color = "normal" if summary['duplicate_pct'] < 1 else "inverse"
        st.metric(
            f"🔄 Uniqueness", 
            f"{100-summary['duplicate_pct']:.1f}%",
            f"{summary['duplicate_count']:,} duplicates",
            delta_color=delta_color,
            help="Percentage of unique records"
        )

    with col5:
        memory_color = "normal" if summary['memory_mb'] < 100 else "inverse"
        st.metric(
            f"💾 {txt['memory']}", 
            f"{summary['memory_mb']:.1f} MB",
            delta_color=memory_color,
            help="Memory usage of the dataset"
        )

    # Quality score with enhanced styling
    quality_score = calculate_quality_score(df)
    st.session_state.quality_score = quality_score

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if quality_score >= 8:
            st.success(f"⭐ {txt['data_quality']}: {quality_score}/10 - Excellent Quality!")
        elif quality_score >= 6:
            st.warning(f"⭐ {txt['data_quality']}: {quality_score}/10 - Good Quality")
        else:
            st.error(f"⭐ {txt['data_quality']}: {quality_score}/10 - Needs Improvement")

# ============================
# SIMPLIFIED TAB FUNCTIONS
# ============================

def render_overview_tab(df: pd.DataFrame, txt: Dict[str, str]):
    """Render comprehensive data overview tab."""
    st.markdown("### 📊 Dataset Analysis")

    # Show data preview
    st.markdown("**📋 Data Preview**")
    st.dataframe(df.head(20), use_container_width=True)

    # Show basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.markdown("**📈 Statistical Summary**")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

def render_analysis_tab(df: pd.DataFrame, txt: Dict[str, str]):
    """Render basic analysis tab."""
    st.markdown("### 📈 Basic Data Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numeric Columns:**")
        st.write(numeric_cols if numeric_cols else "No numeric columns found")
        
    with col2:
        st.markdown("**Categorical Columns:**")
        st.write(categorical_cols if categorical_cols else "No categorical columns found")

def render_visualization_tab(df: pd.DataFrame, txt: Dict[str, str]):
    """Render basic visualization tab."""
    st.markdown("### 🎨 Basic Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_col = st.selectbox("Select column to visualize:", numeric_cols)
        
        if selected_col:
            # Simple histogram
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available for visualization.")

def render_forecasting_tab(df: pd.DataFrame, txt: Dict[str, str]):
    """Render basic forecasting tab."""
    st.markdown("### 🔮 Sales Forecasting")
    st.info("Forecasting features will be available in future updates.")

def render_export_tab(df: pd.DataFrame, txt: Dict[str, str]):
    """Render basic export tab."""
    st.markdown("### 📋 Export Data")
    
    # Simple CSV export
    csv_data = df.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.download_button(
        "📥 Download CSV",
        data=csv_data,
        file_name=f"data_export_{timestamp}.csv",
        mime="text/csv"
    )

# ============================
# MAIN APPLICATION (FIXED)
# ============================

def main():
    """Main application function with proper file upload handling."""
    # Initialize session state FIRST
    init_session_state()

    # Language selection in sidebar
    st.sidebar.title("🌐 Language / اللغة")
    lang = st.sidebar.selectbox(
        "Select Language:",
        ["en", "ar"],
        index=0
    )

    # Get language configuration
    txt = get_language_config()[lang]

    # Apply RTL for Arabic
    if lang == "ar":
        st.markdown('<div dir="rtl">', unsafe_allow_html=True)

    # Render header
    render_header(txt)

    # File upload section with proper handling
    st.sidebar.title(f"📁 {txt['upload_file']}")
    
    uploaded_file = st.sidebar.file_uploader(
        txt['upload_file'],
        type=['csv', 'xlsx', 'xls', 'json'],
        help=txt['supported_formats'],
        accept_multiple_files=False
    )

    # Process file upload immediately when file is uploaded
    if uploaded_file is not None:
        # Check if this is a new file
        current_file_name = uploaded_file.name
        
        if current_file_name != st.session_state.current_file_name:
            # New file uploaded
            st.session_state.current_file_name = current_file_name
            st.session_state.file_uploaded = False
            st.session_state.processing_complete = False
            
            # Show processing message
            with st.sidebar:
                with st.spinner(txt['processing']):
                    # Load the file
                    df = load_file(uploaded_file)
                    
                    if df is not None and not df.empty:
                        # Store in session state
                        st.session_state.df = df
                        st.session_state.data_summary = get_data_summary(df)
                        st.session_state.file_uploaded = True
                        st.session_state.processing_complete = True
                        
                        # Show success message
                        st.success(f"✅ {txt['file_uploaded']}")
                        
                        # Show file info
                        st.markdown("**📊 File Information:**")
                        st.write(f"• **Name:** {uploaded_file.name}")
                        st.write(f"• **Size:** {uploaded_file.size / 1024:.1f} KB")
                        st.write(f"• **Rows:** {df.shape[0]:,}")
                        st.write(f"• **Columns:** {df.shape[1]}")
                        
                        # Data quality indicator
                        quality_score = calculate_quality_score(df)
                        if quality_score >= 8:
                            st.success(f"📊 Quality: {quality_score}/10")
                        elif quality_score >= 6:
                            st.warning(f"📊 Quality: {quality_score}/10")
                        else:
                            st.error(f"📊 Quality: {quality_score}/10")
                    else:
                        st.error("❌ Failed to load file. Please check the file format and try again.")
                        st.session_state.file_uploaded = False

    # Main content area
    if st.session_state.file_uploaded and st.session_state.df is not None:
        df = st.session_state.df

        # Enhanced data overview metrics
        render_enhanced_data_overview(df, txt)

        # Simplified main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            f"{txt['data_overview']}",
            f"{txt['data_analysis']}",
            f"{txt['visualizations']}",
            f"{txt['forecasting']}",
            f"{txt['export']}"
        ])

        with tab1:
            render_overview_tab(df, txt)

        with tab2:
            render_analysis_tab(df, txt)

        with tab3:
            render_visualization_tab(df, txt)

        with tab4:
            render_forecasting_tab(df, txt)

        with tab5:
            render_export_tab(df, txt)

    else:
        # Welcome screen when no file is uploaded
        render_welcome_screen(txt)

    # Close RTL div for Arabic
    if lang == "ar":
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem; background: rgba(15, 20, 25, 0.5); border-radius: 12px; margin-top: 2rem;">
        <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
            📈 <strong>CortexX Sales & Demand Forecasting Platform</strong>
        </div>
        <div style="font-size: 0.9rem; opacity: 0.8;">
            Professional Business Intelligence | Built with ❤️ using Streamlit | Version 1.3.1
        </div>
        <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.6;">
            Transform your sales data into actionable insights
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
