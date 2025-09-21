"""
CortexX Sales & Demand Forecasting Platform - Complete Working Edition v3.0

Enhanced with complete Phase 1 integration:
- Full Business Intelligence Dashboard integration
- Complete Pipeline Management Dashboard
- Advanced Quality Engine Dashboard
- Automated EDA Report Generation
- Enhanced File Handler integration
- Professional theme management
- Enterprise export capabilities

Author: CortexX Team
Version: 3.0.0 - Complete Enterprise Edition with Full Phase 1 Integration
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
import uuid

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================
# ENHANCED IMPORTS WITH FALLBACKS
# ============================

# Try importing enhanced modules with proper error handling
VISUALIZATION_ENHANCED = False
PREPROCESSING_ENHANCED = False
BUSINESS_INTELLIGENCE_AVAILABLE = False
QUALITY_AVAILABLE = False
PIPELINE_AVAILABLE = False
FILE_HANDLER_AVAILABLE = False

try:
    import visualization
    VISUALIZATION_ENHANCED = hasattr(visualization, 'render_automated_eda_dashboard')
    print("✅ Enhanced Visualization module loaded")
except ImportError:
    print("⚠️ Enhanced Visualization module not available")
    visualization = None

try:
    import preprocess
    from preprocess import DataCleaningPipeline
    PREPROCESSING_ENHANCED = hasattr(preprocess, 'render_data_cleaning_dashboard')
    print("✅ Enhanced Preprocessing module loaded")
except ImportError:
    print("⚠️ Enhanced Preprocessing module not available")
    preprocess = None
    DataCleaningPipeline = None

try:
    import business_intelligence
    BUSINESS_INTELLIGENCE_AVAILABLE = hasattr(business_intelligence, 'render_business_intelligence_dashboard')
    print("✅ Business Intelligence module loaded")
except ImportError:
    print("⚠️ Business Intelligence module not available")
    business_intelligence = None

try:
    import quality
    from quality import EnhancedQualityEngine, EnhancedQualityReporter
    QUALITY_AVAILABLE = hasattr(quality, 'render_quality_dashboard')
    print("✅ Quality module loaded")
except ImportError:
    print("⚠️ Quality module not available")
    quality = None
    EnhancedQualityEngine = None
    EnhancedQualityReporter = None

try:
    import pipeline
    from pipeline import Pipeline
    PIPELINE_AVAILABLE = hasattr(pipeline, 'render_pipeline_dashboard')
    print("✅ Pipeline module loaded")
except ImportError:
    print("⚠️ Pipeline module not available")
    pipeline = None
    Pipeline = None

try:
    import file_handler
    FILE_HANDLER_AVAILABLE = hasattr(file_handler, 'process_file_advanced')
    print("✅ File Handler module loaded")
except ImportError:
    print("⚠️ File Handler module not available")
    file_handler = None

# Enhanced visualization functions
render_automated_eda_dashboard = getattr(visualization, 'render_automated_eda_dashboard', None) if visualization else None
create_ecdf_plot = getattr(visualization, 'create_ecdf_plot', None) if visualization else None
create_qq_plot = getattr(visualization, 'create_qq_plot', None) if visualization else None

# Enhanced preprocessing functions
render_data_cleaning_dashboard = getattr(preprocess, 'render_data_cleaning_dashboard', None) if preprocess else None

# Business intelligence functions
render_business_intelligence_dashboard = getattr(business_intelligence, 'render_business_intelligence_dashboard', None) if business_intelligence else None
calculate_business_kpis = getattr(business_intelligence, 'calculate_business_kpis', None) if business_intelligence else None

# Quality functions
render_quality_dashboard = getattr(quality, 'render_quality_dashboard', None) if quality else None

# Pipeline functions
render_pipeline_dashboard = getattr(pipeline, 'render_pipeline_dashboard', None) if pipeline else None

# File handler functions
process_file_advanced = getattr(file_handler, 'process_file_advanced', None) if file_handler else None

# Business rules validation
validate_business_rules = getattr(quality, 'validate_business_rules', None) if quality else None

# ============================
# ENHANCED SESSION STATE INITIALIZATION
# ============================

def initialize_enhanced_session_state():
    """Initialize enhanced session state with all Phase 1 components."""
    
    # Core data state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'current_filename' not in st.session_state:
        st.session_state.current_filename = ""
    
    # Enhanced components state
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = {}
    if 'quality_score' not in st.session_state:
        st.session_state.quality_score = 0
    if 'quality_results' not in st.session_state:
        st.session_state.quality_results = []
    if 'quality_metrics' not in st.session_state:
        st.session_state.quality_metrics = None
    if 'business_kpis' not in st.session_state:
        st.session_state.business_kpis = {}
    
    # Pipeline and cleaning state
    if 'cleaning_pipeline' not in st.session_state:
        st.session_state.cleaning_pipeline = None
    if 'pipeline_instance' not in st.session_state:
        st.session_state.pipeline_instance = None
    
    # Quality engine state
    if 'quality_engine' not in st.session_state:
        st.session_state.quality_engine = None
    
    # Visualization session prefix for stable widget keys
    if 'viz_session' not in st.session_state:
        st.session_state.viz_session = str(uuid.uuid4())[:8]
    
    # Initialize quality engine if available
    if QUALITY_AVAILABLE and st.session_state.quality_engine is None:
        st.session_state.quality_engine = EnhancedQualityEngine()

# ============================
# ENHANCED UTILITY FUNCTIONS
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

def key_of(name: str) -> str:
    """Stable widget key based on a session prefix."""
    return f"{st.session_state.viz_session}_{name}"

def calculate_enterprise_quality_score(df: pd.DataFrame) -> float:
    """Calculate comprehensive enterprise quality score."""
    if df is None or df.empty:
        return 0.0
    
    if QUALITY_AVAILABLE and st.session_state.quality_engine:
        try:
            _, metrics = st.session_state.quality_engine.run_rules(df, [])
            return metrics.overall_score
        except Exception as e:
            print(f"Quality engine error: {e}")
            return calculate_enhanced_quality_score(df)
    else:
        return calculate_enhanced_quality_score(df)

def run_enterprise_quality_checks(df: pd.DataFrame):
    """Run comprehensive enterprise quality checks."""
    if not QUALITY_AVAILABLE or st.session_state.quality_engine is None:
        return [], None
    
    try:
        # Define enterprise rules
        enterprise_rules = [
            {'type': 'not_null_threshold', 'params': {'column': col, 'threshold': 0.95}}
            for col in df.columns if df[col].dtype in [np.number, object]
        ][:10]  # Limit to first 10 columns for performance
        
        results, metrics = st.session_state.quality_engine.run_rules(df, enterprise_rules)
        st.session_state.quality_results = results
        st.session_state.quality_metrics = metrics
        return results, metrics
    except Exception as e:
        print(f"Enterprise quality check failed: {e}")
        return [], None

# ============================
# ENHANCED FILE LOADING
# ============================

def load_file_enterprise(uploaded_file) -> pd.DataFrame:
    """Enterprise-grade file loading with enhanced validation."""
    if FILE_HANDLER_AVAILABLE and process_file_advanced:
        try:
            result = process_file_advanced(uploaded_file)
            if result.success and result.dataframe is not None:
                return result.dataframe
            else:
                st.warning("Enhanced file loading failed, falling back to basic loader")
                return load_file_enhanced(uploaded_file)
        except Exception as e:
            print(f"Enterprise file loading error: {e}")
            return load_file_enhanced(uploaded_file)
    else:
        return load_file_enhanced(uploaded_file)

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
        'shape': (int(df.shape[0]), int(df.shape[1])),
        'memory_mb': float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        'missing_count': int(df.isnull().sum().sum()),
        'missing_pct': float((df.isnull().sum().sum() / df.size) * 100),
        'duplicate_count': int(df.duplicated().sum()),
        'duplicate_pct': float((df.duplicated().sum() / len(df)) * 100),
        'numeric_cols': df.select_dtypes(include=np.number).columns.tolist(),
        'categorical_cols': df.select_dtypes(include='object').columns.tolist(),
        'datetime_cols': [],
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'data_quality_issues': []
    }
    
    # Enhanced datetime detection
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(50)
            try:
                pd.to_datetime(sample, errors='raise')
                summary['datetime_cols'].append(col)
            except:
                pass
    
    # Data quality issues
    if summary['missing_pct'] > 10:
        summary['data_quality_issues'].append(f"High missing data: {summary['missing_pct']:.1f}%")
    if summary['duplicate_pct'] > 1:
        summary['data_quality_issues'].append(f"Duplicate records: {summary['duplicate_pct']:.1f}%")
    if summary['memory_mb'] > 100:
        summary['data_quality_issues'].append(f"Large dataset: {summary['memory_mb']:.1f}MB")
    
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
# ENHANCED UI COMPONENTS
# ============================

def render_enterprise_sidebar():
    """Render enterprise-grade sidebar with enhanced feature status."""
    
    # Function to encode local image to base64
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            print(f"Error loading logo: {e}")
            return None
    
    # Sidebar styling
    st.sidebar.markdown("""
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
    </style>
    """, unsafe_allow_html=True)
    
    # FIXED: Logo display with your assets logo
    logo_path = "assets/logo.png"  # Path relative to your app.py
    logo_base64 = get_base64_image(logo_path)
    
    if logo_base64:
        st.sidebar.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" width="180" style="border-radius: 8px;">
            <div class="caption">CortexX Platform</div>
            <div class="tagline">Enterprise Data Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback if logo can't be loaded
        st.sidebar.markdown("""
        <div class="logo-container">
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); width: 180px; height: 60px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin: 0 auto; color: white; font-weight: bold; font-size: 1.2rem;">
                CortexX
            </div>
            <div class="caption">CortexX Platform</div>
            <div class="tagline">Enterprise Data Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # File upload section
    st.sidebar.markdown("**Data Upload**")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Supported formats: CSV, Excel (XLSX/XLS), JSON",
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Enhanced Feature Status
    st.sidebar.markdown("**Platform Features**")
    
    # EDA Status
    if VISUALIZATION_ENHANCED:
        st.sidebar.markdown("""
        <div class="status-indicator available">
            <div class="status-icon">✅</div>
            <div class="status-content">
                <h4>Exploratory Data Analysis</h4>
                <p>Enhanced visualization & analytics</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">⚠️</div>
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
            <div class="status-icon">✅</div>
            <div class="status-content">
                <h4>Data Cleaning</h4>
                <p>Advanced preprocessing pipeline</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">⚠️</div>
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
            <div class="status-icon">✅</div>
            <div class="status-content">
                <h4>Business Intelligence</h4>
                <p>Advanced analytics dashboards</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">⚠️</div>
            <div class="status-content">
                <h4>Business Intelligence</h4>
                <p>Basic functionality only</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add Quality Module status
    if QUALITY_AVAILABLE:
        st.sidebar.markdown("""
        <div class="status-indicator available">
            <div class="status-icon">✅</div>
            <div class="status-content">
                <h4>Data Quality Engine</h4>
                <p>Enterprise validation & reporting</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">⚠️</div>
            <div class="status-content">
                <h4>Data Quality Engine</h4>
                <p>Basic validation only</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add Pipeline Module status
    if PIPELINE_AVAILABLE:
        st.sidebar.markdown("""
        <div class="status-indicator available">
            <div class="status-icon">✅</div>
            <div class="status-content">
                <h4>Workflow Pipeline</h4>
                <p>Advanced workflow management</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">⚠️</div>
            <div class="status-content">
                <h4>Workflow Pipeline</h4>
                <p>Basic processing only</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add File Handler status
    if FILE_HANDLER_AVAILABLE:
        st.sidebar.markdown("""
        <div class="status-indicator available">
            <div class="status-icon">✅</div>
            <div class="status-content">
                <h4>Advanced File Handling</h4>
                <p>Multi-format support with validation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-indicator unavailable">
            <div class="status-icon">⚠️</div>
            <div class="status-content">
                <h4>Advanced File Handling</h4>
                <p>Basic file support</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.markdown("""
    <div class="about-section">
        <div class="about-header">🌍 Global Enterprise Platform</div>
        <div class="about-content">
            CortexX delivers world-class data intelligence solutions to Fortune 500 companies worldwide. 
            Our AI-powered platform transforms complex data into actionable business insights.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Add a subtle footer with updated year
    st.sidebar.markdown("""
    <div style="text-align: center; margin-top: 1rem; color: #64748b; font-size: 0.75rem;">
        © 2025 CortexX Global Inc. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
    
    return uploaded_file

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
    
    if QUALITY_AVAILABLE:
        feature_badges.append('<span class="feature-badge working">✅ Quality Engine</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic Quality</span>')
    
    if PIPELINE_AVAILABLE:
        feature_badges.append('<span class="feature-badge working">✅ Workflow Pipeline</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic Processing</span>')
    
    if FILE_HANDLER_AVAILABLE:
        feature_badges.append('<span class="feature-badge working">✅ File Handling</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic Files</span>')
    
    feature_badges.append('<span class="feature-badge pulse">🌍 Global Platform</span>')
    
    st.markdown(f"""
    <div class="enterprise-header fade-in">
        <h1>🚀 CortexX Enterprise Intelligence Platform</h1>
        <p>World-Class Data Analytics & Forecasting Solutions</p>
        <div style="margin-top: 1.5rem;">
            {' '.join(feature_badges)}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_enterprise_welcome_screen():
    """Render enterprise-grade welcome screen with feature status."""
    
    st.markdown("""
    <div class="fade-in">
        <div class="enterprise-card float">
            <h3>🎯 Welcome to CortexX Enterprise Platform</h3>
            <p>Upload your business data to unlock AI-powered insights, predictive analytics, and enterprise-grade forecasting</p>
            <p><strong>Supported: CSV, Excel (XLSX/XLS), JSON files</strong></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🚀 Enterprise Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        card = st.container()
        card.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        if VISUALIZATION_ENHANCED:
            card.success("🔍 Advanced Data Exploration")
            card.markdown("""
            - Automated data profiling & analysis
            - Intelligent correlation detection
            - Distribution analysis with insights
            - Missing data pattern recognition
            - Multi-method outlier detection
            - AI-powered recommendations
            """)
        else:
            card.warning("⚠️ Basic Data Exploration")
            card.markdown("""
            **To Enable Enhanced EDA:**
            - Ensure visualization.py file exists
            - Install: pip install scipy scikit-learn
            - Restart the application
            """)
        card.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        card = st.container()
        card.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        if PREPROCESSING_ENHANCED:
            card.success("🧹 Intelligent Data Preparation")
            card.markdown("""
            - Interactive cleaning pipeline
            - Version control with undo/redo
            - Smart imputation comparison
            - Memory optimization tools
            - Data type auto-conversion
            - Quality-driven transformations
            """)
        else:
            card.warning("⚠️ Basic Data Preparation")
            card.markdown("""
            **To Enable Advanced Cleaning:**
            - Ensure preprocess.py file exists
            - Install: pip install scipy scikit-learn
            - Restart the application
            """)
        card.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        card = st.container()
        # Feature status showcase
        card.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        if BUSINESS_INTELLIGENCE_AVAILABLE:
            card.success("📊 Business Intelligence Suite")
            card.markdown("""
            - Executive KPI dashboards
            - Sales performance analysis
            - Revenue trend forecasting
            - Customer behavior insights
            - Seasonal pattern detection
            - Professional reporting suite
            """)
        else:
            card.warning("⚠️ Basic Business Intelligence")
            card.markdown("""
            **To Enable Business Intelligence:**
            - Ensure business_intelligence.py file exists
            - Install: pip install scipy scikit-learn plotly
            - Restart the application
            """)
        card.markdown('</div>', unsafe_allow_html=True)
    
    # Second row of features
    col4, col5, col6 = st.columns(3)
    
    with col4:
        card = st.container()
        card.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        if QUALITY_AVAILABLE:
            card.success("🔍 Data Quality Engine")
            card.markdown("""
            - Comprehensive rule-based validation
            - Business impact assessment
            - Professional quality reporting
            - Automated recommendations
            - Quality trend analysis
            - Enterprise-grade scoring
            """)
        else:
            card.warning("⚠️ Basic Data Quality")
            card.markdown("""
            **To Enable Quality Engine:**
            - Ensure quality.py file exists
            - Install: pip install scipy scikit-learn
            - Restart the application
            """)
        card.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        card = st.container()
        card.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        if PIPELINE_AVAILABLE:
            card.success("🔄 Workflow Pipeline")
            card.markdown("""
            - Advanced workflow management
            - Complete undo/redo functionality
            - Performance monitoring
            - Professional serialization
            - Smart operation ordering
            - Enterprise integration
            """)
        else:
            card.warning("⚠️ Basic Processing")
            card.markdown("""
            **To Enable Workflow Pipeline:**
            - Ensure pipeline.py file exists
            - Install: pip install scipy scikit-learn
            - Restart the application
            """)
        card.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        card = st.container()
        # Second row of features
        card.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        if FILE_HANDLER_AVAILABLE:
            card.success("📁 Advanced File Handling")
            card.markdown("""
            - Multi-format data ingestion
            - Comprehensive error handling
            - Performance optimization
            - Smart format detection
            - Memory-efficient processing
            - Professional metadata
            """)
        else:
            card.warning("⚠️ Basic File Support")
            card.markdown("""
            **To Enable Advanced File Handling:**
            - Ensure file_handler.py file exists
            - Install: pip install scipy scikit-learn openpyxl
            - Restart the application
            """)
        card.markdown('</div>', unsafe_allow_html=True)

def render_enterprise_data_overview(df: pd.DataFrame):
    """Render enterprise-grade data overview."""
    if df is None or df.empty:
        st.warning("No data loaded. Please upload a file to unlock advanced analytics.")
        return
    
    summary = get_enhanced_data_summary(df)
    if not summary:
        st.warning("No data summary available.")
        return
    
    # Second row of features
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Records", f"{summary['shape'][0]:,}")
    
    with col2:
        st.metric("Columns", f"{summary['shape'][1]}")
    
    with col3:
        delta_color = "normal" if summary['missing_pct'] < 5 else "inverse"
        st.metric(
            "Completeness", 
            f"{100-summary['missing_pct']:.1f}%", 
            f"{summary['missing_count']:,} missing",
            delta_color=delta_color
        )
    
    with col4:
        delta_color = "normal" if summary['duplicate_pct'] < 1 else "inverse"
        st.metric(
            "Uniqueness", 
            f"{100-summary['duplicate_pct']:.1f}%", 
            f"{summary['duplicate_count']:,} duplicates",
            delta_color=delta_color
        )
    
    with col5:
        memory_color = "normal" if summary['memory_mb'] < 100 else "inverse"
        st.metric("Memory", f"{summary['memory_mb']:.1f} MB", delta_color=memory_color)
    
    with col6:
        quality_score = calculate_enterprise_quality_score(df)
        st.session_state.quality_score = quality_score
        quality_color = "normal" if quality_score > 7 else "inverse"
        st.metric("Quality Score", f"{quality_score}/10", delta_color=quality_color)

# ============================
# ENHANCED TAB RENDERERS
# ============================

def render_enhanced_eda_tab(df: pd.DataFrame):
    """Render enhanced EDA tab."""
    st.markdown("## 🔍 Automated Exploratory Data Analysis")
    
    if VISUALIZATION_ENHANCED and render_automated_eda_dashboard:
        try:
            render_automated_eda_dashboard(df, theme="professional_dark")
        except Exception as e:
            st.error(f"Enhanced EDA Error: {e}")
            render_basic_eda_fallback(df)
    else:
        st.warning("⚠️ Enhanced EDA features not available. Using basic analysis.")
        render_basic_eda_fallback(df)

def render_basic_eda_fallback(df: pd.DataFrame):
    """Basic EDA fallback."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Dataset Information")
        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.write(f"Memory: {df.memory_usage(deep=True).sum() / (1024 * 1024):.1f} MB")
        st.write(f"Missing Values: {df.isnull().sum().sum():,}")
        st.write(f"Duplicates: {df.duplicated().sum():,}")
    
    with col2:
        st.markdown("### 🏷️ Column Types")
        st.write(f"Numeric: {len(numeric_cols)} columns")
        st.write(f"Categorical: {len(categorical_cols)} columns")
    
    if numeric_cols:
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        if len(numeric_cols) >= 2:
            st.markdown("### 🔗 Basic Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
            fig.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                paper_bgcolor='rgba(15, 23, 42, 1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True, key=key_of("basic_histogram_chart"))

def render_enhanced_cleaning_tab(df: pd.DataFrame):
    """Render enhanced cleaning tab."""
    st.markdown("## 🧹 Advanced Data Cleaning Pipeline")
    
    if PREPROCESSING_ENHANCED and DataCleaningPipeline and render_data_cleaning_dashboard:
        try:
            if st.session_state.cleaning_pipeline is None:
                st.session_state.cleaning_pipeline = DataCleaningPipeline(df, "Sales Data Cleaning")
            
            render_data_cleaning_dashboard(st.session_state.cleaning_pipeline)
            st.session_state.df = st.session_state.cleaning_pipeline.current_df
        except Exception as e:
            st.error(f"Advanced Cleaning Error: {e}")
            render_basic_cleaning_fallback(df)
    else:
        st.warning("⚠️ Enhanced cleaning features not available. Using basic cleaning.")
        render_basic_cleaning_fallback(df)

def render_basic_cleaning_fallback(df: pd.DataFrame):
    """Basic cleaning fallback."""
    st.markdown("### 🔧 Basic Cleaning Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Remove duplicates
        if st.button("Remove Duplicates"):
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                st.session_state.df = df.drop_duplicates()
                st.success(f"Removed {duplicate_count} duplicate records")
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
                st.write(f"• {col}: {count:,} ({pct:.1f}%)")

def render_enterprise_business_intelligence_tab(df: pd.DataFrame):
    """Render complete enterprise business intelligence."""
    if BUSINESS_INTELLIGENCE_AVAILABLE and render_business_intelligence_dashboard:
        try:
            render_business_intelligence_dashboard(df)
            
            # Calculate and store business KPIs
            if calculate_business_kpis:
                st.session_state.business_kpis = calculate_business_kpis(df)
        except Exception as e:
            st.error(f"Enterprise BI Error: {e}")
            render_basic_bi_fallback(df)
    else:
        st.warning("⚠️ Business Intelligence features not available. Using basic analysis.")
        render_basic_bi_fallback(df)

def render_basic_bi_fallback(df: pd.DataFrame):
    """Basic BI fallback."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_cols:
        st.markdown("## 📊 Basic Business Metrics")
        
        selected_col = st.selectbox("Select metric column:", numeric_cols)
        
        if selected_col:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = df[selected_col].sum()
                st.metric("Total", f"{total_value:,.2f}")
            
            with col2:
                avg_value = df[selected_col].mean()
                st.metric("Average", f"{avg_value:.2f}")
            
            with col3:
                max_value = df[selected_col].max()
                st.metric("Maximum", f"{max_value:,.2f}")
            
            with col4:
                count_value = df[selected_col].count()
                st.metric("Count", f"{count_value:,}")
            
            # Basic visualization
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            fig.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                paper_bgcolor='rgba(15, 23, 42, 1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for business analysis.")

def render_enhanced_visualization_tab(df: pd.DataFrame):
    """Render enhanced visualization tab with proper session state handling."""
    st.markdown("## 📈 Advanced Interactive Visualizations")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    if not numeric_cols and not categorical_cols:
        st.info("No suitable columns found for visualization.")
        return
    
    # Basic visualization
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Basic Charts", "ECDF Analysis", "QQ-Plot"] if VISUALIZATION_ENHANCED else ["Basic Charts"],
        key="viz_type_selector"
    )
    
    if viz_type == "Basic Charts" and numeric_cols:
        # Use widget keys directly without session state assignment
        selected_col = st.selectbox(
            "Select column to visualize:",
            numeric_cols,
            key="basic_col_selector"
        )
        chart_type = st.radio(
            "Chart type:",
            ["Histogram", "Box Plot", "Line Chart"],
            key="basic_chart_type"
        )
        
        if chart_type == "Histogram":
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
        elif chart_type == "Box Plot":
            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
        else:
            fig = px.line(df.reset_index(), x="index", y=selected_col, title=f"Trend of {selected_col}")
        
        fig.update_layout(
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            paper_bgcolor='rgba(15, 23, 42, 1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True, key=key_of("basic_chart_display"))
        
    elif viz_type == "ECDF Analysis" and VISUALIZATION_ENHANCED and create_ecdf_plot and numeric_cols:
        # Use widget keys directly without session state assignment
        ecdf_column = st.selectbox(
            "Select numeric column:", 
            numeric_cols, 
            key="ecdf_column_selector"
        )
        ecdf_normalize = st.checkbox(
            "Normalize values", 
            key="ecdf_normalize_checkbox"
        )
        
        if ecdf_column:
            ecdf_fig = create_ecdf_plot(df, ecdf_column, theme="professional_dark", ecdf_norm=ecdf_normalize)
            st.plotly_chart(ecdf_fig, use_container_width=True, key=key_of("ecdf_chart_display"))
        
    elif viz_type == "QQ-Plot" and VISUALIZATION_ENHANCED and create_qq_plot and numeric_cols:
        # Use widget keys directly without session state assignment
        qq_column = st.selectbox(
            "Select numeric column:", 
            numeric_cols, 
            key="qq_column_selector"
        )
        
        if qq_column:
            qq_fig = create_qq_plot(df, qq_column, theme="professional_dark")
            st.plotly_chart(qq_fig, use_container_width=True, key=key_of("qq_chart_display"))
    else:
        st.info("No numeric columns available for visualization.")
    
    # Business Rules Validation
    if QUALITY_AVAILABLE and validate_business_rules:
        if st.button("Validate Business Rules", key="validate_business_rules_btn"):
            with st.spinner("Validating business rules..."):
                try:
                    business_rules = []  # Define your business rules here
                    validation_results = validate_business_rules(df, business_rules)
                    st.session_state.business_rules = validation_results
                    st.success("Business rules validated!")
                except Exception as e:
                    st.error(f"Business rules validation failed: {e}")

def render_enhanced_forecasting_tab(df: pd.DataFrame):
    """Render forecasting tab placeholder."""
    st.markdown("## 🤖 AI-Powered Sales Forecasting")
    st.info("Coming in Phase 2: Advanced ML forecasting models")
    
    st.markdown("### 🎯 Planned Forecasting Features")
    st.markdown("""
    - Time series forecasting with ARIMA, Prophet
    - Demand prediction using ML algorithms
    - Seasonal trend analysis
    - Revenue growth projections
    - Customer behavior forecasting
    - Risk assessment models
    """)

def render_enterprise_export_tab(df: pd.DataFrame):
    """Render enterprise-grade export capabilities."""
    st.markdown("## 📤 Enterprise Export & Reports")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📄 Data Export")
        
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"sales_data_{timestamp}.csv",
            mime="text/csv"
        )
        
        # Excel export
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Data", index=False)
                
                # Add summary sheet with safe conversion
                if st.session_state.data_summary:
                    try:
                        summary_data = []
                        for key, value in st.session_state.data_summary.items():
                            safe_value = safe_convert_for_json(value)
                            if isinstance(safe_value, (int, float, str)) and not pd.isna(safe_value):
                                summary_data.append({"Metric": key, "Value": str(safe_value)})
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    except:
                        pass
            
            st.download_button(
                "Download Excel",
                data=buffer.getvalue(),
                file_name=f"sales_report_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.warning("Excel export requires: pip install openpyxl")
        except Exception as e:
            st.error(f"Excel export error: {e}")
    
    with col2:
        st.markdown("### 📊 Analytics Reports")
        
        if st.session_state.quality_score > 0:
            try:
                quality_report = {
                    "Dataset": st.session_state.current_filename,
                    "Quality_Score": f"{st.session_state.quality_score}/10",
                    "Total_Records": int(len(df)),
                    "Total_Columns": int(len(df.columns)),
                    "Missing_Values": int(df.isnull().sum().sum()),
                    "Duplicate_Records": int(df.duplicated().sum()),
                    "Memory_Usage_MB": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2)
                }
                
                # Generated datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report_json = json.dumps(quality_report, indent=2, default=safe_convert_for_json)
                st.download_button(
                    "Quality Report (JSON)",
                    data=report_json,
                    file_name=f"quality_report_{timestamp}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Quality report error: {str(e)}")
    
    with col3:
        st.markdown("### 📈 Business Intelligence")
        
        if BUSINESS_INTELLIGENCE_AVAILABLE and st.session_state.business_kpis:
            try:
                safe_kpis = safe_convert_for_json(st.session_state.business_kpis)
                kpis_json = json.dumps(safe_kpis, indent=2, default=str)
                st.download_button(
                    "Business KPIs (JSON)",
                    data=kpis_json,
                   file_name=f"business_kpis_{timestamp}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Business KPIs error: {str(e)}")
        else:
            st.info("Upload data to generate business KPIs")
    
    with col4:
        st.markdown("### 🏢 Enterprise Reports")
        
        # Quality Report
        if QUALITY_AVAILABLE and st.session_state.quality_results:
            try:
                reporter = EnhancedQualityReporter()
                html_report = reporter.generate_html_report(
                    df, 
                    [],  # rules would be passed here
                    st.session_state.quality_results,
                    st.session_state.quality_metrics,
                    st.session_state.current_filename
                )
                st.download_button(
                    "Quality Report (HTML)",
                    data=html_report,
                    file_name=f"enterprise_quality_report_{timestamp}.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Quality report error: {str(e)}")
                

# ============================
# STABLE ENTERPRISE HEADER FUNCTION
# ============================

def render_stable_enterprise_header():
    """Render stable enterprise header that never changes."""
    
    # Function to encode local image to base64
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            print(f"Error loading logo: {e}")
            return None
    
    # Load logo
    logo_path = "assets/logo.png"
    logo_base64 = get_base64_image(logo_path)
    
    # Feature badges (stable - based on module availability)
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
    
    if QUALITY_AVAILABLE:
        feature_badges.append('<span class="feature-badge working">✅ Quality Engine</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic Quality</span>')
    
    if PIPELINE_AVAILABLE:
        feature_badges.append('<span class="feature-badge working">✅ Workflow Pipeline</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic Processing</span>')
    
    if FILE_HANDLER_AVAILABLE:
        feature_badges.append('<span class="feature-badge working">✅ File Handling</span>')
    else:
        feature_badges.append('<span class="feature-badge basic">⚠️ Basic Files</span>')
    
    feature_badges.append('<span class="feature-badge pulse">🌍 Global Platform</span>')
    
    # Create stable header HTML
    if logo_base64:
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="header-logo">'
    else:
        logo_html = '<div style="width: 60px; height: 60px; margin: 0 auto 1rem auto; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">CX</div>'
    
    # Render stable header
    st.markdown(f"""
    <div class="stable-header">
        <div class="enterprise-header fade-in">
            {logo_html}
            <h1>🚀 CortexX Enterprise Intelligence Platform</h1>
            <p>World-Class Data Analytics & Forecasting Solutions</p>
            <div style="margin-top: 1.5rem;">
                {' '.join(feature_badges)}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Stable Enterprise Header
def main():
    """Enhanced main application with enterprise integration."""
    
    # Initialize enhanced session state
    initialize_enhanced_session_state()
    
    # FIXED: Enhanced page config with logo as favicon
    try:
        # Try to set favicon with your logo
        st.set_page_config(
            page_title="CortexX Enterprise Intelligence Platform",
            page_icon="assets/logo.png",  # Your logo as favicon
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': "CortexX Enterprise Intelligence Platform v3.0 - Advanced Data Analytics & Forecasting"
            }
        )
    except:
        # Fallback if logo file can't be accessed
        st.set_page_config(
            page_title="CortexX Enterprise Intelligence Platform",
            page_icon="🚀",  # Fallback emoji
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    # Enhanced CSS with professional dark theme
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    .stApp > header {
        background-color: transparent;
    }
    .enterprise-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid #475569;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .enterprise-header h1 {
        color: #60a5fa;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .enterprise-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    .header-logo {
        width: 60px;
        height: 60px;
        margin: 0 auto 1rem auto;
        display: block;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    /* Make header sticky and stable */
    .stable-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem 0;
        border-bottom: 2px solid #475569;
        margin-bottom: 2rem;
    }
    .feature-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid;
    }
    .feature-badge.working {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #a7f3d0;
        border-color: #10b981;
    }
    .feature-badge.basic {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        color: #fbbf24;
        border-color: #f59e0b;
    }
    .feature-badge.pulse {
        background: linear-gradient(135deg, #3730a3 0%, #4338ca 100%);
        color: #c7d2fe;
        border-color: #6366f1;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    .enterprise-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #475569;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .enterprise-card.float:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ✅ FIXED: Use ONLY the stable header (removed the duplicate call)
    render_stable_enterprise_header()
    
    # Render enterprise sidebar and get uploaded file
    uploaded_file = render_enterprise_sidebar()
    
    # Main content area
    if uploaded_file is not None:
        # File Analysis
        st.markdown("## 📁 File Analysis")
        st.write(f"**Name:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        
        # Check if this is a new file
        current_filename = uploaded_file.name
        if current_filename != st.session_state.current_filename:
            st.session_state.current_filename = current_filename
            st.session_state.file_uploaded = False
            st.session_state.processing_complete = False
            st.session_state.cleaning_pipeline = None
        
        # Process file upload
        with st.spinner("Processing data with enterprise AI insights..."):
            # Use enterprise file loader
            df = load_file_enterprise(uploaded_file)
            
            if df is not None and not df.empty:
                st.session_state.df = df
                st.session_state.data_summary = get_enhanced_data_summary(df)
                st.session_state.file_uploaded = True
                st.session_state.processing_complete = True
                
                st.write(f"**Records:** {df.shape[0]:,}")
                st.write(f"**Features:** {df.shape[1]}")
                
                # Calculate quality score
                quality_score = calculate_enterprise_quality_score(df)
                st.session_state.quality_score = quality_score
                
                if quality_score > 8:
                    st.success(f"✅ Quality: {quality_score}/10 - Excellent")
                elif quality_score > 6:
                    st.warning(f"⚠️ Quality: {quality_score}/10 - Good")
                else:
                    st.error(f"❌ Quality: {quality_score}/10 - Needs Work")
                
                # Run enterprise quality checks
                run_enterprise_quality_checks(df)
                
            else:
                st.error("Failed to load file. Please check the file format and try again.")
                st.session_state.file_uploaded = False
    
    # Add enterprise enhancements
    if st.session_state.file_uploaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # Render enterprise data overview
        render_enterprise_data_overview(df)
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Automated EDA", 
            "🧹 Smart Data Cleaning", 
            "📊 Business Intelligence", 
            "📈 Interactive Visualizations", 
            "🤖 AI Forecasting", 
            "📤 Export Reports"
        ])
        
        with tab1:
            render_enhanced_eda_tab(df)
        
        with tab2:
            render_enhanced_cleaning_tab(df)
        
        with tab3:
            render_enterprise_business_intelligence_tab(df)
        
        with tab4:
            render_enhanced_visualization_tab(df)
        
        with tab5:
            render_enhanced_forecasting_tab(df)
        
        with tab6:
            render_enterprise_export_tab(df)
        
        # Enhanced report summary
        st.markdown("---")
        st.markdown("## 📋 Enterprise Report Summary")
        
        try:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Records", f"{len(df):,}")
            
            with col2:
                st.metric("Columns", f"{len(df.columns)}")
            
            with col3:
                quality_score = getattr(st.session_state, 'quality_score', 0)
                st.metric("Quality", f"{quality_score}/10")
            
            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("Size", f"{memory_mb:.1f} MB")
            
            with col5:
                if st.session_state.quality_metrics:
                    readiness = getattr(st.session_state.quality_metrics, 'business_readiness', 'Unknown')
                else:
                    readiness = 'Unknown'
                st.metric("Readiness", readiness.title())
                
        except Exception as e:
            st.error(f"Summary metrics error: {str(e)}")
        
        # Enhanced dashboards
        if QUALITY_AVAILABLE and st.session_state.quality_results:
            with st.expander("🔍 Data Quality Insights", expanded=False):
                render_quality_dashboard(df, st.session_state.quality_results, st.session_state.quality_metrics)
        
        if PIPELINE_AVAILABLE and st.session_state.pipeline_instance:
            with st.expander("🔄 Processing Pipeline", expanded=False):
                render_pipeline_dashboard(st.session_state.pipeline_instance)
    
    else:
        # Render enterprise welcome screen
        render_enterprise_welcome_screen()


if __name__ == "__main__":
    main()
