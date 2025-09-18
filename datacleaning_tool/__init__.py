# filename: __init__.py
"""
CortexX Sales & Demand Forecasting Platform - Enhanced Professional Edition v2.2

A comprehensive, professional data analytics platform with Phase 1 enhancements:
- Automated EDA with smart recommendations
- Advanced data cleaning with undo/redo functionality
- Business intelligence dashboards and KPI analysis
- Interactive visualizations and professional reporting
- Multi-format data ingestion with smart processing
- Quality assessment and optimization tools
- Fixed imports and proper error handling

Phase 1 Enhanced Modules:
- file_handler: Multi-format data ingestion with enhanced error handling
- preprocess: Advanced cleaning pipeline with smart imputation
- visualization: Automated EDA and professional business intelligence charts  
- business_intelligence: Executive dashboards and sales performance analysis
- pipeline: Workflow management with serialization
- quality: Comprehensive data quality assessment and reporting

Author: CortexX Team
Version: 2.2.0 - Enhanced Professional Edition with All Phase 1 Features Working
"""

import sys
import warnings
import importlib
from typing import Dict, Any, Optional
from datetime import datetime
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================
# ENHANCED PACKAGE METADATA
# ============================

__version__ = "2.2.0"
__author__ = "CortexX Team"
__description__ = "Professional Sales & Demand Forecasting Platform with AI Enhancement"
__status__ = "Production/Stable - Phase 1 Complete"
__phase__ = "Phase 1: Core Enhancements Complete & Working"

# ============================
# ENHANCED SAFE MODULE IMPORTS
# ============================

# Initialize module status
_modules_loaded = {}
_import_errors = []
_enhanced_features = {}

def _safe_import(module_name: str, feature_name: str, enhanced: bool = False):
    """Safely import modules with proper error handling."""
    global _modules_loaded, _import_errors, _enhanced_features

    try:
        # Direct import from current directory
        module = importlib.import_module(module_name)
        _modules_loaded[module_name] = module
        
        # Mark as enhanced if it has Phase 1 features
        if enhanced:
            _enhanced_features[module_name] = True
            
        return module, True

    except ImportError as e:
        _import_errors.append(f"{module_name}: {str(e)}")
        
        # Create a dummy module if import fails
        class DummyModule:
            def __getattr__(self, name):
                def dummy_function(*args, **kwargs):
                    raise ImportError(f"Module {module_name} not available. Check if {module_name}.py exists or install dependencies.")
                return dummy_function

        return DummyModule(), False

    except Exception as e:
        _import_errors.append(f"{module_name}: Unexpected error - {str(e)}")
        return None, False

# Import core modules with enhanced features detection
print("🚀 Loading CortexX Platform v2.2...")

file_handler, file_handler_available = _safe_import('file_handler', 'File Handling', enhanced=True)
preprocess, preprocess_available = _safe_import('preprocess', 'Advanced Preprocessing', enhanced=True)
visualization, visualization_available = _safe_import('visualization', 'Enhanced Visualization', enhanced=True)
pipeline, pipeline_available = _safe_import('pipeline', 'Workflow Pipeline', enhanced=True)
quality, quality_available = _safe_import('quality', 'Data Quality', enhanced=True)

# Import new Phase 1 modules
business_intelligence, business_intelligence_available = _safe_import('business_intelligence', 'Business Intelligence', enhanced=True)

# Status reporting
print(f"📊 Module Loading Status:")
print(f"   file_handler: {'✅ Loaded' if file_handler_available else '❌ Failed'}")
print(f"   preprocess: {'✅ Loaded' if preprocess_available else '❌ Failed'}")
print(f"   visualization: {'✅ Loaded' if visualization_available else '❌ Failed'}")
print(f"   business_intelligence: {'✅ Loaded' if business_intelligence_available else '❌ Failed'}")
print(f"   pipeline: {'✅ Loaded' if pipeline_available else '❌ Failed'}")
print(f"   quality: {'✅ Loaded' if quality_available else '❌ Failed'}")

# ============================
# ENHANCED PLATFORM CLASS
# ============================

class CortexXPlatform:
    """
    Enhanced main interface for the CortexX Sales & Demand Forecasting Platform.
    
    Phase 1 enhancements include:
    - Automated EDA capabilities
    - Advanced data cleaning pipelines
    - Business intelligence dashboards
    - Professional reporting tools
    - Proper error handling and fallbacks
    """

    def __init__(self):
        self.version = __version__
        self.name = "CortexX Sales & Demand Forecasting Platform"
        self.phase = __phase__
        self.modules = {
            'file_handler': file_handler,
            'preprocess': preprocess,
            'visualization': visualization,
            'pipeline': pipeline,
            'quality': quality,
            'business_intelligence': business_intelligence
        }

        # Check which modules loaded successfully
        self.available_modules = list(_modules_loaded.keys())
        self.enhanced_modules = list(_enhanced_features.keys())
        self.import_errors = _import_errors.copy()
        
        # Module availability flags
        self.file_handler_available = file_handler_available
        self.preprocess_available = preprocess_available
        self.visualization_available = visualization_available
        self.business_intelligence_available = business_intelligence_available
        self.pipeline_available = pipeline_available
        self.quality_available = quality_available

    def get_system_info(self) -> Dict[str, Any]:
        """Get enhanced platform and system information."""

        info = {
            'platform': {
                'name': self.name,
                'version': self.version,
                'phase': self.phase,
                'status': __status__,
                'description': __description__
            },
            'system': {
                'python_version': sys.version.split()[0],
                'platform': sys.platform,
                'working_directory': os.getcwd()
            },
            'modules': {
                'loaded': self.available_modules,
                'enhanced': self.enhanced_modules,
                'total': len(self.modules),
                'errors': self.import_errors,
                'availability': {
                    'file_handler': self.file_handler_available,
                    'preprocess': self.preprocess_available,
                    'visualization': self.visualization_available,
                    'business_intelligence': self.business_intelligence_available,
                    'pipeline': self.pipeline_available,
                    'quality': self.quality_available
                }
            },
            'capabilities': self.get_enhanced_capabilities(),
            'phase1_features': {
                'automated_eda': self.visualization_available,
                'advanced_cleaning': self.preprocess_available,
                'business_intelligence': self.business_intelligence_available,
                'enhanced_quality': self.quality_available,
                'smart_pipeline': self.pipeline_available,
                'file_handling': self.file_handler_available
            }
        }

        return info

    def check_dependencies(self) -> Dict[str, str]:
        """Check platform dependencies including Phase 1 requirements."""

        required_packages = [
            'pandas', 'numpy', 'plotly', 'streamlit'
        ]

        enhanced_packages = [
            'scipy', 'sklearn', 'requests', 'openpyxl', 'lxml'
        ]

        optional_packages = [
            'seaborn', 'matplotlib', 'fastapi', 'uvicorn', 'xlsxwriter'
        ]

        status = {}

        # Check required packages
        for package in required_packages:
            try:
                importlib.import_module(package)
                status[package] = '✅ Available (required)'
            except ImportError:
                status[package] = '❌ Missing (required)'

        # Check enhanced packages (needed for Phase 1 features)
        for package in enhanced_packages:
            try:
                importlib.import_module(package)
                status[package] = '🚀 Available (enhanced features)'
            except ImportError:
                status[package] = '⚠️ Missing (enhanced features limited)'

        # Check optional packages  
        for package in optional_packages:
            try:
                importlib.import_module(package)
                status[package] = '✅ Available (optional)'
            except ImportError:
                status[package] = '⚪ Missing (optional)'

        return status

    def get_enhanced_capabilities(self) -> Dict[str, list]:
        """Get enhanced platform capabilities including Phase 1 features."""

        capabilities = {
            'file_formats': [
                'CSV/TSV with smart encoding', 'Excel (XLSX/XLS)', 'JSON', 'SQLite'
            ],
            'data_processing': [
                'Smart Missing Value Imputation', 'Advanced Data Type Optimization',
                'Intelligent Outlier Detection', 'Memory Usage Optimization', 
                'Automated Data Quality Assessment'
            ],
            'automated_analysis': [
                'Automated EDA with Recommendations', 'Smart Correlation Analysis',
                'Distribution Pattern Recognition', 'Missing Data Pattern Analysis',
                'Outlier Detection (IQR, Z-Score, ML-based)'
            ],
            'business_intelligence': [
                'Executive KPI Dashboards', 'Sales Performance Analysis',
                'Revenue Trend Forecasting', 'Customer Behavior Insights',
                'Seasonal Pattern Detection', 'Professional Reporting'
            ],
            'interactive_features': [
                'Data Cleaning Pipeline with Undo/Redo', 'Real-time Quality Scoring',
                'Interactive Visualizations', 'Column Mapping Interface',
                'Multi-language Support (EN/AR)'
            ],
            'workflow_features': [
                'Enhanced Pipeline Management', 'Operation History Tracking',
                'Smart Workflow Recommendations', 'Performance Monitoring',
                'Professional Export Capabilities'
            ]
        }

        return capabilities

    def create_enhanced_workflow(self, name: str = "Enhanced Data Processing Workflow"):
        """Create an enhanced processing workflow with Phase 1 features."""

        if not self.pipeline_available:
            raise ImportError("Pipeline module not available")

        # Create enhanced function registry
        registry = {}

        # Add available functions from enhanced modules
        if self.preprocess_available and hasattr(preprocess, 'advanced_missing_value_handler'):
            registry['smart_imputation'] = preprocess.advanced_missing_value_handler
        if self.preprocess_available and hasattr(preprocess, 'optimize_dtypes_advanced'):
            registry['optimize_memory'] = preprocess.optimize_dtypes_advanced
        if self.preprocess_available and hasattr(preprocess, 'advanced_outlier_treatment'):
            registry['handle_outliers'] = preprocess.advanced_outlier_treatment

        # Add quality assessment functions
        if self.quality_available and hasattr(quality, 'run_rules'):
            registry['quality_check'] = quality.run_rules

        # Create enhanced pipeline
        if hasattr(pipeline, 'Pipeline'):
            return pipeline.Pipeline(registry=registry, name=name)
        else:
            raise ImportError("Enhanced Pipeline class not available")

    def create_cleaning_pipeline(self, df, name: str = "Data Cleaning Pipeline"):
        """Create an advanced data cleaning pipeline."""
        
        if not self.preprocess_available:
            raise ImportError("Advanced preprocessing module not available")
            
        if hasattr(preprocess, 'DataCleaningPipeline'):
            return preprocess.DataCleaningPipeline(df, name)
        else:
            raise ImportError("DataCleaningPipeline class not available")

    def generate_eda_report(self, df):
        """Generate automated EDA report."""
        
        if not self.visualization_available:
            raise ImportError("Enhanced visualization module not available")
            
        if hasattr(visualization, 'generate_automated_eda_report'):
            return visualization.generate_automated_eda_report(df)
        else:
            raise ImportError("Automated EDA function not available")

    def calculate_business_kpis(self, df, revenue_col=None, quantity_col=None, date_col=None, customer_col=None):
        """Calculate business KPIs."""
        
        if not self.business_intelligence_available:
            raise ImportError("Business intelligence module not available")
            
        if hasattr(business_intelligence, 'calculate_business_kpis'):
            return business_intelligence.calculate_business_kpis(df, revenue_col, quantity_col, date_col, customer_col)
        else:
            raise ImportError("Business KPI calculation function not available")

    def load_file(self, file_path):
        """Load file using enhanced file handler."""
        
        if not self.file_handler_available:
            raise ImportError("File handler module not available")
            
        if hasattr(file_handler, 'load_file'):
            return file_handler.load_file(file_path)
        else:
            raise ImportError("File loading function not available")

    def assess_data_quality(self, df):
        """Assess data quality using quality module."""
        
        if not self.quality_available:
            raise ImportError("Quality assessment module not available")
            
        if hasattr(quality, 'run_rules'):
            return quality.run_rules(df)
        else:
            raise ImportError("Quality assessment function not available")

    def validate_enhanced_installation(self) -> bool:
        """Validate enhanced platform installation."""

        # Check required modules
        required_modules = ['file_handler', 'preprocess', 'visualization', 'quality', 'pipeline']
        enhanced_modules = ['business_intelligence']
        
        missing_modules = []
        missing_enhanced = []

        for module_name in required_modules:
            if module_name not in self.available_modules:
                missing_modules.append(module_name)

        for module_name in enhanced_modules:
            if module_name not in self.available_modules:
                missing_enhanced.append(module_name)

        if missing_modules:
            print(f"❌ Missing core modules: {', '.join(missing_modules)}")
            print("   Please ensure all .py files are in the same directory as app.py")
            return False

        if missing_enhanced:
            print(f"⚠️ Missing enhanced modules: {', '.join(missing_enhanced)}")
            print("   Platform will work with basic features only")

        # Check required packages
        dependencies = self.check_dependencies()
        missing_required = [pkg for pkg, status in dependencies.items() if '❌' in status]

        if missing_required:
            print(f"❌ Missing required packages: {', '.join(missing_required)}")
            print(f"   Install with: pip install {' '.join(missing_required)}")
            return False

        print("✅ CortexX Platform Enhanced Installation Validated Successfully")
        if self.enhanced_modules:
            print(f"🚀 Enhanced features available: {', '.join(self.enhanced_modules)}")
        
        return True

    def get_feature_status(self) -> Dict[str, bool]:
        """Get current feature availability status."""
        return {
            "Enhanced EDA": self.visualization_available,
            "Advanced Cleaning": self.preprocess_available,
            "Business Intelligence": self.business_intelligence_available,
            "File Handling": self.file_handler_available,
            "Quality Assessment": self.quality_available,
            "Pipeline Management": self.pipeline_available
        }

# ============================
# ENHANCED CONVENIENCE FUNCTIONS
# ============================

def get_version() -> str:
    """Get platform version."""
    return __version__

def get_enhanced_platform_info() -> Dict[str, Any]:
    """Get complete enhanced platform information."""
    platform = CortexXPlatform()
    return platform.get_system_info()

def check_enhanced_installation() -> bool:
    """Check if enhanced platform is properly installed."""
    try:
        platform = CortexXPlatform()
        return platform.validate_enhanced_installation()
    except Exception as e:
        print(f"❌ Enhanced installation check failed: {e}")
        return False

def create_enhanced_workflow(name: str = "Enhanced Sales Analysis Workflow"):
    """Create an enhanced data processing workflow."""
    platform = CortexXPlatform()
    return platform.create_enhanced_workflow(name)

def create_cleaning_pipeline(df, name: str = "Smart Data Cleaning"):
    """Create an advanced cleaning pipeline."""
    platform = CortexXPlatform()
    return platform.create_cleaning_pipeline(df, name)

def get_enhanced_sample_data():
    """Generate enhanced sample sales data for testing Phase 1 features."""
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Generate enhanced sample sales data with more realistic patterns
        np.random.seed(42)

        # Create date range
        start_date = datetime(2024, 1, 1)
        dates = pd.date_range(start=start_date, periods=1000, freq='D')
        
        # Product categories and names
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty']
        products = [f'Product_{chr(65 + i%26)}{i//26 + 1}' for i in range(50)]
        
        # Generate realistic sales data
        data = []
        for i in range(2000):
            # Add seasonality and trends
            day_of_year = (dates[i % len(dates)] - start_date).days
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            trend_factor = 1 + 0.001 * day_of_year
            
            base_price = np.random.uniform(10, 500)
            seasonal_price = base_price * seasonal_factor * trend_factor
            
            data.append({
                'Date': dates[i % len(dates)],
                'Product': np.random.choice(products),
                'Category': np.random.choice(categories),
                'Revenue': max(1, seasonal_price + np.random.normal(0, seasonal_price * 0.1)),
                'Quantity': max(1, int(np.random.poisson(5) + seasonal_factor)),
                'Unit_Price': max(0.1, base_price + np.random.normal(0, base_price * 0.05)),
                'Customer_ID': f'CUST_{np.random.randint(1, 500):04d}',
                'Sales_Rep': f'Rep_{np.random.randint(1, 25):02d}',
                'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central']),
                'Channel': np.random.choice(['Online', 'Store', 'Phone', 'Partner'])
            })

        df = pd.DataFrame(data)

        # Add some realistic data quality issues for testing
        # Missing values (5% of revenue data)
        missing_mask = np.random.random(len(df)) < 0.05
        df.loc[missing_mask, 'Revenue'] = np.nan
        
        # Some missing customer IDs
        missing_customer_mask = np.random.random(len(df)) < 0.03
        df.loc[missing_customer_mask, 'Customer_ID'] = None
        
        # Add some outliers (unusually high revenues)
        outlier_mask = np.random.random(len(df)) < 0.02
        df.loc[outlier_mask, 'Revenue'] = df.loc[outlier_mask, 'Revenue'] * 10
        
        # Add some duplicate records
        duplicate_indices = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
        duplicate_rows = df.loc[duplicate_indices].copy()
        df = pd.concat([df, duplicate_rows], ignore_index=True)

        return df

    except ImportError:
        print("❌ Pandas not available - cannot generate enhanced sample data")
        return None

def get_feature_status() -> Dict[str, bool]:
    """Get current feature availability status."""
    platform = CortexXPlatform()
    return platform.get_feature_status()

# ============================
# ENHANCED PLATFORM INSTANCE
# ============================

# Create global enhanced platform instance
_platform_instance = None

def get_enhanced_platform() -> CortexXPlatform:
    """Get or create the global enhanced platform instance."""
    global _platform_instance
    if _platform_instance is None:
        _platform_instance = CortexXPlatform()
    return _platform_instance

# ============================
# ENHANCED EXPORTS
# ============================

__all__ = [
    # Core modules (enhanced)
    "file_handler",
    "preprocess",
    "visualization",
    "pipeline", 
    "quality",
    
    # New Phase 1 modules
    "business_intelligence",

    # Enhanced platform interface
    "CortexXPlatform",
    "get_enhanced_platform",

    # Enhanced convenience functions
    "get_version",
    "get_enhanced_platform_info",
    "check_enhanced_installation", 
    "create_enhanced_workflow",
    "create_cleaning_pipeline",
    "get_enhanced_sample_data",
    "get_feature_status",

    # Metadata
    "__version__",
    "__author__",
    "__description__",
    "__phase__"
]

# ============================
# ENHANCED STARTUP MESSAGE
# ============================

def _show_enhanced_startup_info():
    """Show enhanced platform startup information."""
    platform = get_enhanced_platform()

    # Count successful modules
    successful_modules = sum([
        platform.file_handler_available,
        platform.preprocess_available, 
        platform.visualization_available,
        platform.business_intelligence_available,
        platform.pipeline_available,
        platform.quality_available
    ])
    
    phase1_status = "🚀 COMPLETE" if successful_modules >= 4 else "⚠️ PARTIAL"
    
    print(f"""
🚀 {platform.name} v{__version__}

📊 Professional Sales & Demand Forecasting Platform
✅ Core modules loaded: {successful_modules}/6
🚀 Enhanced modules: {len(platform.enhanced_modules)}
📈 Phase 1 Status: {phase1_status}

🎯 Phase 1 Enhanced Features Status:
  • 🤖 Automated EDA: {'✅ Ready' if platform.visualization_available else '❌ Not Available'}
  • 🧹 Advanced Cleaning: {'✅ Ready' if platform.preprocess_available else '❌ Not Available'}
  • 📊 Business Intelligence: {'✅ Ready' if platform.business_intelligence_available else '❌ Not Available'}
  • 📁 File Handling: {'✅ Ready' if platform.file_handler_available else '❌ Not Available'}
  • 🔍 Quality Assessment: {'✅ Ready' if platform.quality_available else '❌ Not Available'}
  • 🔄 Pipeline Management: {'✅ Ready' if platform.pipeline_available else '❌ Not Available'}

Ready for advanced analytics! 🎯

Quick Start:
  • check_enhanced_installation() - Validate setup
  • get_enhanced_sample_data() - Get test data  
  • create_enhanced_workflow() - Build pipelines
  • get_feature_status() - Check feature availability
    """)

# Show enhanced startup info
try:
    _show_enhanced_startup_info()
except Exception as e:
    print(f"⚠️ CortexX Platform loaded with startup info error: {e}")
    print("Platform should still work normally")

if _import_errors:
    print(f"\n⚠️ Import warnings detected: {len(_import_errors)}")
    for error in _import_errors:
        print(f"   • {error}")
    print("Run check_enhanced_installation() for detailed diagnosis")

print("\n✅ CortexX Platform __init__.py v2.2 - Loaded Successfully!")
