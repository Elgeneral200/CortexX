# filename: __init__.py
"""
CortexX Sales & Demand Forecasting Platform - Fixed Professional Edition

A streamlined, reliable data analytics platform for sales and demand forecasting
with clean architecture, essential features, and professional reliability.

This platform provides:
- Multi-format data ingestion (CSV, Excel, JSON)
- Professional data cleaning and preprocessing
- Interactive visualizations and dashboards  
- Data quality assessment and reporting
- Pipeline workflow management
- Multilingual support (English/Arabic)

Modules:
--------
file_handler : Multi-format data ingestion
    - CSV, Excel, JSON file support
    - Smart encoding detection
    - Error handling and validation

preprocess : Data cleaning and transformation
    - Missing value handling
    - Data type detection and conversion
    - Outlier detection and treatment
    - Statistical operations and scaling

visualization : Professional dashboards and charts
    - Interactive business intelligence dashboards
    - Statistical analysis visualizations
    - Data quality visual assessments
    - Professional themes and styling

pipeline : Workflow management
    - Step recording and replay
    - Undo/redo functionality
    - Pipeline serialization and templates
    - Performance monitoring

quality : Data quality management
    - Comprehensive quality assessment
    - Professional HTML reporting
    - Quality scoring and recommendations
    - Business impact analysis

Author: CortexX Team
Version: 1.2.0 - Fixed Professional Edition
"""

import sys
import warnings
import importlib
from typing import Dict, Any, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================
# PACKAGE METADATA
# ============================

__version__ = "1.2.0"
__author__ = "CortexX Team"
__description__ = "Professional Sales & Demand Forecasting Platform"
__status__ = "Production/Stable"

# ============================
# SAFE MODULE IMPORTS
# ============================

# Initialize module status
_modules_loaded = {}
_import_errors = []

def _safe_import(module_name: str, fallback_name: Optional[str] = None):
    """Safely import modules with error handling."""
    global _modules_loaded, _import_errors

    try:
        # Try relative import from core package
        if module_name in ['file_handler', 'preprocessing', 'visualization', 'pipeline', 'quality']:
            module = importlib.import_module(f'.{module_name}', package='core')
            _modules_loaded[module_name] = module
            return module
        else:
            # Direct import
            module = importlib.import_module(module_name)
            _modules_loaded[module_name] = module
            return module

    except ImportError as e:
        _import_errors.append(f"{module_name}: {str(e)}")

        if fallback_name:
            try:
                module = importlib.import_module(fallback_name)
                _modules_loaded[fallback_name] = module
                return module
            except ImportError:
                pass

        # Create a dummy module if all else fails
        class DummyModule:
            def __getattr__(self, name):
                def dummy_function(*args, **kwargs):
                    raise ImportError(f"Module {module_name} not available")
                return dummy_function

        return DummyModule()

# Import core modules with safe handling
file_handler = _safe_import('file_handler')
preprocessing = _safe_import('preprocessing')
visualization = _safe_import('visualization')
pipeline = _safe_import('pipeline')
quality = _safe_import('quality')

# ============================
# PLATFORM CLASS
# ============================

class CortexXPlatform:
    """
    Main interface for the CortexX Sales & Demand Forecasting Platform.

    Provides unified access to all platform capabilities with
    clean architecture and professional reliability.
    """

    def __init__(self):
        self.version = __version__
        self.name = "CortexX Sales & Demand Forecasting Platform"
        self.modules = {
            'file_handler': file_handler,
            'preprocessing': preprocessing,
            'visualization': visualization,
            'pipeline': pipeline,
            'quality': quality
        }

        # Check which modules loaded successfully
        self.available_modules = list(_modules_loaded.keys())
        self.import_errors = _import_errors.copy()

    def get_system_info(self) -> Dict[str, Any]:
        """Get platform and system information."""

        info = {
            'platform': {
                'name': self.name,
                'version': self.version,
                'status': __status__,
                'description': __description__
            },
            'system': {
                'python_version': sys.version.split()[0],
                'platform': sys.platform
            },
            'modules': {
                'loaded': self.available_modules,
                'total': len(self.modules),
                'errors': self.import_errors
            },
            'capabilities': self.get_capabilities()
        }

        return info

    def check_dependencies(self) -> Dict[str, str]:
        """Check platform dependencies."""

        required_packages = [
            'pandas', 'numpy', 'plotly', 'streamlit'
        ]

        optional_packages = [
            'scipy', 'sklearn', 'requests', 'openpyxl', 'lxml'
        ]

        status = {}

        # Check required packages
        for package in required_packages:
            try:
                importlib.import_module(package)
                status[package] = '✅ Available (required)'
            except ImportError:
                status[package] = '❌ Missing (required)'

        # Check optional packages  
        for package in optional_packages:
            try:
                importlib.import_module(package)
                status[package] = '✅ Available (optional)'
            except ImportError:
                status[package] = '⚠️ Missing (optional)'

        return status

    def get_capabilities(self) -> Dict[str, list]:
        """Get platform capabilities."""

        return {
            'file_formats': [
                'CSV/TSV', 'Excel (XLSX/XLS)', 'JSON', 'SQLite'
            ],
            'data_processing': [
                'Missing Value Handling', 'Data Type Conversion',
                'Outlier Detection', 'Data Cleaning', 'Statistical Operations'
            ],
            'visualizations': [
                'Interactive Dashboards', 'Business Intelligence Charts',
                'Statistical Analysis Plots', 'Data Quality Reports'
            ],
            'quality_management': [
                'Data Quality Scoring', 'Professional HTML Reports',
                'Business Impact Assessment', 'Quality Recommendations'
            ],
            'workflow_features': [
                'Pipeline Management', 'Undo/Redo Operations',
                'Workflow Serialization', 'Progress Tracking'
            ]
        }

    def create_workflow(self, name: str = "Data Processing Workflow"):
        """Create a new processing workflow."""

        # Create basic function registry for pipeline
        registry = {}

        # Add available functions from modules
        if hasattr(preprocessing, 'handle_missing_values'):
            registry['handle_missing'] = preprocessing.handle_missing_values
        if hasattr(preprocessing, 'remove_duplicates'):
            registry['remove_duplicates'] = preprocessing.remove_duplicates
        if hasattr(preprocessing, 'detect_and_convert_types'):
            registry['convert_types'] = preprocessing.detect_and_convert_types

        # Create pipeline with registry
        if hasattr(pipeline, 'Pipeline'):
            return pipeline.Pipeline(registry=registry, name=name)
        else:
            raise ImportError("Pipeline module not available")

    def validate_installation(self) -> bool:
        """Validate platform installation."""

        # Check required modules
        required_modules = ['file_handler', 'preprocessing', 'visualization', 'quality', 'pipeline']
        missing_modules = []

        for module_name in required_modules:
            if module_name not in self.available_modules:
                missing_modules.append(module_name)

        if missing_modules:
            print(f"❌ Missing modules: {', '.join(missing_modules)}")
            return False

        # Check required packages
        dependencies = self.check_dependencies()
        missing_required = [pkg for pkg, status in dependencies.items() if '❌' in status]

        if missing_required:
            print(f"❌ Missing required packages: {', '.join(missing_required)}")
            return False

        print("✅ CortexX Platform validation successful")
        return True

# ============================
# CONVENIENCE FUNCTIONS
# ============================

def get_version() -> str:
    """Get platform version."""
    return __version__

def get_platform_info() -> Dict[str, Any]:
    """Get complete platform information."""
    platform = CortexXPlatform()
    return platform.get_system_info()

def check_installation() -> bool:
    """Check if platform is properly installed."""
    try:
        platform = CortexXPlatform()
        return platform.validate_installation()
    except Exception as e:
        print(f"❌ Installation check failed: {e}")
        return False

def create_workflow(name: str = "Sales Analysis Workflow"):
    """Create a new data processing workflow."""
    platform = CortexXPlatform()
    return platform.create_workflow(name)

def get_sample_data():
    """Generate sample sales data for testing."""
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Generate sample sales data
        np.random.seed(42)

        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        regions = ['North', 'South', 'East', 'West', 'Central']

        data = []
        for i in range(500):
            data.append({
                'Date': np.random.choice(dates),
                'Product': np.random.choice(products),
                'Region': np.random.choice(regions),
                'Sales_Amount': np.random.uniform(100, 5000),
                'Quantity': np.random.randint(1, 50),
                'Customer_ID': f'CUST_{np.random.randint(1000, 9999)}',
                'Sales_Rep': f'Rep_{np.random.randint(1, 20)}'
            })

        df = pd.DataFrame(data)

        # Add some missing values for testing
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, 'Sales_Amount'] = None

        return df

    except ImportError:
        print("❌ Pandas not available - cannot generate sample data")
        return None

# ============================
# PLATFORM INSTANCE
# ============================

# Create global platform instance
_platform_instance = None

def get_platform() -> CortexXPlatform:
    """Get or create the global platform instance."""
    global _platform_instance
    if _platform_instance is None:
        _platform_instance = CortexXPlatform()
    return _platform_instance

# ============================
# EXPORTS
# ============================

__all__ = [
    # Core modules
    "file_handler",
    "preprocessing", 
    "visualization",
    "pipeline",
    "quality",

    # Platform interface
    "CortexXPlatform",
    "get_platform",

    # Convenience functions
    "get_version",
    "get_platform_info",
    "check_installation", 
    "create_workflow",
    "get_sample_data",

    # Metadata
    "__version__",
    "__author__",
    "__description__"
]

# ============================
# STARTUP MESSAGE
# ============================

def _show_startup_info():
    """Show platform startup information."""
    platform = get_platform()

    print(f"""
🚀 {platform.name} v{__version__}

📊 Professional Sales & Demand Forecasting Platform
✅ Modules loaded: {len(platform.available_modules)}/{len(platform.modules)}

Ready for data analysis! 🎯

Quick Start:
  • check_installation() - Validate your setup
  • get_sample_data() - Get test data  
  • create_workflow() - Build processing pipelines
  • get_platform_info() - View system details
    """)

# Show startup info only in interactive sessions
if hasattr(sys, 'ps1') and len(_import_errors) == 0:
    _show_startup_info()
elif _import_errors:
    print(f"⚠️ CortexX Platform loaded with {len(_import_errors)} import warnings")
    print("Run check_installation() for details")
