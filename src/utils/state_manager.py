"""
Centralized State Management for CortexX Forecasting Platform.

PHASE 3 - SESSION 1: Enhanced with filter state management
- Date range filters
- Product/category filters
- Comparison mode
- Quick filter presets
"""

import streamlit as st
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """
    Centralized session state manager for the CortexX platform.
    
    ENHANCED: Phase 3 - Added comprehensive filter state management
    
    Benefits:
    - Single place to define all state variables
    - Consistent initialization across app
    - Easy to add validation and cleanup
    - Type hints for better IDE support
    """
    
    # Define all state keys and their default values
    STATE_DEFAULTS = {
        # Data state
        'data_loaded': False,
        'current_data': None,
        'date_column': None,
        'value_column': None,
        'data_hash': None,  # Track data changes
        
        # Model state
        'trained_models': {},
        'model_results': {},
        'best_model_name': None,
        'backtest_results': {},
        'optimization_results': {},
        
        # UI state
        'current_page': 'Dashboard',
        'selected_models': ['XGBoost', 'LightGBM', 'Random Forest'],
        'chatbot_open': False,          # whether the chat panel is visible
        'chat_history': [],
        
        # Feature engineering state
        'engineered_features': [],
        'selected_features': [],
        
        # Training configuration
        'training_in_progress': False,
        'last_training_time': None,
        
        # Forecast state
        'forecast_results': {},
        'forecast_dates': None,
        
        # ✅ PHASE 3 - SESSION 1: Filter state
        'filter_enabled': False,
        'filter_start_date': None,
        'filter_end_date': None,
        'filter_preset': None,  # 'last_7d', 'last_30d', 'mtd', 'qtd', 'ytd'
        'filter_products': [],  # Selected product IDs
        'filter_categories': [],  # Selected categories
        'comparison_enabled': False,
        'comparison_period': 'previous',  # 'previous' or 'previous_year'
        'filtered_data': None,  # Cached filtered data
        'comparison_data': None,  # Cached comparison period data
        
    }
    
    @classmethod
    def initialize(cls, force: bool = False):
        """
        Initialize all session state variables with defaults.
        
        Args:
            force: If True, reinitialize even if already set
        """
        for key, default_value in cls.STATE_DEFAULTS.items():
            if force or key not in st.session_state:
                st.session_state[key] = default_value
        logger.info("Session state initialized")
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a value from session state with optional default.
        
        Args:
            key: State key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any):
        """
        Set a value in session state.
        
        Args:
            key: State key to set
            value: Value to store
        """
        st.session_state[key] = value
    
    @classmethod
    def update(cls, updates: Dict[str, Any]):
        """
        Update multiple state values at once.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            st.session_state[key] = value
    
    @classmethod
    def clear_data(cls):
        """Clear all data-related state."""
        cls.update({
            'data_loaded': False,
            'current_data': None,
            'date_column': None,
            'value_column': None,
            'data_hash': None
        })
        logger.info("Data state cleared")
    
    @classmethod
    def clear_models(cls):
        """Clear all model-related state."""
        cls.update({
            'trained_models': {},
            'model_results': {},
            'best_model_name': None,
            'backtest_results': {},
            'optimization_results': {}
        })
        logger.info("Model state cleared")
    
    @classmethod
    def clear_filters(cls):
        """
        Clear all filter-related state.
        
        ✅ NEW: Phase 3 - Session 1
        """
        cls.update({
            'filter_enabled': False,
            'filter_start_date': None,
            'filter_end_date': None,
            'filter_preset': None,
            'filter_products': [],
            'filter_categories': [],
            'comparison_enabled': False,
            'filtered_data': None,
            'comparison_data': None
        })
        logger.info("Filter state cleared")
    
    @classmethod
    def clear_all(cls):
        """Clear all session state (nuclear option)."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        cls.initialize()
        logger.info("All session state cleared and reinitialized")
    
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current session state.
        
        Returns:
            Dict with state summary (useful for debugging)
        """
        return {
            'data_loaded': cls.get('data_loaded'),
            'num_trained_models': len(cls.get('trained_models', {})),
            'current_page': cls.get('current_page'),
            'training_in_progress': cls.get('training_in_progress'),
            'has_forecasts': len(cls.get('forecast_results', {})) > 0,
            'filter_enabled': cls.get('filter_enabled'),
            'comparison_enabled': cls.get('comparison_enabled')
        }
    
    @classmethod
    def validate_state(cls) -> Dict[str, Any]:
        """
        Validate current session state integrity.
        
        Returns:
            Dict with validation results
        """
        issues = []
        
        # Check if data is loaded but column references are missing
        if cls.get('data_loaded') and cls.get('current_data') is None:
            issues.append("data_loaded is True but current_data is None")
        
        # Check if models exist but no results
        trained_models = cls.get('trained_models', {})
        model_results = cls.get('model_results', {})
        
        if len(trained_models) > 0 and len(model_results) == 0:
            issues.append("Models exist but no results recorded")
        
        # Check for orphaned results
        if len(model_results) > len(trained_models):
            issues.append("More results than trained models")
        
        # ✅ NEW: Validate filter state
        if cls.get('filter_enabled'):
            if cls.get('filter_start_date') is None or cls.get('filter_end_date') is None:
                issues.append("Filter enabled but date range not set")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    # ============================================================================
    # ✅ NEW: FILTER-SPECIFIC METHODS (Phase 3 - Session 1)
    # ============================================================================
    
    @classmethod
    def set_date_filter(cls, start_date: datetime, end_date: datetime):
        """
        Set date range filter.
        
        Args:
            start_date: Filter start date
            end_date: Filter end date
        """
        cls.update({
            'filter_enabled': True,
            'filter_start_date': start_date,
            'filter_end_date': end_date,
            'filter_preset': None  # Clear preset when custom dates set
        })
        logger.info(f"Date filter set: {start_date} to {end_date}")
    
       
    @classmethod
    def set_quick_filter(cls, preset: str):
        """
        Set quick filter preset.
        
        ✅ FIXED: Uses data's actual max date, not current date
        
        Args:
            preset: One of 'last_7d', 'last_30d', 'last_90d', 'mtd', 'qtd', 'ytd'
        """
        # ✅ FIX: Get data and date column first
        df = cls.get('current_data')
        date_col = cls.get('date_column')
        
        if df is None or date_col is None:
            logger.warning("Cannot set quick filter: no data or date column")
            return
        
        try:
            # ✅ FIX: Ensure date column exists
            if date_col not in df.columns:
                logger.error(f"Date column '{date_col}' not found in data")
                return
            
            # ✅ FIX: Ensure date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
            
            # ✅ FIX: Use data's max date as reference (NOT datetime.now()!)
            end_date = pd.Timestamp(df[date_col].max())
            
            logger.info(f"Quick filter: Using data max date as reference: {end_date.date()}")
            
            # Calculate start date based on preset
            if preset == 'last_7d':
                start_date = end_date - timedelta(days=7)
            elif preset == 'last_30d':
                start_date = end_date - timedelta(days=30)
            elif preset == 'last_90d':
                start_date = end_date - timedelta(days=90)
            elif preset == 'mtd':
                start_date = end_date.replace(day=1)
            elif preset == 'qtd':
                quarter_start_month = ((end_date.month - 1) // 3) * 3 + 1
                start_date = end_date.replace(month=quarter_start_month, day=1)
            elif preset == 'ytd':
                start_date = end_date.replace(month=1, day=1)
            else:
                logger.warning(f"Unknown preset: {preset}")
                return
            
            # Update state with calculated dates
            cls.update({
                'filter_enabled': True,
                'filter_preset': preset,
                'filter_start_date': start_date,
                'filter_end_date': end_date
            })
            
            logger.info(f"Quick filter set: {preset} → {start_date.date()} to {end_date.date()}")
            
        except Exception as e:
            logger.error(f"Error setting quick filter '{preset}': {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
   

    @classmethod
    def set_product_filter(cls, products: List[str]):
        """
        Set product filter.
        
        Args:
            products: List of product IDs to filter
        """
        cls.update({
            'filter_products': products
        })
        logger.info(f"Product filter set: {len(products)} products")
    
    @classmethod
    def set_category_filter(cls, categories: List[str]):
        """
        Set category filter.
        
        Args:
            categories: List of categories to filter
        """
        cls.update({
            'filter_categories': categories
        })
        logger.info(f"Category filter set: {len(categories)} categories")
    
    @classmethod
    def toggle_comparison(cls, enabled: bool = None):
        """
        Toggle comparison mode.
        
        Args:
            enabled: If provided, set to this value. Otherwise toggle.
        """
        if enabled is None:
            enabled = not cls.get('comparison_enabled', False)
        
        cls.set('comparison_enabled', enabled)
        logger.info(f"Comparison mode: {'enabled' if enabled else 'disabled'}")
    
    @classmethod
    def get_filter_summary(cls) -> Dict[str, Any]:
        """
        Get summary of active filters.
        
        Returns:
            Dict with filter information
        """
        return {
            'enabled': cls.get('filter_enabled'),
            'start_date': cls.get('filter_start_date'),
            'end_date': cls.get('filter_end_date'),
            'preset': cls.get('filter_preset'),
            'products': cls.get('filter_products', []),
            'categories': cls.get('filter_categories', []),
            'comparison_enabled': cls.get('comparison_enabled'),
            'has_filters': cls.is_filtered()
        }
    
    @classmethod
    def is_filtered(cls) -> bool:
        """
        Check if any filters are active.
        
        Returns:
            True if filters are applied
        """
        return (
            cls.get('filter_enabled', False) or
            len(cls.get('filter_products', [])) > 0 or
            len(cls.get('filter_categories', [])) > 0
        )
    
    @classmethod
    def get_filtered_data(cls) -> Optional[pd.DataFrame]:
        """
        Get filtered data (from cache or compute).
        
        Returns:
            Filtered DataFrame or None
        """
        # Check cache first
        cached = cls.get('filtered_data')
        if cached is not None:
            return cached
        
        # If no cache, return original data
        return cls.get('current_data')
    @staticmethod
    def get_trained_models():
        """Get trained models from session state."""
        return st.session_state.get('trained_models', {})

    @staticmethod
    def get_model_results():
        """Get model results from session state."""
        return st.session_state.get('model_results', {})

    @staticmethod
    def get_trained_model(model_name):
        """Get specific trained model."""
        models = StateManager.get_trained_models()
        return models.get(model_name)


# ============================================================================
# CONVENIENCE FUNCTIONS (Original + New)
# ============================================================================

def is_data_loaded() -> bool:
    """Check if data is loaded."""
    return StateManager.get('data_loaded', False)


def get_current_data():
    """Get current dataframe."""
    return StateManager.get('current_data')


def get_trained_models() -> Dict:
    """Get all trained models."""
    return StateManager.get('trained_models', {})


def set_training_progress(in_progress: bool):
    """Update training progress flag."""
    StateManager.set('training_in_progress', in_progress)


# ✅ NEW: Filter convenience functions
def is_filtered() -> bool:
    """Check if any filters are active."""
    return StateManager.is_filtered()


def get_filter_summary() -> Dict[str, Any]:
    """Get summary of active filters."""
    return StateManager.get_filter_summary()


def get_filtered_data():
    """Get filtered dataframe."""
    return StateManager.get_filtered_data()
