"""
Centralized State Management for CortexX Forecasting Platform.

NEW: Single source of truth for all session state management.
Eliminates duplicate state initialization across files.
"""

import streamlit as st
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """
    Centralized session state manager for the CortexX platform.
    
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
        
        # Feature engineering state
        'engineered_features': [],
        'selected_features': [],
        
        # Training configuration
        'training_in_progress': False,
        'last_training_time': None,
        
        # Forecast state
        'forecast_results': {},
        'forecast_dates': None,
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
            'has_forecasts': len(cls.get('forecast_results', {})) > 0
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
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


# Convenience functions for common operations
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
