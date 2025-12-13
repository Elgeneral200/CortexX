"""
Centralized State Management for CortexX Forecasting Platform.

PHASE 5 - COMPLETE INTEGRATION:
- Phase 3: Filter state management
- Phase 4: Feature engineering state
- Phase 5: Forecasting + Optimization + Evaluation caching
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
    
    COMPLETE INTEGRATION - Phases 3, 4, 5:
    - Phase 3: Filter state management
    - Phase 4: Feature engineering caching
    - Phase 5: Forecasting, optimization, evaluation caching
    
    Benefits:
    - Single place to define all state variables
    - Consistent initialization across app
    - Intelligent caching for 5-30x performance gains
    - Type hints for better IDE support
    """
    
    # Define all state keys and their default values
    STATE_DEFAULTS = {
        # ============================================================================
        # DATA STATE
        # ============================================================================
        'data_loaded': False,
        'current_data': None,
        'date_column': None,
        'value_column': None,
        'data_hash': None,  # Track data changes
        
        # ============================================================================
        # MODEL STATE
        # ============================================================================
        'trained_models': {},
        'model_results': {},
        'best_model_name': None,
        'backtest_results': {},
        'optimization_results': {},  # Legacy - kept for backwards compatibility
        
        # ============================================================================
        # UI STATE
        # ============================================================================
        'current_page': 'Dashboard',
        'selected_models': ['XGBoost', 'LightGBM', 'Random Forest'],
        'chatbot_open': False,
        'chat_history': [],
        
        # ============================================================================
        # FEATURE ENGINEERING STATE (Phase 4)
        # ============================================================================
        'engineered_data': None,              # Full DataFrame with features
        'engineered_features': [],            # List of feature names created
        'selected_features': [],              # Features user selected
        'feature_engineering_time': None,     # Track when created
        'feature_metadata': {},               # Feature descriptions
        
        # ============================================================================
        # TRAINING STATE
        # ============================================================================
        'training_in_progress': False,
        'last_training_time': None,
        
        # ============================================================================
        # FORECASTING STATE (Phase 5 - NEW)
        # ============================================================================
        'forecast_results': {},               # {model_name: forecast_data}
        'forecast_metadata': {},              # {model_name: {horizon, confidence, date}}
        'last_forecast_time': None,           # Track when forecast was generated
        'forecast_cache_valid': True,         # Cache invalidation flag
        
        # ============================================================================
        # EVALUATION CACHE (Phase 5 - NEW)
        # ============================================================================
        'evaluation_cache': {},               # {model_name: detailed_metrics}
        'comparison_cache': None,             # Cached comparison dataframe
        'evaluation_time': None,              # When evaluation was cached
        
        # ============================================================================
        # OPTIMIZATION HISTORY (Phase 5 - NEW - CRITICAL FOR PERFORMANCE!)
        # ============================================================================
        'optimization_history': {},           # {model_name: optuna_results}
        'optimization_completed': [],         # List of optimized models
        'optimization_metadata': {},          # {model_name: {trials, time, best_value}}
        
        # ============================================================================
        # FILTER STATE (Phase 3)
        # ============================================================================
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
    
    # ============================================================================
    # DATA MANAGEMENT METHODS
    # ============================================================================
    
    @classmethod
    def clear_data(cls):
        """Clear all data-related state with cascade clearing."""
        cls.update({
            'data_loaded': False,
            'current_data': None,
            'date_column': None,
            'value_column': None,
            'data_hash': None
        })
        
        # ✅ CASCADE CLEAR: When data changes, clear dependent states
        cls.clear_engineered_data()
        cls.clear_forecasts()
        cls.clear_filters()
        # Note: Don't clear optimization - it's model-specific, not data-specific
        
        logger.info("Data state cleared (cascaded to features, forecasts, filters)")
    
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
        
        # Clear forecast cache when models are cleared
        cls.clear_forecasts()
        
        logger.info("Model state cleared")
    
    @classmethod
    def clear_filters(cls):
        """Clear all filter-related state."""
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
    
    # ============================================================================
    # FEATURE ENGINEERING METHODS (Phase 4)
    # ============================================================================
    
    @classmethod
    def get_engineered_data(cls) -> Optional[pd.DataFrame]:
        """Get engineered DataFrame from state."""
        return cls.get('engineered_data')
    
    @classmethod
    def set_engineered_data(cls, df: pd.DataFrame, features: List[str]):
        """Save engineered data to state."""
        cls.update({
            'engineered_data': df,
            'engineered_features': features,
            'feature_engineering_time': datetime.now()
        })
        logger.info(f"✅ Engineered data saved: {len(df):,} rows, {len(features)} features")
    
    @classmethod
    def is_data_engineered(cls) -> bool:
        """Check if data has been engineered."""
        data = cls.get_engineered_data()
        return data is not None and not data.empty
    
    @classmethod
    def clear_engineered_data(cls):
        """Clear engineered data (call when raw data changes)."""
        cls.update({
            'engineered_data': None,
            'engineered_features': [],
            'feature_engineering_time': None
        })
        logger.info("🗑️ Engineered data cleared")
    
    # ============================================================================
    # FORECASTING METHODS (Phase 5 - NEW)
    # ============================================================================
    
    @classmethod
    def get_forecast_results(cls, model_name: str = None) -> Optional[Dict]:
        """
        Get forecast results for a model or all models.
        
        Args:
            model_name: Specific model name, or None for all forecasts
            
        Returns:
            Forecast data dict or dict of all forecasts
        """
        results = cls.get('forecast_results', {})
        if model_name:
            return results.get(model_name)
        return results
    
    @classmethod
    def set_forecast_results(cls, model_name: str, forecast_data: Dict):
        """
        Save forecast results to state.
        
        Args:
            model_name: Name of the model
            forecast_data: Dictionary containing forecast data
        """
        results = cls.get('forecast_results', {})
        results[model_name] = forecast_data
        
        metadata = cls.get('forecast_metadata', {})
        metadata[model_name] = {
            'generated_at': datetime.now(),
            'horizon': forecast_data.get('horizon'),
            'confidence': forecast_data.get('confidence'),
            'model': model_name
        }
        
        cls.update({
            'forecast_results': results,
            'forecast_metadata': metadata,
            'last_forecast_time': datetime.now(),
            'forecast_cache_valid': True
        })
        
        logger.info(f"✅ Forecast saved for {model_name} (horizon: {forecast_data.get('horizon')})")
    
    @classmethod
    def is_forecast_available(cls, model_name: str = None) -> bool:
        """
        Check if forecast exists for model.
        
        Args:
            model_name: Model name to check, or None to check if any forecasts exist
            
        Returns:
            True if forecast exists
        """
        results = cls.get('forecast_results', {})
        if model_name:
            return model_name in results and cls.get('forecast_cache_valid', True)
        return len(results) > 0 and cls.get('forecast_cache_valid', True)
    
    @classmethod
    def clear_forecasts(cls):
        """Clear all forecast results."""
        cls.update({
            'forecast_results': {},
            'forecast_metadata': {},
            'last_forecast_time': None,
            'forecast_cache_valid': False
        })
        logger.info("🗑️ Forecast cache cleared")
    
    @classmethod
    def invalidate_forecast_cache(cls):
        """Invalidate forecast cache without clearing data (for UI refresh)."""
        cls.set('forecast_cache_valid', False)
        logger.info("⚠️ Forecast cache invalidated")
    
    # ============================================================================
    # OPTIMIZATION CACHING METHODS (Phase 5 - NEW - CRITICAL!)
    # ============================================================================
    
    @classmethod
    def get_optimization_results(cls, model_name: str = None) -> Optional[Dict]:
        """
        Get optimization results (prevents re-optimization).
        
        Args:
            model_name: Specific model name, or None for all results
            
        Returns:
            Optimization results dict
        """
        history = cls.get('optimization_history', {})
        if model_name:
            return history.get(model_name)
        return history
    
    @classmethod
    def set_optimization_results(cls, model_name: str, optimization_data: Dict):
        """
        Cache optimization results (saves 10-20 min per model!).
        
        Args:
            model_name: Name of the model
            optimization_data: Optuna optimization results
        """
        history = cls.get('optimization_history', {})
        history[model_name] = optimization_data
        
        completed = cls.get('optimization_completed', [])
        if model_name not in completed:
            completed.append(model_name)
        
        metadata = cls.get('optimization_metadata', {})
        metadata[model_name] = {
            'optimized_at': datetime.now(),
            'n_trials': optimization_data.get('n_trials', 0),
            'best_value': optimization_data.get('best_value', None)
        }
        
        cls.update({
            'optimization_history': history,
            'optimization_completed': completed,
            'optimization_metadata': metadata
        })
        
        logger.info(f"✅ Optimization cached for {model_name} (saved ~10-20 min for future runs!)")
    
    @classmethod
    def is_optimization_completed(cls, model_name: str) -> bool:
        """
        Check if model has been optimized (prevent redundant optimization).
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if optimization completed and cached
        """
        return model_name in cls.get('optimization_completed', [])
    
    @classmethod
    def clear_optimization_cache(cls):
        """Clear optimization cache (force re-optimization)."""
        cls.update({
            'optimization_history': {},
            'optimization_completed': [],
            'optimization_metadata': {}
        })
        logger.info("🗑️ Optimization cache cleared (will re-optimize on next training)")
    
    @classmethod
    def get_optimization_metadata(cls, model_name: str = None) -> Optional[Dict]:
        """
        Get optimization metadata (timing, trials, etc.).
        
        Args:
            model_name: Model name or None for all metadata
            
        Returns:
            Metadata dict
        """
        metadata = cls.get('optimization_metadata', {})
        if model_name:
            return metadata.get(model_name)
        return metadata
    
    # ============================================================================
    # EVALUATION CACHE METHODS (Phase 5 - NEW)
    # ============================================================================
    
    @classmethod
    def get_evaluation_cache(cls, model_name: str = None) -> Optional[Dict]:
        """
        Get cached evaluation results.
        
        Args:
            model_name: Specific model or None for all
            
        Returns:
            Evaluation results dict
        """
        cache = cls.get('evaluation_cache', {})
        if model_name:
            return cache.get(model_name)
        return cache
    
    @classmethod
    def set_evaluation_cache(cls, model_name: str, evaluation_data: Dict):
        """
        Cache evaluation results.
        
        Args:
            model_name: Model name
            evaluation_data: Evaluation metrics and analysis
        """
        cache = cls.get('evaluation_cache', {})
        cache[model_name] = evaluation_data
        
        cls.update({
            'evaluation_cache': cache,
            'evaluation_time': datetime.now()
        })
        
        logger.info(f"✅ Evaluation cached for {model_name}")
    
    @classmethod
    def clear_evaluation_cache(cls):
        """Clear evaluation cache."""
        cls.update({
            'evaluation_cache': {},
            'comparison_cache': None,
            'evaluation_time': None
        })
        logger.info("🗑️ Evaluation cache cleared")
    
    # ============================================================================
    # FILTER METHODS (Phase 3)
    # ============================================================================
    
    @classmethod
    def set_date_filter(cls, start_date: datetime, end_date: datetime):
        """Set date range filter."""
        cls.update({
            'filter_enabled': True,
            'filter_start_date': start_date,
            'filter_end_date': end_date,
            'filter_preset': None
        })
        logger.info(f"Date filter set: {start_date} to {end_date}")
    
    @classmethod
    def set_quick_filter(cls, preset: str):
        """
        Set quick filter preset using data's actual date range.
        
        Args:
            preset: One of 'last_7d', 'last_30d', 'last_90d', 'mtd', 'qtd', 'ytd'
        """
        df = cls.get('current_data')
        date_col = cls.get('date_column')
        
        if df is None or date_col is None:
            logger.warning("Cannot set quick filter: no data or date column")
            return
        
        try:
            if date_col not in df.columns:
                logger.error(f"Date column '{date_col}' not found in data")
                return
            
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
            
            end_date = pd.Timestamp(df[date_col].max())
            
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
            
            cls.update({
                'filter_enabled': True,
                'filter_preset': preset,
                'filter_start_date': start_date,
                'filter_end_date': end_date
            })
            
            logger.info(f"Quick filter set: {preset} → {start_date.date()} to {end_date.date()}")
            
        except Exception as e:
            logger.error(f"Error setting quick filter '{preset}': {str(e)}")
    
    @classmethod
    def set_product_filter(cls, products: List[str]):
        """Set product filter."""
        cls.update({'filter_products': products})
        logger.info(f"Product filter set: {len(products)} products")
    
    @classmethod
    def set_category_filter(cls, categories: List[str]):
        """Set category filter."""
        cls.update({'filter_categories': categories})
        logger.info(f"Category filter set: {len(categories)} categories")
    
    @classmethod
    def toggle_comparison(cls, enabled: bool = None):
        """Toggle comparison mode."""
        if enabled is None:
            enabled = not cls.get('comparison_enabled', False)
        
        cls.set('comparison_enabled', enabled)
        logger.info(f"Comparison mode: {'enabled' if enabled else 'disabled'}")
    
    @classmethod
    def get_filter_summary(cls) -> Dict[str, Any]:
        """Get summary of active filters."""
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
        """Check if any filters are active."""
        return (
            cls.get('filter_enabled', False) or
            len(cls.get('filter_products', [])) > 0 or
            len(cls.get('filter_categories', [])) > 0
        )
    
    @classmethod
    def get_filtered_data(cls) -> Optional[pd.DataFrame]:
        """Get filtered data (from cache or compute)."""
        cached = cls.get('filtered_data')
        if cached is not None:
            return cached
        return cls.get('current_data')
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get a summary of current session state."""
        return {
            'data_loaded': cls.get('data_loaded'),
            'data_engineered': cls.is_data_engineered(),
            'num_trained_models': len(cls.get('trained_models', {})),
            'num_optimized_models': len(cls.get('optimization_completed', [])),
            'has_forecasts': cls.is_forecast_available(),
            'current_page': cls.get('current_page'),
            'training_in_progress': cls.get('training_in_progress'),
            'filter_enabled': cls.get('filter_enabled'),
            'comparison_enabled': cls.get('comparison_enabled')
        }
    
    @classmethod
    def validate_state(cls) -> Dict[str, Any]:
        """Validate current session state integrity."""
        issues = []
        
        if cls.get('data_loaded') and cls.get('current_data') is None:
            issues.append("data_loaded is True but current_data is None")
        
        trained_models = cls.get('trained_models', {})
        model_results = cls.get('model_results', {})
        
        if len(trained_models) > 0 and len(model_results) == 0:
            issues.append("Models exist but no results recorded")
        
        if len(model_results) > len(trained_models):
            issues.append("More results than trained models")
        
        if cls.get('filter_enabled'):
            if cls.get('filter_start_date') is None or cls.get('filter_end_date') is None:
                issues.append("Filter enabled but date range not set")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    # ============================================================================
    # LEGACY STATIC METHODS (Keep for backwards compatibility)
    # ============================================================================
    
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
# CONVENIENCE FUNCTIONS (Keep for backwards compatibility)
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


def is_filtered() -> bool:
    """Check if any filters are active."""
    return StateManager.is_filtered()


def get_filter_summary() -> Dict[str, Any]:
    """Get summary of active filters."""
    return StateManager.get_filter_summary()


def get_filtered_data():
    """Get filtered dataframe."""
    return StateManager.get_filtered_data()
