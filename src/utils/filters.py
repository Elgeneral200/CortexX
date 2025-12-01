"""
Filter Utilities for CortexX Forecasting Platform

PHASE 3 - SESSION 1: Comprehensive filtering system
- Date range filters with presets
- Product/category filters
- Comparison period calculations
- Data filtering logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DateFilterPresets:
    """
    Pre-defined date filter presets.
    
    ✅ NEW: Phase 3 - Session 1
    """
    
    LAST_7_DAYS = 'last_7d'
    LAST_30_DAYS = 'last_30d'
    LAST_90_DAYS = 'last_90d'
    MONTH_TO_DATE = 'mtd'
    QUARTER_TO_DATE = 'qtd'
    YEAR_TO_DATE = 'ytd'
    
    @staticmethod
    def get_date_range(preset: str, reference_date: datetime = None) -> Tuple[datetime, datetime]:
        """
        Calculate date range for a given preset.
        
        Args:
            preset: One of the preset constants
            reference_date: Reference date (defaults to today)
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        end_date = reference_date
        
        if preset == DateFilterPresets.LAST_7_DAYS:
            start_date = end_date - timedelta(days=7)
        elif preset == DateFilterPresets.LAST_30_DAYS:
            start_date = end_date - timedelta(days=30)
        elif preset == DateFilterPresets.LAST_90_DAYS:
            start_date = end_date - timedelta(days=90)
        elif preset == DateFilterPresets.MONTH_TO_DATE:
            start_date = end_date.replace(day=1)
        elif preset == DateFilterPresets.QUARTER_TO_DATE:
            quarter_start_month = ((end_date.month - 1) // 3) * 3 + 1
            start_date = end_date.replace(month=quarter_start_month, day=1)
        elif preset == DateFilterPresets.YEAR_TO_DATE:
            start_date = end_date.replace(month=1, day=1)
        else:
            logger.warning(f"Unknown preset: {preset}, defaulting to last 30 days")
            start_date = end_date - timedelta(days=30)
        
        return start_date, end_date
    
    @staticmethod
    def get_preset_label(preset: str) -> str:
        """
        Get human-readable label for preset.
        
        Args:
            preset: Preset constant
            
        Returns:
            Human-readable label
        """
        labels = {
            'last_7d': 'Last 7 Days',
            'last_30d': 'Last 30 Days',
            'last_90d': 'Last 90 Days',
            'mtd': 'Month to Date',
            'qtd': 'Quarter to Date',
            'ytd': 'Year to Date'
        }
        return labels.get(preset, preset)
    
    @staticmethod
    def get_all_presets() -> List[Dict[str, str]]:
        """
        Get all available presets.
        
        Returns:
            List of dicts with 'value' and 'label' keys
        """
        return [
            {'value': 'last_7d', 'label': 'Last 7 Days'},
            {'value': 'last_30d', 'label': 'Last 30 Days'},
            {'value': 'last_90d', 'label': 'Last 90 Days'},
            {'value': 'mtd', 'label': 'Month to Date'},
            {'value': 'qtd', 'label': 'Quarter to Date'},
            {'value': 'ytd', 'label': 'Year to Date'}
        ]


class DataFilter:
    """
    Main data filtering class.
    
    ✅ NEW: Phase 3 - Session 1
    """
    
    def __init__(self, df: pd.DataFrame, date_column: str = None):
        """
        Initialize data filter.
        
        Args:
            df: DataFrame to filter
            date_column: Name of date column
        """
        self.df = df.copy() if df is not None else None
        self.date_column = date_column
    
    def apply_date_filter(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Apply date range filter.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None or self.date_column is None:
            logger.warning("Cannot apply date filter: missing data or date column")
            return self.df
        
        if self.date_column not in self.df.columns:
            logger.warning(f"Date column '{self.date_column}' not found in dataframe")
            return self.df
        
        try:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_column]):
                self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
            
            # Apply filter
            mask = (self.df[self.date_column] >= start_date) & (self.df[self.date_column] <= end_date)
            filtered_df = self.df[mask].copy()
            
            logger.info(f"Date filter applied: {len(filtered_df)} of {len(self.df)} records retained")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error applying date filter: {str(e)}")
            return self.df
    
    def apply_product_filter(self, products: List[str], product_column: str = 'product_id') -> pd.DataFrame:
        """
        Apply product filter.
        
        Args:
            products: List of product IDs to include
            product_column: Name of product column
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None or not products:
            return self.df
        
        if product_column not in self.df.columns:
            logger.warning(f"Product column '{product_column}' not found")
            return self.df
        
        try:
            filtered_df = self.df[self.df[product_column].isin(products)].copy()
            logger.info(f"Product filter applied: {len(filtered_df)} of {len(self.df)} records retained")
            return filtered_df
        except Exception as e:
            logger.error(f"Error applying product filter: {str(e)}")
            return self.df
    
    def apply_category_filter(self, categories: List[str], category_column: str = 'category') -> pd.DataFrame:
        """
        Apply category filter.
        
        Args:
            categories: List of categories to include
            category_column: Name of category column
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None or not categories:
            return self.df
        
        if category_column not in self.df.columns:
            logger.warning(f"Category column '{category_column}' not found")
            return self.df
        
        try:
            filtered_df = self.df[self.df[category_column].isin(categories)].copy()
            logger.info(f"Category filter applied: {len(filtered_df)} of {len(self.df)} records retained")
            return filtered_df
        except Exception as e:
            logger.error(f"Error applying category filter: {str(e)}")
            return self.df
    
    def apply_all_filters(self, 
                         start_date: datetime = None,
                         end_date: datetime = None,
                         products: List[str] = None,
                         categories: List[str] = None,
                         product_column: str = 'product_id',
                         category_column: str = 'category') -> pd.DataFrame:
        """
        Apply all filters at once.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            products: List of products to include
            categories: List of categories to include
            product_column: Name of product column
            category_column: Name of category column
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.copy() if self.df is not None else None
        
        # Apply date filter
        if start_date and end_date:
            self.df = filtered_df
            filtered_df = self.apply_date_filter(start_date, end_date)
        
        # Apply product filter
        if products and len(products) > 0:
            self.df = filtered_df
            filtered_df = self.apply_product_filter(products, product_column)
        
        # Apply category filter
        if categories and len(categories) > 0:
            self.df = filtered_df
            filtered_df = self.apply_category_filter(categories, category_column)
        
        return filtered_df


class ComparisonPeriodCalculator:
    """
    Calculate comparison periods for analysis.
    
    ✅ NEW: Phase 3 - Session 1
    """
    
    @staticmethod
    def get_previous_period(start_date: datetime, end_date: datetime) -> Tuple[datetime, datetime]:
        """
        Get the previous period of same duration.
        
        Args:
            start_date: Current period start
            end_date: Current period end
            
        Returns:
            Tuple of (comparison_start, comparison_end)
        """
        period_duration = end_date - start_date
        comparison_end = start_date - timedelta(days=1)
        comparison_start = comparison_end - period_duration
        
        return comparison_start, comparison_end
    
    @staticmethod
    def get_previous_year_period(start_date: datetime, end_date: datetime) -> Tuple[datetime, datetime]:
        """
        Get the same period from previous year.
        
        Args:
            start_date: Current period start
            end_date: Current period end
            
        Returns:
            Tuple of (comparison_start, comparison_end)
        """
        try:
            comparison_start = start_date.replace(year=start_date.year - 1)
            comparison_end = end_date.replace(year=end_date.year - 1)
            return comparison_start, comparison_end
        except ValueError:
            # Handle leap year edge case
            comparison_start = start_date - timedelta(days=365)
            comparison_end = end_date - timedelta(days=365)
            return comparison_start, comparison_end
    
    @staticmethod
    def calculate_period_change(current_value: float, comparison_value: float) -> Dict[str, Any]:
        """
        Calculate change between periods.
        
        Args:
            current_value: Current period value
            comparison_value: Comparison period value
            
        Returns:
            Dict with change metrics
        """
        if comparison_value == 0:
            return {
                'absolute_change': current_value - comparison_value,
                'percent_change': 0 if current_value == 0 else float('inf'),
                'direction': 'up' if current_value > 0 else 'flat'
            }
        
        absolute_change = current_value - comparison_value
        percent_change = (absolute_change / comparison_value) * 100
        
        return {
            'absolute_change': absolute_change,
            'percent_change': percent_change,
            'direction': 'up' if percent_change > 0 else 'down' if percent_change < 0 else 'flat'
        }


def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """
    Get unique values from a column for filter options.
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        Sorted list of unique values
    """
    if df is None or column not in df.columns:
        return []
    
    try:
        unique_values = df[column].dropna().unique().tolist()
        return sorted(unique_values)
    except Exception as e:
        logger.error(f"Error getting unique values from {column}: {str(e)}")
        return []


def format_filter_summary(filter_info: Dict[str, Any]) -> str:
    """
    Format filter information into human-readable string.
    
    Args:
        filter_info: Dictionary with filter information
        
    Returns:
        Formatted summary string
    """
    if not filter_info.get('has_filters', False):
        return "No filters applied"
    
    parts = []
    
    # Date filter
    if filter_info.get('start_date') and filter_info.get('end_date'):
        start = filter_info['start_date'].strftime('%Y-%m-%d')
        end = filter_info['end_date'].strftime('%Y-%m-%d')
        
        if filter_info.get('preset'):
            preset_label = DateFilterPresets.get_preset_label(filter_info['preset'])
            parts.append(f"📅 {preset_label}")
        else:
            parts.append(f"📅 {start} to {end}")
    
    # Product filter
    products = filter_info.get('products', [])
    if products:
        parts.append(f"🏷️ {len(products)} product(s)")
    
    # Category filter
    categories = filter_info.get('categories', [])
    if categories:
        parts.append(f"📂 {len(categories)} category(ies)")
    
    # Comparison
    if filter_info.get('comparison_enabled'):
        parts.append("🔄 Comparison enabled")
    
    return " | ".join(parts) if parts else "No filters applied"


def apply_filters_from_state(df: pd.DataFrame, 
                             date_column: str,
                             state_manager) -> pd.DataFrame:
    """
    Apply filters based on StateManager settings.
    
    ✅ FIXED: Only applies filters for columns that exist in the data
    
    Args:
        df: DataFrame to filter
        date_column: Name of date column
        state_manager: StateManager instance
        
    Returns:
        Filtered DataFrame
    """
    if df is None:
        return None
    
    filter_summary = state_manager.get_filter_summary()
    
    # If no filters active, return original data
    if not filter_summary.get('has_filters'):
        return df
    
    # Start with full dataset
    filtered_df = df.copy()
    
    # Get filter parameters
    start_date = filter_summary.get('start_date')
    end_date = filter_summary.get('end_date')
    products = filter_summary.get('products', [])
    categories = filter_summary.get('categories', [])
    
    logger.info(f"Starting filter: {len(filtered_df)} records")
    
    # ============================================================================
    # APPLY DATE FILTER (if dates are set)
    # ============================================================================
    if start_date and end_date and date_column:
        data_filter = DataFilter(filtered_df, date_column)
        filtered_df = data_filter.apply_date_filter(start_date, end_date)
        logger.info(f"After date filter: {len(filtered_df)} records")
    
    # ============================================================================
    # APPLY PRODUCT FILTER (only if column exists)
    # ============================================================================
    if products and len(products) > 0:
        # Check for product column (multiple possible names)
        product_col = None
        possible_cols = ['product_id', 'Product ID', 'product', 'Product', 'ProductID']
        for col in possible_cols:
            if col in filtered_df.columns:
                product_col = col
                logger.info(f"Found product column: {col}")
                break
        
        if product_col:
            data_filter = DataFilter(filtered_df, date_column)
            filtered_df = data_filter.apply_product_filter(products, product_col)
            logger.info(f"After product filter: {len(filtered_df)} records")
        else:
            logger.warning(f"Product filter skipped - no product column found. Available columns: {list(filtered_df.columns)}")
    
    # ============================================================================
    # APPLY CATEGORY FILTER (only if column exists)
    # ============================================================================
    if categories and len(categories) > 0:
        # Check for category column (multiple possible names)
        category_col = None
        possible_cols = ['category', 'Category', 'product_category', 'Product Category', 'ProductCategory']
        for col in possible_cols:
            if col in filtered_df.columns:
                category_col = col
                logger.info(f"Found category column: {col}")
                break
        
        if category_col:
            data_filter = DataFilter(filtered_df, date_column)
            filtered_df = data_filter.apply_category_filter(categories, category_col)
            logger.info(f"After category filter: {len(filtered_df)} records")
        else:
            logger.warning(f"Category filter skipped - no category column found. Available columns: {list(filtered_df.columns)}")
    
    logger.info(f"Final filtered result: {len(df)} → {len(filtered_df)} records ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df
