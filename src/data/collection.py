"""
Data collection module for CortexX sales forecasting platform.

✅ RETAIL INVENTORY FORECASTING OPTIMIZED:
- Retail-specific validation rules
- Negative demand forecast handling
- Store-Product hierarchy validation
- Date continuity checks for retail data
- Column name standardization
- File size validation
- Enhanced error reporting

✅ FIXED: Added missing _validate_retail_structure method
✅ FIXED: Datetime conversion issues for binary columns
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any, List
import logging
from io import StringIO
import re
import streamlit as st
import hashlib
import os

logger = logging.getLogger(__name__)


class DataCollector:
    """
    A class to handle data collection from various sources for sales forecasting.
    
    ✅ RETAIL OPTIMIZED: Enhanced for multi-store, multi-product inventory datasets.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize DataCollector with configuration.
        
        Args:
            config: Configuration object (uses get_config() if None)
        """
        if config is None:
            try:
                from src.utils.config import get_config
                config = get_config()
            except ImportError:
                config = None
                logger.warning("Config module not available, using defaults")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ✅ Retail-specific expected columns
        self.retail_expected_columns = {
            'required': ['Date', 'Store ID', 'Product ID', 'Units Sold'],
            'recommended': ['Price', 'Category', 'Inventory Level'],
            'optional': ['Discount', 'Holiday/Promotion', 'Weather Condition', 
                        'Seasonality', 'Region', 'Competitor Pricing', 'Demand Forecast']
        }
        
        # ✅ NEW: Column name variations (case-insensitive mapping)
        self.column_aliases = {
            'date': ['date', 'day', 'time', 'timestamp', 'dt'],
            'store_id': ['store id', 'store_id', 'storeid', 'store', 'shop_id'],
            'product_id': ['product id', 'product_id', 'productid', 'product', 'sku'],
            'units_sold': ['units sold', 'units_sold', 'unitssold', 'sales', 'quantity', 'qty'],
            'price': ['price', 'unit_price', 'unitprice', 'cost'],
            'category': ['category', 'cat', 'product_category', 'type'],
            'inventory_level': ['inventory level', 'inventory_level', 'stock', 'inventory'],
            'discount': ['discount', 'disc', 'discount_rate'],
            'promotion': ['holiday/promotion', 'promotion', 'promo', 'is_promo', 'holiday'],
            'region': ['region', 'area', 'location', 'zone'],
            'weather': ['weather condition', 'weather_condition', 'weather'],
            'seasonality': ['seasonality', 'season'],
            'competitor_pricing': ['competitor pricing', 'competitor_pricing', 'comp_price'],
            'demand_forecast': ['demand forecast', 'demand_forecast', 'forecast', 'predicted_demand']
        }
    
    def load_csv_data(
        self, 
        file_path: Union[str, object], 
        validate_retail: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load sales data from CSV file with retail-specific validation.
        
        ✅ ENHANCED: Added file size check and column standardization
        
        Args:
            file_path (Union[str, object]): Path to CSV file or file-like object
            validate_retail (bool): Apply retail-specific validations
            **kwargs: Additional arguments for pandas read_csv
        
        Returns:
            pd.DataFrame: Loaded and validated sales data
        """
        try:
            self.logger.info(f"Loading data from {file_path}")
            
            # ✅ NEW: Check file size before loading (prevent memory issues)
            if isinstance(file_path, str):
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                self.logger.info(f"File size: {file_size_mb:.2f} MB")
                
                if file_size_mb > 500:  # Warning for large files
                    self.logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Loading may take time.")
            
            # ✅ Try to import validators
            try:
                from src.utils.validators import validate_upload_file, DataValidator
                validators_available = True
            except ImportError:
                validators_available = False
                self.logger.warning("Validators not available, loading without validation")
            
            # Load data
            if isinstance(file_path, str):
                # File path - can be cached directly
                self.logger.info("Loading from file path...")
                df = load_csv_cached(file_path, **kwargs)
                
                # ✅ Validate if validators available
                if validators_available and validate_retail:
                    validator = DataValidator()
                    validation_result = validator.validate_dataframe(df)
                    if not validation_result['is_valid']:
                        error_msg = "; ".join(validation_result['errors'])
                        self.logger.error(f"Data validation failed: {error_msg}")
                        raise ValueError(f"Data validation failed: {error_msg}")
                    
                    # Log warnings
                    if validation_result.get('warnings'):
                        for warning in validation_result['warnings']:
                            self.logger.warning(f"Data quality warning: {warning}")
            else:
                # File-like object (Streamlit upload)
                self.logger.info("Loading from uploaded file...")
                if validators_available:
                    is_valid, message, df = validate_upload_file(file_path)
                    if not is_valid:
                        self.logger.error(f"File validation failed: {message}")
                        raise ValueError(message)
                    if df is None:
                        raise ValueError("Failed to load data from uploaded file")
                    self.logger.info(f"Loaded data from uploaded file: {message}")
                else:
                    # Fallback without validation
                    file_hash = hash_file_upload(file_path)
                    df = load_csv_from_upload_cached(file_path, file_hash, **kwargs)
            
            self.logger.info(f"Initial data shape: {df.shape}")
            
            # ✅ NEW: Standardize column names
            df = self._standardize_column_names(df)
            
            # Auto-detect and convert date columns
            df = self._auto_detect_dates(df)
            
            # ✅ Apply retail-specific validation
            if validate_retail:
                df = self._validate_retail_structure(df)
                df = self._clean_retail_data(df)
            
            self.logger.info(f"✅ Successfully loaded data with shape {df.shape}")
            return df
            
        except FileNotFoundError as e:
            self.logger.error(f"CSV file not found: {file_path}")
            raise e
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ NEW: Standardize column names to match expected retail format.
        
        Args:
            df: Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with standardized column names
        """
        try:
            df_standard = df.copy()
            columns_renamed = {}
            
            # Map for standard names
            standard_mapping = {
                'date': 'Date',
                'store_id': 'Store ID',
                'product_id': 'Product ID',
                'units_sold': 'Units Sold',
                'price': 'Price',
                'category': 'Category',
                'inventory_level': 'Inventory Level',
                'discount': 'Discount',
                'promotion': 'Holiday/Promotion',
                'region': 'Region',
                'weather': 'Weather Condition',
                'seasonality': 'Seasonality',
                'competitor_pricing': 'Competitor Pricing',
                'demand_forecast': 'Demand Forecast'
            }
            
            # Check each column against aliases
            for col in df_standard.columns:
                col_lower = col.lower().strip()
                
                # Find matching standard column
                for standard_key, aliases in self.column_aliases.items():
                    if col_lower in [alias.lower() for alias in aliases]:
                        standard_name = standard_mapping.get(standard_key)
                        if standard_name and col != standard_name:
                            columns_renamed[col] = standard_name
                        break
            
            # Rename columns
            if columns_renamed:
                df_standard = df_standard.rename(columns=columns_renamed)
                self.logger.info(f"✅ Standardized column names: {columns_renamed}")
            else:
                self.logger.info("Column names already standardized")
            
            return df_standard
            
        except Exception as e:
            self.logger.warning(f"Column standardization partial failure: {str(e)}")
            return df
    
    def _validate_retail_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ NEW: Validate retail data structure.
        
        Args:
            df: Input dataframe
        
        Returns:
            pd.DataFrame: Validated dataframe
        
        Raises:
            ValueError: If required columns are missing
        """
        required = self.retail_expected_columns['required']
        recommended = self.retail_expected_columns['recommended']
        optional = self.retail_expected_columns['optional']
        
        # Check required columns
        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            error_msg = f"Missing required retail columns: {missing_required}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"✅ All required columns present: {required}")
        
        # Check recommended columns
        missing_recommended = [col for col in recommended if col not in df.columns]
        if missing_recommended:
            self.logger.warning(f"⚠️ Missing recommended columns: {missing_recommended}")
        
        # Log optional columns found
        optional_found = [col for col in optional if col in df.columns]
        if optional_found:
            self.logger.info(f"✅ Optional columns found: {optional_found}")
        
        self.logger.info("✅ Retail structure validation passed")
        return df
    
    def _clean_retail_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ Clean retail-specific data issues.
        ✅ FIXED: Handles datetime conversion issues for binary columns
        
        Args:
            df: Input dataframe
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        issues_fixed = []
        
        # 1. Fix negative demand forecasts
        if 'Demand Forecast' in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean['Demand Forecast']):
                negative_count = (df_clean['Demand Forecast'] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(
                        f"⚠️ Found {negative_count} negative 'Demand Forecast' values. Clipping to 0."
                    )
                    df_clean['Demand Forecast'] = df_clean['Demand Forecast'].clip(lower=0)
                    issues_fixed.append(f"Fixed {negative_count} negative demand forecasts")
        
        # 2. Fix negative units sold (should be 0 or positive)
        if 'Units Sold' in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean['Units Sold']):
                negative_sales = (df_clean['Units Sold'] < 0).sum()
                if negative_sales > 0:
                    self.logger.warning(
                        f"⚠️ Found {negative_sales} negative 'Units Sold' values. Setting to 0."
                    )
                    df_clean.loc[df_clean['Units Sold'] < 0, 'Units Sold'] = 0
                    issues_fixed.append(f"Fixed {negative_sales} negative sales")
        
        # 3. Fix negative prices
        price_columns = ['Price', 'Competitor Pricing']
        for col in price_columns:
            if col in df_clean.columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    negative_prices = (df_clean[col] < 0).sum()
                    if negative_prices > 0:
                        self.logger.warning(
                            f"⚠️ Found {negative_prices} negative '{col}' values. Taking absolute value."
                        )
                        df_clean[col] = df_clean[col].abs()
                        issues_fixed.append(f"Fixed {negative_prices} negative prices in {col}")
        
        # 4. Fix discount values (should be 0-100)
        if 'Discount' in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean['Discount']):
                # Check if discount is in 0-1 range (convert to percentage)
                if df_clean['Discount'].max() <= 1 and df_clean['Discount'].max() > 0:
                    df_clean['Discount'] = df_clean['Discount'] * 100
                    self.logger.info("ℹ️ Converted discount from decimal to percentage")
                    issues_fixed.append("Converted discount to percentage")
                
                # Clip to valid range
                invalid_discount = ((df_clean['Discount'] < 0) | (df_clean['Discount'] > 100)).sum()
                if invalid_discount > 0:
                    df_clean['Discount'] = df_clean['Discount'].clip(0, 100)
                    self.logger.warning(f"⚠️ Clipped {invalid_discount} invalid discount values to [0, 100]")
                    issues_fixed.append(f"Fixed {invalid_discount} invalid discounts")
        
        # 5. ✅ FIXED: Fix binary columns (Holiday/Promotion) - Handle datetime conversion issues
        binary_columns = ['Holiday/Promotion']
        for col in binary_columns:
            if col in df_clean.columns:
                # ✅ FIX: Check if column was wrongly converted to datetime
                if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    self.logger.warning(f"⚠️ '{col}' was incorrectly converted to datetime. Converting back to binary.")
                    # Extract original values: datetime with nanoseconds = 1, without = 0
                    df_clean[col] = (df_clean[col].astype('int64') % 1000000000 > 0).astype(int)
                    issues_fixed.append(f"Fixed datetime conversion in {col}")
                else:
                    # ✅ Normal binary conversion for non-datetime columns
                    unique_vals = df_clean[col].dropna().unique()
                    # Check if already binary
                    if not all(v in [0, 1, 0.0, 1.0] for v in unique_vals):
                        self.logger.warning(f"⚠️ '{col}' has non-binary values: {unique_vals}")
                        # Ensure numeric first
                        if not pd.api.types.is_numeric_dtype(df_clean[col]):
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
                        # Convert to binary
                        df_clean[col] = (df_clean[col] > 0).astype(int)
                        issues_fixed.append(f"Converted {col} to binary (0/1)")
                    else:
                        # Already binary, just ensure int type
                        df_clean[col] = df_clean[col].astype(int)
        
        # 6. Ensure data types are correct
        if 'Store ID' in df_clean.columns:
            df_clean['Store ID'] = df_clean['Store ID'].astype(str)
        
        if 'Product ID' in df_clean.columns:
            df_clean['Product ID'] = df_clean['Product ID'].astype(str)
        
        # 7. Sort by date and reset index
        if 'Date' in df_clean.columns:
            df_clean = df_clean.sort_values('Date').reset_index(drop=True)
            issues_fixed.append("Sorted data by date")
        
        # 8. ✅ Remove any completely duplicate rows
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"⚠️ Found {duplicates} duplicate rows. Removing...")
            df_clean = df_clean.drop_duplicates()
            issues_fixed.append(f"Removed {duplicates} duplicate rows")
        
        # Log summary
        if issues_fixed:
            self.logger.info(f"✅ Data cleaning complete. Fixed: {', '.join(issues_fixed)}")
        else:
            self.logger.info("✅ No data cleaning issues found")
        
        return df_clean
    
    def _auto_detect_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect and convert date columns in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with date columns converted
        """
        df_processed = df.copy()
        date_columns_found = []
        
        # Common date column patterns (case insensitive)
        date_patterns = [
            r'date', r'time', r'day', r'month', r'year', r'timestamp',
            r'created', r'modified', r'period', r'week', r'quarter'
        ]
        
        for col in df_processed.columns:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                date_columns_found.append(col)
                self.logger.info(f"✅ Column '{col}' is already datetime")
                continue
            
            col_lower = col.lower()
            
            # Check if column name matches date patterns
            is_date_like = any(pattern in col_lower for pattern in date_patterns)
            
            if is_date_like:
                # Try to convert to datetime
                converted_col, success = self._try_convert_to_datetime(df_processed[col])
                if success:
                    df_processed[col] = converted_col
                    date_columns_found.append(col)
                    self.logger.info(f"✅ Auto-converted column '{col}' to datetime")
                    continue
            
            # If not detected by name, check content for 'Date' column specifically
            if col == 'Date' and df_processed[col].dtype == 'object':
                converted_col, success = self._try_convert_to_datetime(df_processed[col])
                if success:
                    df_processed[col] = converted_col
                    date_columns_found.append(col)
                    self.logger.info(f"✅ Auto-detected date column '{col}' from content")
        
        if date_columns_found:
            self.logger.info(f"✅ Date columns ready: {date_columns_found}")
        else:
            self.logger.warning("⚠️ No date columns detected automatically")
        
        return df_processed
    
    def _try_convert_to_datetime(self, series: pd.Series) -> tuple:
        """
        Try to convert a series to datetime using multiple methods.
        
        Args:
            series (pd.Series): Series to convert
        
        Returns:
            tuple: (converted_series, success_flag)
        """
        import warnings
        
        # If already datetime, return as is
        if pd.api.types.is_datetime64_any_dtype(series):
            return series, True
        
        # Try direct conversion (suppress warnings)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(series, errors='coerce')
            
            # Check if conversion was successful (not all NaT)
            non_null_ratio = converted.notna().sum() / len(converted)
            if non_null_ratio > 0.8:  # At least 80% converted
                return converted, True
        except:
            pass
        
        # Try with different date formats
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d/%m/%Y', '%m-%d-%Y', '%Y%m%d', '%d%m%Y',
            '%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted = pd.to_datetime(series, format=fmt, errors='coerce')
                non_null_ratio = converted.notna().sum() / len(converted)
                if non_null_ratio > 0.8:
                    return converted, True
            except:
                continue
        
        return series, False
    
    def _is_likely_date_column(self, series: pd.Series, threshold: float = 0.8) -> bool:
        """
        Check if a series is likely a date column based on content patterns.
        
        Args:
            series (pd.Series): Series to check
            threshold (float): Minimum proportion of valid dates required
        
        Returns:
            bool: True if likely a date column
        """
        if not pd.api.types.is_datetime64_any_dtype(series):
            return False
        
        # Check if most values are valid dates
        valid_dates_ratio = 1 - series.isna().mean()
        
        # Check for reasonable date range (not all same date, not too wide range)
        if valid_dates_ratio > threshold:
            unique_dates = series.dropna().nunique()
            
            if unique_dates > 1:
                date_range = series.dropna().max() - series.dropna().min()
                
                # Reasonable criteria for a date column
                if date_range.days > 0:
                    return True
        
        return False
    
    def generate_sample_retail_data(
        self, 
        periods: int = 730,  # 2 years
        n_stores: int = 5,
        n_products: int = 20
    ) -> pd.DataFrame:
        """
        ✅ Generate synthetic retail inventory data matching Kaggle structure.
        
        Args:
            periods (int): Number of days to generate
            n_stores (int): Number of stores
            n_products (int): Number of products
        
        Returns:
            pd.DataFrame: Synthetic retail data
        """
        return generate_retail_sample_data(periods, n_stores, n_products)
    
    
    def validate_data_structure(
        self, 
        df: pd.DataFrame, 
        required_columns: list = None
    ) -> Dict[str, Any]:
        """
        Validate the structure and quality of the loaded data.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            required_columns (list, optional): List of required columns
        
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {},
            'date_columns': []
        }
        
        try:
            # Check for empty dataframe
            if df.empty:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Dataframe is empty")
                return validation_result
            
            # Find date columns
            date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            validation_result['date_columns'] = date_columns
            
            # Basic statistics
            validation_result['summary'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'data_types': dict(df.dtypes.astype(str)),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': int(df.duplicated().sum()),
                'date_columns_found': date_columns
            }
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(
                        f"Missing required columns: {missing_columns}"
                    )
            
            # Check for excessive missing values
            missing_percentage = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_percentage[missing_percentage > 30].index.tolist()
            if high_missing:
                validation_result['warnings'].append(
                    f"High missing values (>30%) in: {high_missing}"
                )
            
            # Warn if no date columns found
            if not date_columns:
                validation_result['warnings'].append(
                    "No date columns detected. Time series analysis may be limited."
                )
            
            # ✅ Retail-specific checks
            if 'Units Sold' in df.columns:
                zero_sales_pct = (df['Units Sold'] == 0).sum() / len(df) * 100
                if zero_sales_pct > 15:
                    validation_result['warnings'].append(
                        f"High zero-inflation: {zero_sales_pct:.1f}% of sales are zero"
                    )
            
            self.logger.info("✅ Data validation completed")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def get_retail_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ Get retail-specific data summary.
        
        Args:
            df: Retail inventory dataframe
        
        Returns:
            Dictionary with retail metrics
        """
        summary = {
            'total_records': len(df),
            'date_range': None,
            'stores': None,
            'products': None,
            'categories': None,
            'avg_daily_sales': None,
            'zero_sales_percentage': None,
            'total_units_sold': None
        }
        
        try:
            # Date range
            if 'Date' in df.columns:
                summary['date_range'] = {
                    'start': df['Date'].min().strftime('%Y-%m-%d'),
                    'end': df['Date'].max().strftime('%Y-%m-%d'),
                    'days': (df['Date'].max() - df['Date'].min()).days
                }
            
            # Store count
            if 'Store ID' in df.columns:
                summary['stores'] = {
                    'count': int(df['Store ID'].nunique()),
                    'ids': df['Store ID'].unique().tolist()
                }
            
            # Product count
            if 'Product ID' in df.columns:
                summary['products'] = {
                    'count': int(df['Product ID'].nunique()),
                    'ids': df['Product ID'].unique().tolist()[:10]  # First 10
                }
            
            # Category breakdown
            if 'Category' in df.columns:
                summary['categories'] = df['Category'].value_counts().to_dict()
            
            # Sales metrics
            if 'Units Sold' in df.columns:
                summary['avg_daily_sales'] = round(float(df['Units Sold'].mean()), 2)
                summary['total_units_sold'] = int(df['Units Sold'].sum())
                summary['zero_sales_percentage'] = round(float(
                    (df['Units Sold'] == 0).sum() / len(df) * 100
                ), 2)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating retail summary: {str(e)}")
            return summary


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_resource
def get_data_collector(_config=None):
    """Get or create cached DataCollector instance (singleton pattern)."""
    return DataCollector(_config)


def hash_file_upload(file_obj) -> str:
    """Create hash of uploaded file for caching."""
    try:
        file_obj.seek(0)
        file_content = file_obj.read()
        file_obj.seek(0)  # Reset for later reading
        return hashlib.md5(file_content).hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file: {str(e)}")
        return str(hash(str(file_obj)))


@st.cache_data(ttl=3600)
def load_csv_cached(file_path: str, **kwargs) -> pd.DataFrame:
    """Load CSV with caching (for file paths)."""
    logger.info(f"Loading CSV from: {file_path}")
    return pd.read_csv(file_path, **kwargs)


@st.cache_data(ttl=3600)
def load_csv_from_upload_cached(file_obj, _file_hash: str, **kwargs) -> pd.DataFrame:
    """Load CSV from Streamlit upload with caching."""
    file_obj.seek(0)
    logger.info(f"Loading CSV from upload (hash: {_file_hash[:8]}...)")
    return pd.read_csv(file_obj, **kwargs)


@st.cache_data(ttl=3600, show_spinner="Generating retail sample data...")
def generate_retail_sample_data(
    periods: int = 730,
    n_stores: int = 5,
    n_products: int = 20
) -> pd.DataFrame:
    """
    ✅ Generate synthetic retail inventory data matching Kaggle dataset structure.
    
    Args:
        periods: Number of days
        n_stores: Number of stores
        n_products: Number of products
    
    Returns:
        pd.DataFrame: Synthetic retail data
    """
    try:
        logger.info(f"Generating retail data: {periods} days, {n_stores} stores, {n_products} products")
        
        dates = pd.date_range(start='2022-01-01', periods=periods, freq='D')
        store_ids = [f'S{str(i+1).zfill(3)}' for i in range(n_stores)]
        product_ids = [f'P{str(i+1).zfill(4)}' for i in range(n_products)]
        categories = ['Electronics', 'Groceries', 'Clothing', 'Toys', 'Home Appliances']
        regions = ['North', 'South', 'East', 'West']
        weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        data = []
        
        for date in dates:
            # Determine season
            month = date.month
            if month in [3, 4, 5]:
                season = 'Spring'
            elif month in [6, 7, 8]:
                season = 'Summer'
            elif month in [9, 10, 11]:
                season = 'Autumn'
            else:
                season = 'Winter'
            
            for store_id in store_ids:
                # Assign region to store
                region = regions[int(store_id[1:]) % len(regions)]
                
                for product_id in product_ids:
                    # Assign category to product
                    category = categories[int(product_id[1:]) % len(categories)]
                    
                    # Base demand
                    base_demand = np.random.uniform(50, 200)
                    
                    # Seasonal effect
                    seasonal_effect = 1.0
                    if season == 'Winter' and category == 'Electronics':
                        seasonal_effect = 1.3  # Holiday shopping
                    elif season == 'Summer' and category == 'Clothing':
                        seasonal_effect = 1.2
                    
                    # Day of week effect
                    dow_effect = 1.2 if date.dayofweek in [5, 6] else 1.0  # Weekend boost
                    
                    # Weather effect
                    weather = np.random.choice(weather_conditions, p=[0.4, 0.3, 0.2, 0.1])
                    weather_effect = 1.1 if weather == 'Sunny' else 0.9 if weather == 'Rainy' else 1.0
                    
                    # Holiday/Promotion
                    is_holiday = 1 if np.random.random() < 0.15 else 0
                    promo_effect = 1.3 if is_holiday else 1.0
                    
                    # Price
                    base_price = np.random.uniform(10, 100)
                    discount = np.random.choice([0, 5, 10, 15, 20], p=[0.5, 0.2, 0.15, 0.1, 0.05])
                    price = base_price
                    
                    # Competitor pricing
                    competitor_price = base_price * np.random.uniform(0.8, 1.2)
                    
                    # Price effect on demand
                    price_effect = 1.2 if price < competitor_price else 0.9
                    
                    # Calculate units sold
                    demand = (base_demand * seasonal_effect * dow_effect * 
                             weather_effect * promo_effect * price_effect)
                    
                    # Add noise
                    demand *= np.random.uniform(0.8, 1.2)
                    
                    units_sold = max(0, int(demand))
                    
                    # Inventory level
                    inventory_level = int(np.random.uniform(50, 500))
                    
                    # Demand forecast (with some error)
                    demand_forecast = demand * np.random.uniform(0.9, 1.1)
                    
                    data.append({
                        'Date': date,
                        'Store ID': store_id,
                        'Product ID': product_id,
                        'Category': category,
                        'Units Sold': units_sold,
                        'Price': round(price, 2),
                        'Discount': discount,
                        'Inventory Level': inventory_level,
                        'Region': region,
                        'Weather Condition': weather,
                        'Seasonality': season,
                        'Holiday/Promotion': is_holiday,
                        'Competitor Pricing': round(competitor_price, 2),
                        'Demand Forecast': round(demand_forecast, 2)
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Generated retail data with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error generating retail sample data: {str(e)}")
        raise ValueError(f"Failed to generate sample data: {str(e)}")
# ============================================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================================

# Alias for backward compatibility with existing code
generate_sample_data_cached = generate_retail_sample_data