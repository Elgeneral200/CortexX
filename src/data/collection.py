"""
Data collection module for CortexX sales forecasting platform.

ENHANCED: 
- Caching strategy
- Config integration
- Optimized date detection
✅ TASK 6: Data validation integration
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
import logging
from io import StringIO
import re
import streamlit as st
import hashlib

logger = logging.getLogger(__name__)


class DataCollector:
    """
    A class to handle data collection from various sources for sales forecasting.
    
    ENHANCED: Integrated with config, optimized for caching, with validation.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize DataCollector with configuration.
        
        Args:
            config: Configuration object (uses get_config() if None)
        """
        if config is None:
            from src.utils.config import get_config
            config = get_config()
        
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_csv_data(self, file_path: Union[str, object], **kwargs) -> pd.DataFrame:
        """
        Load sales data from CSV file or file-like object with automatic date detection.
        
        ✅ TASK 6: Enhanced with validation checks
        
        Args:
            file_path (Union[str, object]): Path to CSV file or file-like object
            **kwargs: Additional arguments for pandas read_csv
        
        Returns:
            pd.DataFrame: Loaded and validated sales data
        """
        try:
            self.logger.info(f"Loading data from {file_path}")
            
            # ✅ TASK 6: Try to import validators
            try:
                from src.utils.validators import validate_upload_file, DataValidator
                validators_available = True
            except ImportError:
                validators_available = False
                self.logger.warning("Validators not available, loading without validation")
            
            if isinstance(file_path, str):
                # File path - can be cached directly
                df = load_csv_cached(file_path, **kwargs)
                
                # ✅ TASK 6: Validate if validators available
                if validators_available:
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
                # ✅ TASK 6: Use validation function if available
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
            
            # Auto-detect and convert date columns
            df = self._auto_detect_dates(df)
            
            self.logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
            
        except FileNotFoundError as e:
            self.logger.error(f"CSV file not found: {file_path}")
            raise e
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")

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
            col_lower = col.lower()
            
            # Check if column name matches date patterns
            is_date_like = any(pattern in col_lower for pattern in date_patterns)
            
            if is_date_like:
                # Try to convert to datetime
                converted_col, success = self._try_convert_to_datetime(df_processed[col])
                if success:
                    df_processed[col] = converted_col
                    date_columns_found.append(col)
                    self.logger.info(f"Auto-converted column '{col}' to datetime")
                    continue
            
            # If not detected by name, check content
            if not is_date_like and df_processed[col].dtype == 'object':
                converted_col, success = self._try_convert_to_datetime(df_processed[col])
                if success and self._is_likely_date_column(converted_col):
                    df_processed[col] = converted_col
                    date_columns_found.append(col)
                    self.logger.info(f"Auto-detected date column '{col}' from content")
        
        if date_columns_found:
            self.logger.info(f"Found date columns: {date_columns_found}")
        else:
            self.logger.warning("No date columns detected automatically")
        
        return df_processed

    def _try_convert_to_datetime(self, series: pd.Series) -> tuple:
        """
        Try to convert a series to datetime using multiple methods.
        
        ✅ FIXED: Removed deprecated infer_datetime_format, suppressed warnings
        
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
            if not converted.isna().all():
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
                if not converted.isna().all():
                    return converted, True
            except:
                continue
        
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(series, errors='coerce')
            if not converted.isna().all():
                return converted, True
        except:
            pass
        
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
            date_range = series.dropna().max() - series.dropna().min()
            
            # Reasonable criteria for a date column
            if unique_dates > 1 and date_range.days > 0:
                return True
        
        return False

    def generate_sample_data(self, periods: int = 365*3, products: int = 3) -> pd.DataFrame:
        """
        Generate synthetic sales data for demonstration purposes.
        
        CHANGED: Now wrapped in caching function for performance.
        
        Args:
            periods (int): Number of time periods to generate
            products (int): Number of different products to generate
        
        Returns:
            pd.DataFrame: Synthetic sales data with realistic patterns
        """
        return generate_sample_data_cached(periods, products)

    def validate_data_structure(self, df: pd.DataFrame, required_columns: list = None) -> Dict[str, Any]:
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
                'data_types': dict(df.dtypes),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'date_columns_found': date_columns
            }
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(f"Missing required columns: {missing_columns}")
            
            # Check for excessive missing values
            missing_percentage = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_percentage[missing_percentage > 30].index.tolist()
            if high_missing:
                validation_result['issues'].append(f"High missing values (>30%) in: {high_missing}")
            
            # Warn if no date columns found
            if not date_columns:
                validation_result['issues'].append("No date columns detected. Time series analysis may be limited.")
            
            self.logger.info("Data validation completed")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result


# ============================================================================
# CACHING FUNCTIONS (NEW)
# ============================================================================

@st.cache_resource
def get_data_collector(_config=None):
    """
    Get or create cached DataCollector instance (singleton pattern).
    
    Args:
        _config: Config object (underscore = don't hash for caching)
    
    Returns:
        DataCollector: Cached data collector instance
    """
    return DataCollector(_config)


def hash_file_upload(file_obj) -> str:
    """
    Create hash of uploaded file for caching.
    
    Args:
        file_obj: Streamlit UploadedFile object
    
    Returns:
        str: MD5 hash of file content
    """
    file_obj.seek(0)
    file_content = file_obj.read()
    file_obj.seek(0)  # Reset for later reading
    return hashlib.md5(file_content).hexdigest()


@st.cache_data(ttl=3600)
def load_csv_cached(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV with caching (for file paths).
    
    Args:
        file_path: Path to CSV file
        **kwargs: pandas read_csv arguments
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_csv(file_path, **kwargs)


@st.cache_data(ttl=3600)
def load_csv_from_upload_cached(file_obj, _file_hash: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV from Streamlit upload with caching.
    
    Args:
        file_obj: Streamlit UploadedFile object
        _file_hash: Hash of file content (underscore prefix = don't include in hash)
        **kwargs: pandas read_csv arguments
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    file_obj.seek(0)
    return pd.read_csv(file_obj, **kwargs)


@st.cache_data(ttl=3600, show_spinner="Generating sample data...")
def generate_sample_data_cached(periods: int = 365*3, products: int = 3) -> pd.DataFrame:
    """
    Generate synthetic sales data with caching.
    
    PERFORMANCE: Cached for 1 hour, prevents regenerating 3,652 rows repeatedly.
    
    Args:
        periods (int): Number of time periods to generate
        products (int): Number of different products to generate
    
    Returns:
        pd.DataFrame: Synthetic sales data with realistic patterns
    """
    try:
        logger.info(f"Generating sample data for {periods} periods and {products} products")
        
        dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
        product_ids = [f'Product_{chr(65+i)}' for i in range(products)]
        
        data = []
        
        for date in dates:
            for product_id in product_ids:
                # Base trend with slight upward trajectory
                trend = 100 + (date - dates[0]).days * 0.1
                
                # Seasonal patterns
                seasonal = 50 * np.sin(2 * np.pi * (date.dayofyear - 1) / 365)
                
                # Weekly pattern
                weekly = 20 * np.sin(2 * np.pi * date.dayofweek / 7)
                
                # Product-specific variations
                product_factor = ord(product_id[-1]) - 64  # A=1, B=2, etc.
                product_base = trend * (0.8 + 0.4 * product_factor / len(product_ids))
                
                # Random noise
                noise = np.random.normal(0, 15)
                
                # Promotional effects (random)
                promotion = np.random.choice([0, 1], p=[0.85, 0.15])
                promo_effect = 40 if promotion else 0
                
                sales = max(0, product_base + seasonal + weekly + noise + promo_effect)
                
                data.append({
                    'date': date,
                    'product_id': product_id,
                    'sales': sales,
                    'price': np.random.uniform(10, 100),
                    'promotion': promotion,
                    'category': f'Category_{(ord(product_id[-1]) - 65) % 3 + 1}'
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated sample data with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        raise ValueError(f"Failed to generate sample data: {str(e)}")
