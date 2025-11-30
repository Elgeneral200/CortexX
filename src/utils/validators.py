"""
Data Validation Module for CortexX

PHASE 2 - TASK 6:
Comprehensive data validation and quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Enterprise-grade data validation for forecasting platform.
    
    Validates:
    - Data schema and types
    - Missing values
    - Data quality metrics
    - Time series requirements
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None) -> Dict:
        """
        Comprehensive dataframe validation.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Basic checks
        if df is None or df.empty:
            results['is_valid'] = False
            results['errors'].append("DataFrame is empty or None")
            return results
        
        # Size validation
        results['metrics']['total_rows'] = len(df)
        results['metrics']['total_columns'] = len(df.columns)
        
        if len(df) < 10:
            results['warnings'].append(f"Very small dataset ({len(df)} rows). Minimum 10 rows recommended.")
        
        # Column validation
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                results['is_valid'] = False
                results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Data type validation
        results['metrics']['numeric_columns'] = len(df.select_dtypes(include=[np.number]).columns)
        results['metrics']['datetime_columns'] = len(df.select_dtypes(include=['datetime64']).columns)
        results['metrics']['object_columns'] = len(df.select_dtypes(include=['object']).columns)
        
        if results['metrics']['numeric_columns'] == 0:
            results['is_valid'] = False
            results['errors'].append("No numeric columns found. At least one numeric column required.")
        
        # Missing value analysis
        missing_analysis = self._analyze_missing_values(df)
        results['metrics']['missing_analysis'] = missing_analysis
        
        if missing_analysis['total_missing_pct'] > 50:
            results['is_valid'] = False
            results['errors'].append(f"Too many missing values ({missing_analysis['total_missing_pct']:.1f}%)")
        elif missing_analysis['total_missing_pct'] > 20:
            results['warnings'].append(f"High missing values ({missing_analysis['total_missing_pct']:.1f}%)")
        
        # Duplicate analysis
        duplicate_analysis = self._analyze_duplicates(df)
        results['metrics']['duplicate_analysis'] = duplicate_analysis
        
        if duplicate_analysis['duplicate_pct'] > 10:
            results['warnings'].append(f"High duplicate rate ({duplicate_analysis['duplicate_pct']:.1f}%)")
        
        # Data quality score
        results['metrics']['quality_score'] = self._calculate_quality_score(
            missing_analysis['total_missing_pct'],
            duplicate_analysis['duplicate_pct']
        )
        
        return results
    
    def validate_time_series(self, df: pd.DataFrame, 
                            date_column: str, 
                            value_column: str) -> Dict:
        """
        Validate time series specific requirements.
        
        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check columns exist
        if date_column not in df.columns:
            results['is_valid'] = False
            results['errors'].append(f"Date column '{date_column}' not found")
            return results
        
        if value_column not in df.columns:
            results['is_valid'] = False
            results['errors'].append(f"Value column '{value_column}' not found")
            return results
        
        # Validate date column
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            results['is_valid'] = False
            results['errors'].append(f"Cannot convert date column to datetime: {str(e)}")
            return results
        
        # Check for date gaps
        df_sorted = df.sort_values(date_column)
        date_diff = df_sorted[date_column].diff()
        
        # Detect frequency
        median_diff = date_diff.median()
        results['metrics']['median_frequency'] = str(median_diff)
        
        # Check for irregular spacing
        irregular_count = (date_diff != median_diff).sum()
        if irregular_count > len(df) * 0.1:
            results['warnings'].append(f"Irregular time spacing detected in {irregular_count} instances")
        
        # Validate value column
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            results['is_valid'] = False
            results['errors'].append(f"Value column '{value_column}' is not numeric")
            return results
        
        # Check for negative values
        if (df[value_column] < 0).any():
            neg_count = (df[value_column] < 0).sum()
            results['warnings'].append(f"Found {neg_count} negative values")
        
        # Check for outliers
        outlier_analysis = self._detect_outliers(df[value_column])
        results['metrics']['outlier_analysis'] = outlier_analysis
        
        if outlier_analysis['outlier_pct'] > 5:
            results['warnings'].append(f"High outlier rate ({outlier_analysis['outlier_pct']:.1f}%)")
        
        # Time range
        results['metrics']['date_range'] = {
            'start': df_sorted[date_column].min().isoformat(),
            'end': df_sorted[date_column].max().isoformat(),
            'total_days': (df_sorted[date_column].max() - df_sorted[date_column].min()).days
        }
        
        # Minimum data requirement
        if len(df) < 30:
            results['warnings'].append(f"Small dataset ({len(df)} periods). 30+ periods recommended for forecasting.")
        
        return results
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values in dataframe."""
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        
        missing_by_column = df.isnull().sum()
        columns_with_missing = missing_by_column[missing_by_column > 0].to_dict()
        
        return {
            'total_missing': total_missing,
            'total_missing_pct': (total_missing / total_cells * 100) if total_cells > 0 else 0,
            'columns_with_missing': columns_with_missing,
            'columns_with_missing_count': len(columns_with_missing)
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate rows in dataframe."""
        duplicate_count = df.duplicated().sum()
        
        return {
            'duplicate_count': duplicate_count,
            'duplicate_pct': (duplicate_count / len(df) * 100) if len(df) > 0 else 0
        }
    
    def _detect_outliers(self, series: pd.Series) -> Dict:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        
        return {
            'outlier_count': outliers,
            'outlier_pct': (outliers / len(series) * 100) if len(series) > 0 else 0,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def _calculate_quality_score(self, missing_pct: float, duplicate_pct: float) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Score components:
        - 50% based on completeness (100 - missing_pct)
        - 30% based on uniqueness (100 - duplicate_pct)
        - 20% base score
        """
        completeness_score = max(0, 100 - missing_pct) * 0.5
        uniqueness_score = max(0, 100 - duplicate_pct) * 0.3
        base_score = 20
        
        total_score = completeness_score + uniqueness_score + base_score
        return round(total_score, 2)


def validate_csv_file(file_path: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate CSV file before loading.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (is_valid, message, dataframe)
    """
    try:
        # Try to read CSV
        df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows for validation
        
        if df.empty:
            return False, "CSV file is empty", None
        
        # Read full file
        df_full = pd.read_csv(file_path)
        
        return True, f"Successfully loaded {len(df_full)} rows", df_full
        
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}", None


def validate_upload_file(uploaded_file) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate uploaded file from Streamlit.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (is_valid, message, dataframe)
    """
    try:
        # Check file type
        if not uploaded_file.name.endswith('.csv'):
            return False, "Only CSV files are supported", None
        
        # Check file size (limit to 100MB)
        if uploaded_file.size > 100 * 1024 * 1024:
            return False, "File too large. Maximum 100MB allowed.", None
        
        # Try to read CSV
        df = pd.read_csv(uploaded_file)
        
        if df.empty:
            return False, "Uploaded file is empty", None
        
        # Basic validation
        validator = DataValidator()
        validation_result = validator.validate_dataframe(df)
        
        if not validation_result['is_valid']:
            error_msg = "; ".join(validation_result['errors'])
            return False, f"Validation failed: {error_msg}", None
        
        # Check for warnings
        warnings = validation_result.get('warnings', [])
        if warnings:
            warning_msg = "; ".join(warnings)
            return True, f"Loaded with warnings: {warning_msg}", df
        
        return True, f"Successfully validated {len(df)} rows", df
        
    except Exception as e:
        return False, f"Error processing file: {str(e)}", None
