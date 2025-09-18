# filename: preprocessing.py
"""
Data Preprocessing Module - Fixed Professional Edition

Clean, reliable data preprocessing with essential functions:
- Missing value handling with multiple strategies
- Data type conversion and detection
- Outlier detection and treatment
- Basic scaling and normalization
- Data validation and quality assessment

Author: CortexX Team
Version: 1.2.0 - Fixed Professional Edition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Literal
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies with fallbacks
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.impute import SimpleImputer, KNNImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================
# UTILITY FUNCTIONS
# ============================

def safe_operation(func):
    """Decorator for safe operations with error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Warning: {func.__name__} failed: {str(e)}")
            # Return original DataFrame if first argument is DataFrame
            if args and isinstance(args[0], pd.DataFrame):
                return args[0].copy()
            return None
    return wrapper

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and ensure we have a proper DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return df.copy()
    return df

# ============================
# MISSING VALUES HANDLING
# ============================

@safe_operation
def check_missing_values(df: pd.DataFrame, as_percentage: bool = False) -> pd.Series:
    """Check missing values in DataFrame columns."""
    df = validate_dataframe(df)

    missing_counts = df.isnull().sum()

    if as_percentage:
        return (missing_counts / len(df) * 100).round(2)

    return missing_counts

@safe_operation  
def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop_rows",
    columns: Optional[List[str]] = None,
    fill_value: Any = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Handle missing values with multiple strategies.

    Parameters:
    - strategy: 'drop_rows', 'drop_columns', 'fill_mean', 'fill_median', 
               'fill_mode', 'fill_forward', 'fill_backward', 'fill_constant'
    - columns: Specific columns to process (None for all)
    - fill_value: Value for 'fill_constant' strategy
    - threshold: Minimum fraction of non-null values for 'drop_columns'
    """
    df = validate_dataframe(df)
    df_result = df.copy()

    # Select columns to process
    target_columns = columns if columns else df.columns.tolist()
    target_columns = [col for col in target_columns if col in df.columns]

    if strategy == "drop_rows":
        # Drop rows with any missing values in target columns
        df_result = df_result.dropna(subset=target_columns)

    elif strategy == "drop_columns":
        # Drop columns with missing ratio above threshold
        for col in target_columns:
            missing_ratio = df_result[col].isnull().sum() / len(df_result)
            if missing_ratio > (1 - threshold):
                df_result = df_result.drop(columns=[col])

    elif strategy in ["fill_mean", "fill_median", "fill_mode"]:
        for col in target_columns:
            if col not in df_result.columns:
                continue

            if strategy == "fill_mean" and pd.api.types.is_numeric_dtype(df_result[col]):
                fill_val = df_result[col].mean()
            elif strategy == "fill_median" and pd.api.types.is_numeric_dtype(df_result[col]):
                fill_val = df_result[col].median()
            elif strategy == "fill_mode":
                mode_values = df_result[col].mode()
                fill_val = mode_values.iloc[0] if not mode_values.empty else None
            else:
                # Fallback to mode for non-numeric columns
                mode_values = df_result[col].mode()
                fill_val = mode_values.iloc[0] if not mode_values.empty else None

            if fill_val is not None and not pd.isna(fill_val):
                df_result[col] = df_result[col].fillna(fill_val)

    elif strategy == "fill_forward":
        df_result[target_columns] = df_result[target_columns].ffill()

    elif strategy == "fill_backward":
        df_result[target_columns] = df_result[target_columns].bfill()

    elif strategy == "fill_constant":
        if fill_value is not None:
            df_result[target_columns] = df_result[target_columns].fillna(fill_value)

    return df_result

# ============================
# DATA TYPE CONVERSION
# ============================

@safe_operation
def detect_and_convert_types(df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Detect and suggest data type conversions.

    Returns dictionary with conversion suggestions and converted DataFrame.
    """
    df = validate_dataframe(df)

    # Sample data for faster analysis
    sample_df = df.sample(n=min(sample_size, len(df))) if len(df) > sample_size else df

    suggestions = {}
    conversions_applied = {}

    for col in df.columns:
        current_type = str(df[col].dtype)
        suggested_type, confidence = _detect_column_type(sample_df[col])

        suggestions[col] = {
            'current_type': current_type,
            'suggested_type': suggested_type,
            'confidence': confidence,
            'sample_values': sample_df[col].dropna().head(5).tolist()
        }

        # Auto-convert high confidence suggestions
        if confidence > 0.8 and suggested_type != current_type:
            try:
                df_converted = _convert_column_type(df, col, suggested_type)
                if df_converted is not None:
                    df = df_converted
                    conversions_applied[col] = suggested_type
            except Exception:
                pass  # Keep original type if conversion fails

    return {
        'dataframe': df,
        'suggestions': suggestions,
        'conversions_applied': conversions_applied
    }

def _detect_column_type(series: pd.Series) -> Tuple[str, float]:
    """Detect the most appropriate data type for a series."""
    non_null_series = series.dropna()

    if len(non_null_series) == 0:
        return 'object', 0.0

    # Check for numeric
    try:
        pd.to_numeric(non_null_series, errors='raise')
        # Check if integers
        numeric_series = pd.to_numeric(non_null_series)
        if (numeric_series % 1 == 0).all():
            return 'int64', 0.9
        else:
            return 'float64', 0.9
    except (ValueError, TypeError):
        pass

    # Check for datetime
    try:
        pd.to_datetime(non_null_series, errors='raise')
        return 'datetime64[ns]', 0.85
    except (ValueError, TypeError):
        pass

    # Check for boolean
    unique_values = set(non_null_series.astype(str).str.lower().str.strip())
    boolean_values = [
        {'true', 'false'},
        {'yes', 'no'},
        {'y', 'n'},
        {'1', '0'},
        {'1.0', '0.0'}
    ]

    for bool_set in boolean_values:
        if unique_values.issubset(bool_set):
            return 'bool', 0.8

    # Check for categorical (low cardinality)
    unique_ratio = non_null_series.nunique() / len(non_null_series)
    if unique_ratio < 0.1 and non_null_series.nunique() < 50:
        return 'category', 0.7

    # Default to object
    return 'object', 0.6

def _convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> Optional[pd.DataFrame]:
    """Convert a column to the target data type."""
    df_result = df.copy()

    try:
        if target_type in ['int64', 'int32']:
            df_result[column] = pd.to_numeric(df_result[column], errors='coerce').astype('Int64')
        elif target_type in ['float64', 'float32']:
            df_result[column] = pd.to_numeric(df_result[column], errors='coerce')
        elif 'datetime' in target_type:
            df_result[column] = pd.to_datetime(df_result[column], errors='coerce')
        elif target_type == 'bool':
            df_result[column] = _convert_to_boolean(df_result[column])
        elif target_type == 'category':
            df_result[column] = df_result[column].astype('category')
        else:
            df_result[column] = df_result[column].astype(target_type)

        return df_result
    except Exception:
        return None

def _convert_to_boolean(series: pd.Series) -> pd.Series:
    """Convert series to boolean with common mappings."""
    # Create mapping for common boolean representations
    bool_map = {}

    # Get string representation of all values
    str_series = series.astype(str).str.lower().str.strip()

    # Define mapping
    true_values = ['true', 'yes', 'y', '1', '1.0']
    false_values = ['false', 'no', 'n', '0', '0.0']

    for val in str_series.unique():
        if val in true_values:
            bool_map[val] = True
        elif val in false_values:
            bool_map[val] = False
        else:
            bool_map[val] = None  # Will become NaN

    return str_series.map(bool_map)

@safe_operation
def convert_column_types(df: pd.DataFrame, type_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Convert multiple columns to specified types.

    Parameters:
    - df: Input DataFrame
    - type_dict: Dictionary mapping column names to target types
    """
    df = validate_dataframe(df)
    df_result = df.copy()

    for column, target_type in type_dict.items():
        if column in df_result.columns:
            converted_df = _convert_column_type(df_result, column, target_type)
            if converted_df is not None:
                df_result = converted_df

    return df_result

# ============================
# OUTLIER DETECTION AND HANDLING
# ============================

@safe_operation
def detect_outliers(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Detect outliers using various methods.

    Parameters:
    - method: 'iqr', 'zscore', 'modified_zscore'
    - threshold: Threshold value (1.5 for IQR, 3.0 for z-score)
    """
    df = validate_dataframe(df)

    # Get numeric columns
    numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_info = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < 4:  # Need minimum data points
            continue

        try:
            if method == "iqr":
                outliers = _detect_outliers_iqr(series, threshold)
            elif method == "zscore":
                outliers = _detect_outliers_zscore(series, threshold)
            elif method == "modified_zscore":
                outliers = _detect_outliers_modified_zscore(series, threshold)
            else:
                outliers = _detect_outliers_iqr(series, threshold)  # Default

            # Get outlier indices and values
            outlier_mask = pd.Series(False, index=df.index)
            outlier_mask.loc[series.index[outliers]] = True

            outlier_info[col] = {
                'outlier_count': int(outliers.sum()),
                'outlier_percentage': float((outliers.sum() / len(series)) * 100),
                'outlier_indices': series.index[outliers].tolist(),
                'outlier_values': series[outliers].tolist(),
                'method': method,
                'threshold': threshold
            }

        except Exception as e:
            outlier_info[col] = {'error': str(e)}

    return outlier_info

def _detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> np.ndarray:
    """Detect outliers using Interquartile Range method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    return (series < lower_bound) | (series > upper_bound)

def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score method."""
    if SCIPY_AVAILABLE:
        z_scores = np.abs(stats.zscore(series))
        return z_scores > threshold
    else:
        # Manual z-score calculation
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return np.zeros(len(series), dtype=bool)
        z_scores = np.abs((series - mean_val) / std_val)
        return z_scores > threshold

def _detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> np.ndarray:
    """Detect outliers using Modified Z-score method."""
    median = series.median()
    mad = np.median(np.abs(series - median))

    if mad == 0:
        return np.zeros(len(series), dtype=bool)

    modified_z_scores = 0.6745 * (series - median) / mad
    return np.abs(modified_z_scores) > threshold

@safe_operation
def handle_outliers(
    df: pd.DataFrame,
    outlier_info: Dict[str, Any],
    method: str = "clip"
) -> pd.DataFrame:
    """
    Handle detected outliers.

    Parameters:
    - method: 'remove', 'clip', 'transform'
    """
    df = validate_dataframe(df)
    df_result = df.copy()

    for col, info in outlier_info.items():
        if 'outlier_indices' not in info or not info['outlier_indices']:
            continue

        try:
            if method == "remove":
                # Remove rows with outliers
                df_result = df_result.drop(info['outlier_indices'])

            elif method == "clip":
                # Clip to 5th and 95th percentiles
                lower_bound = df_result[col].quantile(0.05)
                upper_bound = df_result[col].quantile(0.95)
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)

            elif method == "transform":
                # Log transform for positive values
                if (df_result[col] > 0).all():
                    df_result[col] = np.log1p(df_result[col])

        except Exception:
            continue  # Skip if handling fails

    return df_result

# ============================
# DATA SCALING AND NORMALIZATION
# ============================

@safe_operation
def scale_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "standard"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scale numeric columns using various methods.

    Parameters:
    - method: 'standard', 'minmax', 'robust'
    """
    df = validate_dataframe(df)

    if not SKLEARN_AVAILABLE:
        # Fallback to manual scaling
        return _manual_scaling(df, columns, method)

    # Get numeric columns
    numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    if not numeric_cols:
        return df.copy(), {"message": "No numeric columns found"}

    df_result = df.copy()
    scaling_info = {"method": method, "columns_scaled": []}

    try:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return df.copy(), {"error": f"Unknown scaling method: {method}"}

        # Apply scaling
        for col in numeric_cols:
            non_null_mask = df_result[col].notna()
            if non_null_mask.sum() < 2:  # Need at least 2 values
                continue

            values = df_result.loc[non_null_mask, col].values.reshape(-1, 1)
            scaled_values = scaler.fit_transform(values)
            df_result.loc[non_null_mask, col] = scaled_values.flatten()
            scaling_info["columns_scaled"].append(col)

    except Exception as e:
        return df.copy(), {"error": str(e)}

    return df_result, scaling_info

def _manual_scaling(df: pd.DataFrame, columns: Optional[List[str]], method: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Manual scaling when sklearn is not available."""
    numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    df_result = df.copy()
    scaling_info = {"method": f"manual_{method}", "columns_scaled": []}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        try:
            non_null_data = df_result[col].dropna()
            if len(non_null_data) < 2:
                continue

            if method == "standard":
                mean_val = non_null_data.mean()
                std_val = non_null_data.std()
                if std_val != 0:
                    df_result[col] = (df_result[col] - mean_val) / std_val

            elif method == "minmax":
                min_val = non_null_data.min()
                max_val = non_null_data.max()
                if max_val != min_val:
                    df_result[col] = (df_result[col] - min_val) / (max_val - min_val)

            elif method == "robust":
                median_val = non_null_data.median()
                q75 = non_null_data.quantile(0.75)
                q25 = non_null_data.quantile(0.25)
                iqr = q75 - q25
                if iqr != 0:
                    df_result[col] = (df_result[col] - median_val) / iqr

            scaling_info["columns_scaled"].append(col)

        except Exception:
            continue

    return df_result, scaling_info

# ============================
# DATA VALIDATION AND QUALITY
# ============================

@safe_operation
def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data quality report."""
    df = validate_dataframe(df)

    if df.empty:
        return {"error": "DataFrame is empty"}

    # Basic statistics
    basic_stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
    }

    # Missing value analysis
    missing_analysis = {}
    total_missing = 0

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        total_missing += missing_count

        missing_analysis[col] = {
            "missing_count": int(missing_count),
            "missing_percentage": round(missing_pct, 2),
            "data_type": str(df[col].dtype),
            "unique_count": int(df[col].nunique()),
            "unique_percentage": round((df[col].nunique() / len(df)) * 100, 2)
        }

    # Overall quality score (0-10)
    completeness = ((len(df) * len(df.columns) - total_missing) / (len(df) * len(df.columns))) * 100
    uniqueness = ((len(df) - df.duplicated().sum()) / len(df)) * 100
    overall_score = (completeness * 0.6 + uniqueness * 0.4) / 10

    return {
        "basic_statistics": basic_stats,
        "missing_value_analysis": missing_analysis,
        "quality_metrics": {
            "completeness_percentage": round(completeness, 2),
            "uniqueness_percentage": round(uniqueness, 2),
            "overall_quality_score": round(overall_score, 2)
        },
        "recommendations": _generate_quality_recommendations(df, completeness, uniqueness)
    }

def _generate_quality_recommendations(df: pd.DataFrame, completeness: float, uniqueness: float) -> List[str]:
    """Generate recommendations based on data quality analysis."""
    recommendations = []

    if completeness < 80:
        recommendations.append("Consider handling missing values - data completeness is below 80%")

    if uniqueness < 95:
        recommendations.append("Consider removing duplicate records to improve data quality")

    # Check for columns with high missing rates
    high_missing_cols = []
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 50:
            high_missing_cols.append(col)

    if high_missing_cols:
        recommendations.append(f"Columns with >50% missing values: {high_missing_cols}")

    # Memory usage check
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if memory_mb > 100:
        recommendations.append("Dataset uses significant memory - consider data type optimization")

    return recommendations

@safe_operation
def remove_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
    """Remove duplicate rows from DataFrame."""
    df = validate_dataframe(df)

    try:
        if columns:
            # Remove duplicates based on specific columns
            return df.drop_duplicates(subset=columns, keep=keep)
        else:
            # Remove completely duplicate rows
            return df.drop_duplicates(keep=keep)
    except Exception:
        return df.copy()

# ============================
# UTILITY FUNCTIONS
# ============================

def get_column_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get detailed information about DataFrame columns."""
    df = validate_dataframe(df)

    column_info = {}

    for col in df.columns:
        info = {
            "data_type": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
            "memory_usage": int(df[col].memory_usage(deep=True))
        }

        # Add type-specific information
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                info.update({
                    "min_value": float(non_null_data.min()),
                    "max_value": float(non_null_data.max()),
                    "mean_value": float(non_null_data.mean()),
                    "std_value": float(non_null_data.std())
                })

        elif pd.api.types.is_object_dtype(df[col]):
            mode_values = df[col].mode()
            if not mode_values.empty:
                info["most_frequent"] = str(mode_values.iloc[0])
                info["most_frequent_count"] = int((df[col] == mode_values.iloc[0]).sum())

        column_info[col] = info

    return column_info

# ============================
# EXPORT ALL FUNCTIONS
# ============================

__all__ = [
    'check_missing_values',
    'handle_missing_values',
    'detect_and_convert_types',
    'convert_column_types',
    'detect_outliers',
    'handle_outliers', 
    'scale_numeric_columns',
    'get_data_quality_report',
    'remove_duplicates',
    'get_column_info'
]
