"""
Data Validation Module for CortexX

✅ RETAIL INVENTORY FORECASTING OPTIMIZED:
- Retail-specific column validation
- Store-Product hierarchy validation
- Business rule validation (negative values, ranges)
- Enhanced time series validation for forecasting
- Zero-inflation detection
- Date gap detection for hierarchical data
- File size and format validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Enterprise-grade data validation for retail forecasting platform.
    
    ✅ ENHANCED: Retail-specific validation capabilities
    
    Validates:
    - Retail data schema and types
    - Store-Product hierarchy
    - Business rules (sales, prices, discounts)
    - Time series requirements for forecasting
    - Data quality metrics
    """
    
    def __init__(self):
        self.validation_results = {}
        
        # ✅ NEW: Define retail-specific expected columns
        self.retail_columns = {
            'required': ['Date', 'Store ID', 'Product ID', 'Units Sold'],
            'recommended': ['Price', 'Category', 'Inventory Level'],
            'optional': ['Discount', 'Holiday/Promotion', 'Weather Condition', 
                        'Seasonality', 'Region', 'Competitor Pricing', 'Demand Forecast']
        }
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        check_retail: bool = True
    ) -> Dict:
        """
        Comprehensive dataframe validation.
        
        ✅ ENHANCED: Added retail-specific validation
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names (overrides defaults)
            check_retail: Whether to apply retail-specific validations
            
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
        
        # ✅ NEW: Retail-specific column validation
        if check_retail:
            retail_validation = self._validate_retail_columns(df)
            results['metrics']['retail_validation'] = retail_validation
            
            if not retail_validation['is_valid']:
                results['is_valid'] = False
                results['errors'].extend(retail_validation['errors'])
            
            results['warnings'].extend(retail_validation.get('warnings', []))
        
        # Column validation (custom or default)
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
        
        # ✅ NEW: Business rule validation
        if check_retail:
            business_rule_validation = self._validate_business_rules(df)
            results['metrics']['business_rules'] = business_rule_validation
            
            if not business_rule_validation['is_valid']:
                results['is_valid'] = False
                results['errors'].extend(business_rule_validation['errors'])
            
            results['warnings'].extend(business_rule_validation.get('warnings', []))
        
        # ✅ NEW: Hierarchy validation
        if check_retail and all(col in df.columns for col in ['Store ID', 'Product ID']):
            hierarchy_validation = self._validate_hierarchy(df)
            results['metrics']['hierarchy'] = hierarchy_validation
            results['warnings'].extend(hierarchy_validation.get('warnings', []))
        
        # Data quality score
        results['metrics']['quality_score'] = self._calculate_quality_score(
            missing_analysis['total_missing_pct'],
            duplicate_analysis['duplicate_pct']
        )
        
        return results
    
    def _validate_retail_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ NEW: Validate retail-specific column requirements.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'found_columns': {
                'required': [],
                'recommended': [],
                'optional': []
            }
        }
        
        # Check required columns
        for col in self.retail_columns['required']:
            if col in df.columns:
                results['found_columns']['required'].append(col)
            else:
                results['is_valid'] = False
                results['errors'].append(f"Missing required retail column: '{col}'")
        
        # Check recommended columns
        for col in self.retail_columns['recommended']:
            if col in df.columns:
                results['found_columns']['recommended'].append(col)
            else:
                results['warnings'].append(f"Recommended column missing: '{col}'")
        
        # Check optional columns
        for col in self.retail_columns['optional']:
            if col in df.columns:
                results['found_columns']['optional'].append(col)
        
        results['coverage'] = {
            'required': f"{len(results['found_columns']['required'])}/{len(self.retail_columns['required'])}",
            'recommended': f"{len(results['found_columns']['recommended'])}/{len(self.retail_columns['recommended'])}",
            'optional': f"{len(results['found_columns']['optional'])}/{len(self.retail_columns['optional'])}"
        }
        
        return results
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ NEW: Validate business rules for retail data.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'violations': []
        }
        
        # Rule 1: Units Sold should be non-negative
        if 'Units Sold' in df.columns:
            negative_sales = (df['Units Sold'] < 0).sum()
            if negative_sales > 0:
                results['is_valid'] = False
                results['errors'].append(
                    f"Found {negative_sales} negative values in 'Units Sold' "
                    f"({negative_sales/len(df)*100:.1f}%)"
                )
                results['violations'].append({
                    'rule': 'non_negative_sales',
                    'column': 'Units Sold',
                    'violation_count': int(negative_sales)
                })
        
        # Rule 2: Price should be positive
        if 'Price' in df.columns:
            non_positive_prices = (df['Price'] <= 0).sum()
            if non_positive_prices > 0:
                results['is_valid'] = False
                results['errors'].append(
                    f"Found {non_positive_prices} non-positive values in 'Price' "
                    f"({non_positive_prices/len(df)*100:.1f}%)"
                )
                results['violations'].append({
                    'rule': 'positive_price',
                    'column': 'Price',
                    'violation_count': int(non_positive_prices)
                })
        
        # Rule 3: Discount should be in range [0, 100]
        if 'Discount' in df.columns:
            if pd.api.types.is_numeric_dtype(df['Discount']):
                invalid_discounts = ((df['Discount'] < 0) | (df['Discount'] > 100)).sum()
                if invalid_discounts > 0:
                    results['warnings'].append(
                        f"Found {invalid_discounts} discount values outside [0, 100] range"
                    )
                    results['violations'].append({
                        'rule': 'discount_range',
                        'column': 'Discount',
                        'violation_count': int(invalid_discounts)
                    })
        
        # Rule 4: Holiday/Promotion should be binary (0 or 1)
        if 'Holiday/Promotion' in df.columns:
            if pd.api.types.is_numeric_dtype(df['Holiday/Promotion']):
                unique_vals = df['Holiday/Promotion'].dropna().unique()
                invalid_vals = [v for v in unique_vals if v not in [0, 1]]
                if invalid_vals:
                    results['warnings'].append(
                        f"'Holiday/Promotion' has non-binary values: {invalid_vals}"
                    )
                    results['violations'].append({
                        'rule': 'binary_promotion',
                        'column': 'Holiday/Promotion',
                        'invalid_values': [float(v) for v in invalid_vals]
                    })
        
        # Rule 5: Competitor Pricing should be positive
        if 'Competitor Pricing' in df.columns:
            if pd.api.types.is_numeric_dtype(df['Competitor Pricing']):
                non_positive_comp = (df['Competitor Pricing'] <= 0).sum()
                if non_positive_comp > 0:
                    results['warnings'].append(
                        f"Found {non_positive_comp} non-positive values in 'Competitor Pricing'"
                    )
        
        # Rule 6: Inventory Level should be non-negative
        if 'Inventory Level' in df.columns:
            if pd.api.types.is_numeric_dtype(df['Inventory Level']):
                negative_inventory = (df['Inventory Level'] < 0).sum()
                if negative_inventory > 0:
                    results['warnings'].append(
                        f"Found {negative_inventory} negative values in 'Inventory Level'"
                    )
        
        # Rule 7: Check for extreme outliers (sales > 99.9th percentile * 5)
        if 'Units Sold' in df.columns:
            p999 = df['Units Sold'].quantile(0.999)
            extreme_values = (df['Units Sold'] > p999 * 5).sum()
            if extreme_values > 0:
                results['warnings'].append(
                    f"Found {extreme_values} extreme outlier values in 'Units Sold' "
                    f"(> {p999*5:.0f})"
                )
        
        results['total_violations'] = len(results['violations'])
        
        return results
    
    def _validate_hierarchy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ NEW: Validate hierarchical structure (Store × Product).
        
        Args:
            df: DataFrame with Store ID and Product ID
        
        Returns:
            Validation results
        """
        results = {
            'warnings': [],
            'metrics': {}
        }
        
        try:
            n_stores = df['Store ID'].nunique()
            n_products = df['Product ID'].nunique()
            n_combinations = len(df.groupby(['Store ID', 'Product ID']))
            expected_combinations = n_stores * n_products
            
            results['metrics'] = {
                'n_stores': n_stores,
                'n_products': n_products,
                'n_combinations': n_combinations,
                'expected_combinations': expected_combinations,
                'coverage_pct': round((n_combinations / expected_combinations * 100), 2)
            }
            
            # Warn if coverage is low
            if results['metrics']['coverage_pct'] < 80:
                results['warnings'].append(
                    f"Low Store×Product coverage: {results['metrics']['coverage_pct']:.1f}%. "
                    f"Expected {expected_combinations} combinations, found {n_combinations}"
                )
            
            # Check for unbalanced data
            records_per_combo = df.groupby(['Store ID', 'Product ID']).size()
            cv = records_per_combo.std() / records_per_combo.mean() if records_per_combo.mean() > 0 else 0
            
            results['metrics']['records_per_combination'] = {
                'mean': round(records_per_combo.mean(), 2),
                'std': round(records_per_combo.std(), 2),
                'cv': round(cv, 2)
            }
            
            if cv > 0.5:
                results['warnings'].append(
                    f"Unbalanced data across Store×Product combinations (CV={cv:.2f})"
                )
        
        except Exception as e:
            logger.error(f"Error validating hierarchy: {e}")
            results['warnings'].append(f"Hierarchy validation error: {str(e)}")
        
        return results
    
    def validate_time_series(
        self, 
        df: pd.DataFrame,
        date_column: str = 'Date',
        value_column: str = 'Units Sold',
        check_forecasting_readiness: bool = True
    ) -> Dict:
        """
        Validate time series specific requirements.
        
        ✅ ENHANCED: Added forecasting readiness checks
        
        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column
            check_forecasting_readiness: Check if data is suitable for forecasting
            
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
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            results['is_valid'] = False
            results['errors'].append(f"Cannot convert date column to datetime: {str(e)}")
            return results
        
        # Sort by date
        df_sorted = df.sort_values(date_column)
        
        # ✅ NEW: Check for date gaps
        date_diff = df_sorted[date_column].diff().dropna()
        
        if len(date_diff) > 0:
            # Detect frequency
            mode_diff = date_diff.mode()[0] if len(date_diff) > 0 else pd.Timedelta(days=1)
            results['metrics']['detected_frequency'] = str(mode_diff)
            
            # Find gaps
            gaps = date_diff[date_diff > mode_diff * 1.5]
            results['metrics']['date_gaps'] = {
                'count': len(gaps),
                'percentage': round(len(gaps) / len(date_diff) * 100, 2) if len(date_diff) > 0 else 0
            }
            
            if len(gaps) > 0:
                results['warnings'].append(
                    f"Found {len(gaps)} date gaps (expected frequency: {mode_diff})"
                )
        
        # Validate value column
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            results['is_valid'] = False
            results['errors'].append(f"Value column '{value_column}' is not numeric")
            return results
        
        # Check for negative values
        if (df[value_column] < 0).any():
            neg_count = (df[value_column] < 0).sum()
            results['warnings'].append(f"Found {neg_count} negative values in '{value_column}'")
        
        # ✅ NEW: Check for zero-inflation
        zero_count = (df[value_column] == 0).sum()
        zero_pct = (zero_count / len(df) * 100) if len(df) > 0 else 0
        results['metrics']['zero_inflation'] = {
            'count': int(zero_count),
            'percentage': round(zero_pct, 2)
        }
        
        if zero_pct > 15:
            results['warnings'].append(
                f"High zero-inflation: {zero_pct:.1f}% of values are zero. "
                "Consider using zero-inflated models."
            )
        
        # Check for outliers
        outlier_analysis = self._detect_outliers(df[value_column])
        results['metrics']['outlier_analysis'] = outlier_analysis
        
        if outlier_analysis['outlier_pct'] > 5:
            results['warnings'].append(
                f"High outlier rate ({outlier_analysis['outlier_pct']:.1f}%)"
            )
        
        # Time range
        results['metrics']['date_range'] = {
            'start': df_sorted[date_column].min().strftime('%Y-%m-%d'),
            'end': df_sorted[date_column].max().strftime('%Y-%m-%d'),
            'total_days': int((df_sorted[date_column].max() - df_sorted[date_column].min()).days),
            'total_periods': len(df_sorted)
        }
        
        # ✅ NEW: Forecasting readiness checks
        if check_forecasting_readiness:
            readiness_check = self._check_forecasting_readiness(
                df_sorted, date_column, value_column
            )
            results['metrics']['forecasting_readiness'] = readiness_check
            
            if not readiness_check['is_ready']:
                results['warnings'].extend(readiness_check['warnings'])
        
        return results
    
    def _check_forecasting_readiness(
        self, 
        df: pd.DataFrame, 
        date_column: str, 
        value_column: str
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Check if data is ready for forecasting.
        
        Args:
            df: Sorted DataFrame
            date_column: Date column name
            value_column: Value column name
        
        Returns:
            Readiness assessment
        """
        readiness = {
            'is_ready': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check 1: Minimum data points (730 days = 2 years)
        total_days = (df[date_column].max() - df[date_column].min()).days
        if total_days < 730:
            readiness['is_ready'] = False
            readiness['warnings'].append(
                f"Only {total_days} days of data. Minimum 730 days (2 years) recommended for reliable forecasting."
            )
            readiness['recommendations'].append("Collect more historical data")
        
        # Check 2: Minimum periods
        if len(df) < 100:
            readiness['warnings'].append(
                f"Only {len(df)} data points. 100+ recommended for robust models."
            )
        
        # Check 3: Recent data freshness
        last_date = df[date_column].max()
        days_old = (pd.Timestamp.now() - last_date).days
        if days_old > 30:
            readiness['warnings'].append(
                f"Data is {days_old} days old. Recent data improves forecast accuracy."
            )
            readiness['recommendations'].append("Update data to include recent periods")
        
        # Check 4: Variance check
        if df[value_column].std() == 0:
            readiness['is_ready'] = False
            readiness['warnings'].append(
                f"'{value_column}' has no variance (constant value). Cannot forecast."
            )
        
        # Check 5: Missing value ratio
        missing_pct = (df[value_column].isnull().sum() / len(df)) * 100
        if missing_pct > 10:
            readiness['warnings'].append(
                f"{missing_pct:.1f}% missing values in '{value_column}'. "
                "Consider imputation before forecasting."
            )
            readiness['recommendations'].append("Handle missing values")
        
        return readiness
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values in dataframe."""
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        
        missing_by_column = df.isnull().sum()
        columns_with_missing = missing_by_column[missing_by_column > 0].to_dict()
        
        return {
            'total_missing': int(total_missing),
            'total_missing_pct': round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0,
            'columns_with_missing': {k: int(v) for k, v in columns_with_missing.items()},
            'columns_with_missing_count': len(columns_with_missing)
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate rows in dataframe."""
        duplicate_count = df.duplicated().sum()
        
        return {
            'duplicate_count': int(duplicate_count),
            'duplicate_pct': round((duplicate_count / len(df) * 100), 2) if len(df) > 0 else 0
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
            'outlier_count': int(outliers),
            'outlier_pct': round((outliers / len(series) * 100), 2) if len(series) > 0 else 0,
            'lower_bound': round(float(lower_bound), 2),
            'upper_bound': round(float(upper_bound), 2)
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


# ============================================================================
# FILE VALIDATION FUNCTIONS
# ============================================================================

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
    
    ✅ ENHANCED: Better error messages and validation
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (is_valid, message, dataframe)
    """
    try:
        # Check file type
        if not uploaded_file.name.endswith('.csv'):
            return False, "❌ Only CSV files are supported", None
        
        # Check file size (limit to 100MB)
        max_size_mb = 100
        if uploaded_file.size > max_size_mb * 1024 * 1024:
            return False, f"❌ File too large ({uploaded_file.size / (1024*1024):.1f}MB). Maximum {max_size_mb}MB allowed.", None
        
        # Try to read CSV
        df = pd.read_csv(uploaded_file)
        
        if df.empty:
            return False, "❌ Uploaded file is empty", None
        
        # Basic validation
        validator = DataValidator()
        validation_result = validator.validate_dataframe(df, check_retail=True)
        
        if not validation_result['is_valid']:
            error_msg = "; ".join(validation_result['errors'])
            return False, f"❌ Validation failed: {error_msg}", None
        
        # Check for warnings
        warnings = validation_result.get('warnings', [])
        if warnings:
            warning_msg = "; ".join(warnings[:3])  # Show first 3 warnings
            if len(warnings) > 3:
                warning_msg += f" ... and {len(warnings) - 3} more warnings"
            return True, f"⚠️ Loaded with warnings: {warning_msg}", df
        
        return True, f"✅ Successfully validated {len(df):,} rows × {len(df.columns)} columns", df
        
    except pd.errors.EmptyDataError:
        return False, "❌ CSV file is empty or corrupted", None
    except pd.errors.ParserError as e:
        return False, f"❌ CSV parsing error: {str(e)}", None
    except Exception as e:
        return False, f"❌ Error processing file: {str(e)}", None


def validate_retail_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    ✅ NEW: Quick validation function for retail datasets.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Validation summary
    """
    validator = DataValidator()
    
    # Run all validations
    basic_validation = validator.validate_dataframe(df, check_retail=True)
    
    # Check if time series validation is possible
    if 'Date' in df.columns and 'Units Sold' in df.columns:
        ts_validation = validator.validate_time_series(df, 'Date', 'Units Sold')
        basic_validation['time_series'] = ts_validation
    
    return basic_validation
