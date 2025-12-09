"""
Data Quality Analysis Module for CortexX Forecasting Platform

✅ RETAIL INVENTORY FORECASTING OPTIMIZED:
- Time series validation (date gaps, frequency)
- Business rule validation (negative values, ranges)
- Zero-inflation detection and analysis
- Feature correlation analysis (multicollinearity)
- Distribution analysis (skewness, kurtosis)
- Enhanced validity checks for retail data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """
    Comprehensive data quality analysis and reporting for retail forecasting.
    
    ✅ ENHANCED: Retail-specific validation capabilities
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with DataFrame.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.total_cells = len(df) * len(df.columns)
        self.total_rows = len(df)
        self.total_cols = len(df.columns)
    
    def calculate_overall_quality_score(self) -> Dict[str, Any]:
        """
        Calculate overall data quality score (0-100).
        
        Returns:
            Dictionary with overall score and component scores
        """
        # Component scores
        completeness_score = self._calculate_completeness_score()
        uniqueness_score = self._calculate_uniqueness_score()
        validity_score = self._calculate_validity_score()
        consistency_score = self._calculate_consistency_score()
        
        # Weighted overall score
        weights = {
            'completeness': 0.35,
            'uniqueness': 0.25,
            'validity': 0.25,
            'consistency': 0.15
        }
        
        overall_score = (
            completeness_score * weights['completeness'] +
            uniqueness_score * weights['uniqueness'] +
            validity_score * weights['validity'] +
            consistency_score * weights['consistency']
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'completeness': round(completeness_score, 2),
            'uniqueness': round(uniqueness_score, 2),
            'validity': round(validity_score, 2),
            'consistency': round(consistency_score, 2),
            'grade': self._get_quality_grade(overall_score),
            'status': self._get_quality_status(overall_score)
        }
    
    def _calculate_completeness_score(self) -> float:
        """Calculate completeness score based on missing values."""
        if self.total_cells == 0:
            return 100.0
        
        missing_cells = self.df.isnull().sum().sum()
        completeness = ((self.total_cells - missing_cells) / self.total_cells) * 100
        return completeness
    
    def _calculate_uniqueness_score(self) -> float:
        """Calculate uniqueness score based on duplicates."""
        if self.total_rows == 0:
            return 100.0
        
        duplicates = self.df.duplicated().sum()
        uniqueness = ((self.total_rows - duplicates) / self.total_rows) * 100
        return uniqueness
    
    def _calculate_validity_score(self) -> float:
        """
        Calculate validity score based on data type consistency.
        
        ✅ ENHANCED: Added retail-specific validity checks
        """
        validity_issues = 0
        total_checks = 0
        
        for col in self.df.columns:
            total_checks += 1
            
            # Check numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check for inf values
                if np.isinf(self.df[col]).any():
                    validity_issues += 1
                
                # ✅ NEW: Check for negative values in sales/revenue/price columns
                if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'price', 'quantity']):
                    if (self.df[col] < 0).any():
                        validity_issues += 0.5
                
                # ✅ NEW: Check for extreme outliers (beyond 6 sigma)
                if self.df[col].std() > 0:
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                    if (z_scores > 6).any():
                        validity_issues += 0.3
            
            # Check datetime columns
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                # Check for future dates (might be invalid)
                future_dates = self.df[col] > pd.Timestamp.now()
                if future_dates.any():
                    validity_issues += 0.5  # Soft penalty
        
        if total_checks == 0:
            return 100.0
        
        validity = ((total_checks - validity_issues) / total_checks) * 100
        return max(0, validity)
    
    def _calculate_consistency_score(self) -> float:
        """Calculate consistency score based on data patterns."""
        consistency_score = 100.0
        
        # Check for columns with high cardinality (might indicate inconsistency)
        for col in self.df.select_dtypes(include=['object']).columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            
            # If unique ratio is between 0.5 and 0.95, might indicate inconsistency
            if 0.5 < unique_ratio < 0.95:
                consistency_score -= 5
        
        return max(0, consistency_score)
    
    def _get_quality_grade(self, score: float) -> str:
        """Get letter grade for quality score."""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'B+'
        elif score >= 80:
            return 'B'
        elif score >= 75:
            return 'C+'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _get_quality_status(self, score: float) -> str:
        """Get status message for quality score."""
        if score >= 90:
            return 'Excellent'
        elif score >= 80:
            return 'Good'
        elif score >= 70:
            return 'Fair'
        elif score >= 60:
            return 'Poor'
        else:
            return 'Critical'
    
    def get_missing_values_summary(self) -> pd.DataFrame:
        """
        Get detailed missing values summary.
        
        Returns:
            DataFrame with missing value statistics per column
        """
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        summary = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': missing_counts.values,
            'Missing %': missing_percentages.values,
            'Present Count': len(self.df) - missing_counts.values,
            'Data Type': self.df.dtypes.values
        })
        
        # Sort by missing percentage descending
        summary = summary.sort_values('Missing %', ascending=False)
        summary = summary[summary['Missing Count'] > 0]  # Only show columns with missing values
        
        return summary
    
    def get_duplicate_analysis(self) -> Dict[str, Any]:
        """
        Analyze duplicate rows.
        
        Returns:
            Dictionary with duplicate statistics
        """
        total_duplicates = self.df.duplicated().sum()
        duplicate_percentage = (total_duplicates / len(self.df)) * 100 if len(self.df) > 0 else 0
        
        # Get duplicate rows
        duplicate_rows = self.df[self.df.duplicated(keep=False)]
        
        return {
            'total_duplicates': int(total_duplicates),
            'duplicate_percentage': round(duplicate_percentage, 2),
            'unique_rows': len(self.df) - total_duplicates,
            'duplicate_rows': duplicate_rows if len(duplicate_rows) > 0 else None,
            'has_duplicates': total_duplicates > 0
        }
    
    def get_column_statistics(self) -> pd.DataFrame:
        """
        Get comprehensive column-level statistics.
        
        Returns:
            DataFrame with column statistics
        """
        stats = []
        
        for col in self.df.columns:
            col_data = self.df[col]
            
            stat = {
                'Column': col,
                'Data Type': str(col_data.dtype),
                'Non-Null': int(col_data.count()),
                'Null': int(col_data.isnull().sum()),
                'Unique': int(col_data.nunique()),
                'Unique %': round((col_data.nunique() / len(col_data)) * 100, 2) if len(col_data) > 0 else 0
            }
            
            # Add numeric-specific stats
            if pd.api.types.is_numeric_dtype(col_data):
                stat['Mean'] = round(col_data.mean(), 2) if col_data.count() > 0 else None
                stat['Std Dev'] = round(col_data.std(), 2) if col_data.count() > 0 else None
                stat['Min'] = round(col_data.min(), 2) if col_data.count() > 0 else None
                stat['Max'] = round(col_data.max(), 2) if col_data.count() > 0 else None
            
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def detect_outliers_summary(self) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns using IQR method.
        
        Returns:
            Dictionary with outlier statistics per numeric column
        """
        outlier_summary = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(col_data)) * 100
                
                if outlier_count > 0:
                    outlier_summary[col] = {
                        'count': int(outlier_count),
                        'percentage': round(outlier_percentage, 2),
                        'lower_bound': round(lower_bound, 2),
                        'upper_bound': round(upper_bound, 2)
                    }
        
        return outlier_summary
    
    def get_data_freshness(self, date_column: str) -> Optional[Dict[str, Any]]:
        """
        Analyze data freshness based on date column.
        
        Args:
            date_column: Name of date column
            
        Returns:
            Dictionary with freshness metrics or None if column not found
        """
        if date_column not in self.df.columns:
            return None
        
        try:
            date_series = pd.to_datetime(self.df[date_column], errors='coerce')
            date_series = date_series.dropna()
            
            if len(date_series) == 0:
                return None
            
            most_recent = date_series.max()
            oldest = date_series.min()
            now = pd.Timestamp.now()
            
            days_since_update = (now - most_recent).days
            data_span_days = (most_recent - oldest).days
            
            # Determine freshness status
            if days_since_update <= 1:
                freshness_status = 'Very Fresh'
                freshness_emoji = '🟢'
            elif days_since_update <= 7:
                freshness_status = 'Fresh'
                freshness_emoji = '🟢'
            elif days_since_update <= 30:
                freshness_status = 'Moderate'
                freshness_emoji = '🟡'
            elif days_since_update <= 90:
                freshness_status = 'Stale'
                freshness_emoji = '🟠'
            else:
                freshness_status = 'Very Stale'
                freshness_emoji = '🔴'
            
            return {
                'most_recent_date': most_recent.strftime('%Y-%m-%d'),
                'oldest_date': oldest.strftime('%Y-%m-%d'),
                'days_since_update': days_since_update,
                'data_span_days': data_span_days,
                'freshness_status': freshness_status,
                'freshness_emoji': freshness_emoji
            }
        
        except Exception as e:
            logger.error(f"Error analyzing data freshness: {e}")
            return None
    
    # ========================================================================
    # ✅ NEW METHODS FOR RETAIL-SPECIFIC VALIDATION
    # ========================================================================
    
    def validate_time_series(
        self, 
        date_column: str, 
        expected_freq: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Validate time series specific requirements.
        
        Args:
            date_column: Name of date column
            expected_freq: Expected frequency ('D', 'W', 'M', etc.)
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            if date_column not in self.df.columns:
                results['is_valid'] = False
                results['issues'].append(f"Date column '{date_column}' not found")
                return results
            
            # Convert to datetime
            date_series = pd.to_datetime(self.df[date_column], errors='coerce')
            
            # Check for invalid dates
            invalid_dates = date_series.isnull().sum()
            if invalid_dates > 0:
                results['warnings'].append(f"{invalid_dates} invalid dates found")
            
            date_series = date_series.dropna().sort_values()
            
            if len(date_series) == 0:
                results['is_valid'] = False
                results['issues'].append("No valid dates found")
                return results
            
            # Check for date gaps
            date_diffs = date_series.diff().dropna()
            mode_diff = date_diffs.mode()[0] if len(date_diffs) > 0 else pd.Timedelta(days=1)
            gaps = date_diffs[date_diffs > mode_diff * 1.5]
            
            if len(gaps) > 0:
                results['warnings'].append(
                    f"{len(gaps)} date gaps detected (expected freq: {mode_diff})"
                )
            
            # Detect frequency
            inferred_freq = pd.infer_freq(date_series)
            results['metrics']['inferred_frequency'] = inferred_freq or 'irregular'
            results['metrics']['date_range_days'] = int((date_series.max() - date_series.min()).days)
            results['metrics']['total_periods'] = len(date_series)
            results['metrics']['missing_dates'] = len(gaps)
            
            # Check expected frequency
            if expected_freq and inferred_freq != expected_freq:
                results['warnings'].append(
                    f"Expected frequency '{expected_freq}', but detected '{inferred_freq}'"
                )
            
            # Check minimum data requirement (at least 2 years for forecasting)
            min_required_days = 730  # 2 years
            if results['metrics']['date_range_days'] < min_required_days:
                results['warnings'].append(
                    f"Only {results['metrics']['date_range_days']} days of data. "
                    f"Recommended: {min_required_days}+ days for reliable forecasting"
                )
            
            # Check for duplicated dates
            duplicate_dates = date_series.duplicated().sum()
            if duplicate_dates > 0:
                results['issues'].append(f"{duplicate_dates} duplicate dates found")
                results['is_valid'] = False
            
            results['metrics']['quality_score'] = self._calculate_time_series_quality(results)
            
            return results
        
        except Exception as e:
            logger.error(f"Error validating time series: {e}")
            results['is_valid'] = False
            results['issues'].append(f"Validation error: {str(e)}")
            return results
    
    def _calculate_time_series_quality(self, validation_results: Dict) -> float:
        """Calculate time series quality score."""
        score = 100.0
        
        # Penalize for issues
        score -= len(validation_results['issues']) * 20
        score -= len(validation_results['warnings']) * 10
        
        return max(0, score)
    
    def validate_business_rules(self, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ✅ NEW: Validate domain-specific business rules.
        
        Args:
            rules: Dictionary of business rules to validate
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'violations': [],
            'summary': {}
        }
        
        # Default rules for sales forecasting
        if rules is None:
            rules = {
                'sales_non_negative': {'columns': ['sales', 'revenue', 'quantity', 'units sold'], 'min': 0},
                'price_positive': {'columns': ['price', 'cost', 'competitor pricing'], 'min': 0.01},
                'promotion_binary': {'columns': ['promotion', 'holiday/promotion', 'discount_flag'], 'values': [0, 1]},
                'discount_range': {'columns': ['discount'], 'min': 0, 'max': 100}
            }
        
        try:
            # Check sales/revenue columns for negative values
            for col_pattern in rules.get('sales_non_negative', {}).get('columns', []):
                matching_cols = [col for col in self.df.columns if col_pattern.lower() in col.lower()]
                for col in matching_cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        negative_count = (self.df[col] < 0).sum()
                        if negative_count > 0:
                            results['violations'].append({
                                'rule': 'sales_non_negative',
                                'column': col,
                                'violation_count': int(negative_count),
                                'percentage': round((negative_count / len(self.df)) * 100, 2)
                            })
                            results['is_valid'] = False
            
            # Check price columns for non-positive values
            for col_pattern in rules.get('price_positive', {}).get('columns', []):
                matching_cols = [col for col in self.df.columns if col_pattern.lower() in col.lower()]
                for col in matching_cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        invalid_count = (self.df[col] <= 0).sum()
                        if invalid_count > 0:
                            results['violations'].append({
                                'rule': 'price_positive',
                                'column': col,
                                'violation_count': int(invalid_count),
                                'percentage': round((invalid_count / len(self.df)) * 100, 2)
                            })
                            results['is_valid'] = False
            
            # Check binary columns
            for col_pattern in rules.get('promotion_binary', {}).get('columns', []):
                matching_cols = [col for col in self.df.columns if col_pattern.lower() in col.lower()]
                for col in matching_cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        unique_vals = self.df[col].dropna().unique()
                        expected_vals = rules['promotion_binary']['values']
                        invalid_vals = [v for v in unique_vals if v not in expected_vals]
                        if invalid_vals:
                            results['violations'].append({
                                'rule': 'promotion_binary',
                                'column': col,
                                'invalid_values': [float(v) for v in invalid_vals],
                                'expected_values': expected_vals
                            })
                            results['is_valid'] = False
            
            # Check discount range
            if 'discount_range' in rules:
                for col_pattern in rules['discount_range']['columns']:
                    matching_cols = [col for col in self.df.columns if col_pattern.lower() in col.lower()]
                    for col in matching_cols:
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            min_val = rules['discount_range']['min']
                            max_val = rules['discount_range']['max']
                            invalid_count = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()
                            if invalid_count > 0:
                                results['violations'].append({
                                    'rule': 'discount_range',
                                    'column': col,
                                    'violation_count': int(invalid_count),
                                    'expected_range': f'[{min_val}, {max_val}]'
                                })
                                results['is_valid'] = False
            
            results['summary']['total_violations'] = len(results['violations'])
            results['summary']['rules_checked'] = len(rules)
            
            return results
        
        except Exception as e:
            logger.error(f"Error validating business rules: {e}")
            results['is_valid'] = False
            results['violations'].append({'error': str(e)})
            return results
    
    def analyze_zero_inflation(self, target_column: str = 'Units Sold') -> Dict[str, Any]:
        """
        ✅ NEW: Analyze zero-inflation in sales data.
        
        Args:
            target_column: Column to analyze for zeros
        
        Returns:
            Zero-inflation analysis
        """
        try:
            if target_column not in self.df.columns:
                return {'error': f'Column {target_column} not found'}
            
            zero_count = (self.df[target_column] == 0).sum()
            zero_percentage = (zero_count / len(self.df)) * 100
            
            analysis = {
                'zero_count': int(zero_count),
                'non_zero_count': int(len(self.df) - zero_count),
                'zero_percentage': round(zero_percentage, 2),
                'severity': self._classify_zero_inflation(zero_percentage),
                'recommendation': self._get_zero_inflation_recommendation(zero_percentage)
            }
            
            # Zero-inflation by store (if applicable)
            if 'Store ID' in self.df.columns:
                zero_by_store = self.df.groupby('Store ID')[target_column].apply(
                    lambda x: (x == 0).sum() / len(x) * 100
                )
                analysis['by_store'] = {
                    'mean': round(zero_by_store.mean(), 2),
                    'std': round(zero_by_store.std(), 2),
                    'worst_stores': zero_by_store.nlargest(5).to_dict()
                }
            
            # Zero-inflation by product (if applicable)
            if 'Product ID' in self.df.columns:
                zero_by_product = self.df.groupby('Product ID')[target_column].apply(
                    lambda x: (x == 0).sum() / len(x) * 100
                )
                analysis['by_product'] = {
                    'mean': round(zero_by_product.mean(), 2),
                    'std': round(zero_by_product.std(), 2),
                    'worst_products': zero_by_product.nlargest(5).to_dict()
                }
            
            # Zero-inflation by category (if applicable)
            if 'Category' in self.df.columns:
                zero_by_category = self.df.groupby('Category')[target_column].apply(
                    lambda x: (x == 0).sum() / len(x) * 100
                )
                analysis['by_category'] = zero_by_category.to_dict()
            
            logger.info(f"✅ Zero-inflation analysis complete: {zero_percentage:.2f}%")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in zero-inflation analysis: {str(e)}")
            return {'error': str(e)}
    
    def _classify_zero_inflation(self, percentage: float) -> str:
        """Classify severity of zero-inflation."""
        if percentage < 5:
            return 'Low (< 5%)'
        elif percentage < 15:
            return 'Moderate (5-15%)'
        elif percentage < 30:
            return 'High (15-30%)'
        else:
            return 'Severe (> 30%)'
    
    def _get_zero_inflation_recommendation(self, percentage: float) -> str:
        """Get model recommendation based on zero-inflation."""
        if percentage < 5:
            return 'Standard models (XGBoost, Random Forest) are suitable'
        elif percentage < 15:
            return 'Consider XGBoost or Random Forest (handle zeros well)'
        elif percentage < 30:
            return 'Recommend: XGBoost, Zero-Inflated models, or Two-Stage approach'
        else:
            return 'Critical: Use Zero-Inflated Poisson/Negative Binomial or Two-Stage models'
    
    def analyze_feature_correlations(self, threshold: float = 0.95) -> Dict[str, Any]:
        """
        ✅ NEW: Detect highly correlated features (multicollinearity).
        
        Args:
            threshold: Correlation threshold (0-1)
        
        Returns:
            Dictionary with correlation analysis
        """
        try:
            numeric_df = self.df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return {
                    'high_correlations': [],
                    'correlation_matrix': None,
                    'warning': 'Insufficient numeric columns for correlation analysis'
                }
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr().abs()
            
            # Find high correlations (exclude diagonal)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] >= threshold:
                        high_corr_pairs.append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': round(corr_matrix.iloc[i, j], 3)
                        })
            
            return {
                'high_correlations': high_corr_pairs,
                'correlation_matrix': corr_matrix,
                'threshold': threshold,
                'recommendation': (
                    'Consider removing one feature from each highly correlated pair '
                    'to avoid multicollinearity' if high_corr_pairs else 'No issues found'
                )
            }
        
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {'error': str(e)}
    
    def analyze_distributions(self) -> Dict[str, Any]:
        """
        ✅ NEW: Analyze data distributions for numeric columns.
        
        Returns:
            Dictionary with distribution metrics
        """
        try:
            numeric_df = self.df.select_dtypes(include=[np.number])
            distributions = {}
            
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                
                if len(col_data) > 0:
                    # Calculate skewness
                    skewness = col_data.skew()
                    
                    # Calculate kurtosis
                    kurtosis = col_data.kurtosis()
                    
                    # Check for zero-inflation
                    zero_count = (col_data == 0).sum()
                    zero_percentage = (zero_count / len(col_data)) * 100
                    
                    # Determine if transformation needed
                    needs_transform = abs(skewness) > 1.0
                    
                    distributions[col] = {
                        'skewness': round(skewness, 3),
                        'kurtosis': round(kurtosis, 3),
                        'zero_percentage': round(zero_percentage, 2),
                        'needs_transformation': needs_transform,
                        'suggested_transform': 'log' if skewness > 1 else 'sqrt' if skewness > 0.5 else None
                    }
            
            return {
                'distributions': distributions,
                'summary': {
                    'highly_skewed': len([d for d in distributions.values() if abs(d['skewness']) > 1]),
                    'zero_inflated': len([d for d in distributions.values() if d['zero_percentage'] > 50])
                }
            }
        
        except Exception as e:
            logger.error(f"Error analyzing distributions: {e}")
            return {'error': str(e)}
    
    def generate_quality_report(self, date_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        ✅ ENHANCED: Now includes new validation methods
        
        Args:
            date_column: Optional date column for time series validation
        
        Returns:
            Dictionary with all quality metrics
        """
        report = {
            'overall_quality': self.calculate_overall_quality_score(),
            'missing_values': self.get_missing_values_summary(),
            'duplicates': self.get_duplicate_analysis(),
            'column_stats': self.get_column_statistics(),
            'outliers': self.detect_outliers_summary(),
            'business_rules': self.validate_business_rules(),
            'correlations': self.analyze_feature_correlations(),
            'distributions': self.analyze_distributions(),
            'total_rows': self.total_rows,
            'total_columns': self.total_cols,
            'total_cells': self.total_cells
        }
        
        # Add time series validation if date column provided
        if date_column:
            report['time_series'] = self.validate_time_series(date_column)
            report['data_freshness'] = self.get_data_freshness(date_column)
        
        # Add zero-inflation analysis if Units Sold exists
        if 'Units Sold' in self.df.columns:
            report['zero_inflation'] = self.analyze_zero_inflation('Units Sold')
        
        return report


def get_quality_analyzer(df: pd.DataFrame) -> DataQualityAnalyzer:
    """Get data quality analyzer instance."""
    return DataQualityAnalyzer(df)
