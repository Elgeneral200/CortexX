"""
Data Quality Analysis Module for CortexX Forecasting Platform
PHASE 3 - SESSION 6: Comprehensive Data Quality Dashboard
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """
    Comprehensive data quality analysis and reporting.
    
    ✅ NEW: Phase 3 - Session 6
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
        """Calculate validity score based on data type consistency."""
        validity_issues = 0
        total_checks = 0
        
        for col in self.df.columns:
            total_checks += 1
            
            # Check numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check for inf values
                if np.isinf(self.df[col]).any():
                    validity_issues += 1
            
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
                'Non-Null': col_data.count(),
                'Null': col_data.isnull().sum(),
                'Unique': col_data.nunique(),
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
                        'count': outlier_count,
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
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Returns:
            Dictionary with all quality metrics
        """
        return {
            'overall_quality': self.calculate_overall_quality_score(),
            'missing_values': self.get_missing_values_summary(),
            'duplicates': self.get_duplicate_analysis(),
            'column_stats': self.get_column_statistics(),
            'outliers': self.detect_outliers_summary(),
            'total_rows': self.total_rows,
            'total_columns': self.total_cols,
            'total_cells': self.total_cells
        }


def get_quality_analyzer(df: pd.DataFrame) -> DataQualityAnalyzer:
    """Get data quality analyzer instance."""
    return DataQualityAnalyzer(df)
