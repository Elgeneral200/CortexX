"""
Comparison Analytics for CortexX Forecasting Platform

PHASE 3 - SESSION 3: Deep comparison analytics
- Period comparison calculations
- Growth rate analysis
- Variance analysis
- Rolling comparisons
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ComparisonAnalytics:
    """
    Advanced comparison analytics engine.
    
    ✅ NEW: Phase 3 - Session 3
    """
    
    def __init__(self, current_df: pd.DataFrame, comparison_df: pd.DataFrame,
                 value_column: str, date_column: str):
        """
        Initialize comparison analytics.
        
        Args:
            current_df: Current period DataFrame
            comparison_df: Comparison period DataFrame
            value_column: Name of value column to analyze
            date_column: Name of date column
        """
        self.current_df = current_df
        self.comparison_df = comparison_df
        self.value_column = value_column
        self.date_column = date_column
        
        logger.info(f"ComparisonAnalytics initialized: Current={len(current_df)}, "
                   f"Comparison={len(comparison_df)} records")
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate comprehensive comparison summary table.
        
        Returns:
            DataFrame with comparison metrics
        """
        try:
            metrics = []
            
            # Total values
            current_total = self.current_df[self.value_column].sum()
            comparison_total = self.comparison_df[self.value_column].sum()
            total_change = self._calculate_change(current_total, comparison_total)
            
            metrics.append({
                'Metric': 'Total Sales',
                'Current Period': f"${current_total:,.0f}",
                'Previous Period': f"${comparison_total:,.0f}",
                'Change': f"{total_change['percent_change']:+.1f}%",
                'Change_Value': total_change['percent_change']  # For sorting
            })
            
            # Average values
            current_avg = self.current_df[self.value_column].mean()
            comparison_avg = self.comparison_df[self.value_column].mean()
            avg_change = self._calculate_change(current_avg, comparison_avg)
            
            metrics.append({
                'Metric': 'Average Sales',
                'Current Period': f"${current_avg:,.0f}",
                'Previous Period': f"${comparison_avg:,.0f}",
                'Change': f"{avg_change['percent_change']:+.1f}%",
                'Change_Value': avg_change['percent_change']
            })
            
            # Record count
            current_count = len(self.current_df)
            comparison_count = len(self.comparison_df)
            count_change = self._calculate_change(current_count, comparison_count)
            
            metrics.append({
                'Metric': 'Number of Records',
                'Current Period': f"{current_count:,}",
                'Previous Period': f"{comparison_count:,}",
                'Change': f"{count_change['percent_change']:+.1f}%",
                'Change_Value': count_change['percent_change']
            })
            
            # Peak day
            current_peak = self.current_df[self.value_column].max()
            comparison_peak = self.comparison_df[self.value_column].max()
            peak_change = self._calculate_change(current_peak, comparison_peak)
            
            metrics.append({
                'Metric': 'Peak Value',
                'Current Period': f"${current_peak:,.0f}",
                'Previous Period': f"${comparison_peak:,.0f}",
                'Change': f"{peak_change['percent_change']:+.1f}%",
                'Change_Value': peak_change['percent_change']
            })
            
            # Standard deviation
            current_std = self.current_df[self.value_column].std()
            comparison_std = self.comparison_df[self.value_column].std()
            std_change = self._calculate_change(current_std, comparison_std)
            
            metrics.append({
                'Metric': 'Volatility (Std Dev)',
                'Current Period': f"${current_std:,.0f}",
                'Previous Period': f"${comparison_std:,.0f}",
                'Change': f"{std_change['percent_change']:+.1f}%",
                'Change_Value': std_change['percent_change']
            })
            
            df = pd.DataFrame(metrics)
            logger.info(f"Summary table generated with {len(metrics)} metrics")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating summary table: {str(e)}")
            return pd.DataFrame()
    
    def calculate_growth_rates(self) -> Dict[str, float]:
        """
        Calculate various growth rates.
        
        Returns:
            Dictionary with growth rate metrics
        """
        try:
            current_total = self.current_df[self.value_column].sum()
            comparison_total = self.comparison_df[self.value_column].sum()
            
            # Simple growth rate
            simple_growth = ((current_total - comparison_total) / comparison_total * 100) if comparison_total != 0 else 0
            
            # Get period lengths
            current_days = (self.current_df[self.date_column].max() - 
                           self.current_df[self.date_column].min()).days + 1
            comparison_days = (self.comparison_df[self.date_column].max() - 
                             self.comparison_df[self.date_column].min()).days + 1
            
            # Daily growth rate
            current_daily = current_total / current_days if current_days > 0 else 0
            comparison_daily = comparison_total / comparison_days if comparison_days > 0 else 0
            daily_growth = ((current_daily - comparison_daily) / comparison_daily * 100) if comparison_daily != 0 else 0
            
            # CAGR (if periods are same length)
            if current_days == comparison_days and comparison_days > 0:
                periods = current_days / 365.25  # Convert to years
                cagr = (((current_total / comparison_total) ** (1 / periods)) - 1) * 100 if comparison_total != 0 and periods > 0 else 0
            else:
                cagr = None
            
            growth_rates = {
                'simple_growth': simple_growth,
                'daily_growth': daily_growth,
                'current_daily_avg': current_daily,
                'comparison_daily_avg': comparison_daily,
                'cagr': cagr,
                'current_days': current_days,
                'comparison_days': comparison_days
            }
            
            logger.info(f"Growth rates calculated: Simple={simple_growth:.1f}%, Daily={daily_growth:.1f}%")
            
            return growth_rates
            
        except Exception as e:
            logger.error(f"Error calculating growth rates: {str(e)}")
            return {}
    
    def calculate_variance_breakdown(self) -> Dict[str, Any]:
        """
        Calculate variance breakdown with categorization.
        
        Returns:
            Dictionary with variance metrics
        """
        try:
            current_total = self.current_df[self.value_column].sum()
            comparison_total = self.comparison_df[self.value_column].sum()
            
            variance = current_total - comparison_total
            variance_pct = (variance / comparison_total * 100) if comparison_total != 0 else 0
            
            # Categorize variance
            if variance_pct > 5:
                category = 'Positive'
                color = 'green'
                emoji = '📈'
            elif variance_pct < -5:
                category = 'Negative'
                color = 'red'
                emoji = '📉'
            else:
                category = 'Neutral'
                color = 'yellow'
                emoji = '➡️'
            
            variance_data = {
                'variance': variance,
                'variance_pct': variance_pct,
                'category': category,
                'color': color,
                'emoji': emoji,
                'current_total': current_total,
                'comparison_total': comparison_total
            }
            
            logger.info(f"Variance calculated: {variance_pct:+.1f}% ({category})")
            
            return variance_data
            
        except Exception as e:
            logger.error(f"Error calculating variance: {str(e)}")
            return {}
    
    def calculate_rolling_comparisons(self, window: int = 7) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate rolling averages for comparison.
        
        Args:
            window: Rolling window size (days)
            
        Returns:
            Tuple of (current_rolling, comparison_rolling)
        """
        try:
            # Ensure date column is datetime
            current_df = self.current_df.copy()
            comparison_df = self.comparison_df.copy()
            
            current_df[self.date_column] = pd.to_datetime(current_df[self.date_column])
            comparison_df[self.date_column] = pd.to_datetime(comparison_df[self.date_column])
            
            # Sort by date
            current_df = current_df.sort_values(self.date_column)
            comparison_df = comparison_df.sort_values(self.date_column)
            
            # Calculate rolling averages
            current_rolling = current_df[self.value_column].rolling(window=window, min_periods=1).mean()
            comparison_rolling = comparison_df[self.value_column].rolling(window=window, min_periods=1).mean()
            
            logger.info(f"Rolling averages calculated with window={window}")
            
            return current_rolling, comparison_rolling
            
        except Exception as e:
            logger.error(f"Error calculating rolling comparisons: {str(e)}")
            return pd.Series(), pd.Series()
    
    def get_top_changes(self, n: int = 5, group_by: str = None) -> pd.DataFrame:
        """
        Get top N changes between periods.
        
        Args:
            n: Number of top items to return
            group_by: Column to group by (e.g., 'product_id', 'category')
            
        Returns:
            DataFrame with top changes
        """
        try:
            if group_by and group_by in self.current_df.columns:
                # Group analysis
                current_grouped = self.current_df.groupby(group_by)[self.value_column].sum()
                comparison_grouped = self.comparison_df.groupby(group_by)[self.value_column].sum()
                
                # Calculate changes
                changes = []
                for item in current_grouped.index:
                    current_val = current_grouped.get(item, 0)
                    comparison_val = comparison_grouped.get(item, 0)
                    
                    if comparison_val > 0:
                        change_pct = ((current_val - comparison_val) / comparison_val) * 100
                        changes.append({
                            group_by: item,
                            'Current': current_val,
                            'Previous': comparison_val,
                            'Change': change_pct,
                            'Absolute Change': current_val - comparison_val
                        })
                
                df = pd.DataFrame(changes)
                df = df.sort_values('Change', ascending=False).head(n)
                
                logger.info(f"Top {n} changes calculated for {group_by}")
                
                return df
            else:
                logger.warning(f"Group by column '{group_by}' not found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting top changes: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _calculate_change(current: float, previous: float) -> Dict[str, float]:
        """
        Calculate change metrics.
        
        Args:
            current: Current value
            previous: Previous value
            
        Returns:
            Dictionary with change metrics
        """
        if previous == 0:
            return {
                'absolute_change': current - previous,
                'percent_change': 0 if current == 0 else float('inf')
            }
        
        return {
            'absolute_change': current - previous,
            'percent_change': ((current - previous) / previous) * 100
        }


def format_comparison_insight(variance_data: Dict[str, Any], growth_rates: Dict[str, float]) -> str:
    """
    Format comparison data into human-readable insight.
    
    Args:
        variance_data: Variance metrics
        growth_rates: Growth rate metrics
        
    Returns:
        Formatted insight string
    """
    emoji = variance_data.get('emoji', '📊')
    variance_pct = variance_data.get('variance_pct', 0)
    category = variance_data.get('category', 'Neutral')
    daily_growth = growth_rates.get('daily_growth', 0)
    
    insight = f"{emoji} **{category} Performance**: {variance_pct:+.1f}% vs previous period\n\n"
    insight += f"📅 Daily average growth: {daily_growth:+.1f}%"
    
    return insight
