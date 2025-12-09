"""
Data exploration module for CortexX sales forecasting platform.
Provides statistical analysis and insights generation capabilities.

✅ RETAIL INVENTORY FORECASTING OPTIMIZED:
- Store-level and Product-level analysis
- Hierarchical aggregation (Store × Product)
- Zero-inflation detection
- Price elasticity analysis
- Promotion/Holiday impact analysis
- Regional performance metrics
- Category-level insights
- Inventory turnover analysis
- Multi-dimensional time series analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataExplorer:
    """
    A class to perform comprehensive exploratory data analysis for retail forecasting.
    
    ✅ ENHANCED: Retail-specific analysis capabilities
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ✅ NEW: Retail-specific column mappings
        self.retail_columns = {
            'date': 'Date',
            'store': 'Store ID',
            'product': 'Product ID',
            'sales': 'Units Sold',
            'price': 'Price',
            'category': 'Category',
            'inventory': 'Inventory Level',
            'discount': 'Discount',
            'promotion': 'Holiday/Promotion',
            'region': 'Region',
            'weather': 'Weather Condition',
            'seasonality': 'Seasonality',
            'competitor_price': 'Competitor Pricing',
            'demand_forecast': 'Demand Forecast'
        }
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for the dataset.
        
        ✅ ENHANCED: Added retail-specific metrics
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, Any]: Dictionary containing summary statistics
        """
        try:
            summary = {
                'dataset_shape': df.shape,
                'data_types': dict(df.dtypes.value_counts()),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'duplicate_rows': int(df.duplicated().sum()),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
            
            # Numerical columns statistics
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                summary['numerical_stats'] = df[numerical_cols].describe().to_dict()
                summary['skewness'] = df[numerical_cols].skew().to_dict()
                summary['kurtosis'] = df[numerical_cols].kurtosis().to_dict()
            
            # Categorical columns statistics
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                categorical_stats = {}
                for col in categorical_cols:
                    categorical_stats[col] = {
                        'unique_count': int(df[col].nunique()),
                        'top_categories': df[col].value_counts().head(5).to_dict(),
                        'missing_count': int(df[col].isnull().sum())
                    }
                summary['categorical_stats'] = categorical_stats
            
            # ✅ NEW: Retail-specific summary
            summary['retail_metrics'] = self._generate_retail_summary(df)
            
            self.logger.info("Generated comprehensive summary statistics")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary statistics: {str(e)}")
            raise ValueError(f"Summary statistics generation failed: {str(e)}")
    
    def _generate_retail_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ NEW: Generate retail-specific summary metrics.
        
        Args:
            df: Input dataframe
        
        Returns:
            Retail metrics dictionary
        """
        retail_summary = {}
        
        try:
            # Store analysis
            if 'Store ID' in df.columns:
                retail_summary['stores'] = {
                    'total_stores': int(df['Store ID'].nunique()),
                    'store_ids': df['Store ID'].unique().tolist()
                }
            
            # Product analysis
            if 'Product ID' in df.columns:
                retail_summary['products'] = {
                    'total_products': int(df['Product ID'].nunique()),
                    'avg_products_per_store': int(len(df) / df['Store ID'].nunique()) if 'Store ID' in df.columns else None
                }
            
            # Category analysis
            if 'Category' in df.columns:
                retail_summary['categories'] = {
                    'total_categories': int(df['Category'].nunique()),
                    'category_distribution': df['Category'].value_counts().to_dict()
                }
            
            # Sales metrics
            if 'Units Sold' in df.columns:
                retail_summary['sales'] = {
                    'total_units_sold': int(df['Units Sold'].sum()),
                    'avg_daily_sales': round(df['Units Sold'].mean(), 2),
                    'median_daily_sales': round(df['Units Sold'].median(), 2),
                    'max_daily_sales': int(df['Units Sold'].max()),
                    'zero_sales_count': int((df['Units Sold'] == 0).sum()),
                    'zero_sales_percentage': round((df['Units Sold'] == 0).sum() / len(df) * 100, 2)
                }
            
            # Price analysis
            if 'Price' in df.columns:
                retail_summary['pricing'] = {
                    'avg_price': round(df['Price'].mean(), 2),
                    'min_price': round(df['Price'].min(), 2),
                    'max_price': round(df['Price'].max(), 2),
                    'price_std': round(df['Price'].std(), 2)
                }
            
            # Promotion analysis
            if 'Holiday/Promotion' in df.columns:
                promo_count = df['Holiday/Promotion'].sum()
                retail_summary['promotions'] = {
                    'total_promo_days': int(promo_count),
                    'promo_percentage': round(promo_count / len(df) * 100, 2)
                }
            
            # Date range
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                retail_summary['time_range'] = {
                    'start_date': df['Date'].min().strftime('%Y-%m-%d'),
                    'end_date': df['Date'].max().strftime('%Y-%m-%d'),
                    'total_days': int((df['Date'].max() - df['Date'].min()).days)
                }
            
            # Regional analysis
            if 'Region' in df.columns:
                retail_summary['regions'] = {
                    'total_regions': int(df['Region'].nunique()),
                    'region_distribution': df['Region'].value_counts().to_dict()
                }
            
            return retail_summary
            
        except Exception as e:
            self.logger.warning(f"Partial retail summary generation: {str(e)}")
            return retail_summary
    
    def analyze_hierarchical_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ NEW: Analyze the hierarchical structure of retail data.
        
        Args:
            df: Input dataframe with Store ID and Product ID
        
        Returns:
            Hierarchical analysis results
        """
        try:
            analysis = {}
            
            # Check for required columns
            if 'Store ID' not in df.columns or 'Product ID' not in df.columns:
                return {'error': 'Missing Store ID or Product ID columns'}
            
            # Store-Product combinations
            store_product_groups = df.groupby(['Store ID', 'Product ID'])
            analysis['store_product_combinations'] = {
                'total_combinations': len(store_product_groups),
                'expected_combinations': df['Store ID'].nunique() * df['Product ID'].nunique(),
                'coverage_percentage': round(
                    len(store_product_groups) / (df['Store ID'].nunique() * df['Product ID'].nunique()) * 100, 2
                )
            }
            
            # Records per combination
            records_per_combo = store_product_groups.size()
            analysis['records_per_combination'] = {
                'mean': round(records_per_combo.mean(), 2),
                'std': round(records_per_combo.std(), 2),
                'min': int(records_per_combo.min()),
                'max': int(records_per_combo.max()),
                'median': int(records_per_combo.median())
            }
            
            # Store-level analysis
            if 'Units Sold' in df.columns:
                store_sales = df.groupby('Store ID')['Units Sold'].agg(['sum', 'mean', 'count'])
                analysis['store_level'] = {
                    'top_stores_by_volume': store_sales.sort_values('sum', ascending=False).head(5)['sum'].to_dict(),
                    'top_stores_by_avg': store_sales.sort_values('mean', ascending=False).head(5)['mean'].to_dict()
                }
                
                # Product-level analysis
                product_sales = df.groupby('Product ID')['Units Sold'].agg(['sum', 'mean', 'count'])
                analysis['product_level'] = {
                    'top_products_by_volume': product_sales.sort_values('sum', ascending=False).head(5)['sum'].to_dict(),
                    'top_products_by_avg': product_sales.sort_values('mean', ascending=False).head(5)['mean'].to_dict()
                }
            
            # Category-level analysis
            if 'Category' in df.columns and 'Units Sold' in df.columns:
                category_sales = df.groupby('Category')['Units Sold'].agg(['sum', 'mean'])
                analysis['category_level'] = {
                    'sales_by_category': category_sales.sort_values('sum', ascending=False)['sum'].to_dict(),
                    'avg_by_category': category_sales.sort_values('mean', ascending=False)['mean'].to_dict()
                }
            
            # Regional analysis
            if 'Region' in df.columns and 'Units Sold' in df.columns:
                region_sales = df.groupby('Region')['Units Sold'].agg(['sum', 'mean'])
                analysis['region_level'] = {
                    'sales_by_region': region_sales.sort_values('sum', ascending=False)['sum'].to_dict(),
                    'avg_by_region': region_sales.sort_values('mean', ascending=False)['mean'].to_dict()
                }
            
            self.logger.info("Completed hierarchical structure analysis")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical analysis: {str(e)}")
            return {'error': str(e)}
    
    def analyze_zero_inflation(self, df: pd.DataFrame, target_column: str = 'Units Sold') -> Dict[str, Any]:
        """
        ✅ NEW: Analyze zero-inflation in sales data.
        
        Args:
            df: Input dataframe
            target_column: Column to analyze for zeros
        
        Returns:
            Zero-inflation analysis
        """
        try:
            if target_column not in df.columns:
                return {'error': f'Column {target_column} not found'}
            
            zero_count = (df[target_column] == 0).sum()
            zero_percentage = (zero_count / len(df)) * 100
            
            analysis = {
                'zero_count': int(zero_count),
                'non_zero_count': int(len(df) - zero_count),
                'zero_percentage': round(zero_percentage, 2),
                'severity': self._classify_zero_inflation(zero_percentage)
            }
            
            # Zero-inflation by store
            if 'Store ID' in df.columns:
                zero_by_store = df.groupby('Store ID')[target_column].apply(
                    lambda x: (x == 0).sum() / len(x) * 100
                )
                analysis['by_store'] = {
                    'mean': round(zero_by_store.mean(), 2),
                    'std': round(zero_by_store.std(), 2),
                    'worst_stores': zero_by_store.nlargest(5).to_dict()
                }
            
            # Zero-inflation by product
            if 'Product ID' in df.columns:
                zero_by_product = df.groupby('Product ID')[target_column].apply(
                    lambda x: (x == 0).sum() / len(x) * 100
                )
                analysis['by_product'] = {
                    'mean': round(zero_by_product.mean(), 2),
                    'std': round(zero_by_product.std(), 2),
                    'worst_products': zero_by_product.nlargest(5).to_dict()
                }
            
            # Zero-inflation by category
            if 'Category' in df.columns:
                zero_by_category = df.groupby('Category')[target_column].apply(
                    lambda x: (x == 0).sum() / len(x) * 100
                )
                analysis['by_category'] = zero_by_category.to_dict()
            
            # Recommendation
            analysis['recommendation'] = self._get_zero_inflation_recommendation(zero_percentage)
            
            self.logger.info(f"Zero-inflation analysis complete: {zero_percentage:.2f}%")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in zero-inflation analysis: {str(e)}")
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
    
    def analyze_price_elasticity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ NEW: Analyze price elasticity and price-sales relationship.
        
        Args:
            df: Input dataframe with Price and Units Sold
        
        Returns:
            Price elasticity analysis
        """
        try:
            if 'Price' not in df.columns or 'Units Sold' not in df.columns:
                return {'error': 'Missing Price or Units Sold columns'}
            
            # Overall correlation
            correlation = df[['Price', 'Units Sold']].corr().iloc[0, 1]
            
            analysis = {
                'overall_correlation': round(correlation, 4),
                'relationship': 'Negative' if correlation < 0 else 'Positive',
                'strength': self._classify_correlation_strength(abs(correlation))
            }
            
            # Price vs Competitor Price
            if 'Competitor Pricing' in df.columns:
                price_diff = df['Price'] - df['Competitor Pricing']
                corr_with_diff = price_diff.corr(df['Units Sold'])
                analysis['price_difference_correlation'] = round(corr_with_diff, 4)
                
                # Competitive positioning
                avg_diff = price_diff.mean()
                analysis['competitive_position'] = {
                    'avg_price_difference': round(avg_diff, 2),
                    'position': 'Premium pricing' if avg_diff > 5 else 'Competitive pricing' if avg_diff > -5 else 'Discount pricing'
                }
            
            # Discount impact
            if 'Discount' in df.columns:
                discount_corr = df[['Discount', 'Units Sold']].corr().iloc[0, 1]
                analysis['discount_impact'] = {
                    'correlation': round(discount_corr, 4),
                    'effectiveness': 'Strong' if abs(discount_corr) > 0.3 else 'Moderate' if abs(discount_corr) > 0.1 else 'Weak'
                }
                
                # Average sales by discount level
                if df['Discount'].nunique() < 20:  # Discrete discount levels
                    avg_sales_by_discount = df.groupby('Discount')['Units Sold'].mean().to_dict()
                    analysis['avg_sales_by_discount'] = avg_sales_by_discount
            
            # Category-level elasticity
            if 'Category' in df.columns:
                category_elasticity = {}
                for category in df['Category'].unique():
                    cat_df = df[df['Category'] == category]
                    if len(cat_df) > 10:  # Minimum sample size
                        cat_corr = cat_df[['Price', 'Units Sold']].corr().iloc[0, 1]
                        category_elasticity[category] = round(cat_corr, 4)
                analysis['by_category'] = category_elasticity
            
            self.logger.info("Price elasticity analysis complete")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in price elasticity analysis: {str(e)}")
            return {'error': str(e)}
    
    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength."""
        if abs_corr < 0.1:
            return 'Very Weak'
        elif abs_corr < 0.3:
            return 'Weak'
        elif abs_corr < 0.5:
            return 'Moderate'
        elif abs_corr < 0.7:
            return 'Strong'
        else:
            return 'Very Strong'
    
    def analyze_promotion_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ NEW: Analyze the impact of promotions/holidays on sales.
        
        Args:
            df: Input dataframe with Holiday/Promotion flag
        
        Returns:
            Promotion impact analysis
        """
        try:
            if 'Holiday/Promotion' not in df.columns or 'Units Sold' not in df.columns:
                return {'error': 'Missing Holiday/Promotion or Units Sold columns'}
            
            # Overall impact
            promo_sales = df[df['Holiday/Promotion'] == 1]['Units Sold']
            regular_sales = df[df['Holiday/Promotion'] == 0]['Units Sold']
            
            analysis = {
                'promo_avg_sales': round(promo_sales.mean(), 2),
                'regular_avg_sales': round(regular_sales.mean(), 2),
                'uplift_percentage': round((promo_sales.mean() / regular_sales.mean() - 1) * 100, 2) if regular_sales.mean() > 0 else 0,
                'promo_days_count': int((df['Holiday/Promotion'] == 1).sum()),
                'promo_days_percentage': round((df['Holiday/Promotion'] == 1).sum() / len(df) * 100, 2)
            }
            
            # Statistical significance test
            try:
                t_stat, p_value = stats.ttest_ind(promo_sales, regular_sales)
                analysis['statistical_significance'] = {
                    't_statistic': round(t_stat, 4),
                    'p_value': round(p_value, 4),
                    'is_significant': p_value < 0.05
                }
            except:
                analysis['statistical_significance'] = {'error': 'Unable to perform test'}
            
            # Category-level promotion impact
            if 'Category' in df.columns:
                category_impact = {}
                for category in df['Category'].unique():
                    cat_df = df[df['Category'] == category]
                    cat_promo_sales = cat_df[cat_df['Holiday/Promotion'] == 1]['Units Sold'].mean()
                    cat_regular_sales = cat_df[cat_df['Holiday/Promotion'] == 0]['Units Sold'].mean()
                    
                    if cat_regular_sales > 0:
                        uplift = (cat_promo_sales / cat_regular_sales - 1) * 100
                        category_impact[category] = round(uplift, 2)
                
                analysis['by_category'] = category_impact
            
            # Promotion with discount interaction
            if 'Discount' in df.columns:
                promo_with_discount = df[(df['Holiday/Promotion'] == 1) & (df['Discount'] > 0)]['Units Sold'].mean()
                promo_without_discount = df[(df['Holiday/Promotion'] == 1) & (df['Discount'] == 0)]['Units Sold'].mean()
                
                analysis['promo_discount_interaction'] = {
                    'promo_with_discount_avg': round(promo_with_discount, 2) if not pd.isna(promo_with_discount) else 0,
                    'promo_without_discount_avg': round(promo_without_discount, 2) if not pd.isna(promo_without_discount) else 0,
                    'synergy_effect': 'Positive' if promo_with_discount > promo_without_discount else 'Negative'
                }
            
            self.logger.info("Promotion impact analysis complete")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in promotion analysis: {str(e)}")
            return {'error': str(e)}
    
    def analyze_time_series_patterns(
        self, 
        df: pd.DataFrame, 
        date_column: str = 'Date',
        value_column: str = 'Units Sold'
    ) -> Dict[str, Any]:
        """
        Analyze time series patterns including trends, seasonality, and stationarity.
        
        ✅ ENHANCED: Improved error handling and retail-specific patterns
        
        Args:
            df (pd.DataFrame): Input dataframe with time series data
            date_column (str): Name of the date column
            value_column (str): Name of the value column
            
        Returns:
            Dict[str, Any]: Time series analysis results
        """
        try:
            if date_column not in df.columns or value_column not in df.columns:
                raise ValueError(f"Required columns not found: {date_column}, {value_column}")
            
            # Ensure date column is datetime
            df_ts = df.copy()
            df_ts[date_column] = pd.to_datetime(df_ts[date_column])
            df_ts = df_ts.sort_values(date_column)
            
            # Aggregate by date if multiple records per date
            df_ts_agg = df_ts.groupby(date_column)[value_column].sum().reset_index()
            
            analysis = {
                'time_period': {
                    'start_date': df_ts_agg[date_column].min().strftime('%Y-%m-%d'),
                    'end_date': df_ts_agg[date_column].max().strftime('%Y-%m-%d'),
                    'total_days': int((df_ts_agg[date_column].max() - df_ts_agg[date_column].min()).days),
                    'data_points': len(df_ts_agg)
                },
                'basic_stats': {
                    'mean': round(df_ts_agg[value_column].mean(), 2),
                    'std': round(df_ts_agg[value_column].std(), 2),
                    'min': round(df_ts_agg[value_column].min(), 2),
                    'max': round(df_ts_agg[value_column].max(), 2),
                    'trend_strength': round(self._calculate_trend_strength(df_ts_agg, date_column, value_column), 4)
                }
            }
            
            # Seasonality analysis
            analysis['seasonality'] = self._analyze_seasonality(df_ts, date_column, value_column)
            
            # Stationarity test
            analysis['stationarity'] = self._test_stationarity(df_ts_agg[value_column])
            
            # Autocorrelation analysis
            analysis['autocorrelation'] = self._analyze_autocorrelation(df_ts_agg[value_column])
            
            # ✅ NEW: Retail-specific temporal patterns
            analysis['retail_patterns'] = self._analyze_retail_temporal_patterns(df_ts, date_column, value_column)
            
            self.logger.info("Completed time series pattern analysis")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing time series patterns: {str(e)}")
            raise ValueError(f"Time series analysis failed: {str(e)}")
    
    def _analyze_retail_temporal_patterns(
        self, 
        df: pd.DataFrame, 
        date_column: str, 
        value_column: str
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Analyze retail-specific temporal patterns.
        """
        try:
            df_temp = df.copy()
            df_temp['day_of_week'] = df_temp[date_column].dt.day_name()
            df_temp['month'] = df_temp[date_column].dt.month_name()
            df_temp['quarter'] = df_temp[date_column].dt.quarter
            df_temp['is_weekend'] = df_temp[date_column].dt.dayofweek.isin([5, 6]).astype(int)
            
            patterns = {}
            
            # Weekend vs Weekday
            weekend_sales = df_temp[df_temp['is_weekend'] == 1][value_column].mean()
            weekday_sales = df_temp[df_temp['is_weekend'] == 0][value_column].mean()
            patterns['weekend_effect'] = {
                'weekend_avg': round(weekend_sales, 2),
                'weekday_avg': round(weekday_sales, 2),
                'uplift_percentage': round((weekend_sales / weekday_sales - 1) * 100, 2) if weekday_sales > 0 else 0
            }
            
            # Best performing day
            day_performance = df_temp.groupby('day_of_week')[value_column].mean().sort_values(ascending=False)
            patterns['best_day_of_week'] = {
                'day': day_performance.index[0],
                'avg_sales': round(day_performance.values[0], 2)
            }
            
            # Best performing month
            month_performance = df_temp.groupby('month')[value_column].mean().sort_values(ascending=False)
            patterns['best_month'] = {
                'month': month_performance.index[0],
                'avg_sales': round(month_performance.values[0], 2)
            }
            
            # Quarterly patterns
            quarterly = df_temp.groupby('quarter')[value_column].mean().to_dict()
            patterns['quarterly_avg'] = {f'Q{k}': round(v, 2) for k, v in quarterly.items()}
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Retail temporal pattern analysis partial failure: {str(e)}")
            return {}
    
    def _calculate_trend_strength(
        self, 
        df: pd.DataFrame, 
        date_column: str,
        value_column: str
    ) -> float:
        """Calculate the strength of trend in time series data."""
        try:
            # Convert dates to numerical values for regression
            dates_numeric = (df[date_column] - df[date_column].min()).dt.days.values
            values = df[value_column].values
            
            # Remove NaN values
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                return 0.0
            
            dates_numeric = dates_numeric[mask]
            values = values[mask]
            
            # Calculate trend using linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
            return abs(r_value)  # Return absolute correlation coefficient as trend strength
            
        except Exception as e:
            self.logger.warning(f"Trend strength calculation failed: {str(e)}")
            return 0.0
    
    def _analyze_seasonality(
        self, 
        df: pd.DataFrame, 
        date_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Analyze seasonal patterns in time series data."""
        try:
            df_seasonal = df.copy()
            df_seasonal['year'] = df_seasonal[date_column].dt.year
            df_seasonal['month'] = df_seasonal[date_column].dt.month
            df_seasonal['day_of_week'] = df_seasonal[date_column].dt.dayofweek
            df_seasonal['week'] = df_seasonal[date_column].dt.isocalendar().week
            
            seasonality = {
                'monthly': df_seasonal.groupby('month')[value_column].mean().to_dict(),
                'weekly': df_seasonal.groupby('day_of_week')[value_column].mean().to_dict(),
                'yearly': df_seasonal.groupby('year')[value_column].mean().to_dict() if df_seasonal['year'].nunique() > 1 else {}
            }
            
            # Calculate seasonality strength
            monthly_values = np.array(list(seasonality['monthly'].values()))
            overall_var = df_seasonal[value_column].var()
            
            if overall_var > 0 and len(monthly_values) > 1:
                monthly_var = monthly_values.var()
                seasonality['strength'] = round(monthly_var / overall_var, 4)
            else:
                seasonality['strength'] = 0
            
            return seasonality
            
        except Exception as e:
            self.logger.warning(f"Seasonality analysis partial failure: {str(e)}")
            return {'monthly': {}, 'weekly': {}, 'yearly': {}, 'strength': 0}
    
    def _test_stationarity(self, series: pd.Series, max_lags: int = 10) -> Dict[str, Any]:
        """Test stationarity of time series using Augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Remove NaN values
            series_clean = series.dropna()
            
            if len(series_clean) < max_lags + 1:
                return {'is_stationary': False, 'p_value': 1.0, 'test_statistic': 0}
            
            # Perform ADF test
            result = adfuller(series_clean, maxlag=max_lags)
            
            return {
                'is_stationary': bool(result[1] <= 0.05),
                'p_value': round(result[1], 4),
                'test_statistic': round(result[0], 4),
                'critical_values': {k: round(v, 4) for k, v in result[4].items()}
            }
            
        except Exception as e:
            self.logger.warning(f"Stationarity test failed: {str(e)}")
            return {'is_stationary': False, 'p_value': 1.0, 'test_statistic': 0, 'error': str(e)}
    
    def _analyze_autocorrelation(self, series: pd.Series, max_lags: int = 20) -> Dict[str, Any]:
        """Analyze autocorrelation in time series data."""
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            series_clean = series.dropna()
            
            if len(series_clean) < max_lags + 1:
                return {'acf': [], 'pacf': [], 'significant_lags': []}
            
            # Calculate ACF and PACF
            acf_values = acf(series_clean, nlags=min(max_lags, len(series_clean) // 2))
            pacf_values = pacf(series_clean, nlags=min(max_lags, len(series_clean) // 2))
            
            # Find significant lags (outside 95% confidence interval)
            confidence_threshold = 1.96 / np.sqrt(len(series_clean))
            significant_acf = [i for i, val in enumerate(acf_values) if abs(val) > confidence_threshold and i > 0]
            significant_pacf = [i for i, val in enumerate(pacf_values) if abs(val) > confidence_threshold and i > 0]
            
            return {
                'acf': [round(x, 4) for x in acf_values.tolist()],
                'pacf': [round(x, 4) for x in pacf_values.tolist()],
                'significant_lags_acf': significant_acf,
                'significant_lags_pacf': significant_pacf,
                'suggested_ar_order': significant_pacf[0] if significant_pacf else 0,
                'suggested_ma_order': significant_acf[0] if significant_acf else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Autocorrelation analysis failed: {str(e)}")
            return {'acf': [], 'pacf': [], 'significant_lags': [], 'error': str(e)}
    
    def identify_data_issues(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify potential data quality issues in the dataset.
        
        ✅ ENHANCED: Added retail-specific checks
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, List[str]]: Dictionary of issues by category
        """
        issues = {
            'missing_data': [],
            'outliers': [],
            'inconsistencies': [],
            'data_quality': [],
            'retail_specific': []  # ✅ NEW
        }
        
        try:
            # Check for missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                for col in missing_cols:
                    missing_pct = (df[col].isnull().sum() / len(df)) * 100
                    issues['missing_data'].append(f"{col}: {missing_pct:.1f}% missing")
            
            # Check for constant columns
            constant_cols = df.columns[df.nunique() <= 1].tolist()
            if constant_cols:
                issues['data_quality'].extend([f"Constant column: {col}" for col in constant_cols])
            
            # Check for high cardinality categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.95:  # More than 95% unique
                    issues['data_quality'].append(
                        f"Very high cardinality in {col}: {df[col].nunique()} unique values ({unique_ratio*100:.1f}%)"
                    )
            
            # Check for potential outliers in numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = (outlier_count / len(df)) * 100
                if outlier_count > 0:
                    issues['outliers'].append(f"{col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
            
            # ✅ NEW: Retail-specific checks
            
            # Check for negative sales
            if 'Units Sold' in df.columns:
                negative_sales = (df['Units Sold'] < 0).sum()
                if negative_sales > 0:
                    issues['retail_specific'].append(f"Negative sales values: {negative_sales}")
            
            # Check for negative prices
            if 'Price' in df.columns:
                negative_prices = (df['Price'] < 0).sum()
                if negative_prices > 0:
                    issues['retail_specific'].append(f"Negative prices: {negative_prices}")
            
            # Check for sales > inventory
            if 'Units Sold' in df.columns and 'Inventory Level' in df.columns:
                sales_exceed_inventory = (df['Units Sold'] > df['Inventory Level']).sum()
                if sales_exceed_inventory > 0:
                    issues['retail_specific'].append(
                        f"Sales exceed inventory: {sales_exceed_inventory} cases"
                    )
            
            # Check discount validity
            if 'Discount' in df.columns:
                invalid_discount = ((df['Discount'] < 0) | (df['Discount'] > 100)).sum()
                if invalid_discount > 0:
                    issues['retail_specific'].append(f"Invalid discount values: {invalid_discount}")
            
            # Check for missing date gaps
            if 'Date' in df.columns:
                df_temp = df.copy()
                df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                date_range = pd.date_range(df_temp['Date'].min(), df_temp['Date'].max(), freq='D')
                missing_dates = len(date_range) - df_temp['Date'].nunique()
                if missing_dates > 0:
                    issues['retail_specific'].append(f"Missing dates in time series: {missing_dates} days")
            
            total_issues = sum(len(v) for v in issues.values())
            self.logger.info(f"Identified {total_issues} data issues")
            return issues
            
        except Exception as e:
            self.logger.error(f"Error identifying data issues: {str(e)}")
            issues['inconsistencies'].append(f"Error during issue identification: {str(e)}")
            return issues
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ NEW: Generate a comprehensive exploration report.
        
        Args:
            df: Input dataframe
        
        Returns:
            Complete exploration report
        """
        try:
            report = {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'summary_statistics': self.generate_summary_statistics(df),
                'hierarchical_structure': self.analyze_hierarchical_structure(df),
                'zero_inflation': self.analyze_zero_inflation(df),
                'price_elasticity': self.analyze_price_elasticity(df),
                'promotion_impact': self.analyze_promotion_impact(df),
                'data_issues': self.identify_data_issues(df)
            }
            
            # Add time series analysis if date column exists
            if 'Date' in df.columns and 'Units Sold' in df.columns:
                report['time_series_patterns'] = self.analyze_time_series_patterns(df)
            
            self.logger.info("Generated comprehensive exploration report")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            return {'error': str(e)}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_data_explorer() -> DataExplorer:
    """Get or create DataExplorer instance."""
    return DataExplorer()
