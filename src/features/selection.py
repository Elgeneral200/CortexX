"""
✅ RETAIL-OPTIMIZED Feature Selection for CortexX.
Hierarchical selection across 100 Store-Product combinations.
Selects features that work for 80%+ of groups.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
from concurrent.futures import ThreadPoolExecutor
import joblib

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    ✅ RETAIL-OPTIMIZED: Hierarchical feature selection for multi-store, multi-product data.
    
    Key improvements:
    - Per-Store-Product Random Forest training (100 models)
    - Aggregated importance across all groups
    - 80%+ group coverage validation
    - Per-group R² performance validation
    """
    
    def __init__(self, n_jobs: int = -1):
        self.logger = logging.getLogger(__name__)
        self.n_jobs = n_jobs
        
    def calculate_grouped_importance(self, df: pd.DataFrame, target_col: str, 
                               group_cols: List[str] = ['Store ID', 'Product ID'],
                               method: str = 'random_forest', 
                               sample_size: int = 1000) -> pd.DataFrame:
        """
        ✅ BULLETPROOF: Calculate feature importance PER Store-Product group.
        """
        try:
            self.logger.info(f"🚀 Hierarchical importance: {len(df)} rows, method={method}")
            
            # Get unique groups
            groups = df[group_cols].drop_duplicates()
            n_groups = len(groups)
            self.logger.info(f"📊 {n_groups} Store-Product groups found")
            
            if n_groups == 0:
                return pd.DataFrame(columns=['feature', 'importance', 'group_coverage_pct'])
            
            all_features = []
            
            # ✅ FIXED: Sequential processing (no ThreadPool issues)
            for group_name, group_data in df.groupby(group_cols):
                if len(group_data) < 10:  # Skip small groups
                    continue
                    
                store_prod = tuple(group_name)
                X = group_data.drop(columns=[target_col] + group_cols, errors='ignore')
                X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
                
                if len(X_numeric) < 10:
                    continue
                    
                X_sample = X_numeric.sample(min(sample_size, len(X_numeric)), random_state=42)
                y = group_data[target_col].loc[X_sample.index].fillna(group_data[target_col].mean())
                
                # Calculate importance
                if method == 'random_forest':
                    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                    rf.fit(X_sample, y)
                    importances = rf.feature_importances_
                elif method == 'f_regression':
                    f_scores, _ = f_regression(X_sample, y)
                    importances = f_scores / (np.sum(f_scores) + 1e-8)
                else:
                    mi_scores = mutual_info_regression(X_sample, y, random_state=42)
                    importances = mi_scores / (np.sum(mi_scores) + 1e-8)
                
                # Store results
                for feature, importance in zip(X_sample.columns, importances):
                    all_features.append({
                        'store_product': store_prod,
                        'feature': feature,
                        'importance': float(importance)
                    })
            
            if not all_features:
                self.logger.warning("No valid group data found")
                return pd.DataFrame(columns=['feature', 'importance', 'group_coverage_pct'])
            
            # Aggregate
            importance_df = pd.DataFrame(all_features)
            avg_importance = importance_df.groupby('feature')['importance'].mean().reset_index()
            avg_importance = avg_importance.sort_values('importance', ascending=False)
            
            coverage = importance_df.groupby('feature')['store_product'].nunique() / n_groups * 100
            avg_importance['group_coverage_pct'] = avg_importance['feature'].map(coverage).fillna(0)
            
            self.logger.info(f"✅ Hierarchical importance: {len(avg_importance)} features")
            return avg_importance
            
        except Exception as e:
            self.logger.error(f"Grouped importance failed: {str(e)}")
            return pd.DataFrame(columns=['feature', 'importance', 'group_coverage_pct'])


    
    def select_hierarchical_features(self, df: pd.DataFrame, target_col: str,
                                   group_cols: List[str] = ['Store ID', 'Product ID'],
                                   n_features: int = 20, min_coverage: float = 80.0,
                                   method: str = 'random_forest') -> List[str]:
        """
        ✅ MAIN SELECTION: Select features with 80%+ group coverage.
        """
        try:
            importance_df = self.calculate_grouped_importance(df, target_col, group_cols, method)
            
            # Filter: High importance + high coverage
            selected = importance_df[
                (importance_df['importance'] > importance_df['importance'].quantile(0.5)) &
                (importance_df['group_coverage_pct'] >= min_coverage)
            ].head(n_features)['feature'].tolist()
            
            self.logger.info(f"✅ Selected {len(selected)} hierarchical features (≥{min_coverage}% coverage)")
            return selected
            
        except Exception as e:
            self.logger.error(f"Hierarchical selection failed: {str(e)}")
            return []
    
    def validate_group_performance(self, df: pd.DataFrame, target_col: str,
                                 selected_features: List[str],
                                 group_cols: List[str] = ['Store ID', 'Product ID']) -> Dict:
        """
        Validate selected features work across Store-Product groups.
        """
        try:
            results = {}
            groups = df[group_cols].drop_duplicates()
            
            r2_scores = []
            for _, group_data in df.groupby(group_cols):
                X = group_data[selected_features].fillna(0)
                y = group_data[target_col]
                
                if len(X) > 10:  # Minimum samples
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(X, y)
                    score = r2_score(y, rf.predict(X))
                    r2_scores.append(score)
            
            results['avg_r2'] = np.mean(r2_scores)
            results['r2_std'] = np.std(r2_scores)
            results['coverage_groups'] = len(r2_scores)
            results['min_r2'] = np.min(r2_scores)
            
            self.logger.info(f"✅ Group validation: R²={results['avg_r2']:.3f}±{results['r2_std']:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Group validation failed: {str(e)}")
            return {'error': str(e)}
    
    def create_retail_selection_report(self, df: pd.DataFrame, target_col: str,
                                     group_cols: List[str] = ['Store ID', 'Product ID']) -> Dict:
        """
        ✅ COMPREHENSIVE RETAIL REPORT.
        """
        try:
            report = {}
            
            # Basic stats
            report['dataset_shape'] = df.shape
            report['n_groups'] = df[group_cols].drop_duplicates().shape[0]
            
            # Hierarchical importance (3 methods)
            methods = ['random_forest', 'f_regression', 'mutual_info']
            report['importance_by_method'] = {}
            
            for method in methods:
                imp_df = self.calculate_grouped_importance(df, target_col, group_cols, method)
                report['importance_by_method'][method] = imp_df.head(20).to_dict('records')
            
            # Recommended features
            selected = self.select_hierarchical_features(df, target_col, group_cols)
            report['recommended_features'] = selected
            
            # Performance validation
            report['group_performance'] = self.validate_group_performance(df, target_col, selected, group_cols)
            
            # Correlation cleanup
            report['features_after_correlation'] = self.remove_correlated_features(df)
            
            self.logger.info("✅ Retail selection report complete")
            return report
            
        except Exception as e:
            self.logger.error(f"Report failed: {str(e)}")
            return {'error': str(e)}
    
    # Legacy methods (backward compatibility)
    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str, method: str = 'random_forest'):
        """Legacy: Global (non-hierarchical) importance."""
        return self.calculate_grouped_importance(df, target_col, group_cols=[], method=method)
    
    def select_features(self, df: pd.DataFrame, target_col: str, n_features: int = 20, method: str = 'random_forest'):
        """Legacy: Global selection."""
        return self.select_hierarchical_features(df, target_col, group_cols=[], n_features=n_features, method=method)
    
    def remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features."""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return df.columns.tolist()
            
            corr_matrix = numeric_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            features_to_keep = [col for col in numeric_df.columns if col not in to_drop]
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            features_to_keep.extend(non_numeric_cols)
            
            self.logger.info(f"Removed {len(to_drop)} correlated features")
            return features_to_keep
            
        except Exception as e:
            self.logger.error(f"Correlation removal failed: {str(e)}")
            return df.columns.tolist()
    
    # Private methods (unchanged)
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            return rf.feature_importances_
        except:
            return np.zeros(X.shape[1])
    
    def _f_regression_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        try:
            f_scores, _ = f_regression(X, y)
            return f_scores / np.sum(f_scores)
        except:
            return np.zeros(X.shape[1])
    
    def _mutual_info_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            return mi_scores / np.sum(mi_scores)
        except:
            return np.zeros(X.shape[1])
