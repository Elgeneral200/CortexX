# filename: preprocess.py
"""
Data Preprocessing Module - Professional Enterprise Edition v3.0

Advanced data cleaning and transformation with:
- Complete integration with Phase 1 enhanced modules
- Professional UI/UX with Streamlit dashboard
- Advanced memory optimization and type conversion
- Smart data imputation with performance comparison
- Batch transformations with undo/redo functionality
- Business rule validation engine integration
- Performance tracking and recommendations
- Comprehensive logging and error handling

Author: CortexX Team
Version: 3.0.0 - Professional Enterprise Edition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Literal, Callable
import warnings
import time
import json
import logging
from datetime import datetime
from copy import deepcopy
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies with proper error handling
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - some statistical features will be limited")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        ITERATIVE_IMPUTER_AVAILABLE = True
    except ImportError:
        ITERATIVE_IMPUTER_AVAILABLE = False
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    ITERATIVE_IMPUTER_AVAILABLE = False
    logger.warning("Scikit-learn not available - some ML-based features will be limited")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit not available - dashboard features will be limited")

try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logger.warning("memory_profiler not available - memory tracking limited")

# ============================
# ADVANCED DATA STRUCTURES
# ============================

class CleaningOperation:
    """Enhanced data cleaning operation with comprehensive metadata."""
    
    def __init__(self, operation_type: str, parameters: Dict[str, Any], 
                 timestamp: Optional[str] = None, description: str = ""):
        self.operation_type = operation_type
        self.parameters = parameters or {}
        self.timestamp = timestamp or datetime.now().isoformat()
        self.description = description
        self.execution_time = 0.0
        self.success = False
        self.error_message = None
        self.before_stats = {}
        self.after_stats = {}
        self.memory_usage = 0.0
        self.operation_id = hashlib.md5(f"{operation_type}_{time.time_ns()}".encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operation to dictionary."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "description": self.description,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
            "before_stats": self.before_stats,
            "after_stats": self.after_stats,
            "memory_usage": self.memory_usage
        }

class DataCleaningPipeline:
    """Advanced data cleaning pipeline with undo/redo functionality and performance tracking."""
    
    def __init__(self, df: pd.DataFrame, name: str = "Data Cleaning Pipeline"):
        self.original_df = df.copy()
        self.current_df = df.copy()
        self.name = name
        self.operations_history: List[CleaningOperation] = []
        self.undo_stack: List[pd.DataFrame] = []
        self.redo_stack: List[pd.DataFrame] = []
        self.max_history = 50  # Increased limit for professional use
        
        # Performance tracking
        self.total_operations = 0
        self.total_execution_time = 0.0
        self.created_at = datetime.now().isoformat()
        self.last_modified = self.created_at
        self.memory_usage_history: List[float] = []
        
        # Initialize with original state
        self._save_state()
        
        logger.info(f"Created DataCleaningPipeline '{name}' with {df.shape[0]} rows, {df.shape[1]} columns")
    
    def _save_state(self):
        """Save current state for undo functionality."""
        if len(self.undo_stack) >= self.max_history:
            self.undo_stack.pop(0)  # Remove oldest state
        self.undo_stack.append(self.current_df.copy())
        
        # Clear redo stack when new operation is performed
        self.redo_stack.clear()
        
        # Record memory usage
        self.memory_usage_history.append(self._get_dataframe_memory(self.current_df))
    
    def _get_dataframe_memory(self, df: pd.DataFrame) -> float:
        """Calculate DataFrame memory usage in MB."""
        try:
            return df.memory_usage(deep=True).sum() / 1024 / 1024
        except:
            return 0.0
    
    def _get_dataframe_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive statistics for a DataFrame."""
        try:
            stats = {
                "shape": df.shape,
                "memory_mb": self._get_dataframe_memory(df),
                "missing_values": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / df.size) * 100 if df.size > 0 else 0,
                "duplicate_rows": df.duplicated().sum(),
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
                "date_columns": len(df.select_dtypes(include=['datetime64']).columns),
                "total_columns": len(df.columns),
                "total_rows": len(df),
                "data_density": (1 - (df.isnull().sum().sum() / df.size)) * 100 if df.size > 0 else 0
            }
            
            # Add column-wise stats
            column_stats = {}
            for col in df.columns:
                col_stats = {
                    "dtype": str(df[col].dtype),
                    "missing": df[col].isnull().sum(),
                    "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
                    "unique_values": df[col].nunique() if df[col].dtype == 'object' else None
                }
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_stats.update({
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": df[col].mean(),
                        "std": df[col].std()
                    })
                
                column_stats[col] = col_stats
            
            stats["column_stats"] = column_stats
            return stats
            
        except Exception as e:
            logger.error(f"Error getting dataframe stats: {e}")
            return {
                "shape": df.shape if hasattr(df, 'shape') else (0, 0),
                "memory_mb": 0,
                "error": str(e)
            }
    
    def execute_operation(self, operation: CleaningOperation, 
                         operation_func: Callable, **kwargs) -> bool:
        """Execute a cleaning operation with comprehensive tracking."""
        try:
            start_time = time.time()
            
            # Record before stats
            operation.before_stats = self._get_dataframe_stats(self.current_df)
            
            # Profile memory usage if available
            if MEMORY_PROFILER_AVAILABLE:
                mem_usage = memory_usage((operation_func, (self.current_df,), {**operation.parameters, **kwargs}), 
                                        interval=0.1, timeout=300, max_usage=True)
                operation.memory_usage = mem_usage if isinstance(mem_usage, float) else max(mem_usage) if mem_usage else 0
            
            # Save current state before operation
            self._save_state()
            
            # Execute operation
            result_df = operation_func(self.current_df, **operation.parameters, **kwargs)
            
            if result_df is not None and isinstance(result_df, pd.DataFrame):
                self.current_df = result_df
                operation.success = True
                
                # Record after stats
                operation.after_stats = self._get_dataframe_stats(self.current_df)
                
                # Update metadata
                operation.execution_time = time.time() - start_time
                self.total_operations += 1
                self.total_execution_time += operation.execution_time
                self.last_modified = datetime.now().isoformat()
                
                # Add to history
                self.operations_history.append(operation)
                
                logger.info(f"Operation '{operation.operation_type}' completed successfully in {operation.execution_time:.2f}s")
                return True
            else:
                operation.success = False
                operation.error_message = "Operation returned invalid result"
                logger.error(f"Operation '{operation.operation_type}' failed: {operation.error_message}")
                return False
                
        except Exception as e:
            operation.success = False
            operation.error_message = str(e)
            operation.execution_time = time.time() - start_time
            logger.error(f"Operation '{operation.operation_type}' failed with error: {e}")
            return False
    
    def undo(self) -> bool:
        """Undo last operation."""
        if len(self.undo_stack) > 1:  # Keep original state
            # Move current state to redo stack
            self.redo_stack.append(self.current_df.copy())
            
            # Restore previous state
            self.undo_stack.pop()  # Remove current state
            self.current_df = self.undo_stack[-1].copy()
            
            # Update memory history
            self.memory_usage_history.append(self._get_dataframe_memory(self.current_df))
            
            self.last_modified = datetime.now().isoformat()
            logger.info("Undo operation performed")
            return True
        return False
    
    def redo(self) -> bool:
        """Redo last undone operation."""
        if self.redo_stack:
            # Save current state to undo stack
            self._save_state()
            
            # Restore from redo stack
            self.current_df = self.redo_stack.pop()
            
            # Update memory history
            self.memory_usage_history.append(self._get_dataframe_memory(self.current_df))
            
            self.last_modified = datetime.now().isoformat()
            logger.info("Redo operation performed")
            return True
        return False
    
    def reset(self):
        """Reset to original state."""
        self.current_df = self.original_df.copy()
        self.operations_history.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._save_state()
        self.last_modified = datetime.now().isoformat()
        self.total_operations = 0
        self.total_execution_time = 0.0
        self.memory_usage_history.clear()
        logger.info("Pipeline reset to original state")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        successful_ops = sum(1 for op in self.operations_history if op.success)
        
        return {
            "name": self.name,
            "total_operations": self.total_operations,
            "successful_operations": successful_ops,
            "failed_operations": self.total_operations - successful_ops,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.total_execution_time / max(self.total_operations, 1),
            "current_shape": self.current_df.shape,
            "original_shape": self.original_df.shape,
            "memory_reduction_mb": round(
                self._get_dataframe_memory(self.original_df) - 
                self._get_dataframe_memory(self.current_df), 2
            ),
            "memory_reduction_pct": round(
                (self._get_dataframe_memory(self.original_df) - 
                 self._get_dataframe_memory(self.current_df)) / 
                self._get_dataframe_memory(self.original_df) * 100, 2
            ) if self._get_dataframe_memory(self.original_df) > 0 else 0,
            "can_undo": len(self.undo_stack) > 1,
            "can_redo": len(self.redo_stack) > 0,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "data_quality_score": self._calculate_data_quality_score()
        }
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate a comprehensive data quality score (0-100)."""
        try:
            df = self.current_df
            
            # Completeness (non-missing ratio)
            completeness = (1 - (df.isnull().sum().sum() / df.size)) * 100 if df.size > 0 else 100
            
            # Uniqueness (non-duplicate ratio)
            uniqueness = (1 - (df.duplicated().sum() / len(df))) * 100 if len(df) > 0 else 100
            
            # Consistency (data type consistency)
            consistency = self._calculate_consistency_score(df)
            
            # Validity (check for valid values)
            validity = self._calculate_validity_score(df)
            
            # Calculate weighted quality score
            weights = [0.3, 0.2, 0.25, 0.25]  # completeness, uniqueness, consistency, validity
            scores = [completeness, uniqueness, consistency * 100, validity * 100]
            
            quality_score = sum(weight * score for weight, score in zip(weights, scores)) / 100
            return round(quality_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            return 0.0
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score (0-1)."""
        try:
            # Check for mixed types in object columns
            object_cols = df.select_dtypes(include=['object']).columns
            consistency_score = 1.0
            
            for col in object_cols:
                # Sample values to check type consistency
                sample = df[col].dropna().sample(min(100, len(df[col])))
                if len(sample) > 0:
                    types = set(type(val) for val in sample)
                    if len(types) > 1:
                        consistency_score -= 0.1 * (len(types) - 1)
            
            return max(0.0, consistency_score)
        except:
            return 0.8  # Default reasonable consistency
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Calculate data validity score (0-1)."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return 1.0  # No numeric columns to check
            
            validity_scores = []
            
            for col in numeric_cols:
                # Check for infinite values
                inf_count = np.isinf(df[col]).sum()
                # Check for extreme outliers (beyond 6 sigma)
                if len(df[col]) > 10:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        extreme_outliers = ((df[col] - mean).abs() > 6 * std).sum()
                        col_score = 1.0 - (inf_count + extreme_outliers) / len(df[col])
                        validity_scores.append(max(0.0, col_score))
            
            return sum(validity_scores) / len(validity_scores) if validity_scores else 1.0
        except:
            return 0.9  # Default reasonable validity
    
    def export_operations_log(self, filepath: str) -> bool:
        """Export operations history to JSON file."""
        try:
            log_data = {
                "pipeline_name": self.name,
                "created_at": self.created_at,
                "last_modified": self.last_modified,
                "operations": [op.to_dict() for op in self.operations_history],
                "summary": self.get_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            logger.info(f"Exported operations log to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting operations log: {e}")
            return False
    
    def optimize_memory(self, aggressive: bool = False) -> bool:
        """Optimize memory usage of the current dataframe."""
        try:
            operation = CleaningOperation(
                operation_type="memory_optimization",
                parameters={"aggressive": aggressive},
                description=f"Applied {'aggressive' if aggressive else 'standard'} memory optimization"
            )
            
            success = self.execute_operation(
                operation,
                lambda df, **kwargs: optimize_dtypes_advanced(df, **kwargs)[0],
                aggressive=aggressive
            )
            
            return success
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
            return False

# ============================
# ENHANCED MISSING VALUE FUNCTIONS
# ============================

def smart_imputation_comparison(df: pd.DataFrame, column: str, 
                               strategies: List[str] = None) -> Dict[str, Any]:
    """Compare different imputation strategies and recommend the best."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    if strategies is None:
        strategies = ['mean', 'median', 'mode', 'drop']
        if SKLEARN_AVAILABLE:
            strategies.extend(['knn', 'iterative'])
    
    series = df[column]
    if series.isnull().sum() == 0:
        return {"warning": "No missing values to impute"}
    
    results = {}
    
    # Split data for validation
    non_missing = series.dropna()
    if len(non_missing) < 10:
        return {"error": "Insufficient non-missing data for comparison"}
    
    # Create artificial missingness for testing
    test_series = series.copy()
    mask = ~test_series.isnull()
    test_indices = non_missing.sample(min(100, len(non_missing) // 4), random_state=42).index
    original_values = test_series.loc[test_indices].copy()
    test_series.loc[test_indices] = np.nan
    
    for strategy in strategies:
        try:
            start_time = time.time()
            
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(series):
                imputed_series = test_series.fillna(test_series.mean())
                imputed_values = imputed_series.loc[test_indices]
                
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(series):
                imputed_series = test_series.fillna(test_series.median())
                imputed_values = imputed_series.loc[test_indices]
                
            elif strategy == 'mode':
                mode_val = test_series.mode().iloc[0] if not test_series.mode().empty else test_series.iloc[0]
                imputed_series = test_series.fillna(mode_val)
                imputed_values = imputed_series.loc[test_indices]
                
            elif strategy == 'drop':
                # For drop strategy, we just remove missing values
                imputed_series = test_series.dropna()
                # For comparison, we need to align with test indices
                imputed_values = pd.Series([np.nan] * len(test_indices), index=test_indices)
                
            elif strategy == 'knn' and SKLEARN_AVAILABLE:
                # Use other numeric columns for KNN
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    knn_data = df[numeric_cols].copy()
                    try:
                        knn_imputer = KNNImputer(n_neighbors=min(5, len(knn_data.dropna())))
                        imputed_data = pd.DataFrame(
                            knn_imputer.fit_transform(knn_data), 
                            columns=numeric_cols, 
                            index=knn_data.index
                        )
                        imputed_values = imputed_data.loc[test_indices, column]
                    except Exception as e:
                        results[strategy] = {"error": f"KNN imputation failed: {str(e)}"}
                        continue
                else:
                    continue
                    
            elif strategy == 'iterative' and SKLEARN_AVAILABLE and ITERATIVE_IMPUTER_AVAILABLE:
                # Use other numeric columns for iterative imputation
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    iter_data = df[numeric_cols].copy()
                    try:
                        iter_imputer = IterativeImputer(random_state=42, max_iter=10)
                        imputed_data = pd.DataFrame(
                            iter_imputer.fit_transform(iter_data),
                            columns=numeric_cols,
                            index=iter_data.index
                        )
                        imputed_values = imputed_data.loc[test_indices, column]
                    except Exception as e:
                        results[strategy] = {"error": f"Iterative imputation failed: {str(e)}"}
                        continue
                else:
                    continue
            else:
                continue
            
            # Calculate error metrics
            if strategy == 'drop':
                # For drop strategy, we measure how much data we lose
                data_loss = (len(test_series) - len(imputed_series)) / len(test_series) * 100
                results[strategy] = {
                    "data_loss_pct": float(data_loss),
                    "execution_time": time.time() - start_time,
                    "quality_score": float(10 - data_loss / 10)  # Custom quality score
                }
            elif pd.api.types.is_numeric_dtype(original_values):
                # Filter out NaN values for comparison
                valid_mask = ~imputed_values.isna() & ~original_values.isna()
                if valid_mask.sum() > 0:
                    valid_imputed = imputed_values[valid_mask]
                    valid_original = original_values[valid_mask]
                    
                    mae = np.mean(np.abs(valid_original - valid_imputed))
                    rmse = np.sqrt(np.mean((valid_original - valid_imputed) ** 2))
                    mape = np.mean(np.abs((valid_original - valid_imputed) / np.maximum(np.abs(valid_original), 1e-10))) * 100
                    
                    results[strategy] = {
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "mape": float(mape),
                        "execution_time": time.time() - start_time,
                        "quality_score": float(10 / (1 + mape/10))  # Custom quality score
                    }
            else:
                # For categorical data, use accuracy
                valid_mask = ~imputed_values.isna() & ~original_values.isna()
                if valid_mask.sum() > 0:
                    valid_imputed = imputed_values[valid_mask]
                    valid_original = original_values[valid_mask]
                    
                    accuracy = (valid_original == valid_imputed).mean()
                    results[strategy] = {
                        "accuracy": float(accuracy),
                        "execution_time": time.time() - start_time,
                        "quality_score": float(accuracy * 10)
                    }
                
        except Exception as e:
            results[strategy] = {"error": str(e)}
    
    # Recommend best strategy
    if results:
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_strategy = max(valid_results.keys(), 
                              key=lambda x: valid_results[x].get('quality_score', 0))
            
            return {
                "comparison_results": results,
                "recommended_strategy": best_strategy,
                "missing_count": series.isnull().sum(),
                "missing_percentage": (series.isnull().sum() / len(series)) * 100,
                "column_type": str(series.dtype)
            }
    
    return {"error": "No valid imputation strategies could be evaluated"}

def advanced_missing_value_handler(df: pd.DataFrame, strategy: str = "smart", 
                                 columns: Optional[List[str]] = None,
                                 threshold: float = 0.5, 
                                 **kwargs) -> pd.DataFrame:
    """Advanced missing value handling with smart recommendations."""
    df_result = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]
    
    processing_log = []
    
    for col in columns:
        missing_count = df_result[col].isnull().sum()
        missing_pct = missing_count / len(df_result)
        
        if missing_count == 0:
            continue
        
        try:
            if strategy == "smart":
                # Use smart imputation comparison
                comparison = smart_imputation_comparison(df_result, col)
                if "recommended_strategy" in comparison:
                    recommended = comparison["recommended_strategy"]
                    
                    if recommended == "drop":
                        # For drop strategy, remove the column if it has high missingness
                        if missing_pct > threshold:
                            df_result = df_result.drop(columns=[col])
                            processing_log.append(f"Column '{col}': Dropped (>{threshold*100}% missing)")
                        else:
                            # For non-high missingness, use a different strategy
                            fallback_strategy = "median" if pd.api.types.is_numeric_dtype(df_result[col]) else "mode"
                            df_result[col] = df_result[col].fillna(
                                df_result[col].median() if fallback_strategy == "median" else df_result[col].mode().iloc[0]
                            )
                            processing_log.append(f"Column '{col}': Applied {fallback_strategy} imputation (fallback)")
                    else:
                        # Apply the recommended strategy
                        if recommended in ["mean", "median", "mode"]:
                            if recommended == "mean" and pd.api.types.is_numeric_dtype(df_result[col]):
                                fill_value = df_result[col].mean()
                            elif recommended == "median" and pd.api.types.is_numeric_dtype(df_result[col]):
                                fill_value = df_result[col].median()
                            elif recommended == "mode":
                                mode_values = df_result[col].mode()
                                fill_value = mode_values.iloc[0] if not mode_values.empty else None
                            
                            if fill_value is not None and not pd.isna(fill_value):
                                df_result[col] = df_result[col].fillna(fill_value)
                                processing_log.append(f"Column '{col}': Applied {recommended} imputation")
                else:
                    # Fallback to basic strategy
                    if pd.api.types.is_numeric_dtype(df_result[col]):
                        df_result[col] = df_result[col].fillna(df_result[col].median())
                    else:
                        mode_val = df_result[col].mode()
                        if not mode_val.empty:
                            df_result[col] = df_result[col].fillna(mode_val.iloc[0])
                    processing_log.append(f"Column '{col}': Applied fallback imputation")
                    
            elif strategy == "drop_high_missing":
                if missing_pct > threshold:
                    df_result = df_result.drop(columns=[col])
                    processing_log.append(f"Column '{col}': Dropped (>{threshold*100}% missing)")
                else:
                    # For columns with low missingness, use a different strategy
                    if pd.api.types.is_numeric_dtype(df_result[col]):
                        df_result[col] = df_result[col].fillna(df_result[col].median())
                    else:
                        mode_val = df_result[col].mode()
                        if not mode_val.empty:
                            df_result[col] = df_result[col].fillna(mode_val.iloc[0])
                    processing_log.append(f"Column '{col}': Applied imputation (low missingness)")
                    
            elif strategy == "interpolate":
                if pd.api.types.is_numeric_dtype(df_result[col]):
                    df_result[col] = df_result[col].interpolate(method='linear')
                    processing_log.append(f"Column '{col}': Applied linear interpolation")
                    
            elif strategy == "forward_fill":
                df_result[col] = df_result[col].fillna(method='ffill')
                processing_log.append(f"Column '{col}': Applied forward fill")
                
            elif strategy == "backward_fill":
                df_result[col] = df_result[col].fillna(method='bfill')
                processing_log.append(f"Column '{col}': Applied backward fill")
                
            elif strategy in ["mean", "median", "mode"]:
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df_result[col]):
                    fill_value = df_result[col].mean()
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df_result[col]):
                    fill_value = df_result[col].median()
                elif strategy == "mode":
                    mode_values = df_result[col].mode()
                    fill_value = mode_values.iloc[0] if not mode_values.empty else None
                else:
                    continue
                    
                if fill_value is not None and not pd.isna(fill_value):
                    df_result[col] = df_result[col].fillna(fill_value)
                    processing_log.append(f"Column '{col}': Applied {strategy} imputation")
                    
            elif strategy == "knn" and SKLEARN_AVAILABLE:
                # Use KNN imputation
                numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    knn_imputer = KNNImputer(n_neighbors=min(5, len(df_result)))
                    imputed_data = knn_imputer.fit_transform(df_result[numeric_cols])
                    df_result[numeric_cols] = pd.DataFrame(imputed_data, columns=numeric_cols, index=df_result.index)
                    processing_log.append(f"Applied KNN imputation to {len(numeric_cols)} numeric columns")
                    
            elif strategy == "iterative" and SKLEARN_AVAILABLE and ITERATIVE_IMPUTER_AVAILABLE:
                # Use iterative imputation
                numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    iter_imputer = IterativeImputer(random_state=42, max_iter=10)
                    imputed_data = iter_imputer.fit_transform(df_result[numeric_cols])
                    df_result[numeric_cols] = pd.DataFrame(imputed_data, columns=numeric_cols, index=df_result.index)
                    processing_log.append(f"Applied iterative imputation to {len(numeric_cols)} numeric columns")
                    
        except Exception as e:
            processing_log.append(f"Column '{col}': Error - {str(e)}")
    
    # Store processing log in DataFrame attributes
    if hasattr(df_result, 'attrs'):
        df_result.attrs['processing_log'] = processing_log
    else:
        df_result.attrs = {'processing_log': processing_log}
    
    logger.info(f"Applied missing value handling with strategy '{strategy}' to {len(columns)} columns")
    return df_result

# ============================
# ADVANCED TYPE CONVERSION
# ============================

def optimize_dtypes_advanced(df: pd.DataFrame, aggressive: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Advanced data type optimization with comprehensive memory analysis."""
    df_result = df.copy()
    optimization_report = {
        "original_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "optimizations": {},
        "memory_saved_mb": 0,
        "optimization_success_rate": 0,
        "total_columns_optimized": 0
    }
    
    successful_optimizations = 0
    total_attempts = 0
    
    for col in df.columns:
        total_attempts += 1
        original_dtype = str(df[col].dtype)
        original_memory = df[col].memory_usage(deep=True)
        
        try:
            # Numeric optimization
            if pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    # Integer optimization
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if pd.isna(min_val) or pd.isna(max_val):
                        continue
                        
                    if min_val >= 0:  # Unsigned integers
                        if max_val <= 255:
                            df_result[col] = df_result[col].astype('uint8')
                        elif max_val <= 65535:
                            df_result[col] = df_result[col].astype('uint16')
                        elif max_val <= 4294967295:
                            df_result[col] = df_result[col].astype('uint32')
                    else:  # Signed integers
                        if min_val >= -128 and max_val <= 127:
                            df_result[col] = df_result[col].astype('int8')
                        elif min_val >= -32768 and max_val <= 32767:
                            df_result[col] = df_result[col].astype('int16')
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            df_result[col] = df_result[col].astype('int32')
                            
                elif pd.api.types.is_float_dtype(df[col]):
                    # Float optimization
                    if aggressive:
                        # Try float32 if precision loss is acceptable
                        try:
                            float32_series = df[col].astype('float32')
                            if np.allclose(df[col].dropna(), float32_series.dropna(), equal_nan=True, rtol=1e-6):
                                df_result[col] = float32_series
                        except:
                            pass
                            
            # Categorical optimization
            elif pd.api.types.is_object_dtype(df[col]):
                unique_count = df[col].nunique()
                total_count = len(df[col])
                
                # Convert to category if cardinality is low
                if unique_count / total_count < 0.5 and unique_count < 1000:
                    df_result[col] = df_result[col].astype('category')
                    
                # String optimization
                elif aggressive:
                    # Try to reduce string memory usage
                    try:
                        max_length = df[col].astype(str).str.len().max()
                        if max_length <= 255:
                            df_result[col] = df_result[col].astype('string')
                    except:
                        pass
            
            # Check if optimization was successful
            new_memory = df_result[col].memory_usage(deep=True)
            if new_memory < original_memory:
                successful_optimizations += 1
                optimization_report["optimizations"][col] = {
                    "original_dtype": original_dtype,
                    "new_dtype": str(df_result[col].dtype),
                    "memory_saved_bytes": original_memory - new_memory,
                    "memory_reduction_pct": ((original_memory - new_memory) / original_memory) * 100
                }
            
        except Exception as e:
            optimization_report["optimizations"][col] = {
                "original_dtype": original_dtype,
                "error": str(e)
            }
    
    # Calculate final statistics
    final_memory_mb = df_result.memory_usage(deep=True).sum() / 1024 / 1024
    optimization_report["final_memory_mb"] = final_memory_mb
    optimization_report["memory_saved_mb"] = optimization_report["original_memory_mb"] - final_memory_mb
    optimization_report["memory_reduction_pct"] = (
        (optimization_report["original_memory_mb"] - final_memory_mb) / 
        optimization_report["original_memory_mb"]
    ) * 100 if optimization_report["original_memory_mb"] > 0 else 0
    optimization_report["optimization_success_rate"] = (successful_optimizations / total_attempts) * 100
    optimization_report["total_columns_optimized"] = successful_optimizations
    
    logger.info(f"Optimized data types: saved {optimization_report['memory_saved_mb']:.2f} MB "
               f"({optimization_report['memory_reduction_pct']:.1f}%)")
    
    return df_result, optimization_report

def smart_datetime_detection(df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Any]:
    """Smart datetime column detection with conversion suggestions."""
    detection_results = {}
    
    # Check all columns, not just object columns
    for col in df.columns:
        # Sample data for performance
        sample_data = df[col].dropna().head(sample_size)
        
        if len(sample_data) == 0:
            continue
            
        datetime_score = 0
        patterns_found = set()
        conversion_success = False
        
        # Skip if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            detection_results[col] = {
                "datetime_score": 100,
                "conversion_recommended": False,
                "already_datetime": True,
                "patterns_detected": ["Already datetime"],
                "sample_values": sample_data.head(3).tolist()
            }
            continue
            
        # Test datetime conversion
        try:
            converted = pd.to_datetime(sample_data, errors='coerce', infer_datetime_format=True)
            non_null_converted = converted.dropna()
            
            if len(non_null_converted) > 0:
                success_rate = len(non_null_converted) / len(sample_data)
                datetime_score = success_rate * 100
                
                if success_rate > 0.8:
                    conversion_success = True
                    
                    # Analyze patterns
                    sample_strings = sample_data.astype(str).head(10)
                    for date_str in sample_strings:
                        if len(date_str) == 10 and '-' in date_str:
                            patterns_found.add("YYYY-MM-DD")
                        elif len(date_str) == 19 and ' ' in date_str and ':' in date_str:
                            patterns_found.add("YYYY-MM-DD HH:MM:SS")
                        elif '/' in date_str:
                            patterns_found.add("MM/DD/YYYY or DD/MM/YYYY")
                        elif len(date_str) == 8 and date_str.isdigit():
                            patterns_found.add("YYYYMMDD")
                    
                    patterns_found = list(patterns_found)
        
        except Exception as e:
            datetime_score = 0
            conversion_success = False
        
        detection_results[col] = {
            "datetime_score": datetime_score,
            "conversion_recommended": conversion_success,
            "patterns_detected": patterns_found,
            "sample_values": sample_data.head(3).tolist(),
            "dtype": str(df[col].dtype)
        }
    
    return detection_results

def convert_to_datetime(df: pd.DataFrame, columns: List[str], 
                       format: Optional[str] = None) -> pd.DataFrame:
    """Convert specified columns to datetime with error handling."""
    df_result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        try:
            if format:
                df_result[col] = pd.to_datetime(df_result[col], format=format, errors='coerce')
            else:
                df_result[col] = pd.to_datetime(df_result[col], infer_datetime_format=True, errors='coerce')
                
            logger.info(f"Converted column '{col}' to datetime")
        except Exception as e:
            logger.error(f"Error converting column '{col}' to datetime: {e}")
    
    return df_result

# ============================
# ADVANCED OUTLIER HANDLING
# ============================

def advanced_outlier_treatment(df: pd.DataFrame, method: str = "iqr_cap",
                              columns: Optional[List[str]] = None,
                              sensitivity: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Advanced outlier treatment with multiple methods and comprehensive reporting."""
    df_result = df.copy()
    treatment_report = {}
    
    numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
            
        original_count = len(series)
        original_stats = {
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "std": series.std()
        }
        
        try:
            if method == "iqr_remove":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - sensitivity * IQR
                upper_bound = Q3 + sensitivity * IQR
                
                outlier_mask = (df_result[col] < lower_bound) | (df_result[col] > upper_bound)
                df_result = df_result[~outlier_mask]
                
            elif method == "iqr_cap":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - sensitivity * IQR
                upper_bound = Q3 + sensitivity * IQR
                
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == "zscore_remove" and SCIPY_AVAILABLE:
                z_scores = np.abs(stats.zscore(series))
                outlier_mask = z_scores > sensitivity
                df_result = df_result[~outlier_mask]
                    
            elif method == "zscore_cap":
                mean_val = series.mean()
                std_val = series.std()
                lower_bound = mean_val - sensitivity * std_val
                upper_bound = mean_val + sensitivity * std_val
                
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == "percentile_cap":
                # Cap at specified percentiles
                lower_pct = (100 - 95) / 2  # 2.5th percentile
                upper_pct = 100 - lower_pct  # 97.5th percentile
                
                lower_bound = series.quantile(lower_pct / 100)
                upper_bound = series.quantile(upper_pct / 100)
                
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == "winsorize" and SCIPY_AVAILABLE:
                # Winsorization using scipy
                try:
                    from scipy.stats.mstats import winsorize
                    winsorized = winsorize(series, limits=[0.05, 0.05])
                    df_result.loc[series.index, col] = winsorized
                except:
                    # Fallback to percentile capping
                    lower_bound = series.quantile(0.05)
                    upper_bound = series.quantile(0.95)
                    df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == "isolation_forest" and SKLEARN_AVAILABLE:
                # Use Isolation Forest for outlier detection
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                preds = iso_forest.fit_predict(series.values.reshape(-1, 1))
                outlier_mask = preds == -1
                
                if method.endswith("_remove"):
                    df_result = df_result[~outlier_mask]
                else:
                    # Cap outliers
                    non_outliers = series[~outlier_mask]
                    if len(non_outliers) > 0:
                        lower_bound = non_outliers.quantile(0.05)
                        upper_bound = non_outliers.quantile(0.95)
                        df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Calculate treatment statistics
            treated_series = df_result[col].dropna()
            new_stats = {
                "min": treated_series.min(),
                "max": treated_series.max(),
                "mean": treated_series.mean(),
                "std": treated_series.std()
            }
            
            if method in ["iqr_remove", "zscore_remove"]:
                outliers_removed = len(df) - len(df_result)
                treatment_report[col] = {
                    "method": method,
                    "outliers_removed": outliers_removed,
                    "removal_percentage": (outliers_removed / len(df)) * 100,
                    "remaining_records": len(df_result),
                    "original_stats": original_stats,
                    "new_stats": new_stats,
                    "sensitivity": sensitivity
                }
            else:
                # For capping methods, count how many values were modified
                if len(series) == len(treated_series):
                    values_modified = (series != treated_series).sum()
                else:
                    values_modified = 0
                    
                treatment_report[col] = {
                    "method": method,
                    "values_modified": int(values_modified),
                    "modification_percentage": (values_modified / len(series)) * 100 if len(series) > 0 else 0,
                    "sensitivity": sensitivity,
                    "original_stats": original_stats,
                    "new_stats": new_stats
                }
                
        except Exception as e:
            treatment_report[col] = {
                "method": method,
                "error": str(e)
            }
    
    logger.info(f"Applied outlier treatment with method '{method}' to {len(numeric_cols)} columns")
    return df_result, treatment_report

# ============================
# ADVANCED ENCODING FUNCTIONS
# ============================

def advanced_encoding(df: pd.DataFrame, strategy: str = "auto", 
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Advanced encoding for categorical variables with multiple strategies."""
    df_result = df.copy()
    
    if columns is None:
        # Auto-detect categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in df.columns]
    
    encoding_report = {}
    
    for col in categorical_cols:
        try:
            unique_count = df_result[col].nunique()
            
            if strategy == "auto":
                # Choose strategy based on cardinality
                if unique_count <= 10:
                    # One-hot encoding for low cardinality
                    encoded = pd.get_dummies(df_result[col], prefix=col)
                    df_result = pd.concat([df_result.drop(columns=[col]), encoded], axis=1)
                    encoding_report[col] = {"strategy": "one_hot", "unique_values": unique_count}
                else:
                    # Label encoding for high cardinality
                    if SKLEARN_AVAILABLE:
                        le = LabelEncoder()
                        df_result[col] = le.fit_transform(df_result[col].astype(str))
                        encoding_report[col] = {"strategy": "label_encoding", "unique_values": unique_count}
                    else:
                        # Fallback to pandas factorize
                        df_result[col] = pd.factorize(df_result[col])[0]
                        encoding_report[col] = {"strategy": "factorize", "unique_values": unique_count}
                        
            elif strategy == "one_hot":
                encoded = pd.get_dummies(df_result[col], prefix=col)
                df_result = pd.concat([df_result.drop(columns=[col]), encoded], axis=1)
                encoding_report[col] = {"strategy": "one_hot", "unique_values": unique_count}
                
            elif strategy == "label" and SKLEARN_AVAILABLE:
                le = LabelEncoder()
                df_result[col] = le.fit_transform(df_result[col].astype(str))
                encoding_report[col] = {"strategy": "label_encoding", "unique_values": unique_count}
                
            elif strategy == "frequency":
                # Frequency encoding
                freq_encoding = df_result[col].value_counts().to_dict()
                df_result[col] = df_result[col].map(freq_encoding)
                encoding_report[col] = {"strategy": "frequency_encoding", "unique_values": unique_count}
                
            elif strategy == "target" and SKLEARN_AVAILABLE:
                # This would typically require a target variable
                # For now, we'll skip and log a warning
                logger.warning("Target encoding requires a target variable, skipping column '{col}'")
                continue
                
        except Exception as e:
            encoding_report[col] = {"strategy": strategy, "error": str(e)}
            logger.error(f"Error encoding column '{col}': {e}")
    
    # Store encoding report in DataFrame attributes
    if hasattr(df_result, 'attrs'):
        if 'encoding_report' in df_result.attrs:
            df_result.attrs['encoding_report'].update(encoding_report)
        else:
            df_result.attrs['encoding_report'] = encoding_report
    
    logger.info(f"Applied {strategy} encoding to {len(categorical_cols)} categorical columns")
    return df_result

# ============================
# STREAMLIT INTEGRATION
# ============================

def render_data_cleaning_dashboard(pipeline: DataCleaningPipeline):
    """Render professional interactive data cleaning dashboard in Streamlit."""
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit not available - cannot render dashboard")
        return None
        
    # Set page configuration
    st.set_page_config(
        page_title="Advanced Data Cleaning",
        page_icon="🧹",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .operation-card {
        background-color: #ffffff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .success-operation {
        border-left-color: #2ecc71;
    }
    .error-operation {
        border-left-color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧹 Advanced Data Cleaning Dashboard</h1>', unsafe_allow_html=True)
    
    # Pipeline summary
    summary = pipeline.get_summary()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "❓ Missing Values", 
        "🎯 Outliers", 
        "🔧 Data Types", 
        "🔄 Duplicates",
        "📈 History"
    ])
    
    with tab1:
        # Overview metrics
        st.markdown("### 📊 Pipeline Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Operations", summary["total_operations"])
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            success_rate = (summary['successful_operations']/max(summary['total_operations'], 1)*100)
            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Memory Saved", f"{summary['memory_reduction_mb']:.1f} MB")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Quality", f"{summary['data_quality_score']}/100")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data shape information
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📐 Data Shape")
            st.write(f"**Original:** {summary['original_shape'][0]} rows × {summary['original_shape'][1]} columns")
            st.write(f"**Current:** {summary['current_shape'][0]} rows × {summary['current_shape'][1]} columns")
            st.write(f"**Change:** {summary['current_shape'][0] - summary['original_shape'][0]} rows, "
                    f"{summary['current_shape'][1] - summary['original_shape'][1]} columns")
        
        with col2:
            st.markdown("### ⚙️ Pipeline Information")
            st.write(f"**Name:** {summary['name']}")
            st.write(f"**Created:** {summary['created_at']}")
            st.write(f"**Last Modified:** {summary['last_modified']}")
            st.write(f"**Total Execution Time:** {summary['total_execution_time']:.2f}s")
        
        # Control buttons
        st.markdown("### 🔧 Pipeline Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("↶ Undo", disabled=not summary["can_undo"], use_container_width=True):
                if pipeline.undo():
                    st.success("Operation undone!")
                    st.rerun()
        
        with col2:
            if st.button("↷ Redo", disabled=not summary["can_redo"], use_container_width=True):
                if pipeline.redo():
                    st.success("Operation redone!")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Reset All", use_container_width=True):
                pipeline.reset()
                st.success("Pipeline reset to original state!")
                st.rerun()
        
        with col4:
            if st.button("💾 Export Log", use_container_width=True):
                # Generate a filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cleaning_log_{timestamp}.json"
                if pipeline.export_operations_log(filename):
                    st.success(f"Log exported to {filename}")
                else:
                    st.error("Failed to export log")
    
    with tab2:
        render_missing_values_tab(pipeline)
        
    with tab3:
        render_outliers_tab(pipeline)
        
    with tab4:
        render_datatypes_tab(pipeline)
        
    with tab5:
        render_duplicates_tab(pipeline)
        
    with tab6:
        render_history_tab(pipeline)

def render_missing_values_tab(pipeline: DataCleaningPipeline):
    """Render missing values cleaning interface."""
    st.markdown("### ❓ Missing Values Treatment")
    
    df = pipeline.current_df
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0].index.tolist()
    
    if not missing_cols:
        st.success("🎉 No missing values detected!")
        return
    
    st.markdown("**Columns with Missing Values:**")
    missing_data = []
    for col in missing_cols:
        missing_count = missing_info[col]
        missing_pct = (missing_count / len(df)) * 100
        missing_data.append({
            "Column": col,
            "Missing Count": missing_count,
            "Missing %": f"{missing_pct:.1f}%",
            "Data Type": str(df[col].dtype)
        })
    
    missing_df = pd.DataFrame(missing_data)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    # Treatment options
    st.markdown("**Treatment Options:**")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_columns = st.multiselect(
            "Select columns to treat:",
            missing_cols,
            default=missing_cols[:min(3, len(missing_cols))] if missing_cols else []
        )
    
    with col2:
        strategies = ["smart", "mean", "median", "mode", "drop_high_missing", "interpolate", "forward_fill", "backward_fill"]
        if SKLEARN_AVAILABLE:
            strategies.extend(["knn", "iterative"])
        
        strategy = st.selectbox("Treatment strategy:", strategies)
    
    # Strategy-specific parameters
    if strategy == "drop_high_missing":
        threshold = st.slider(
            "Missing threshold for dropping:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Columns with missing percentage above this threshold will be dropped"
        )
    else:
        threshold = 0.5
    
    if st.button("🚀 Apply Missing Values Treatment", type="primary"):
        if selected_columns:
            operation = CleaningOperation(
                operation_type="missing_values_treatment",
                parameters={"columns": selected_columns, "strategy": strategy, "threshold": threshold},
                description=f"Applied {strategy} strategy to {len(selected_columns)} columns"
            )
            
            success = pipeline.execute_operation(
                operation, 
                advanced_missing_value_handler,
                columns=selected_columns,
                strategy=strategy,
                threshold=threshold
            )
            
            if success:
                st.success(f"✅ Missing values treatment applied successfully!")
                st.rerun()
            else:
                st.error(f"❌ Treatment failed: {operation.error_message}")
        else:
            st.warning("Please select at least one column to treat")

def render_outliers_tab(pipeline: DataCleaningPipeline):
    """Render outliers treatment interface."""
    st.markdown("### 🎯 Outlier Detection & Treatment")
    
    df = pipeline.current_df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns found for outlier analysis.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_columns = st.multiselect(
            "Select columns:",
            numeric_cols,
            default=numeric_cols[:min(2, len(numeric_cols))] if numeric_cols else []
        )
    
    with col2:
        methods = ["iqr_cap", "iqr_remove", "zscore_cap", "percentile_cap"]
        if SCIPY_AVAILABLE:
            methods.extend(["zscore_remove", "winsorize"])
        if SKLEARN_AVAILABLE:
            methods.append("isolation_forest")
        
        method = st.selectbox("Treatment method:", methods)
    
    with col3:
        sensitivity = st.slider(
            "Sensitivity:",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Lower values detect more outliers"
        )
    
    # Show column statistics
    if selected_columns:
        st.markdown("**Column Statistics:**")
        stats_data = []
        for col in selected_columns:
            stats_data.append({
                "Column": col,
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Mean": df[col].mean(),
                "Std": df[col].std(),
                "Missing": df[col].isnull().sum()
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    if st.button("🚀 Apply Outlier Treatment", type="primary"):
        if selected_columns:
            operation = CleaningOperation(
                operation_type="outlier_treatment",
                parameters={"columns": selected_columns, "method": method, "sensitivity": sensitivity},
                description=f"Applied {method} outlier treatment to {len(selected_columns)} columns"
            )
            
            success = pipeline.execute_operation(
                operation,
                lambda df, **kwargs: advanced_outlier_treatment(df, **kwargs)[0],
                columns=selected_columns,
                method=method,
                sensitivity=sensitivity
            )
            
            if success:
                st.success(f"✅ Outlier treatment applied successfully!")
                st.rerun()
            else:
                st.error(f"❌ Treatment failed: {operation.error_message}")
        else:
            st.warning("Please select at least one column to treat")

def render_datatypes_tab(pipeline: DataCleaningPipeline):
    """Render data types optimization interface."""
    st.markdown("### 🔧 Data Type Optimization")
    
    df = pipeline.current_df
    
    # Memory usage analysis
    memory_mb = pipeline._get_dataframe_memory(df)
    st.metric("Current Memory Usage", f"{memory_mb:.2f} MB")
    
    # Data type summary
    dtype_summary = df.dtypes.value_counts()
    st.markdown("**Current Data Types:**")
    for dtype, count in dtype_summary.items():
        st.write(f"• **{dtype}**: {count} columns")
    
    # Datetime detection
    st.markdown("### 📅 Datetime Detection")
    if st.button("Detect Datetime Columns"):
        datetime_results = smart_datetime_detection(df)
        datetime_cols = [col for col, res in datetime_results.items() 
                        if res.get('conversion_recommended', False)]
        
        if datetime_cols:
            st.success(f"Found {len(datetime_cols)} columns that can be converted to datetime")
            for col in datetime_cols:
                st.write(f"• **{col}**: {datetime_results[col]['datetime_score']:.1f}% confidence")
            
            if st.button("Convert to Datetime"):
                operation = CleaningOperation(
                    operation_type="datetime_conversion",
                    parameters={"columns": datetime_cols},
                    description=f"Converted {len(datetime_cols)} columns to datetime"
                )
                
                success = pipeline.execute_operation(
                    operation,
                    convert_to_datetime,
                    columns=datetime_cols
                )
                
                if success:
                    st.success("✅ Datetime conversion applied successfully!")
                    st.rerun()
                else:
                    st.error("❌ Conversion failed")
        else:
            st.info("No datetime columns detected")
    
    # Memory optimization
    st.markdown("### 💾 Memory Optimization")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Optimize Data Types", use_container_width=True):
            operation = CleaningOperation(
                operation_type="dtype_optimization",
                parameters={"aggressive": False},
                description="Applied automatic data type optimization"
            )
            
            success = pipeline.execute_operation(
                operation,
                lambda df, **kwargs: optimize_dtypes_advanced(df, **kwargs)[0],
                aggressive=False
            )
            
            if success:
                st.success("✅ Data types optimized successfully!")
                st.rerun()
            else:
                st.error(f"❌ Optimization failed")
    
    with col2:
        if st.button("⚡ Aggressive Optimization", use_container_width=True):
            operation = CleaningOperation(
                operation_type="dtype_optimization_aggressive",
                parameters={"aggressive": True},
                description="Applied aggressive data type optimization"
            )
            
            success = pipeline.execute_operation(
                operation,
                lambda df, **kwargs: optimize_dtypes_advanced(df, **kwargs)[0],
                aggressive=True
            )
            
            if success:
                st.success("✅ Aggressive optimization applied successfully!")
                st.rerun()
            else:
                st.error(f"❌ Optimization failed")

def render_duplicates_tab(pipeline: DataCleaningPipeline):
    """Render duplicates removal interface."""
    st.markdown("### 🔄 Duplicate Records Management")
    
    df = pipeline.current_df
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count == 0:
        st.success("🎉 No duplicate records found!")
        return
    
    st.warning(f"⚠️ Found {duplicate_count:,} duplicate records ({(duplicate_count/len(df)*100):.1f}%)")
    
    # Show sample of duplicates
    if duplicate_count > 0:
        st.markdown("**Sample Duplicate Records:**")
        duplicates_sample = df[df.duplicated(keep=False)].head(10)
        st.dataframe(duplicates_sample)
    
    col1, col2 = st.columns(2)
    
    with col1:
        keep_option = st.selectbox(
            "Keep which record:",
            ["first", "last", "none"]
        )
    
    with col2:
        subset_cols = st.multiselect(
            "Check duplicates based on columns (optional):",
            df.columns.tolist()
        )
    
    if st.button("🗑️ Remove Duplicates", type="primary"):
        operation = CleaningOperation(
            operation_type="remove_duplicates",
            parameters={"keep": keep_option, "subset": subset_cols if subset_cols else None},
            description=f"Removed duplicates keeping '{keep_option}' record"
        )
        
        def remove_duplicates_func(df, **kwargs):
            if kwargs.get("subset"):
                return df.drop_duplicates(subset=kwargs["subset"], keep=kwargs["keep"])
            else:
                return df.drop_duplicates(keep=kwargs["keep"])
        
        success = pipeline.execute_operation(
            operation,
            remove_duplicates_func,
            keep=keep_option,
            subset=subset_cols if subset_cols else None
        )
        
        if success:
            st.success("✅ Duplicate records removed successfully!")
            st.rerun()
        else:
            st.error(f"❌ Removal failed: {operation.error_message}")

def render_history_tab(pipeline: DataCleaningPipeline):
    """Render operations history."""
    st.markdown("### 📈 Operations History")
    
    if not pipeline.operations_history:
        st.info("No operations performed yet.")
        return
    
    # Operations timeline
    history_data = []
    for i, op in enumerate(pipeline.operations_history):
        status_class = "success-operation" if op.success else "error-operation"
        history_data.append({
            "Step": i + 1,
            "Operation": op.operation_type.replace("_", " ").title(),
            "Description": op.description,
            "Status": "✅ Success" if op.success else "❌ Failed",
            "Execution Time": f"{op.execution_time:.3f}s",
            "Memory Usage": f"{op.memory_usage:.1f} MB" if op.memory_usage > 0 else "N/A",
            "Timestamp": op.timestamp.split("T")[1][:8] if "T" in op.timestamp else op.timestamp
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Summary statistics
    summary = pipeline.get_summary()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Operations", summary["total_operations"])
    with col2:
        success_rate = (summary['successful_operations']/max(summary['total_operations'], 1)*100)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col3:
        st.metric("Avg Execution Time", f"{summary['average_execution_time']:.3f}s")
    
    # Export button
    if st.button("💾 Export Operations Log"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaning_operations_{timestamp}.json"
        if pipeline.export_operations_log(filename):
            st.success(f"Operations log exported to {filename}")
        else:
            st.error("Failed to export operations log")

# ============================
# EXPORTS
# ============================

__all__ = [
    'CleaningOperation',
    'DataCleaningPipeline', 
    'smart_imputation_comparison',
    'advanced_missing_value_handler',
    'optimize_dtypes_advanced',
    'smart_datetime_detection',
    'convert_to_datetime',
    'advanced_outlier_treatment',
    'advanced_encoding',
    'render_data_cleaning_dashboard'
]

# Print module load status
logger.info("Enhanced Preprocessing Module v3.0 - Loaded Successfully!")
print("✅ Enhanced Preprocessing Module v3.0 - Loaded Successfully!")
print(f"   📊 SciPy Available: {SCIPY_AVAILABLE}")
print(f"   🤖 Scikit-learn Available: {SKLEARN_AVAILABLE}")
print(f"   🔄 Iterative Imputer Available: {ITERATIVE_IMPUTER_AVAILABLE}")
print(f"   🎨 Streamlit Available: {STREAMLIT_AVAILABLE}")
print(f"   📈 Memory Profiler Available: {MEMORY_PROFILER_AVAILABLE}")
print("   🚀 All functions ready for import!")