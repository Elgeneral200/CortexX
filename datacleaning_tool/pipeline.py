# filename: pipeline.py
"""
Workflow Pipeline Module - Enhanced Professional Edition v3.0

Advanced workflow management and pipeline orchestration with:
- Enhanced pipeline execution with smart operations
- Complete integration with all Phase 1 enhanced modules
- Professional UI/UX with Streamlit
- Advanced undo/redo functionality and state management
- Performance monitoring and optimization
- Robust serialization and persistence capabilities
- Comprehensive error handling and recovery mechanisms
- Business rule validation integration
- Professional reporting and analytics

Author: CortexX Team  
Version: 3.0.0 - Professional Enterprise Edition with Complete Phase 1 Integration
"""

import json
import pickle
import time
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import OrderedDict, deque
import hashlib
import inspect
import functools

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available - some features may be limited")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("streamlit not available - UI features disabled")

try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logger.warning("memory_profiler not available - memory tracking limited")

# ============================
# ENHANCED DATA STRUCTURES
# ============================

@dataclass
class PipelineStep:
    """Enhanced pipeline step with comprehensive metadata and validation."""
    name: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    # Execution metadata
    execution_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    step_id: str = field(default="")
    
    # Data impact metadata
    input_shape: Tuple[int, int] = (0, 0)
    output_shape: Tuple[int, int] = (0, 0)
    memory_impact: float = 0.0
    quality_impact: float = 0.0
    memory_usage: float = 0.0
    
    # Validation metadata
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate unique step ID after initialization."""
        if not self.step_id:
            self.step_id = hashlib.md5(f"{self.name}_{time.time_ns()}".encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return asdict(self)
    
    def validate_parameters(self) -> bool:
        """Validate step parameters against function signature."""
        try:
            sig = inspect.signature(self.function)
            # Check if all required parameters are provided
            for param_name, param in sig.parameters.items():
                if (param.default == inspect.Parameter.empty and 
                    param_name not in self.parameters and 
                    param_name != 'df'):
                    self.validation_errors.append(f"Missing required parameter: {param_name}")
                    return False
            return True
        except Exception as e:
            self.validation_errors.append(f"Validation error: {str(e)}")
            return False

@dataclass
class PipelineExecutionResult:
    """Comprehensive result of pipeline execution with enhanced metrics."""
    success: bool
    dataframe: Optional[pd.DataFrame]
    pipeline_name: str
    execution_time: float
    steps_executed: int
    steps_succeeded: int
    steps_failed: int
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    final_data_quality: float = 0.0
    memory_usage_mb: float = 0.0
    data_transformation_ratio: float = 1.0
    
    def get_success_rate(self) -> float:
        """Calculate step success rate."""
        if self.steps_executed == 0:
            return 0.0
        return (self.steps_succeeded / self.steps_executed) * 100
    
    def to_summary(self) -> Dict[str, Any]:
        """Generate execution summary."""
        return {
            "pipeline_name": self.pipeline_name,
            "success": self.success,
            "execution_time": round(self.execution_time, 4),
            "steps_executed": self.steps_executed,
            "steps_succeeded": self.steps_succeeded,
            "success_rate": round(self.get_success_rate(), 2),
            "final_shape": self.dataframe.shape if self.dataframe is not None else (0, 0),
            "data_quality": round(self.final_data_quality, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "transformation_ratio": round(self.data_transformation_ratio, 4),
            "has_errors": len(self.error_messages) > 0,
            "has_warnings": len(self.warnings) > 0
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to DataFrame for analysis."""
        if self.dataframe is None:
            return pd.DataFrame()
        
        # Add execution metadata as columns
        df = self.dataframe.copy()
        df.attrs['pipeline_execution'] = self.to_summary()
        return df

# ============================
# ENHANCED REGISTRY MANAGER
# ============================

class EnhancedRegistryManager:
    """Advanced operation registry with Phase 1 integration and validation."""
    
    def __init__(self):
        self.registry = {}
        self.operation_metadata = {}
        self._initialize_enhanced_operations()
    
    def _initialize_enhanced_operations(self):
        """Initialize enhanced operations from Phase 1 modules with fallbacks."""
        enhanced_ops = {}
        
        # Enhanced preprocessing operations
        preprocessing_ops = {
            'smart_imputation': ('preprocess', 'advanced_missing_value_handler'),
            'optimize_memory': ('preprocess', 'optimize_dtypes_advanced'),
            'handle_outliers': ('preprocess', 'advanced_outlier_treatment'),
            'advanced_encoding': ('preprocess', 'advanced_encoding'),
            'feature_engineering': ('preprocess', 'advanced_feature_engineering')
        }
        
        # Visualization operations
        visualization_ops = {
            'generate_eda': ('visualization', 'generate_automated_eda_report'),
            'create_dashboard': ('visualization', 'create_interactive_dashboard'),
            'export_visualization': ('visualization', 'export_visualization_report')
        }
        
        # Business intelligence operations
        business_ops = {
            'calculate_kpis': ('business_intelligence', 'calculate_business_kpis'),
            'trend_analysis': ('business_intelligence', 'perform_trend_analysis'),
            'anomaly_detection': ('business_intelligence', 'detect_business_anomalies')
        }
        
        # Quality operations
        quality_ops = {
            'assess_quality': ('quality', 'run_rules'),
            'validate_data': ('quality', 'validate_dataset'),
            'data_profiling': ('quality', 'generate_data_profile')
        }
        
        # File handling operations
        file_ops = {
            'load_file': ('file_handler', 'process_file_advanced'),
            'export_data': ('file_handler', 'export_data_advanced'),
            'multi_file_processing': ('file_handler', 'process_multiple_files')
        }
        
        # Machine learning operations (if available)
        ml_ops = {
            'train_model': ('ml_pipeline', 'train_ml_model'),
            'predict': ('ml_pipeline', 'make_predictions'),
            'evaluate_model': ('ml_pipeline', 'evaluate_model_performance')
        }
        
        # Combine all operation groups
        all_ops = {
            **preprocessing_ops, 
            **visualization_ops,
            **business_ops,
            **quality_ops,
            **file_ops,
            **ml_ops
        }
        
        # Try to import each operation
        for op_name, (module, func_name) in all_ops.items():
            try:
                module_obj = __import__(module, fromlist=[func_name])
                func = getattr(module_obj, func_name)
                enhanced_ops[op_name] = func
                self.operation_metadata[op_name] = {
                    'module': module,
                    'function': func_name,
                    'description': func.__doc__ or f"{op_name} operation",
                    'signature': inspect.signature(func)
                }
                logger.info(f"✅ Loaded enhanced operation: {op_name}")
            except ImportError as e:
                logger.warning(f"⚠️ Could not load {op_name} from {module}: {e}")
            except AttributeError as e:
                logger.warning(f"⚠️ Function {func_name} not found in {module}: {e}")
        
        # Add basic pandas operations as fallbacks
        basic_ops = self._create_basic_operations()
        enhanced_ops.update(basic_ops)
        
        self.registry.update(enhanced_ops)
        logger.info(f"Registry initialized with {len(self.registry)} operations")
    
    def _create_basic_operations(self) -> Dict[str, Callable]:
        """Create basic pandas operations with enhanced metadata."""
        basic_ops = {}
        
        operations = {
            'drop_duplicates': lambda df, **kwargs: df.drop_duplicates(**kwargs),
            'fillna_mean': lambda df, columns=None, **kwargs: df.fillna(df[columns].mean() if columns else df.mean(), **kwargs),
            'fillna_median': lambda df, columns=None, **kwargs: df.fillna(df[columns].median() if columns else df.median(), **kwargs),
            'fillna_mode': lambda df, columns=None, **kwargs: df.fillna(df[columns].mode().iloc[0] if columns else df.mode().iloc[0], **kwargs),
            'fillna_value': lambda df, value=0, **kwargs: df.fillna(value, **kwargs),
            'drop_columns': lambda df, columns=None, **kwargs: df.drop(columns=columns, **kwargs) if columns else df,
            'select_columns': lambda df, columns=None, **kwargs: df[columns] if columns else df,
            'filter_rows': lambda df, condition=None, **kwargs: df.query(condition, **kwargs) if condition else df,
            'sort_values': lambda df, by=None, **kwargs: df.sort_values(by=by, **kwargs) if by else df,
            'reset_index': lambda df, **kwargs: df.reset_index(**kwargs),
            'sample_data': lambda df, n=1000, **kwargs: df.sample(n=min(n, len(df)), **kwargs) if len(df) > n else df,
            'rename_columns': lambda df, columns=None, **kwargs: df.rename(columns=columns, **kwargs) if columns else df,
            'change_dtype': lambda df, dtype_map=None, **kwargs: df.astype(dtype_map, **kwargs) if dtype_map else df
        }
        
        for name, func in operations.items():
            basic_ops[name] = func
            self.operation_metadata[name] = {
                'module': 'pandas',
                'function': name,
                'description': f"Pandas {name} operation",
                'signature': inspect.signature(func)
            }
        
        return basic_ops
    
    def get_registry(self) -> Dict[str, Callable]:
        """Get the operation registry."""
        return self.registry.copy()
    
    def get_operation_metadata(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific operation."""
        return self.operation_metadata.get(operation_name, None)
    
    def list_operations(self, category: Optional[str] = None) -> List[str]:
        """List all available operations, optionally filtered by category."""
        if category:
            # Categorize operations based on their functionality
            categories = {
                'preprocessing': ['smart_imputation', 'optimize_memory', 'handle_outliers', 
                                 'advanced_encoding', 'feature_engineering', 'drop_duplicates',
                                 'fillna_mean', 'fillna_median', 'fillna_mode', 'fillna_value',
                                 'drop_columns', 'select_columns', 'change_dtype'],
                'visualization': ['generate_eda', 'create_dashboard', 'export_visualization'],
                'business': ['calculate_kpis', 'trend_analysis', 'anomaly_detection'],
                'quality': ['assess_quality', 'validate_data', 'data_profiling'],
                'file': ['load_file', 'export_data', 'multi_file_processing'],
                'ml': ['train_model', 'predict', 'evaluate_model'],
                'basic': ['filter_rows', 'sort_values', 'reset_index', 'sample_data', 'rename_columns']
            }
            
            if category in categories:
                return [op for op in self.registry.keys() if op in categories[category]]
            else:
                return list(self.registry.keys())
        
        return list(self.registry.keys())
    
    def add_operation(self, name: str, function: Callable, metadata: Optional[Dict[str, Any]] = None):
        """Add operation to registry with metadata."""
        self.registry[name] = function
        self.operation_metadata[name] = metadata or {
            'module': 'custom',
            'function': name,
            'description': function.__doc__ or f"Custom {name} operation",
            'signature': inspect.signature(function)
        }
        logger.info(f"Added custom operation: {name}")
    
    def validate_operation(self, operation_name: str, parameters: Dict[str, Any]) -> List[str]:
        """Validate operation parameters against function signature."""
        errors = []
        
        if operation_name not in self.registry:
            return [f"Operation '{operation_name}' not found in registry"]
        
        try:
            sig = inspect.signature(self.registry[operation_name])
            
            # Check for required parameters
            for param_name, param in sig.parameters.items():
                if (param.default == inspect.Parameter.empty and 
                    param_name not in parameters and 
                    param_name != 'df'):
                    errors.append(f"Missing required parameter: {param_name}")
            
            # Check for unknown parameters
            for param_name in parameters.keys():
                if param_name not in sig.parameters and param_name != 'df':
                    errors.append(f"Unknown parameter: {param_name}")
                    
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors

# ============================
# ENHANCED PIPELINE CLASS
# ============================

class Pipeline:
    """Professional Pipeline class with complete Phase 1 integration."""
    
    def __init__(self, registry: Optional[Dict[str, Callable]] = None, name: str = "Enhanced Pipeline"):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.registry = registry or global_registry_manager.get_registry()
        
        # Enhanced state management
        self.execution_history: deque[PipelineExecutionResult] = deque(maxlen=20)
        self.state_history: deque[pd.DataFrame] = deque(maxlen=10)  # Limit memory usage
        self.undo_stack: deque[pd.DataFrame] = deque(maxlen=10)
        self.redo_stack: deque[pd.DataFrame] = deque(maxlen=10)
        
        # Performance tracking
        self.total_executions = 0
        self.total_execution_time = 0.0
        self.created_at = datetime.now().isoformat()
        self.last_modified = self.created_at
        self.last_execution: Optional[PipelineExecutionResult] = None
        
        # Configuration
        self.config = {
            'auto_save': True,
            'max_history_size': 10,
            'enable_profiling': True,
            'stop_on_error': True
        }
    
    def add_step(self, operation_name: str, parameters: Optional[Dict[str, Any]] = None, 
                 description: str = "") -> 'Pipeline':
        """Add a step to the pipeline with enhanced validation."""
        if operation_name not in self.registry:
            available_ops = list(self.registry.keys())
            raise ValueError(f"Operation '{operation_name}' not found in registry. Available: {available_ops[:10]}{'...' if len(available_ops) > 10 else ''}")
        
        # Validate parameters
        validation_errors = global_registry_manager.validate_operation(operation_name, parameters or {})
        if validation_errors:
            error_msg = f"Parameter validation failed for '{operation_name}': {', '.join(validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        step = PipelineStep(
            name=operation_name,
            function=self.registry[operation_name],
            parameters=parameters or {},
            description=description or f"Execute {operation_name} operation"
        )
        
        # Validate step
        if not step.validate_parameters():
            error_msg = f"Step validation failed: {', '.join(step.validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.steps.append(step)
        self.last_modified = datetime.now().isoformat()
        logger.info(f"Added step: {operation_name}")
        return self
    
    def remove_step(self, index: int) -> 'Pipeline':
        """Remove a step from the pipeline."""
        if 0 <= index < len(self.steps):
            removed_step = self.steps.pop(index)
            self.last_modified = datetime.now().isoformat()
            logger.info(f"Removed step: {removed_step.name}")
        else:
            raise IndexError(f"Step index {index} out of range")
        return self
    
    def insert_step(self, index: int, operation_name: str, 
                   parameters: Optional[Dict[str, Any]] = None, description: str = "") -> 'Pipeline':
        """Insert a step at a specific position with validation."""
        if operation_name not in self.registry:
            raise ValueError(f"Operation '{operation_name}' not found in registry")
        
        # Validate parameters
        validation_errors = global_registry_manager.validate_operation(operation_name, parameters or {})
        if validation_errors:
            raise ValueError(f"Parameter validation failed: {', '.join(validation_errors)}")
        
        step = PipelineStep(
            name=operation_name,
            function=self.registry[operation_name],
            parameters=parameters or {},
            description=description
        )
        
        # Validate step
        if not step.validate_parameters():
            raise ValueError(f"Step validation failed: {', '.join(step.validation_errors)}")
        
        self.steps.insert(index, step)
        self.last_modified = datetime.now().isoformat()
        logger.info(f"Inserted step: {operation_name} at position {index}")
        return self
    
    def execute(self, dataframe: pd.DataFrame, 
                save_state: bool = True,
                stop_on_error: bool = True,
                profile_memory: bool = True) -> PipelineExecutionResult:
        """Execute the pipeline with enhanced monitoring, profiling, and error handling."""
        start_time = time.time()
        initial_memory = self._get_dataframe_memory(dataframe)
        
        # Initialize execution result
        result = PipelineExecutionResult(
            success=True,
            dataframe=dataframe.copy() if dataframe is not None else None,
            pipeline_name=self.name,
            execution_time=0.0,
            steps_executed=0,
            steps_succeeded=0,
            steps_failed=0,
            memory_usage_mb=initial_memory
        )
        
        if not self.steps:
            result.execution_time = time.time() - start_time
            result.warnings.append("No steps to execute in pipeline")
            logger.warning("Pipeline execution skipped: no steps configured")
            return result
        
        if dataframe is None or dataframe.empty:
            result.success = False
            result.error_messages.append("Input dataframe is None or empty")
            result.execution_time = time.time() - start_time
            logger.error("Pipeline execution failed: input dataframe is empty")
            return result
        
        # Save initial state if requested
        if save_state:
            self._save_state(dataframe)
            self.undo_stack.append(dataframe.copy())
        
        current_df = dataframe.copy()
        step_results = []
        
        # Execute each step
        for step_index, step in enumerate(self.steps):
            step_start_time = time.time()
            step.timestamp = datetime.now().isoformat()
            step.input_shape = current_df.shape
            input_memory = self._get_dataframe_memory(current_df)
            
            try:
                # Profile memory usage if enabled
                if profile_memory and MEMORY_PROFILER_AVAILABLE:
                    mem_usage = memory_usage((step.function, (current_df,), step.parameters), 
                                            interval=0.1, timeout=60, max_usage=True)
                    step.memory_usage = mem_usage if isinstance(mem_usage, float) else max(mem_usage) if mem_usage else 0
                
                # Execute the step
                if step.parameters:
                    step_result = step.function(current_df, **step.parameters)
                else:
                    step_result = step.function(current_df)
                
                # Handle different return types
                if isinstance(step_result, pd.DataFrame):
                    current_df = step_result
                elif isinstance(step_result, tuple) and len(step_result) > 0:
                    # Handle functions that return multiple values
                    if isinstance(step_result[0], pd.DataFrame):
                        current_df = step_result[0]
                    # Store additional results if they exist
                    if len(step_result) > 1:
                        step_results.append(step_result[1:])
                elif step_result is None:
                    # Functions that modify dataframe in place
                    pass
                else:
                    result.warnings.append(f"Step '{step.name}' returned unexpected type: {type(step_result)}")
                    logger.warning(f"Step '{step.name}' returned unexpected type: {type(step_result)}")
                
                # Update step metadata
                step.success = True
                step.output_shape = current_df.shape
                output_memory = self._get_dataframe_memory(current_df)
                step.memory_impact = output_memory - input_memory
                
                result.steps_succeeded += 1
                logger.info(f"Step {step_index+1}/{len(self.steps)} completed: {step.name}")
                
            except Exception as e:
                step.success = False
                step.error_message = str(e)
                step.output_shape = step.input_shape  # No change due to error
                
                result.steps_failed += 1
                error_msg = f"Step '{step.name}': {str(e)}"
                result.error_messages.append(error_msg)
                logger.error(f"Step {step_index+1}/{len(self.steps)} failed: {error_msg}")
                
                if stop_on_error:
                    result.success = False
                    logger.error("Stopping pipeline execution due to error")
                    break
            
            finally:
                step.execution_time = time.time() - step_start_time
                result.steps_executed += 1
        
        # Finalize execution result
        result.dataframe = current_df
        result.execution_time = time.time() - start_time
        result.success = result.success and (result.steps_failed == 0)
        
        # Calculate final metrics
        final_memory = self._get_dataframe_memory(current_df)
        result.memory_usage_mb = final_memory
        result.data_transformation_ratio = self._calculate_transformation_ratio(dataframe, current_df)
        result.final_data_quality = self._calculate_data_quality(current_df)
        
        # Add performance metrics
        result.performance_metrics = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_change_mb': final_memory - initial_memory,
            'transformation_ratio': result.data_transformation_ratio,
            'steps_per_second': len(self.steps) / result.execution_time if result.execution_time > 0 else 0,
            'data_quality_score': result.final_data_quality
        }
        
        # Update pipeline statistics
        self.total_executions += 1
        self.total_execution_time += result.execution_time
        self.execution_history.append(result)
        self.last_execution = result
        self.last_modified = datetime.now().isoformat()
        
        # Clear redo stack since we've made new changes
        self.redo_stack.clear()
        
        logger.info(f"Pipeline execution completed: {result.success}, "
                   f"time: {result.execution_time:.2f}s, "
                   f"steps: {result.steps_succeeded}/{result.steps_executed}")
        
        return result
    
    def _get_dataframe_memory(self, df: pd.DataFrame) -> float:
        """Calculate DataFrame memory usage in MB."""
        if df is None or not isinstance(df, pd.DataFrame):
            return 0.0
        try:
            return df.memory_usage(deep=True).sum() / 1024 / 1024
        except:
            return 0.0
    
    def _calculate_transformation_ratio(self, initial_df: pd.DataFrame, final_df: pd.DataFrame) -> float:
        """Calculate how much the data has been transformed."""
        if initial_df is None or final_df is None or initial_df.empty or final_df.empty:
            return 0.0
        
        try:
            # Simple transformation metric based on shape change and memory usage
            shape_change = abs(final_df.shape[0] - initial_df.shape[0]) / initial_df.shape[0]
            memory_change = abs(self._get_dataframe_memory(final_df) - self._get_dataframe_memory(initial_df)) / self._get_dataframe_memory(initial_df)
            
            # Combine metrics (weighted average)
            return min(1.0, (shape_change * 0.7 + memory_change * 0.3))
        except:
            return 0.0
    
    def _save_state(self, dataframe: pd.DataFrame):
        """Save dataframe state for undo functionality."""
        self.state_history.append(dataframe.copy())
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate comprehensive data quality score."""
        if df is None or df.empty:
            return 0.0
        
        try:
            # Calculate multiple quality metrics
            total_cells = len(df) * len(df.columns) if len(df.columns) > 0 else 0
            
            # Completeness (non-null ratio)
            completeness = 1.0 - (df.isnull().sum().sum() / total_cells) if total_cells > 0 else 1.0
            
            # Uniqueness (non-duplicate ratio)
            uniqueness = 1.0 - (df.duplicated().sum() / len(df)) if len(df) > 0 else 1.0
            
            # Consistency (data type consistency)
            type_consistency = self._calculate_type_consistency(df)
            
            # Validity (check basic validity of numeric columns)
            validity = self._calculate_validity_score(df)
            
            # Calculate weighted quality score (0-10)
            weights = [0.3, 0.2, 0.25, 0.25]  # completeness, uniqueness, consistency, validity
            scores = [completeness * 10, uniqueness * 10, type_consistency * 10, validity * 10]
            
            quality_score = sum(weight * score for weight, score in zip(weights, scores))
            return round(quality_score, 1)
        except Exception as e:
            logger.warning(f"Error calculating data quality: {e}")
            return 5.0  # Default medium quality
    
    def _calculate_type_consistency(self, df: pd.DataFrame) -> float:
        """Calculate data type consistency score."""
        try:
            # Check for mixed types in object columns
            object_cols = df.select_dtypes(include=['object']).columns
            consistency_score = 1.0
            
            for col in object_cols:
                # Sample some values to check type consistency
                sample = df[col].dropna().sample(min(100, len(df[col])))
                if len(sample) > 0:
                    types = set(type(val) for val in sample)
                    if len(types) > 1:
                        consistency_score -= 0.1 * (len(types) - 1)
            
            return max(0.0, consistency_score)
        except:
            return 0.8  # Default reasonable consistency
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Calculate basic validity score for numeric columns."""
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
    
    def undo(self) -> Optional[pd.DataFrame]:
        """Undo the last operation and return the previous state."""
        if self.undo_stack:
            current_state = self.undo_stack.pop()
            self.redo_stack.append(current_state)
            return self.undo_stack[-1] if self.undo_stack else None
        return None
    
    def redo(self) -> Optional[pd.DataFrame]:
        """Redo the last undone operation."""
        if self.redo_stack:
            previous_state = self.redo_stack.pop()
            self.undo_stack.append(previous_state)
            return previous_state
        return None
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        success_rate = self._calculate_success_rate()
        
        return {
            "name": self.name,
            "total_steps": len(self.steps),
            "total_executions": self.total_executions,
            "average_execution_time": round(self.total_execution_time / max(self.total_executions, 1), 4),
            "success_rate": round(success_rate, 2),
            "last_execution_status": self.last_execution.success if self.last_execution else "Never executed",
            "available_operations": len(self.registry),
            "enhanced_operations": self._get_enhanced_operations(),
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "can_undo": len(self.undo_stack) > 0,
            "can_redo": len(self.redo_stack) > 0,
            "memory_usage_mb": round(sum(self._get_dataframe_memory(state) for state in self.state_history), 2)
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall pipeline success rate."""
        if not self.execution_history:
            return 0.0
        
        successful_executions = sum(1 for result in self.execution_history if result.success)
        return (successful_executions / len(self.execution_history)) * 100
    
    def _get_enhanced_operations(self) -> List[str]:
        """Get list of enhanced operations from Phase 1 modules."""
        enhanced_ops = [
            'smart_imputation', 'optimize_memory', 'handle_outliers', 'advanced_encoding',
            'generate_eda', 'create_dashboard', 'calculate_kpis', 'assess_quality', 
            'load_file', 'export_data', 'train_model'
        ]
        return [op for op in enhanced_ops if op in self.registry]
    
    def get_step_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all steps."""
        return [step.to_dict() for step in self.steps]
    
    def clear_steps(self) -> 'Pipeline':
        """Clear all steps from the pipeline."""
        self.steps.clear()
        self.last_modified = datetime.now().isoformat()
        logger.info("Cleared all pipeline steps")
        return self
    
    def clear_history(self) -> 'Pipeline':
        """Clear execution history and state history."""
        self.execution_history.clear()
        self.state_history.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        logger.info("Cleared pipeline history")
        return self
    
    def clone(self, new_name: Optional[str] = None) -> 'Pipeline':
        """Create a copy of the pipeline."""
        new_pipeline = Pipeline(
            registry=self.registry.copy(),
            name=new_name or f"{self.name}_copy"
        )
        
        # Copy steps
        for step in self.steps:
            new_pipeline.add_step(
                operation_name=step.name,
                parameters=step.parameters.copy(),
                description=step.description
            )
        
        # Copy configuration
        new_pipeline.config = self.config.copy()
        
        logger.info(f"Cloned pipeline: {self.name} -> {new_pipeline.name}")
        return new_pipeline
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """Validate pipeline configuration with comprehensive checks."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "step_validations": []
        }
        
        if not self.steps:
            validation_result["warnings"].append("Pipeline has no steps")
        
        # Validate each step
        for i, step in enumerate(self.steps):
            step_validation = {
                "step_index": i,
                "step_name": step.name,
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check if operation exists
            if step.name not in self.registry:
                step_validation["valid"] = False
                step_validation["errors"].append(f"Operation not found in registry")
                validation_result["valid"] = False
            
            # Validate parameters
            param_errors = global_registry_manager.validate_operation(step.name, step.parameters)
            if param_errors:
                step_validation["valid"] = False
                step_validation["errors"].extend(param_errors)
                validation_result["valid"] = False
            
            validation_result["step_validations"].append(step_validation)
        
        # Check for potential issues
        step_names = [step.name for step in self.steps]
        
        # Check for multiple duplicate removal steps
        if step_names.count('drop_duplicates') > 1:
            validation_result["warnings"].append("Multiple duplicate removal steps detected")
        
        # Check for conflicting imputation strategies
        imputation_ops = ['smart_imputation', 'fillna_mean', 'fillna_median', 'fillna_mode', 'fillna_value']
        if sum(1 for op in step_names if op in imputation_ops) > 1:
            validation_result["warnings"].append("Multiple imputation strategies detected")
        
        # Check for inefficient ordering
        memory_ops = ['optimize_memory']
        if any(op in step_names for op in memory_ops) and step_names.index('optimize_memory') != 0:
            validation_result["recommendations"].append("Consider placing memory optimization earlier in the pipeline")
        
        # Recommendations based on common best practices
        if 'drop_duplicates' not in step_names:
            validation_result["recommendations"].append("Consider adding duplicate removal step")
        
        if not any(op in step_names for op in imputation_ops):
            validation_result["recommendations"].append("Consider adding missing value handling")
        
        if not any(op in step_names for op in ['assess_quality', 'validate_data']):
            validation_result["recommendations"].append("Consider adding data quality assessment")
        
        return validation_result
    
    def optimize_order(self) -> 'Pipeline':
        """Optimize the order of pipeline steps for better performance."""
        # Simple optimization: place memory optimization first, quality assessment last
        # This is a placeholder for more sophisticated optimization logic
        memory_ops = ['optimize_memory']
        quality_ops = ['assess_quality', 'validate_data', 'data_profiling']
        
        # Extract memory and quality steps
        memory_steps = [step for step in self.steps if step.name in memory_ops]
        quality_steps = [step for step in self.steps if step.name in quality_ops]
        other_steps = [step for step in self.steps if step.name not in memory_ops + quality_ops]
        
        # Reorder: memory steps first, then other steps, then quality steps
        self.steps = memory_steps + other_steps + quality_steps
        self.last_modified = datetime.now().isoformat()
        
        logger.info("Optimized pipeline step order")
        return self
    
    # ============================
    # SERIALIZATION METHODS
    # ============================
    
    def save_to_json(self, filepath: str) -> bool:
        """Save pipeline configuration to JSON file with enhanced metadata."""
        try:
            pipeline_config = {
                "name": self.name,
                "created_at": self.created_at,
                "last_modified": self.last_modified,
                "config": self.config,
                "steps": [
                    {
                        "name": step.name,
                        "parameters": step.parameters,
                        "description": step.description
                    }
                    for step in self.steps
                ],
                "summary": self.get_pipeline_summary(),
                "version": "3.0.0"
            }
            
            with open(filepath, 'w') as f:
                json.dump(pipeline_config, f, indent=2, default=str)
            
            logger.info(f"Saved pipeline configuration to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving pipeline to JSON: {e}")
            return False
    
    def save_to_pickle(self, filepath: str) -> bool:
        """Save complete pipeline object to pickle file."""
        if not JOBLIB_AVAILABLE:
            logger.warning("joblib not available - using pickle instead")
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self, f)
                logger.info(f"Saved pipeline to pickle file: {filepath}")
                return True
            except Exception as e:
                logger.error(f"Error saving pipeline to pickle: {e}")
                return False
        
        try:
            joblib.dump(self, filepath)
            logger.info(f"Saved pipeline to joblib file: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving pipeline to joblib: {e}")
            return False
    
    @classmethod
    def load_from_json(cls, filepath: str, registry: Optional[Dict[str, Callable]] = None) -> 'Pipeline':
        """Load pipeline configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            pipeline = cls(registry=registry, name=config.get("name", "Loaded Pipeline"))
            
            # Load configuration if available
            if 'config' in config:
                pipeline.config.update(config['config'])
            
            # Load steps
            for step_config in config.get("steps", []):
                try:
                    pipeline.add_step(
                        operation_name=step_config["name"],
                        parameters=step_config.get("parameters", {}),
                        description=step_config.get("description", "")
                    )
                except ValueError as e:
                    logger.warning(f"Skipping step {step_config['name']}: {e}")
                    continue
            
            logger.info(f"Loaded pipeline from JSON: {filepath}")
            return pipeline
        except Exception as e:
            logger.error(f"Error loading pipeline from JSON: {e}")
            raise
    
    @classmethod
    def load_from_pickle(cls, filepath: str) -> 'Pipeline':
        """Load complete pipeline object from pickle file."""
        if not JOBLIB_AVAILABLE:
            try:
                with open(filepath, 'rb') as f:
                    pipeline = pickle.load(f)
                logger.info(f"Loaded pipeline from pickle file: {filepath}")
                return pipeline
            except Exception as e:
                logger.error(f"Error loading pipeline from pickle: {e}")
                raise
        
        try:
            pipeline = joblib.load(filepath)
            logger.info(f"Loaded pipeline from joblib file: {filepath}")
            return pipeline
        except Exception as e:
            logger.error(f"Error loading pipeline from joblib: {e}")
            raise

# ============================
# PIPELINE BUILDER CLASS
# ============================

class PipelineBuilder:
    """Enhanced pipeline builder with smart recommendations and templates."""
    
    def __init__(self, registry: Optional[Dict[str, Callable]] = None):
        self.registry = registry or global_registry_manager.get_registry()
        self.current_pipeline = Pipeline(registry=self.registry)
    
    def for_data_cleaning(self, name: str = "Data Cleaning Pipeline") -> 'PipelineBuilder':
        """Create a pipeline optimized for data cleaning."""
        self.current_pipeline = Pipeline(registry=self.registry, name=name)
        
        # Add common data cleaning steps
        cleaning_steps = [
            ("drop_duplicates", {}, "Remove duplicate records"),
            ("smart_imputation", {"strategy": "auto"}, "Handle missing values intelligently"),
            ("optimize_memory", {"aggressive": True}, "Optimize data types for memory efficiency"),
            ("handle_outliers", {"method": "auto", "sensitivity": 1.5}, "Handle outliers using automated methods"),
            ("advanced_encoding", {"strategy": "auto"}, "Encode categorical variables")
        ]
        
        for op_name, params, desc in cleaning_steps:
            if op_name in self.registry:
                self.current_pipeline.add_step(op_name, params, desc)
        
        return self
    
    def for_analysis(self, name: str = "Data Analysis Pipeline") -> 'PipelineBuilder':
        """Create a pipeline optimized for data analysis."""
        self.current_pipeline = Pipeline(registry=self.registry, name=name)
        
        # Add common analysis steps
        analysis_steps = [
            ("assess_quality", {}, "Assess data quality"),
            ("generate_eda", {}, "Generate automated EDA report"),
            ("calculate_kpis", {}, "Calculate business KPIs"),
            ("trend_analysis", {}, "Perform trend analysis")
        ]
        
        for op_name, params, desc in analysis_steps:
            if op_name in self.registry:
                self.current_pipeline.add_step(op_name, params, desc)
        
        return self
    
    def for_ml_preparation(self, name: str = "ML Preparation Pipeline") -> 'PipelineBuilder':
        """Create a pipeline optimized for machine learning preparation."""
        self.current_pipeline = Pipeline(registry=self.registry, name=name)
        
        # Add ML preparation steps
        ml_steps = [
            ("drop_duplicates", {}, "Remove duplicate records"),
            ("smart_imputation", {"strategy": "ml"}, "Handle missing values using ML"),
            ("handle_outliers", {"method": "isolation_forest"}, "Detect and handle outliers"),
            ("advanced_encoding", {"strategy": "target"}, "Encode categorical variables using target encoding"),
            ("feature_engineering", {"auto": True}, "Perform automated feature engineering")
        ]
        
        for op_name, params, desc in ml_steps:
            if op_name in self.registry:
                self.current_pipeline.add_step(op_name, params, desc)
        
        return self
    
    def add_custom_step(self, operation_name: str, parameters: Optional[Dict[str, Any]] = None,
                       description: str = "") -> 'PipelineBuilder':
        """Add a custom step to the current pipeline."""
        self.current_pipeline.add_step(operation_name, parameters, description)
        return self
    
    def build(self) -> Pipeline:
        """Build and return the current pipeline."""
        return self.current_pipeline

# ============================
# STREAMLIT INTEGRATION
# ============================

def render_pipeline_dashboard(pipeline: Pipeline):
    """Render professional interactive pipeline dashboard in Streamlit."""
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit not available - cannot render dashboard")
        return None
    
    # Set page configuration
    st.set_page_config(
        page_title="Enhanced Pipeline Manager",
        page_icon="🔧",
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
    .step-card {
        background-color: #ffffff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .success-step {
        border-left-color: #2ecc71;
    }
    .error-step {
        border-left-color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔄 Enhanced Pipeline Management</h1>', unsafe_allow_html=True)
    
    # Pipeline summary
    summary = pipeline.get_pipeline_summary()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Configuration", "Execution", "Validation", "History"])
    
    with tab1:
        # Overview metrics
        st.markdown("### 📊 Pipeline Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Steps", summary["total_steps"])
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Executions", summary["total_executions"])
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Time", f"{summary['average_execution_time']:.2f}s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Pipeline information
        st.markdown("### ℹ️ Pipeline Information")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write(f"**Name:** {summary['name']}")
            st.write(f"**Created:** {summary['created_at']}")
            st.write(f"**Last Modified:** {summary['last_modified']}")
        
        with info_col2:
            st.write(f"**Available Operations:** {summary['available_operations']}")
            st.write(f"**Enhanced Operations:** {len(summary['enhanced_operations'])}")
            st.write(f"**Memory Usage:** {summary['memory_usage_mb']} MB")
    
    with tab2:
        # Pipeline configuration
        st.markdown("### 🔧 Pipeline Configuration")
        
        if summary["total_steps"] > 0:
            steps_df = pd.DataFrame(pipeline.get_step_details())
            
            # Display steps with enhanced formatting
            for i, step in enumerate(steps_df.to_dict('records')):
                status_class = "success-step" if step.get('success', False) else "error-step" if step.get('error_message') else ""
                st.markdown(f'<div class="step-card {status_class}">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i+1}. {step['name']}**")
                    st.write(f"*{step['description']}*")
                    if step.get('parameters'):
                        st.json(step['parameters'])
                with col2:
                    if step.get('execution_time', 0) > 0:
                        st.metric("Time", f"{step['execution_time']:.2f}s")
                    if step.get('success') is not None:
                        status = "✅" if step['success'] else "❌"
                        st.write(f"**Status:** {status}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No steps configured in pipeline")
        
        # Step management
        st.markdown("### ➕ Add Pipeline Step")
        
        col1, col2 = st.columns(2)
        with col1:
            # Categorize operations for better selection
            operation_categories = {
                "All": pipeline.registry.keys(),
                "Preprocessing": global_registry_manager.list_operations('preprocessing'),
                "Visualization": global_registry_manager.list_operations('visualization'),
                "Business Intelligence": global_registry_manager.list_operations('business'),
                "Quality": global_registry_manager.list_operations('quality'),
                "File Operations": global_registry_manager.list_operations('file'),
                "ML Operations": global_registry_manager.list_operations('ml')
            }
            
            selected_category = st.selectbox("Category:", list(operation_categories.keys()))
            available_ops = list(operation_categories[selected_category])
            selected_op = st.selectbox("Operation:", available_ops)
        
        with col2:
            step_description = st.text_input("Description:", value=f"Execute {selected_op}")
        
        # Show operation metadata if available
        op_metadata = global_registry_manager.get_operation_metadata(selected_op)
        if op_metadata:
            with st.expander("Operation Details"):
                st.write(f"**Module:** {op_metadata.get('module', 'Unknown')}")
                st.write(f"**Description:** {op_metadata.get('description', 'No description available')}")
                st.write("**Signature:**")
                st.code(str(op_metadata.get('signature', 'No signature available')))
        
        # Parameters input
        with st.expander("Step Parameters (JSON format)"):
            default_params = {}
            if op_metadata and 'signature' in op_metadata:
                sig = op_metadata['signature']
                for param_name, param in sig.parameters.items():
                    if param_name != 'df' and param.default != inspect.Parameter.empty:
                        default_params[param_name] = param.default
            
            params_json = st.text_area("Parameters:", value=json.dumps(default_params, indent=2), height=200)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Step", type="primary", use_container_width=True):
                try:
                    params = json.loads(params_json) if params_json.strip() else {}
                    pipeline.add_step(selected_op, params, step_description)
                    st.success(f"Added step: {selected_op}")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in parameters")
                except Exception as e:
                    st.error(f"Error adding step: {e}")
        
        with col2:
            if st.button("Clear All Steps", use_container_width=True):
                pipeline.clear_steps()
                st.success("Cleared all pipeline steps")
                st.rerun()
    
    with tab3:
        # Pipeline execution
        st.markdown("### ▶️ Execute Pipeline")
        
        if summary["total_steps"] > 0:
            execution_col1, execution_col2 = st.columns(2)
            
            with execution_col1:
                st.markdown("#### Execution Options")
                stop_on_error = st.checkbox("Stop on Error", value=True)
                save_state = st.checkbox("Save State", value=True)
                profile_memory = st.checkbox("Profile Memory", value=True)
                
                # Execution history
                if pipeline.execution_history:
                    st.markdown("#### Recent Executions")
                    for i, result in enumerate(reversed(pipeline.execution_history)):
                        if i >= 3:  # Show only last 3 executions
                            break
                        status = "✅" if result.success else "❌"
                        st.write(f"{status} {result.pipeline_name} - {result.execution_time:.2f}s")
            
            with execution_col2:
                if st.button("Execute Pipeline", type="primary", use_container_width=True):
                    if 'df' in st.session_state and st.session_state.df is not None:
                        with st.spinner("Executing pipeline..."):
                            result = pipeline.execute(
                                st.session_state.df, 
                                save_state=save_state,
                                stop_on_error=stop_on_error,
                                profile_memory=profile_memory
                            )
                        
                        # Display results
                        if result.success:
                            st.success(f"Pipeline executed successfully in {result.execution_time:.2f}s")
                            st.session_state.df = result.dataframe
                            
                            # Show execution summary
                            with st.expander("Execution Summary"):
                                st.json(result.to_summary())
                            
                            # Show performance metrics
                            st.markdown("#### Performance Metrics")
                            metrics = result.performance_metrics
                            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                            with mcol1:
                                st.metric("Initial Memory", f"{metrics['initial_memory_mb']:.2f} MB")
                            with mcol2:
                                st.metric("Final Memory", f"{metrics['final_memory_mb']:.2f} MB")
                            with mcol3:
                                st.metric("Memory Change", f"{metrics['memory_change_mb']:+.2f} MB")
                            with mcol4:
                                st.metric("Transformation", f"{metrics['transformation_ratio']:.2%}")
                        else:
                            st.error("Pipeline execution failed")
                            for error in result.error_messages:
                                st.error(error)
                    else:
                        st.warning("No data available. Please upload data first.")
            
            # Undo/Redo functionality
            if pipeline.undo_stack or pipeline.redo_stack:
                st.markdown("#### History Management")
                undo_col, redo_col = st.columns(2)
                
                with undo_col:
                    if st.button("Undo Last Operation", disabled=not pipeline.undo_stack):
                        previous_state = pipeline.undo()
                        if previous_state is not None and 'df' in st.session_state:
                            st.session_state.df = previous_state
                            st.success("Undo successful")
                            st.rerun()
                
                with redo_col:
                    if st.button("Redo Operation", disabled=not pipeline.redo_stack):
                        next_state = pipeline.redo()
                        if next_state is not None and 'df' in st.session_state:
                            st.session_state.df = next_state
                            st.success("Redo successful")
                            st.rerun()
        else:
            st.info("No steps configured in pipeline. Add steps before execution.")
    
    with tab4:
        # Pipeline validation
        st.markdown("### ✅ Pipeline Validation")
        validation = pipeline.validate_pipeline()
        
        if validation["valid"]:
            st.success("✅ Pipeline configuration is valid")
        else:
            st.error("❌ Pipeline has validation errors")
        
        # Display validation results
        if validation["errors"]:
            st.markdown("#### Errors")
            for error in validation["errors"]:
                st.error(error)
        
        if validation["warnings"]:
            st.markdown("#### Warnings")
            for warning in validation["warnings"]:
                st.warning(warning)
        
        if validation["recommendations"]:
            st.markdown("#### Recommendations")
            for rec in validation["recommendations"]:
                st.info(rec)
        
        # Step-level validation
        if validation["step_validations"]:
            st.markdown("#### Step Validation Details")
            for step_validation in validation["step_validations"]:
                if not step_validation["valid"] or step_validation["warnings"]:
                    status = "❌" if not step_validation["valid"] else "⚠️"
                    st.write(f"{status} Step {step_validation['step_index']+1}: {step_validation['step_name']}")
                    
                    for error in step_validation["errors"]:
                        st.error(f"Error: {error}")
                    
                    for warning in step_validation["warnings"]:
                        st.warning(f"Warning: {warning}")
        
        # Optimization suggestions
        if st.button("Optimize Pipeline Order"):
            pipeline.optimize_order()
            st.success("Pipeline order optimized")
            st.rerun()
    
    with tab5:
        # Execution history
        st.markdown("### 📋 Execution History")
        
        if pipeline.execution_history:
            history_data = []
            for result in pipeline.execution_history:
                summary = result.to_summary()
                history_data.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Status": "Success" if result.success else "Failed",
                    "Time (s)": summary["execution_time"],
                    "Steps": f"{summary['steps_succeeded']}/{summary['steps_executed']}",
                    "Success Rate": f"{summary['success_rate']}%",
                    "Shape": str(summary["final_shape"]),
                    "Quality": summary["data_quality"]
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Detailed view of selected execution
            if st.checkbox("Show Detailed Execution Results"):
                selected_index = st.selectbox("Select Execution", range(len(history_data)), format_func=lambda x: history_data[x]["Timestamp"])
                if 0 <= selected_index < len(pipeline.execution_history):
                    selected_result = list(pipeline.execution_history)[selected_index]
                    st.json(selected_result.to_summary())
        else:
            st.info("No execution history available")

# ============================
# GLOBAL INSTANCES AND CONVENIENCE FUNCTIONS
# ============================

# Initialize global registry manager
global_registry_manager = EnhancedRegistryManager()

def create_enhanced_sales_pipeline(registry: Dict[str, Callable], name: str = "Enhanced Sales Analysis Pipeline") -> Pipeline:
    """Create enhanced sales pipeline - Compatible with existing function name."""
    pipeline = Pipeline(registry=registry, name=name)
    
    # Add common sales analysis steps
    common_ops = [
        ("drop_duplicates", {}, "Remove duplicate records"),
        ("smart_imputation", {"strategy": "auto"}, "Smart missing value handling"),
        ("optimize_memory", {"aggressive": True}, "Optimize data types"),
        ("handle_outliers", {"method": "auto", "sensitivity": 1.5}, "Handle outliers"),
        ("calculate_kpis", {}, "Calculate business KPIs"),
        ("generate_eda", {}, "Generate EDA report"),
        ("trend_analysis", {}, "Perform trend analysis")
    ]
    
    for op_name, params, desc in common_ops:
        if op_name in registry:
            pipeline.add_step(op_name, params, desc)
    
    return pipeline

def create_pipeline(registry: Optional[Dict[str, Callable]] = None, name: str = "Data Pipeline") -> Pipeline:
    """Create a basic pipeline."""
    if registry is None:
        registry = global_registry_manager.get_registry()
    return Pipeline(registry=registry, name=name)

def get_available_operations() -> List[str]:
    """Get list of available operations."""
    return global_registry_manager.list_operations()

def get_enhanced_operations() -> List[str]:
    """Get list of enhanced Phase 1 operations."""
    enhanced_ops = [
        'smart_imputation', 'optimize_memory', 'handle_outliers', 'advanced_encoding',
        'generate_eda', 'create_dashboard', 'calculate_kpis', 'assess_quality', 
        'validate_data', 'load_file', 'export_data', 'train_model', 'predict'
    ]
    available = global_registry_manager.list_operations()
    return [op for op in enhanced_ops if op in available]

def add_custom_operation(name: str, function: Callable, metadata: Optional[Dict[str, Any]] = None):
    """Add custom operation to global registry."""
    global_registry_manager.add_operation(name, function, metadata)

def create_data_cleaning_pipeline(name: str = "Data Cleaning Pipeline") -> Pipeline:
    """Create pre-configured cleaning pipeline."""
    builder = PipelineBuilder(registry=global_registry_manager.get_registry())
    return builder.for_data_cleaning(name).build()

def create_analysis_pipeline(name: str = "Analysis Pipeline") -> Pipeline:
    """Create pre-configured analysis pipeline."""
    builder = PipelineBuilder(registry=global_registry_manager.get_registry())
    return builder.for_analysis(name).build()

def create_ml_preparation_pipeline(name: str = "ML Preparation Pipeline") -> Pipeline:
    """Create pre-configured ML preparation pipeline."""
    builder = PipelineBuilder(registry=global_registry_manager.get_registry())
    return builder.for_ml_preparation(name).build()

# ============================
# EXPORTS
# ============================

__all__ = [
    # Enhanced classes
    'Pipeline',
    'PipelineStep',
    'PipelineExecutionResult',
    'PipelineBuilder',
    'EnhancedRegistryManager',
    
    # Main functions (compatible with existing)
    'create_enhanced_sales_pipeline',
    'create_pipeline',
    'get_available_operations',
    'get_enhanced_operations',
    'add_custom_operation',
    'create_data_cleaning_pipeline',
    'create_analysis_pipeline',
    'create_ml_preparation_pipeline',
    
    # Streamlit integration
    'render_pipeline_dashboard'
]

# Initialize logging and print status
logger.info("Enhanced Pipeline Module v3.0 - Loaded Successfully!")
print("✅ Enhanced Pipeline Module v3.0 - Loaded Successfully!")
print(f"   🔄 Available Operations: {len(global_registry_manager.list_operations())}")
print(f"   🚀 Enhanced Operations: {len(get_enhanced_operations())}")
print(f"   📊 Streamlit Support: {STREAMLIT_AVAILABLE}")
print(f"   💾 Serialization Support: {JOBLIB_AVAILABLE}")
print(f"   🧠 Memory Profiling: {MEMORY_PROFILER_AVAILABLE}")
print("   🚀 All functions ready for import!")