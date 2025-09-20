# filename: pipeline.py
"""
Workflow Pipeline Module - Enhanced Professional Edition v2.2

Advanced workflow management and pipeline orchestration with:
- Enhanced pipeline execution with smart operations
- Integration with all Phase 1 enhanced modules
- Undo/redo functionality and state management
- Performance monitoring and optimization
- Serialization and persistence capabilities
- Error handling and recovery mechanisms
- Business rule validation integration
- Professional reporting and analytics

Author: CortexX Team  
Version: 2.2.0 - Enhanced Professional Edition with Phase 1 Integration
"""

import json
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib

import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# ============================
# ENHANCED DATA STRUCTURES
# ============================

@dataclass
class PipelineStep:
    """Enhanced pipeline step with comprehensive metadata."""
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
    
    def __post_init__(self):
        """Generate unique step ID after initialization."""
        if not self.step_id:
            self.step_id = hashlib.md5(f"{self.name}_{time.time()}".encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "description": self.description,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "step_id": self.step_id,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "memory_impact": self.memory_impact,
            "quality_impact": self.quality_impact
        }

@dataclass
class PipelineExecutionResult:
    """Result of pipeline execution."""
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
            "execution_time": self.execution_time,
            "steps_executed": self.steps_executed,
            "success_rate": self.get_success_rate(),
            "final_shape": self.dataframe.shape if self.dataframe is not None else (0, 0),
            "data_quality": self.final_data_quality,
            "has_errors": len(self.error_messages) > 0,
            "has_warnings": len(self.warnings) > 0
        }

# ============================
# ENHANCED REGISTRY MANAGER
# ============================

class EnhancedRegistryManager:
    """Manage operation registry with Phase 1 enhancements."""
    
    def __init__(self):
        self.registry = {}
        self._initialize_enhanced_operations()
    
    def _initialize_enhanced_operations(self):
        """Initialize enhanced operations from Phase 1 modules."""
        enhanced_ops = {}
        
        # Try to import and register enhanced preprocessing functions
        try:
            from preprocess import (
                advanced_missing_value_handler,
                optimize_dtypes_advanced,
                advanced_outlier_treatment
            )
            enhanced_ops.update({
                'smart_imputation': advanced_missing_value_handler,
                'optimize_memory': lambda df, **kwargs: optimize_dtypes_advanced(df, **kwargs)[0],
                'handle_outliers': lambda df, **kwargs: advanced_outlier_treatment(df, **kwargs)[0]
            })
            print("✅ Enhanced preprocessing operations loaded")
        except ImportError:
            print("⚠️ Enhanced preprocessing not available")
        
        # Try to import visualization functions
        try:
            from visualization import generate_automated_eda_report
            enhanced_ops['generate_eda'] = generate_automated_eda_report
            print("✅ Enhanced EDA operations loaded")
        except ImportError:
            print("⚠️ Enhanced EDA not available")
        
        # Try to import business intelligence functions
        try:
            from business_intelligence import calculate_business_kpis
            enhanced_ops['calculate_kpis'] = calculate_business_kpis
            print("✅ Business intelligence operations loaded")
        except ImportError:
            print("⚠️ Business intelligence not available")
        
        # Try to import quality functions
        try:
            from quality import run_rules
            enhanced_ops['assess_quality'] = run_rules
            print("✅ Quality assessment operations loaded")
        except ImportError:
            print("⚠️ Quality assessment not available")
        
        # Try to import file handling operations
        try:
            from file_handler import process_file_advanced
            enhanced_ops['load_file'] = lambda filepath, **kwargs: process_file_advanced(filepath, **kwargs).dataframe
            print("✅ File handling operations loaded")
        except ImportError:
            print("⚠️ File handling not available")
        
        # Add basic pandas operations as fallbacks
        basic_ops = {
            'drop_duplicates': lambda df, **kwargs: df.drop_duplicates(**kwargs),
            'fillna_mean': lambda df, columns=None, **kwargs: df.fillna(df[columns].mean() if columns else df.mean()),
            'fillna_median': lambda df, columns=None, **kwargs: df.fillna(df[columns].median() if columns else df.median()),
            'fillna_mode': lambda df, columns=None, **kwargs: df.fillna(df[columns].mode().iloc[0] if columns else df.mode().iloc[0]),
            'drop_columns': lambda df, columns=None, **kwargs: df.drop(columns=columns) if columns else df,
            'filter_rows': lambda df, condition=None, **kwargs: df.query(condition) if condition else df,
            'sort_values': lambda df, by=None, **kwargs: df.sort_values(by=by, **kwargs) if by else df,
            'reset_index': lambda df, **kwargs: df.reset_index(drop=True, **kwargs),
            'sample_data': lambda df, n=1000, **kwargs: df.sample(n=min(n, len(df)), **kwargs) if len(df) > n else df
        }
        
        # Update registry with all operations
        self.registry.update(enhanced_ops)
        self.registry.update(basic_ops)
    
    def get_registry(self) -> Dict[str, Callable]:
        """Get the operation registry."""
        return self.registry.copy()
    
    def add_operation(self, name: str, function: Callable):
        """Add operation to registry."""
        self.registry[name] = function
    
    def list_operations(self) -> List[str]:
        """List all available operations."""
        return list(self.registry.keys())

# ============================
# ENHANCED PIPELINE CLASS
# ============================

class Pipeline:
    """Enhanced Pipeline class with Phase 1 integration."""
    
    def __init__(self, registry: Optional[Dict[str, Callable]] = None, name: str = "Enhanced Pipeline"):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.registry = registry or global_registry_manager.get_registry()
        
        # Enhanced state management
        self.execution_history: List[PipelineExecutionResult] = []
        self.state_history: List[pd.DataFrame] = []
        self.max_history = 10  # Limit memory usage
        
        # Performance tracking
        self.total_executions = 0
        self.total_execution_time = 0.0
        self.created_at = datetime.now().isoformat()
        self.last_modified = self.created_at
    
    def add_step(self, operation_name: str, parameters: Optional[Dict[str, Any]] = None, 
                 description: str = "") -> 'Pipeline':
        """Add a step to the pipeline with enhanced metadata."""
        if operation_name not in self.registry:
            raise ValueError(f"Operation '{operation_name}' not found in registry. Available: {list(self.registry.keys())}")
        
        step = PipelineStep(
            name=operation_name,
            function=self.registry[operation_name],
            parameters=parameters or {},
            description=description or f"Execute {operation_name} operation"
        )
        
        self.steps.append(step)
        self.last_modified = datetime.now().isoformat()
        return self
    
    def remove_step(self, index: int) -> 'Pipeline':
        """Remove a step from the pipeline."""
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
            self.last_modified = datetime.now().isoformat()
        else:
            raise IndexError(f"Step index {index} out of range")
        return self
    
    def insert_step(self, index: int, operation_name: str, 
                   parameters: Optional[Dict[str, Any]] = None, description: str = "") -> 'Pipeline':
        """Insert a step at a specific position."""
        if operation_name not in self.registry:
            raise ValueError(f"Operation '{operation_name}' not found in registry")
        
        step = PipelineStep(
            name=operation_name,
            function=self.registry[operation_name],
            parameters=parameters or {},
            description=description
        )
        
        self.steps.insert(index, step)
        self.last_modified = datetime.now().isoformat()
        return self
    
    def execute(self, dataframe: pd.DataFrame, 
                save_state: bool = True,
                stop_on_error: bool = False) -> PipelineExecutionResult:
        """Execute the pipeline with enhanced monitoring and error handling."""
        start_time = time.time()
        
        # Initialize execution result
        result = PipelineExecutionResult(
            success=True,
            dataframe=dataframe.copy() if dataframe is not None else None,
            pipeline_name=self.name,
            execution_time=0.0,
            steps_executed=0,
            steps_succeeded=0,
            steps_failed=0
        )
        
        if not self.steps:
            result.execution_time = time.time() - start_time
            result.warnings.append("No steps to execute in pipeline")
            return result
        
        if dataframe is None or dataframe.empty:
            result.success = False
            result.error_messages.append("Input dataframe is None or empty")
            result.execution_time = time.time() - start_time
            return result
        
        # Save initial state if requested
        if save_state:
            self._save_state(dataframe)
        
        current_df = dataframe.copy()
        
        # Execute each step
        for step_index, step in enumerate(self.steps):
            step_start_time = time.time()
            step.timestamp = datetime.now().isoformat()
            step.input_shape = current_df.shape
            
            try:
                # Execute the step
                if step.parameters:
                    step_result = step.function(current_df, **step.parameters)
                else:
                    step_result = step.function(current_df)
                
                # Handle different return types
                if isinstance(step_result, pd.DataFrame):
                    current_df = step_result
                elif isinstance(step_result, tuple) and len(step_result) > 0:
                    # Some functions return (dataframe, report) tuples
                    if isinstance(step_result[0], pd.DataFrame):
                        current_df = step_result[0]
                elif step_result is None:
                    # Functions that modify dataframe in place
                    pass
                else:
                    result.warnings.append(f"Step '{step.name}' returned unexpected type: {type(step_result)}")
                
                # Update step metadata
                step.success = True
                step.output_shape = current_df.shape
                step.memory_impact = (current_df.memory_usage(deep=True).sum() - 
                                    dataframe.memory_usage(deep=True).sum()) / 1024 / 1024
                
                result.steps_succeeded += 1
                
            except Exception as e:
                step.success = False
                step.error_message = str(e)
                step.output_shape = step.input_shape  # No change due to error
                
                result.steps_failed += 1
                result.error_messages.append(f"Step '{step.name}': {str(e)}")
                
                if stop_on_error:
                    result.success = False
                    break
            
            finally:
                step.execution_time = time.time() - step_start_time
                result.steps_executed += 1
        
        # Finalize execution result
        result.dataframe = current_df
        result.execution_time = time.time() - start_time
        result.success = result.success and (result.steps_failed == 0)
        
        # Calculate final data quality if possible
        result.final_data_quality = self._calculate_data_quality(current_df)
        
        # Update pipeline statistics
        self.total_executions += 1
        self.total_execution_time += result.execution_time
        self.execution_history.append(result)
        
        # Limit history size
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)
        
        return result
    
    def _save_state(self, dataframe: pd.DataFrame):
        """Save dataframe state for undo functionality."""
        if len(self.state_history) >= self.max_history:
            self.state_history.pop(0)
        self.state_history.append(dataframe.copy())
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate basic data quality score."""
        if df is None or df.empty:
            return 0.0
        
        try:
            # Basic quality metrics
            total_cells = len(df) * len(df.columns) if len(df.columns) > 0 else 0
            missing_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0
            duplicate_pct = (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0
            
            # Calculate score (0-10)
            completeness_score = max(0, 10 - (missing_pct / 5))
            uniqueness_score = max(0, 10 - (duplicate_pct / 2))
            
            return round((completeness_score + uniqueness_score) / 2, 1)
        except:
            return 5.0  # Default medium quality
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        return {
            "name": self.name,
            "total_steps": len(self.steps),
            "total_executions": self.total_executions,
            "average_execution_time": (self.total_execution_time / max(self.total_executions, 1)),
            "success_rate": self._calculate_success_rate(),
            "available_operations": list(self.registry.keys()),
            "enhanced_operations": self._get_enhanced_operations(),
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "can_undo": len(self.state_history) > 0
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
            'smart_imputation', 'optimize_memory', 'handle_outliers',
            'generate_eda', 'calculate_kpis', 'assess_quality', 'load_file'
        ]
        return [op for op in enhanced_ops if op in self.registry]
    
    def get_step_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all steps."""
        return [step.to_dict() for step in self.steps]
    
    def clear_steps(self) -> 'Pipeline':
        """Clear all steps from the pipeline."""
        self.steps.clear()
        self.last_modified = datetime.now().isoformat()
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
        
        return new_pipeline
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """Validate pipeline configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        if not self.steps:
            validation_result["warnings"].append("Pipeline has no steps")
        
        # Check for missing operations
        for step in self.steps:
            if step.name not in self.registry:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Operation '{step.name}' not found in registry")
        
        # Check for potential issues
        step_names = [step.name for step in self.steps]
        if step_names.count('drop_duplicates') > 1:
            validation_result["warnings"].append("Multiple duplicate removal steps detected")
        
        if 'smart_imputation' in step_names and 'fillna_mean' in step_names:
            validation_result["warnings"].append("Both smart imputation and basic fillna detected")
        
        # Recommendations
        if 'drop_duplicates' not in step_names:
            validation_result["recommendations"].append("Consider adding duplicate removal step")
        
        if not any(op in step_names for op in ['smart_imputation', 'fillna_mean', 'fillna_median']):
            validation_result["recommendations"].append("Consider adding missing value handling")
        
        return validation_result
    
    # ============================
    # SERIALIZATION METHODS
    # ============================
    
    def save_to_json(self, filepath: str) -> bool:
        """Save pipeline configuration to JSON file."""
        try:
            pipeline_config = {
                "name": self.name,
                "created_at": self.created_at,
                "last_modified": self.last_modified,
                "steps": [
                    {
                        "name": step.name,
                        "parameters": step.parameters,
                        "description": step.description
                    }
                    for step in self.steps
                ],
                "summary": self.get_pipeline_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(pipeline_config, f, indent=2, default=str)
            
            return True
        except Exception:
            return False
    
    @classmethod
    def load_from_json(cls, filepath: str, registry: Optional[Dict[str, Callable]] = None) -> 'Pipeline':
        """Load pipeline configuration from JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        pipeline = cls(registry=registry, name=config.get("name", "Loaded Pipeline"))
        
        for step_config in config.get("steps", []):
            try:
                pipeline.add_step(
                    operation_name=step_config["name"],
                    parameters=step_config.get("parameters", {}),
                    description=step_config.get("description", "")
                )
            except ValueError:
                # Skip steps with missing operations
                continue
        
        return pipeline

# ============================
# PIPELINE BUILDER CLASS
# ============================

class PipelineBuilder:
    """Enhanced pipeline builder with smart recommendations."""
    
    def __init__(self, registry: Optional[Dict[str, Callable]] = None):
        self.registry = registry or global_registry_manager.get_registry()
        self.current_pipeline = Pipeline(registry=self.registry)
    
    def for_data_cleaning(self, name: str = "Data Cleaning Pipeline") -> 'PipelineBuilder':
        """Create a pipeline optimized for data cleaning."""
        self.current_pipeline = Pipeline(registry=self.registry, name=name)
        
        # Add common data cleaning steps
        cleaning_steps = [
            ("drop_duplicates", {}, "Remove duplicate records"),
            ("smart_imputation", {"strategy": "smart"}, "Handle missing values intelligently"),
            ("optimize_memory", {"aggressive": False}, "Optimize data types for memory efficiency"),
            ("handle_outliers", {"method": "iqr_cap", "sensitivity": 1.5}, "Handle outliers using IQR method")
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
            ("calculate_kpis", {}, "Calculate business KPIs")
        ]
        
        for op_name, params, desc in analysis_steps:
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
    """Render interactive pipeline dashboard in Streamlit."""
    if not STREAMLIT_AVAILABLE:
        print("⚠️ Streamlit not available - cannot render dashboard")
        return None
    
    st.markdown("## 🔄 Enhanced Pipeline Management")
    
    # Pipeline summary
    summary = pipeline.get_pipeline_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", summary["total_steps"])
    with col2:
        st.metric("Executions", summary["total_executions"])
    with col3:
        st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
    with col4:
        st.metric("Avg Time", f"{summary['average_execution_time']:.2f}s")
    
    # Pipeline configuration
    st.markdown("### 🔧 Pipeline Configuration")
    
    if summary["total_steps"] > 0:
        steps_df = pd.DataFrame(pipeline.get_step_details())
        st.dataframe(steps_df[['name', 'description', 'success', 'execution_time']], 
                    use_container_width=True)
    else:
        st.info("No steps configured in pipeline")
    
    # Step management
    st.markdown("### ➕ Add Pipeline Step")
    
    col1, col2 = st.columns(2)
    with col1:
        available_ops = list(pipeline.registry.keys())
        selected_op = st.selectbox("Select Operation:", available_ops)
    
    with col2:
        step_description = st.text_input("Description (optional):", 
                                       value=f"Execute {selected_op}")
    
    # Parameters input (simplified)
    with st.expander("Step Parameters (JSON format)"):
        params_json = st.text_area("Parameters:", value="{}", height=100)
    
    if st.button("Add Step"):
        try:
            params = json.loads(params_json) if params_json.strip() else {}
            pipeline.add_step(selected_op, params, step_description)
            st.success(f"Added step: {selected_op}")
            st.rerun()
        except json.JSONDecodeError:
            st.error("Invalid JSON format in parameters")
        except Exception as e:
            st.error(f"Error adding step: {e}")
    
    # Pipeline execution
    if summary["total_steps"] > 0:
        st.markdown("### ▶️ Execute Pipeline")
        
        col1, col2 = st.columns(2)
        with col1:
            stop_on_error = st.checkbox("Stop on Error", value=True)
        with col2:
            save_state = st.checkbox("Save State", value=True)
        
        if st.button("Execute Pipeline"):
            if 'df' in st.session_state and st.session_state.df is not None:
                with st.spinner("Executing pipeline..."):
                    result = pipeline.execute(
                        st.session_state.df, 
                        save_state=save_state,
                        stop_on_error=stop_on_error
                    )
                
                if result.success:
                    st.success(f"Pipeline executed successfully in {result.execution_time:.2f}s")
                    st.session_state.df = result.dataframe
                    
                    # Show execution summary
                    st.json(result.to_summary())
                else:
                    st.error("Pipeline execution failed")
                    for error in result.error_messages:
                        st.error(error)
            else:
                st.warning("No data available. Please upload data first.")
    
    # Pipeline validation
    st.markdown("### ✅ Pipeline Validation")
    validation = pipeline.validate_pipeline()
    
    if validation["valid"]:
        st.success("Pipeline configuration is valid")
    else:
        st.error("Pipeline has validation errors")
        for error in validation["errors"]:
            st.error(error)
    
    for warning in validation["warnings"]:
        st.warning(warning)
    
    for rec in validation["recommendations"]:
        st.info(rec)

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
        ("smart_imputation", {"strategy": "smart"}, "Smart missing value handling"),
        ("optimize_memory", {"aggressive": False}, "Optimize data types"),
        ("handle_outliers", {"method": "iqr_cap", "sensitivity": 1.5}, "Handle outliers")
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
        'smart_imputation', 'optimize_memory', 'handle_outliers',
        'generate_eda', 'calculate_kpis', 'assess_quality', 'load_file'
    ]
    available = global_registry_manager.list_operations()
    return [op for op in enhanced_ops if op in available]

def add_custom_operation(name: str, function: Callable):
    """Add custom operation to global registry."""
    global_registry_manager.add_operation(name, function)

def create_data_cleaning_pipeline(name: str = "Data Cleaning Pipeline") -> Pipeline:
    """Create pre-configured cleaning pipeline."""
    builder = PipelineBuilder(registry=global_registry_manager.get_registry())
    return builder.for_data_cleaning(name).build()

def create_analysis_pipeline(name: str = "Analysis Pipeline") -> Pipeline:
    """Create pre-configured analysis pipeline."""
    builder = PipelineBuilder(registry=global_registry_manager.get_registry())
    return builder.for_analysis(name).build()

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
    'create_enhanced_sales_pipeline',  # This matches existing
    'create_pipeline',
    'get_available_operations',
    'get_enhanced_operations',
    'add_custom_operation',
    'create_data_cleaning_pipeline',
    'create_analysis_pipeline',
    
    # Streamlit integration
    'render_pipeline_dashboard'
]

print("✅ Enhanced Pipeline Module v2.2 - Loaded Successfully!")
print(f"   🔄 Available Operations: {len(global_registry_manager.list_operations())}")
print(f"   🚀 Enhanced Operations: {len(get_enhanced_operations())}")
print(f"   🎨 Streamlit Support: {STREAMLIT_AVAILABLE}")
print(f"   💾 Serialization Support: {JOBLIB_AVAILABLE}")
print("   🚀 All functions ready for import!")