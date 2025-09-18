# filename: pipeline.py
"""
Pipeline Module - Fixed Professional Edition

Simple, reliable pipeline system for recording and managing data transformation steps.
Clean architecture with essential features:
- Step recording and replay
- Undo/redo functionality  
- Pipeline serialization
- Error handling
- Performance monitoring

Author: CortexX Team
Version: 1.2.0 - Fixed Professional Edition
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from datetime import datetime
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================
# SIMPLE DATA STRUCTURES
# ============================

class Step:
    """Simple step representation for pipeline operations."""

    def __init__(self, op: str, params: Dict[str, Any], timestamp: Optional[str] = None, label: Optional[str] = None):
        self.op = op
        self.params = params or {}
        self.timestamp = timestamp or datetime.now().isoformat()
        self.label = label or op
        self.execution_time = 0.0
        self.success = False
        self.error_message = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "op": self.op,
            "params": self.params,
            "timestamp": self.timestamp,
            "label": self.label,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Step':
        """Create step from dictionary."""
        step = cls(
            op=data.get("op", "unknown"),
            params=data.get("params", {}),
            timestamp=data.get("timestamp"),
            label=data.get("label")
        )
        step.execution_time = data.get("execution_time", 0.0)
        step.success = data.get("success", False)
        step.error_message = data.get("error_message")
        return step

# ============================
# PIPELINE CLASS
# ============================

class Pipeline:
    """
    Simple, reliable pipeline for data transformation workflows.

    Features:
    - Step recording and execution
    - Undo/redo functionality
    - JSON serialization 
    - Error handling
    - Performance monitoring
    """

    def __init__(self, registry: Optional[Dict[str, Callable]] = None, name: str = "Pipeline"):
        """Initialize pipeline with optional function registry."""
        self.steps: List[Step] = []
        self._undone: List[Step] = []
        self.registry: Dict[str, Callable] = registry or {}
        self.name = name
        self.version = "1.2.0"
        self.created_at = datetime.now().isoformat()
        self.modified_at = self.created_at

        # Execution tracking
        self.total_executions = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0
        self.success_count = 0
        self.error_count = 0

    @property
    def has_steps(self) -> bool:
        """Return True if the pipeline contains at least one step."""
        return len(self.steps) > 0

    @property
    def can_undo(self) -> bool:
        """Return True if there are steps that can be undone."""
        return len(self.steps) > 0

    @property
    def can_redo(self) -> bool:
        """Return True if there are steps that can be redone."""
        return len(self._undone) > 0

    @property
    def step_count(self) -> int:
        """Return the total number of steps in the pipeline."""
        return len(self.steps)

    @property
    def is_empty(self) -> bool:
        """Return True if the pipeline has no steps."""
        return len(self.steps) == 0

    def add_step(self, op: str, params: Optional[Dict[str, Any]] = None, label: Optional[str] = None) -> None:
        """
        Add a step to the pipeline.

        Parameters:
        - op: Operation name (must exist in registry for execution)
        - params: Parameters for the operation
        - label: Optional human-readable label for the step
        """
        if not op:
            raise ValueError("Operation name cannot be empty")

        step = Step(op=op, params=params or {}, label=label)
        self.steps.append(step)
        self._undone.clear()  # Clear redo stack when adding new step
        self.modified_at = datetime.now().isoformat()

    def insert_step(self, index: int, op: str, params: Optional[Dict[str, Any]] = None, label: Optional[str] = None) -> None:
        """Insert a step at a specific position."""
        if not op:
            raise ValueError("Operation name cannot be empty")

        if index < 0 or index > len(self.steps):
            raise IndexError(f"Index {index} out of range for pipeline with {len(self.steps)} steps")

        step = Step(op=op, params=params or {}, label=label)
        self.steps.insert(index, step)
        self._undone.clear()
        self.modified_at = datetime.now().isoformat()

    def remove_step(self, index: int) -> bool:
        """Remove a step by index."""
        if index < 0 or index >= len(self.steps):
            return False

        removed_step = self.steps.pop(index)
        self._undone.append(removed_step)
        self.modified_at = datetime.now().isoformat()
        return True

    def get_step(self, index: int) -> Optional[Step]:
        """Get a step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def update_step(self, index: int, op: Optional[str] = None, params: Optional[Dict[str, Any]] = None, label: Optional[str] = None) -> bool:
        """Update step parameters."""
        if index < 0 or index >= len(self.steps):
            return False

        step = self.steps[index]
        if op is not None:
            step.op = op
        if params is not None:
            step.params = params
        if label is not None:
            step.label = label

        self.modified_at = datetime.now().isoformat()
        return True

    def undo(self) -> bool:
        """Remove the last step and add it to the redo stack."""
        if not self.steps:
            return False

        undone_step = self.steps.pop()
        self._undone.append(undone_step)
        self.modified_at = datetime.now().isoformat()
        return True

    def redo(self) -> bool:
        """Restore the last undone step."""
        if not self._undone:
            return False

        redone_step = self._undone.pop()
        self.steps.append(redone_step)
        self.modified_at = datetime.now().isoformat()
        return True

    def clear(self) -> None:
        """Clear all steps from the pipeline."""
        self.steps.clear()
        self._undone.clear()
        self.modified_at = datetime.now().isoformat()

    def apply(
        self, 
        df: pd.DataFrame, 
        on_error: str = "skip",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        return_details: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Apply all pipeline steps to a DataFrame.

        Parameters:
        - df: Input DataFrame
        - on_error: 'skip' to continue on errors, 'raise' to stop on first error
        - progress_callback: Optional callback function(current, total, message)
        - return_details: If True, return (result_df, execution_details)

        Returns:
        - DataFrame or (DataFrame, execution_details) tuple
        """
        if df.empty:
            if return_details:
                return df, {"error": "Input DataFrame is empty"}
            return df

        start_time = time.time()
        result_df = df.copy()

        # Execution tracking
        execution_details = {
            "total_steps": len(self.steps),
            "completed_steps": 0,
            "failed_steps": 0,
            "skipped_steps": 0,
            "step_results": [],
            "errors": [],
            "warnings": [],
            "total_time": 0.0
        }

        # Execute each step
        for i, step in enumerate(self.steps):
            step_start_time = time.time()

            # Progress callback
            if progress_callback:
                try:
                    progress_callback(i, len(self.steps), f"Executing: {step.label}")
                except Exception:
                    pass  # Don't let callback errors stop execution

            try:
                # Check if operation exists in registry
                if step.op not in self.registry:
                    error_msg = f"Operation '{step.op}' not found in registry"
                    step.error_message = error_msg
                    step.success = False
                    execution_details["errors"].append(error_msg)
                    execution_details["failed_steps"] += 1

                    if on_error == "raise":
                        raise ValueError(error_msg)

                    execution_details["skipped_steps"] += 1
                    continue

                # Execute the operation
                func = self.registry[step.op]

                # Handle different function signatures
                try:
                    if step.params:
                        step_result_df = func(result_df, **step.params)
                    else:
                        step_result_df = func(result_df)
                except TypeError:
                    # Try without unpacking params
                    step_result_df = func(result_df)

                # Validate result
                if not isinstance(step_result_df, pd.DataFrame):
                    raise ValueError(f"Operation '{step.op}' did not return a DataFrame")

                # Update result
                result_df = step_result_df

                # Record success
                step.success = True
                step.error_message = None
                step.execution_time = time.time() - step_start_time
                execution_details["completed_steps"] += 1

            except Exception as e:
                error_msg = f"Step '{step.label}' failed: {str(e)}"
                step.error_message = error_msg
                step.success = False
                step.execution_time = time.time() - step_start_time

                execution_details["errors"].append(error_msg)
                execution_details["failed_steps"] += 1

                if on_error == "raise":
                    raise RuntimeError(error_msg) from e

            # Record step result
            rows_before = len(df) if i == 0 else len(result_df)
            execution_details["step_results"].append({
                "step": step.label,
                "operation": step.op,
                "success": step.success,
                "execution_time": step.execution_time,
                "error": step.error_message,
                "rows_before": rows_before,
                "rows_after": len(result_df),
                "columns_before": len(df.columns) if i == 0 else len(result_df.columns),
                "columns_after": len(result_df.columns)
            })

        # Update execution statistics
        total_time = time.time() - start_time
        execution_details["total_time"] = total_time

        self.total_executions += 1
        self.total_execution_time += total_time
        self.last_execution_time = total_time

        if execution_details["failed_steps"] == 0:
            self.success_count += 1
        else:
            self.error_count += 1

        # Final progress callback
        if progress_callback:
            try:
                progress_callback(len(self.steps), len(self.steps), "Pipeline execution completed")
            except Exception:
                pass

        if return_details:
            return result_df, execution_details
        else:
            return result_df

    def get_step_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all steps in the pipeline."""
        return [
            {
                "index": i,
                "operation": step.op,
                "label": step.label,
                "params": step.params,
                "timestamp": step.timestamp,
                "last_execution_time": step.execution_time,
                "last_success": step.success,
                "last_error": step.error_message
            }
            for i, step in enumerate(self.steps)
        ]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        avg_execution_time = (self.total_execution_time / self.total_executions) if self.total_executions > 0 else 0.0
        success_rate = (self.success_count / self.total_executions * 100) if self.total_executions > 0 else 0.0

        return {
            "total_executions": self.total_executions,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate_percent": round(success_rate, 1),
            "total_execution_time": round(self.total_execution_time, 2),
            "average_execution_time": round(avg_execution_time, 2),
            "last_execution_time": round(self.last_execution_time, 2)
        }

    # ============================
    # SERIALIZATION
    # ============================

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "steps": [step.to_dict() for step in self.steps],
            "undone_steps": [step.to_dict() for step in self._undone],
            "performance_stats": self.get_performance_stats(),
            "meta": {
                "step_count": len(self.steps),
                "undone_count": len(self._undone),
                "exported_at": datetime.now().isoformat()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], registry: Optional[Dict[str, Callable]] = None) -> 'Pipeline':
        """Create pipeline from dictionary."""
        pipeline = cls(registry=registry, name=data.get("name", "Pipeline"))

        # Restore metadata
        pipeline.version = data.get("version", "1.2.0")
        pipeline.created_at = data.get("created_at", datetime.now().isoformat())
        pipeline.modified_at = data.get("modified_at", datetime.now().isoformat())

        # Restore steps
        steps_data = data.get("steps", [])
        pipeline.steps = [Step.from_dict(step_data) for step_data in steps_data]

        # Restore undone steps
        undone_data = data.get("undone_steps", [])
        pipeline._undone = [Step.from_dict(step_data) for step_data in undone_data]

        # Restore performance stats if available
        perf_stats = data.get("performance_stats", {})
        pipeline.total_executions = perf_stats.get("total_executions", 0)
        pipeline.success_count = perf_stats.get("success_count", 0)
        pipeline.error_count = perf_stats.get("error_count", 0)
        pipeline.total_execution_time = perf_stats.get("total_execution_time", 0.0)
        pipeline.last_execution_time = perf_stats.get("last_execution_time", 0.0)

        return pipeline

    def save_to_file(self, filepath: Union[str, Path]) -> bool:
        """Save pipeline to JSON file."""
        try:
            filepath = Path(filepath)

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Failed to save pipeline: {str(e)}")
            return False

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path], registry: Optional[Dict[str, Callable]] = None) -> Optional['Pipeline']:
        """Load pipeline from JSON file."""
        try:
            filepath = Path(filepath)

            if not filepath.exists():
                print(f"Pipeline file not found: {filepath}")
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return cls.from_dict(data, registry)

        except Exception as e:
            print(f"Failed to load pipeline: {str(e)}")
            return None

    def export_to_json(self) -> str:
        """Export pipeline as JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def import_from_json(cls, json_str: str, registry: Optional[Dict[str, Callable]] = None) -> Optional['Pipeline']:
        """Import pipeline from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data, registry)
        except Exception as e:
            print(f"Failed to import pipeline: {str(e)}")
            return None

    # ============================
    # UTILITY METHODS
    # ============================

    def copy(self, new_name: Optional[str] = None) -> 'Pipeline':
        """Create a copy of the pipeline."""
        data = self.to_dict()
        copied = self.from_dict(data, self.registry)

        if new_name:
            copied.name = new_name
        else:
            copied.name = f"{self.name}_copy"

        copied.created_at = datetime.now().isoformat()
        copied.modified_at = copied.created_at

        # Reset performance stats for the copy
        copied.total_executions = 0
        copied.total_execution_time = 0.0
        copied.last_execution_time = 0.0
        copied.success_count = 0
        copied.error_count = 0

        return copied

    def find_steps_by_operation(self, operation: str) -> List[int]:
        """Find all step indices with a specific operation."""
        return [i for i, step in enumerate(self.steps) if step.op == operation]

    def replace_operation(self, old_op: str, new_op: str) -> int:
        """Replace all occurrences of an operation with another."""
        count = 0
        for step in self.steps:
            if step.op == old_op:
                step.op = new_op
                count += 1

        if count > 0:
            self.modified_at = datetime.now().isoformat()

        return count

    def validate(self) -> Dict[str, Any]:
        """Validate pipeline structure and operations."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }

        # Check for empty pipeline
        if self.is_empty:
            validation_result["warnings"].append("Pipeline is empty")
            return validation_result

        # Check for unknown operations
        unknown_ops = []
        for step in self.steps:
            if step.op not in self.registry:
                unknown_ops.append(step.op)

        if unknown_ops:
            unique_unknown = list(set(unknown_ops))
            validation_result["errors"].extend([f"Unknown operation: {op}" for op in unique_unknown])
            validation_result["is_valid"] = False

        # Performance suggestions
        if len(self.steps) > 10:
            validation_result["suggestions"].append("Large pipeline - consider breaking into smaller pipelines")

        # Check for duplicate operations that could be combined
        op_counts = {}
        for step in self.steps:
            op_counts[step.op] = op_counts.get(step.op, 0) + 1

        duplicates = [op for op, count in op_counts.items() if count > 1]
        if duplicates:
            validation_result["suggestions"].append(f"Consider combining duplicate operations: {duplicates}")

        return validation_result

    def __str__(self) -> str:
        """String representation of the pipeline."""
        return f"Pipeline('{self.name}', steps={len(self.steps)}, version={self.version})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Pipeline(name='{self.name}', steps={len(self.steps)}, "
                f"undone={len(self._undone)}, version='{self.version}')")

# ============================
# CONVENIENCE FUNCTIONS
# ============================

def create_basic_pipeline(registry: Dict[str, Callable], name: str = "Basic Pipeline") -> Pipeline:
    """Create a basic pipeline with common data cleaning steps."""
    pipeline = Pipeline(registry=registry, name=name)

    # Add common steps if operations exist in registry
    common_ops = [
        ("remove_duplicates", {}),
        ("handle_missing", {"strategy": "drop"}),
        ("standardize_columns", {}),
    ]

    for op, params in common_ops:
        if op in registry:
            pipeline.add_step(op, params)

    return pipeline

def merge_pipelines(pipeline1: Pipeline, pipeline2: Pipeline, name: str = "Merged Pipeline") -> Pipeline:
    """Merge two pipelines into one."""
    # Use registry from first pipeline
    merged = Pipeline(registry=pipeline1.registry, name=name)

    # Add steps from both pipelines
    for step in pipeline1.steps:
        merged.add_step(step.op, step.params.copy(), step.label)

    for step in pipeline2.steps:
        merged.add_step(step.op, step.params.copy(), step.label)

    return merged

# ============================
# EXPORTS
# ============================

__all__ = [
    'Step',
    'Pipeline',
    'create_basic_pipeline',
    'merge_pipelines'
]
