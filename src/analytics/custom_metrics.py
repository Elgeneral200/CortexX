"""
Custom Metrics Builder for CortexX Forecasting Platform
PHASE 3 - SESSION 7: User-Defined Calculated Fields
✅ FIXED: Handles column names with spaces properly
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class CustomMetric:
    """
    Represents a custom calculated metric.
    
    ✅ NEW: Phase 3 - Session 7
    """
    
    def __init__(self, name: str, formula: str, description: str = ""):
        """
        Initialize custom metric.
        
        Args:
            name: Metric name
            formula: Calculation formula
            description: Optional description
        """
        self.name = name
        self.formula = formula
        self.description = description
        self.created_at = pd.Timestamp.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'formula': self.formula,
            'description': self.description,
            'created_at': str(self.created_at)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomMetric':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            formula=data['formula'],
            description=data.get('description', '')
        )


class FormulaParser:
    """
    Parse and validate formulas for custom metrics.
    
    ✅ FIXED: Phase 3 - Session 7 - Handles column names with spaces
    """
    
    OPERATORS = ['+', '-', '*', '/', '%', '(', ')']
    FUNCTIONS = ['SUM', 'AVG', 'MIN', 'MAX', 'COUNT', 'ABS', 'ROUND']
    
    @staticmethod
    def extract_column_references(formula: str, available_columns: List[str]) -> List[str]:
        """
        ✅ FIXED: Extract column names from formula, handling spaces properly.
        
        Args:
            formula: Formula string
            available_columns: List of actual column names from DataFrame
            
        Returns:
            List of column names found in formula
        """
        found_columns = []
        
        # Sort columns by length (longest first) to match multi-word columns first
        # This prevents "Units Sold" from matching "Units" and "Sold" separately
        sorted_columns = sorted(available_columns, key=len, reverse=True)
        
        # Check if each column name appears in the formula
        for col in sorted_columns:
            if col in formula:
                found_columns.append(col)
        
        return list(set(found_columns))
    
    @staticmethod
    def validate_formula_syntax(formula: str) -> tuple:
        """
        Validate formula syntax.
        
        Args:
            formula: Formula string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not formula or formula.strip() == '':
            return False, "Formula cannot be empty"
        
        # Check balanced parentheses
        if formula.count('(') != formula.count(')'):
            return False, "Unbalanced parentheses"
        
        # Check for consecutive operators (excluding negative numbers)
        for i in range(len(formula) - 1):
            if formula[i] in ['+', '-', '*', '/', '%'] and formula[i+1] in ['+', '*', '/', '%']:
                return False, f"Invalid consecutive operators at position {i}"
        
        # Check for division by zero patterns
        if '/0' in formula.replace(' ', ''):
            return False, "Potential division by zero"
        
        return True, "Valid"
    
    @staticmethod
    def convert_to_pandas_expression(formula: str, columns: List[str]) -> str:
        """
        ✅ FIXED: Convert user formula to pandas-compatible expression.
        
        Args:
            formula: User-friendly formula
            columns: List of column names to replace
            
        Returns:
            Pandas-compatible expression
        """
        expr = formula
        
        # Sort by length (longest first) to avoid partial replacements
        # e.g., "Units Sold" should be replaced before "Units"
        sorted_columns = sorted(columns, key=len, reverse=True)
        
        # Replace column names with pandas column references
        for col in sorted_columns:
            # Replace the column name with df['column_name']
            expr = expr.replace(col, f"df['{col}']")
        
        # Convert functions to pandas equivalents
        expr = expr.replace('SUM(', 'sum(')
        expr = expr.replace('AVG(', 'mean(')
        expr = expr.replace('MIN(', 'min(')
        expr = expr.replace('MAX(', 'max(')
        expr = expr.replace('COUNT(', 'count(')
        expr = expr.replace('ABS(', 'abs(')
        expr = expr.replace('ROUND(', 'round(')
        
        return expr


class MetricsManager:
    """
    Manage custom metrics and apply them to DataFrames.
    
    ✅ FIXED: Phase 3 - Session 7
    """
    
    def __init__(self):
        """Initialize metrics manager."""
        self.metrics: Dict[str, CustomMetric] = {}
    
    def add_metric(self, metric: CustomMetric) -> bool:
        """
        Add a custom metric.
        
        Args:
            metric: CustomMetric instance
            
        Returns:
            Success status
        """
        if metric.name in self.metrics:
            logger.warning(f"Metric '{metric.name}' already exists. Overwriting.")
        
        self.metrics[metric.name] = metric
        logger.info(f"Added metric: {metric.name}")
        return True
    
    def remove_metric(self, name: str) -> bool:
        """
        Remove a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Success status
        """
        if name in self.metrics:
            del self.metrics[name]
            logger.info(f"Removed metric: {name}")
            return True
        return False
    
    def get_metric(self, name: str) -> Optional[CustomMetric]:
        """Get metric by name."""
        return self.metrics.get(name)
    
    def list_metrics(self) -> List[Dict[str, Any]]:
        """List all metrics."""
        return [metric.to_dict() for metric in self.metrics.values()]
    
    def apply_metric(self, df: pd.DataFrame, metric: CustomMetric) -> pd.Series:
        """
        ✅ FIXED: Apply a custom metric to a DataFrame.
        
        Args:
            df: DataFrame to apply metric to
            metric: CustomMetric to apply
            
        Returns:
            Series with calculated values
            
        Raises:
            ValueError: If formula is invalid or references missing columns
        """
        # Validate formula syntax
        is_valid, error_msg = FormulaParser.validate_formula_syntax(metric.formula)
        if not is_valid:
            raise ValueError(f"Invalid formula: {error_msg}")
        
        # ✅ FIX: Pass available columns to extract_column_references
        columns = FormulaParser.extract_column_references(
            metric.formula,
            df.columns.tolist()
        )
        
        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")
        
        # ✅ FIX: Convert formula to pandas expression with proper column handling
        pandas_expr = FormulaParser.convert_to_pandas_expression(metric.formula, columns)
        
        try:
            # Evaluate expression
            result = eval(pandas_expr)
            
            # Handle scalar results
            if isinstance(result, (int, float)):
                result = pd.Series([result] * len(df), index=df.index)
            
            return result
        
        except Exception as e:
            raise ValueError(f"Error evaluating formula: {str(e)}")
    
    def apply_metrics_to_dataframe(
        self,
        df: pd.DataFrame,
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply multiple metrics to DataFrame and return new DataFrame with calculated columns.
        
        Args:
            df: Original DataFrame
            metric_names: List of metric names to apply (None = all metrics)
            
        Returns:
            DataFrame with additional calculated columns
        """
        result_df = df.copy()
        
        # Determine which metrics to apply
        if metric_names is None:
            metrics_to_apply = self.metrics.values()
        else:
            metrics_to_apply = [self.metrics[name] for name in metric_names if name in self.metrics]
        
        # Apply each metric
        for metric in metrics_to_apply:
            try:
                result_df[metric.name] = self.apply_metric(df, metric)
                logger.info(f"Applied metric: {metric.name}")
            except Exception as e:
                logger.error(f"Failed to apply metric '{metric.name}': {e}")
        
        return result_df
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export all metrics as JSON-serializable list."""
        return [metric.to_dict() for metric in self.metrics.values()]
    
    def import_metrics(self, metrics_data: List[Dict[str, Any]]) -> int:
        """
        Import metrics from list of dictionaries.
        
        Args:
            metrics_data: List of metric dictionaries
            
        Returns:
            Number of metrics imported
        """
        imported = 0
        for data in metrics_data:
            try:
                metric = CustomMetric.from_dict(data)
                self.add_metric(metric)
                imported += 1
            except Exception as e:
                logger.error(f"Failed to import metric: {e}")
        
        return imported


def get_metrics_manager() -> MetricsManager:
    """Get metrics manager instance."""
    return MetricsManager()


# Example predefined metrics
# ✅ NOTE: These use generic column names - users should create their own
PREDEFINED_METRICS = [
    CustomMetric(
        name="Example: Profit Margin %",
        formula="(Revenue - Cost) / Revenue * 100",
        description="Calculate profit margin as percentage (requires Revenue and Cost columns)"
    ),
    CustomMetric(
        name="Example: Growth Rate %",
        formula="(Current - Previous) / Previous * 100",
        description="Calculate growth rate between two values (requires Current and Previous columns)"
    ),
    CustomMetric(
        name="Example: ROI %",
        formula="(Gain - Investment) / Investment * 100",
        description="Return on investment percentage (requires Gain and Investment columns)"
    )
]
