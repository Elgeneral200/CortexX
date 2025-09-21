# filename: quality.py
"""
Data Quality Module - Professional Enterprise Edition v3.0

Enhanced data quality validation and reporting with complete Phase 1 integration:
- Comprehensive quality rules (null, unique, range, pattern validation)
- Advanced quality scoring and metrics with business intelligence
- Professional HTML and interactive reporting
- Complete integration with enhanced preprocessing and visualization modules
- Real-time quality monitoring and alerts
- Business rule validation engine
- Quality trend analysis and recommendations
- Professional logging and error handling

Author: CortexX Team
Version: 3.0.0 - Professional Enterprise Edition
"""

import pandas as pd
import numpy as np
import json
import re
import html
import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quality.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies for Phase 1 integration
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - some statistical features will be limited")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - some ML-based features will be limited")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit not available - dashboard features will be limited")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - advanced visualization features will be limited")

# ============================
# ENHANCED DATA STRUCTURES
# ============================

@dataclass
class QualityResult:
    """Enhanced quality check result with comprehensive metadata."""
    rule_id: str
    label: str
    passed: bool
    failed_count: int
    total_count: int
    failed_indices: List[int] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced attributes for Phase 1
    severity: str = "medium"  # low, medium, high, critical
    business_impact: str = ""
    recommendation: str = ""
    fix_strategy: str = ""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    
    def __post_init__(self):
        """Calculate pass rate and determine severity."""
        self.pass_rate = ((self.total_count - self.failed_count) / max(self.total_count, 1)) * 100
        
        # Auto-determine severity based on failure rate
        if self.pass_rate >= 95:
            self.severity = "low"
        elif self.pass_rate >= 85:
            self.severity = "medium" 
        elif self.pass_rate >= 70:
            self.severity = "high"
        else:
            self.severity = "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "label": self.label,
            "passed": self.passed,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "pass_rate": round(self.pass_rate, 2),
            "severity": self.severity,
            "business_impact": self.business_impact,
            "recommendation": self.recommendation,
            "fix_strategy": self.fix_strategy,
            "execution_time": round(self.execution_time, 4),
            "memory_usage": round(self.memory_usage, 2),
            "stats": self.stats
        }

@dataclass 
class QualityMetrics:
    """Enhanced quality metrics with business intelligence."""
    total_rows: int = 0
    total_columns: int = 0
    passed_rules: int = 0
    failed_rules: int = 0
    overall_score: float = 0.0
    execution_time: float = 0.0
    missing_values: int = 0
    duplicate_rows: int = 0
    
    # Enhanced Phase 1 metrics
    data_completeness: float = 0.0
    data_uniqueness: float = 0.0
    data_consistency: float = 0.0
    data_validity: float = 0.0
    memory_usage_mb: float = 0.0
    quality_trend: str = "stable"  # improving, stable, declining
    business_readiness: str = "unknown"  # ready, needs_attention, critical
    
    def calculate_enhanced_score(self) -> float:
        """Calculate enhanced quality score using multiple dimensions."""
        # Weighted scoring system
        completeness_weight = 0.3
        uniqueness_weight = 0.25
        consistency_weight = 0.25
        validity_weight = 0.2
        
        weighted_score = (
            self.data_completeness * completeness_weight +
            self.data_uniqueness * uniqueness_weight +
            self.data_consistency * consistency_weight +
            self.data_validity * validity_weight
        )
        
        self.overall_score = round(weighted_score, 1)
        return self.overall_score
    
    def determine_business_readiness(self) -> str:
        """Determine business readiness based on quality metrics."""
        if self.overall_score >= 8.5:
            self.business_readiness = "ready"
        elif self.overall_score >= 7.0:
            self.business_readiness = "needs_attention"
        else:
            self.business_readiness = "critical"
        
        return self.business_readiness

# ============================
# ENHANCED RULE FUNCTIONS
# ============================

def rule_label(rule: Dict[str, Any]) -> str:
    """Generate user-friendly labels for quality rules."""
    rule_type = rule.get('type', 'unknown')
    params = rule.get('params', {})
    
    if rule_type == 'not_null':
        return f"Not Null: {params.get('column', 'N/A')}"
    elif rule_type == 'not_null_threshold':
        threshold = params.get('threshold', 1.0) * 100
        return f"Completeness ≥{threshold:.0f}%: {params.get('column', 'N/A')}"
    elif rule_type == 'unique':
        return f"Unique Values: {params.get('column', 'N/A')}"
    elif rule_type == 'unique_multi':
        cols = params.get('columns', [])
        return f"Unique Combination: {', '.join(cols)}"
    elif rule_type == 'min':
        return f"Min Value ≥{params.get('min')}: {params.get('column', 'N/A')}"
    elif rule_type == 'max':
        return f"Max Value ≤{params.get('max')}: {params.get('column', 'N/A')}"
    elif rule_type == 'between':
        return f"Range [{params.get('min')}, {params.get('max')}]: {params.get('column', 'N/A')}"
    elif rule_type == 'allowed':
        allowed_count = len(params.get('allowed', []))
        return f"Allowed Values ({allowed_count} items): {params.get('column', 'N/A')}"
    elif rule_type == 'regex':
        pattern = params.get('pattern', '')[:15]
        pattern = pattern + '...' if len(params.get('pattern', '')) > 15 else pattern
        return f"Pattern Match '{pattern}': {params.get('column', 'N/A')}"
    elif rule_type == 'dtype':
        return f"Data Type '{params.get('dtype')}': {params.get('column', 'N/A')}"
    elif rule_type == 'custom':
        return f"Custom Rule: {params.get('name', 'Unnamed')}"
    else:
        return f"Unknown Rule: {rule_type}"

# Enhanced check functions with better error handling and performance
def check_not_null(df: pd.DataFrame, column: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check for non-null values in a column."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    mask = df[column].notna()
    missing_count = int((~mask).sum())
    missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
    
    return mask, {
        "missing_count": missing_count,
        "missing_percentage": round(missing_pct, 2),
        "total_rows": len(df)
    }

def check_not_null_threshold(df: pd.DataFrame, column: str, threshold: float = 0.95) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check if column completeness meets threshold."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    mask = df[column].notna()
    completeness = mask.sum() / len(df) if len(df) > 0 else 0
    passes_threshold = completeness >= threshold
    
    # If threshold is met, all rows pass; otherwise, missing rows fail
    if passes_threshold:
        result_mask = pd.Series([True] * len(df), index=df.index)
    else:
        result_mask = mask
    
    return result_mask, {
        "completeness": round(completeness, 4),
        "threshold": threshold,
        "passes_threshold": passes_threshold,
        "missing_count": int((~mask).sum())
    }

def check_unique(df: pd.DataFrame, column: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check for unique values in a column."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    # Handle null values - they don't violate uniqueness
    non_null_mask = df[column].notna()
    dup_mask = df[column].duplicated(keep=False) & non_null_mask
    unique_mask = ~dup_mask
    
    duplicate_count = int(dup_mask.sum())
    unique_values = df[column].nunique()
    total_non_null = int(non_null_mask.sum())
    
    return unique_mask, {
        "duplicate_count": duplicate_count,
        "unique_values": unique_values,
        "total_non_null": total_non_null,
        "duplicate_percentage": round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0
    }

def check_unique_multi(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check for unique combinations across multiple columns."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Columns not found: {missing_columns}"}
    
    dup_mask = df.duplicated(subset=columns, keep=False)
    unique_mask = ~dup_mask
    
    duplicate_count = int(dup_mask.sum())
    unique_combinations = len(df.drop_duplicates(subset=columns))
    
    return unique_mask, {
        "duplicate_count": duplicate_count,
        "unique_combinations": unique_combinations,
        "total_rows": len(df),
        "columns_checked": columns
    }

def check_min(df: pd.DataFrame, column: str, min_val: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check minimum value constraint."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    try:
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        mask = (numeric_series >= min_val) | numeric_series.isna()  # Allow nulls to pass
        violations = int((~mask & numeric_series.notna()).sum())
        actual_min = numeric_series.min()
        
        return mask, {
            "violations": violations,
            "actual_min": float(actual_min) if not pd.isna(actual_min) else None,
            "expected_min": min_val
        }
    except Exception as e:
        return pd.Series([False] * len(df), index=df.index), {"error": str(e)}

def check_max(df: pd.DataFrame, column: str, max_val: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check maximum value constraint."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    try:
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        mask = (numeric_series <= max_val) | numeric_series.isna()  # Allow nulls to pass
        violations = int((~mask & numeric_series.notna()).sum())
        actual_max = numeric_series.max()
        
        return mask, {
            "violations": violations,
            "actual_max": float(actual_max) if not pd.isna(actual_max) else None,
            "expected_max": max_val
        }
    except Exception as e:
        return pd.Series([False] * len(df), index=df.index), {"error": str(e)}

def check_between(df: pd.DataFrame, column: str, min_val: Any, max_val: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check value range constraint."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    try:
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        mask = ((numeric_series >= min_val) & (numeric_series <= max_val)) | numeric_series.isna()
        violations = int((~mask & numeric_series.notna()).sum())
        below_min = int((numeric_series < min_val).sum())
        above_max = int((numeric_series > max_val).sum())
        
        return mask, {
            "violations": violations,
            "below_minimum": below_min,
            "above_maximum": above_max,
            "valid_range": [min_val, max_val]
        }
    except Exception as e:
        return pd.Series([False] * len(df), index=df.index), {"error": str(e)}

def check_allowed(df: pd.DataFrame, column: str, allowed: List[Any]) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check if values are in allowed list."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    mask = df[column].isin(allowed) | df[column].isna()  # Allow nulls to pass
    violations = int((~mask).sum())
    
    # Get sample of invalid values
    invalid_values = df.loc[~mask, column].value_counts().head(5).to_dict()
    
    return mask, {
        "violations": violations,
        "allowed_count": len(allowed),
        "unique_values_found": df[column].nunique(),
        "invalid_values_sample": invalid_values
    }

def check_regex(df: pd.DataFrame, column: str, pattern: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check if values match regex pattern."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Invalid regex pattern: {str(e)}"}
    
    try:
        series = df[column].astype(str).fillna('')
        mask = series.apply(lambda x: bool(regex.search(x))) | df[column].isna()
        violations = int((~mask).sum())
        match_examples = series[mask & df[column].notna()].head(3).tolist()
        
        return mask, {
            "violations": violations,
            "pattern": pattern,
            "match_rate": round((len(df) - violations) / len(df), 4) if len(df) > 0 else 0,
            "match_examples": match_examples
        }
    except Exception as e:
        return pd.Series([False] * len(df), index=df.index), {"error": str(e)}

def check_dtype(df: pd.DataFrame, column: str, dtype: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check if column has expected data type."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    actual_dtype = str(df[column].dtype)
    type_matches = actual_dtype == dtype
    
    # All rows pass or fail based on column type
    mask = pd.Series([type_matches] * len(df), index=df.index)
    
    return mask, {
        "expected_type": dtype,
        "actual_type": actual_dtype,
        "type_matches": type_matches,
        "null_count": int(df[column].isna().sum()),
        "unique_values": df[column].nunique()
    }

def check_custom(df: pd.DataFrame, column: str, func: Callable) -> Tuple[pd.Series, Dict[str, Any]]:
    """Check custom function rule."""
    if df.empty:
        return pd.Series([], dtype=bool), {"warning": "Empty DataFrame"}
    
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {"error": f"Column '{column}' not found"}
    
    try:
        mask = df[column].apply(func)
        violations = int((~mask).sum())
        
        return mask, {
            "violations": violations,
            "function_name": func.__name__ if hasattr(func, '__name__') else 'anonymous',
            "pass_rate": round((len(df) - violations) / len(df), 4) if len(df) > 0 else 0
        }
    except Exception as e:
        return pd.Series([False] * len(df), index=df.index), {"error": str(e)}

# ============================
# ENHANCED QUALITY ENGINE
# ============================

class EnhancedQualityEngine:
    """Enhanced quality rule execution engine with complete Phase 1 integration."""
    
    def __init__(self):
        self.rule_functions = {
            'not_null': check_not_null,
            'not_null_threshold': check_not_null_threshold,
            'unique': check_unique,
            'unique_multi': check_unique_multi,
            'min': check_min,
            'max': check_max,
            'between': check_between,
            'allowed': check_allowed,
            'regex': check_regex,
            'dtype': check_dtype,
            'custom': check_custom,
        }
        
        # Enhanced rule configurations with business context
        self.rule_templates = {
            'completeness': {'type': 'not_null_threshold', 'threshold': 0.95},
            'uniqueness': {'type': 'unique'},
            'email_format': {'type': 'regex', 'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
            'phone_format': {'type': 'regex', 'pattern': r'^(\+\d{1,3})?[\s-]?KATEX_INLINE_OPEN?\d{3}KATEX_INLINE_CLOSE?[\s-]?\d{3}[\s-]?\d{4}$'},
            'positive_values': {'type': 'min', 'min': 0},
            'percentage_range': {'type': 'between', 'min': 0, 'max': 100},
            'date_format': {'type': 'regex', 'pattern': r'^\d{4}-\d{2}-\d{2}$'}
        }
        
        logger.info("EnhancedQualityEngine initialized with %d rule functions", len(self.rule_functions))
    
    def run_rules(self, df: pd.DataFrame, rules: List[Dict[str, Any]], 
                  progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[QualityResult], QualityMetrics]:
        """Execute quality rules and return enhanced results with performance tracking."""
        start_time = time.time()
        results = []
        failed_indices_all = set()
        
        for i, rule in enumerate(rules):
            if progress_callback:
                try:
                    progress_callback(i, len(rules), f"Checking {rule.get('type', 'unknown')}")
                except Exception as e:
                    logger.warning("Progress callback failed: %s", e)
            
            try:
                rule_start_time = time.time()
                result = self.execute_rule(df, rule)
                result.execution_time = time.time() - rule_start_time
                results.append(result)
                
                if not result.passed:
                    failed_indices_all.update(result.failed_indices)
                    
            except Exception as e:
                # Create error result
                error_result = QualityResult(
                    rule_id=rule.get('id', f'error_{i}'),
                    label=f"ERROR: {rule_label(rule)}",
                    passed=False,
                    failed_count=len(df),
                    total_count=len(df),
                    stats={"error": str(e)}
                )
                results.append(error_result)
                logger.error("Error executing rule %s: %s", rule.get('type', 'unknown'), e)
        
        # Calculate enhanced metrics
        metrics = self.calculate_enhanced_metrics(df, results, time.time() - start_time)
        
        logger.info("Quality check completed: %d rules executed, %d passed, %d failed",
                   len(results), metrics.passed_rules, metrics.failed_rules)
        
        return results, metrics
    
    def execute_rule(self, df: pd.DataFrame, rule: Dict[str, Any]) -> QualityResult:
        """Execute a single quality rule with enhanced context."""
        rule_type = rule.get('type', 'unknown')
        params = rule.get('params', {})
        rule_id = rule.get('id', f'rule_{rule_type}_{hashlib.md5(str(rule).encode()).hexdigest()[:6]}')
        
        # Get the check function
        check_function = self.rule_functions.get(rule_type)
        if not check_function:
            return QualityResult(
                rule_id=rule_id,
                label=f"Unknown rule: {rule_type}",
                passed=False,
                failed_count=len(df),
                total_count=len(df),
                stats={"error": f"Unknown rule type: {rule_type}"}
            )
        
        try:
            # Execute the check
            mask, stats = check_function(df, **params)
            
            if "error" in stats:
                return QualityResult(
                    rule_id=rule_id,
                    label=rule_label(rule),
                    passed=False,
                    failed_count=len(df),
                    total_count=len(df),
                    stats=stats
                )
            
            # Calculate results
            failed_mask = (~mask).fillna(True)  # Treat NaN as failed
            failed_indices = df.index[failed_mask].tolist()
            
            result = QualityResult(
                rule_id=rule_id,
                label=rule_label(rule),
                passed=len(failed_indices) == 0,
                failed_count=len(failed_indices),
                total_count=len(df),
                failed_indices=failed_indices[:100],  # Limit for memory
                stats=stats
            )
            
            # Enhanced context
            result.business_impact = self._assess_business_impact(rule_type, result)
            result.recommendation = self._generate_recommendation(rule_type, result, stats)
            result.fix_strategy = self._suggest_fix_strategy(rule_type, result, stats)
            
            return result
            
        except Exception as e:
            logger.error("Error executing rule %s: %s", rule_type, e)
            return QualityResult(
                rule_id=rule_id,
                label=rule_label(rule),
                passed=False,
                failed_count=len(df),
                total_count=len(df),
                stats={"error": str(e)}
            )
    
    def _assess_business_impact(self, rule_type: str, result: QualityResult) -> str:
        """Assess business impact of quality issues."""
        if result.passed:
            return "No impact"
        
        impact_map = {
            'not_null': f"Missing critical data affects {result.failed_count} records",
            'not_null_threshold': f"Incomplete data affects analysis in {result.failed_count} records",
            'unique': f"Duplicate data may cause analysis errors in {result.failed_count} records",
            'unique_multi': f"Non-unique combinations affect data integrity in {result.failed_count} records",
            'min': f"Values below minimum threshold affect {result.failed_count} records",
            'max': f"Values above maximum threshold affect {result.failed_count} records",
            'between': f"Out of range values affect data reliability in {result.failed_count} records",
            'allowed': f"Invalid values affect data quality in {result.failed_count} records",
            'regex': f"Invalid format affects data processing in {result.failed_count} records",
            'dtype': f"Data type mismatch affects {result.failed_count} records",
            'custom': f"Custom validation failed in {result.failed_count} records"
        }
        
        return impact_map.get(rule_type, f"Data quality issue affects {result.failed_count} records")
    
    def _generate_recommendation(self, rule_type: str, result: QualityResult, stats: Dict) -> str:
        """Generate actionable recommendations."""
        if result.passed:
            return "Continue current data quality practices"
        
        recommendations = {
            'not_null': "Implement mandatory field validation or use smart imputation",
            'not_null_threshold': "Improve data collection processes or use imputation techniques",
            'unique': "Add unique constraints or use duplicate removal from cleaning pipeline", 
            'unique_multi': "Review business rules for unique combination requirements",
            'min': "Implement minimum value validation at data entry",
            'max': "Implement maximum value validation at data entry",
            'between': "Review business rules and add input validation",
            'allowed': "Implement allowed values validation at data entry",
            'regex': "Implement format validation at data entry point",
            'dtype': "Add data type validation and conversion in preprocessing",
            'custom': "Review custom validation function and business rules"
        }
        
        return recommendations.get(rule_type, "Review and implement data validation rules")
    
    def _suggest_fix_strategy(self, rule_type: str, result: QualityResult, stats: Dict) -> str:
        """Suggest specific fix strategies using Phase 1 modules."""
        if result.passed:
            return "No action needed"
        
        strategies = {
            'not_null': "Use pipeline.add_step('smart_imputation', {'strategy': 'smart'})",
            'not_null_threshold': "Use pipeline.add_step('smart_imputation', {'strategy': 'auto'})",
            'unique': "Use pipeline.add_step('drop_duplicates', {})",
            'unique_multi': "Use pipeline.add_step('drop_duplicates', {'subset': ['column1', 'column2']})",
            'min': "Use pipeline.add_step('handle_outliers', {'method': 'iqr_cap'})",
            'max': "Use pipeline.add_step('handle_outliers', {'method': 'iqr_cap'})",
            'between': "Use pipeline.add_step('handle_outliers', {'method': 'iqr_cap'})",
            'allowed': "Filter or replace invalid values using custom transformation",
            'regex': "Clean data format before validation using string operations",
            'dtype': "Use pipeline.add_step('optimize_memory', {'aggressive': False})",
            'custom': "Implement data transformation to fix validation issues"
        }
        
        return strategies.get(rule_type, "Apply appropriate data cleaning operations")
    
    def calculate_enhanced_metrics(self, df: pd.DataFrame, results: List[QualityResult], execution_time: float) -> QualityMetrics:
        """Calculate enhanced quality metrics with business intelligence."""
        metrics = QualityMetrics()
        
        # Basic metrics
        metrics.total_rows = len(df)
        metrics.total_columns = len(df.columns)
        metrics.passed_rules = sum(1 for r in results if r.passed)
        metrics.failed_rules = sum(1 for r in results if not r.passed)
        metrics.execution_time = execution_time
        
        if not df.empty:
            metrics.missing_values = int(df.isnull().sum().sum())
            metrics.duplicate_rows = int(df.duplicated().sum())
            metrics.memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Enhanced Phase 1 metrics
        total_cells = len(df) * len(df.columns) if len(df.columns) > 0 else 1
        
        # Data completeness (0-10 scale)
        completeness_pct = ((total_cells - metrics.missing_values) / total_cells) * 100
        metrics.data_completeness = min(10, max(0, completeness_pct / 10))
        
        # Data uniqueness (0-10 scale)  
        uniqueness_pct = ((len(df) - metrics.duplicate_rows) / max(len(df), 1)) * 100
        metrics.data_uniqueness = min(10, max(0, uniqueness_pct / 10))
        
        # Data consistency (based on rule pass rates)
        if results:
            avg_pass_rate = sum(r.pass_rate for r in results) / len(results)
            metrics.data_consistency = avg_pass_rate / 10
        else:
            metrics.data_consistency = 10.0
        
        # Data validity (based on critical rules)
        critical_rules = [r for r in results if r.severity == "critical"]
        if critical_rules:
            critical_pass_rate = sum(1 for r in critical_rules if r.passed) / len(critical_rules) * 100
            metrics.data_validity = critical_pass_rate / 10
        else:
            metrics.data_validity = metrics.data_consistency
        
        # Calculate overall score
        metrics.calculate_enhanced_score()
        
        # Determine business readiness
        metrics.determine_business_readiness()
        
        return metrics

# ============================
# ENHANCED QUALITY REPORTER
# ============================

class EnhancedQualityReporter:
    """Generate comprehensive quality reports with Phase 1 integration."""
    
    def generate_html_report(self, df: pd.DataFrame, rules: List[Dict[str, Any]], 
                           results: List[QualityResult], metrics: QualityMetrics,
                           dataset_name: str = "Dataset", theme: str = "dark") -> str:
        """Generate a comprehensive HTML quality report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate CSS
        css = self.get_report_css(theme)
        
        # Generate sections
        summary_html = self.generate_summary_section(metrics, len(results))
        rules_html = self.generate_rules_table(rules)
        results_html = self.generate_results_table(results)
        recommendations_html = self.generate_enhanced_recommendations(results, metrics)
        overview_html = self.generate_data_overview(df, metrics)
        phase1_html = self.generate_phase1_integration_section(results, metrics)
        
        # Combine into complete HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Quality Report - {html.escape(dataset_name)}</title>
    {css}
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-content">
                <h1>📊 Data Quality Report</h1>
                <div class="header-meta">
                    <div class="dataset-name">{html.escape(dataset_name)}</div>
                    <div class="report-date">{timestamp}</div>
                </div>
            </div>
            <div class="quality-score">
                <div class="score-circle">
                    <span class="score-number">{metrics.overall_score:.1f}</span>
                    <span class="score-label">Quality Score</span>
                </div>
            </div>
        </header>

        <!-- Summary Section -->
        {summary_html}

        <!-- Data Overview -->
        {overview_html}

        <!-- Phase 1 Integration -->
        {phase1_html}

        <!-- Rules Configuration -->
        <section class="card">
            <h2>📋 Quality Rules Configuration</h2>
            {rules_html}
        </section>

        <!-- Results -->
        <section class="card">
            <h2>✅ Quality Check Results</h2>
            {results_html}
        </section>

        <!-- Recommendations -->
        <section class="card">
            <h2>💡 Enhanced Recommendations</h2>
            {recommendations_html}
        </section>

        <!-- Footer -->
        <footer class="footer">
            <div class="footer-content">
                <p>Generated by CortexX Data Quality Engine v3.0</p>
                <p>Execution time: {metrics.execution_time:.2f} seconds</p>
            </div>
        </footer>
    </div>
</body>
</html>"""

        return html_content
    
    def generate_phase1_integration_section(self, results: List[QualityResult], metrics: QualityMetrics) -> str:
        """Generate Phase 1 integration section."""
        # Count issues by fix strategy
        cleaning_fixes = [r for r in results if not r.passed and 'pipeline' in r.fix_strategy]
        critical_issues = [r for r in results if r.severity == "critical"]
        
        return f"""
        <section class="card">
            <h2>🚀 Phase 1 Integration</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-icon">🧹</div>
                    <div class="summary-value">{len(cleaning_fixes)}</div>
                    <div class="summary-label">Issues Fixable by Cleaning Pipeline</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">⚠️</div>
                    <div class="summary-value">{len(critical_issues)}</div>
                    <div class="summary-label">Critical Issues</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">📊</div>
                    <div class="summary-value">{metrics.business_readiness.title()}</div>
                    <div class="summary-label">Business Readiness</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">💾</div>
                    <div class="summary-value">{metrics.memory_usage_mb:.1f} MB</div>
                    <div class="summary-label">Memory Usage</div>
                </div>
            </div>
        </section>"""
    
    def get_report_css(self, theme: str = "dark") -> str:
        """Generate CSS for the enhanced report."""
        return """<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a;
    color: #f8fafc;
    line-height: 1.6;
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.header-meta { color: #cbd5e1; }
.dataset-name { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; }
.report-date { font-size: 0.9rem; color: #94a3b8; }

.quality-score { text-align: center; }

.score-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #10b981);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
}

.score-number { font-size: 1.8rem; font-weight: 700; color: white; }
.score-label { font-size: 0.7rem; color: rgba(255, 255, 255, 0.9); }

.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

.card h2 {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #334155;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
}

.summary-item {
    background: #334155;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    border: 1px solid #475569;
}

.summary-icon { font-size: 2rem; margin-bottom: 8px; }
.summary-value { font-size: 1.5rem; font-weight: 600; color: #3b82f6; margin-bottom: 4px; }
.summary-label { color: #cbd5e1; font-size: 0.9rem; }

.table-wrapper { overflow-x: auto; margin-top: 16px; }

table { width: 100%; border-collapse: collapse; background: #334155; border-radius: 8px; overflow: hidden; }

th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #475569; }

th { background: #475569; font-weight: 600; color: #f8fafc; }
td { color: #e2e8f0; }

.status-pass { color: #10b981; font-weight: 600; }
.status-fail { color: #ef4444; font-weight: 600; }
.severity-critical { color: #ef4444; font-weight: 600; }
.severity-high { color: #f59e0b; font-weight: 600; }
.severity-medium { color: #3b82f6; font-weight: 600; }
.severity-low { color: #10b981; font-weight: 600; }

.recommendations { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }

.recommendation {
    background: #334155;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 16px;
    display: flex;
    align-items: start;
    gap: 12px;
}

.rec-icon { font-size: 1.2rem; margin-top: 2px; }
.rec-text { color: #cbd5e1; font-size: 0.9rem; }

.footer { text-align: center; padding: 24px; color: #94a3b8; border-top: 1px solid #334155; margin-top: 40px; }
.footer p { margin-bottom: 4px; font-size: 0.9rem; }

@media (max-width: 768px) {
    .header { flex-direction: column; gap: 20px; text-align: center; }
    .summary-grid { grid-template-columns: 1fr; }
    .recommendations { grid-template-columns: 1fr; }
}
</style>"""
    
    def generate_summary_section(self, metrics: QualityMetrics, total_rules: int) -> str:
        """Generate enhanced executive summary section."""
        success_rate = (metrics.passed_rules / max(total_rules, 1)) * 100
        
        return f"""
        <section class="card">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-icon">📊</div>
                    <div class="summary-value">{metrics.total_rows:,}</div>
                    <div class="summary-label">Total Rows</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">📋</div>
                    <div class="summary-value">{metrics.total_columns}</div>
                    <div class="summary-label">Total Columns</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">✅</div>
                    <div class="summary-value">{metrics.passed_rules}/{total_rules}</div>
                    <div class="summary-label">Rules Passed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">📈</div>
                    <div class="summary-value">{success_rate:.1f}%</div>
                    <div class="summary-label">Success Rate</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">❓</div>
                    <div class="summary-value">{metrics.missing_values:,}</div>
                    <div class="summary-label">Missing Values</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">🔄</div>
                    <div class="summary-value">{metrics.duplicate_rows:,}</div>
                    <div class="summary-label">Duplicate Rows</div>
                </div>
            </div>
        </section>"""
    
    def generate_data_overview(self, df: pd.DataFrame, metrics: QualityMetrics) -> str:
        """Generate enhanced data overview section."""
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        text_cols = len(df.select_dtypes(include=['object']).columns)
        date_cols = len(df.select_dtypes(include=['datetime64']).columns)
        bool_cols = len(df.select_dtypes(include=['bool']).columns)
        
        return f"""
        <section class="card">
            <h2>🔍 Enhanced Data Overview</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-icon">💾</div>
                    <div class="summary-value">{metrics.memory_usage_mb:.1f} MB</div>
                    <div class="summary-label">Memory Usage</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">🔢</div>
                    <div class="summary-value">{numeric_cols}</div>
                    <div class="summary-label">Numeric Columns</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">📝</div>
                    <div class="summary-value">{text_cols}</div>
                    <div class="summary-label">Text Columns</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">📅</div>
                    <div class="summary-value">{date_cols}</div>
                    <div class="summary-label">Date Columns</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">✅</div>
                    <div class="summary-value">{bool_cols}</div>
                    <div class="summary-label">Boolean Columns</div>
                </div>
                <div class="summary-item">
                    <div class="summary-icon">🎯</div>
                    <div class="summary-value">{metrics.overall_score:.1f}/10</div>
                    <div class="summary-label">Quality Score</div>
                </div>
            </div>
        </section>"""
    
    def generate_rules_table(self, rules: List[Dict[str, Any]]) -> str:
        """Generate enhanced rules configuration table."""
        rows = []
        for i, rule in enumerate(rules, 1):
            rule_type = html.escape(rule.get('type', 'unknown'))
            params = html.escape(str(rule.get('params', {})))
            label = html.escape(rule_label(rule))
            
            rows.append(f"""
            <tr>
                <td>{i}</td>
                <td>{label}</td>
                <td><code>{rule_type}</code></td>
                <td><small>{params}</small></td>
            </tr>""")
        
        return f"""
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Rule Description</th>
                        <th>Type</th>
                        <th>Parameters</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>"""
    
    def generate_results_table(self, results: List[QualityResult]) -> str:
        """Generate enhanced results table with severity indicators."""
        rows = []
        for result in results:
            status_class = "status-pass" if result.passed else "status-fail"
            status_text = "PASS" if result.passed else "FAIL"
            severity_class = f"severity-{result.severity}"
            
            rows.append(f"""
            <tr>
                <td>{html.escape(result.label)}</td>
                <td>{result.failed_count:,}</td>
                <td>{result.pass_rate:.1f}%</td>
                <td class="{status_class}">{status_text}</td>
                <td class="{severity_class}">{result.severity.title()}</td>
                <td><small>{html.escape(result.business_impact[:50])}</small></td>
            </tr>""")
        
        return f"""
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>Rule</th>
                        <th>Failed Rows</th>
                        <th>Pass Rate</th>
                        <th>Status</th>
                        <th>Severity</th>
                        <th>Business Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>"""
    
    def generate_enhanced_recommendations(self, results: List[QualityResult], metrics: QualityMetrics) -> str:
        """Generate enhanced recommendations with Phase 1 integration."""
        recommendations = []
        
        # Quality-based recommendations
        if metrics.overall_score < 7:
            recommendations.append("🔧 Overall data quality needs improvement - consider implementing comprehensive data validation")
        
        if metrics.data_completeness < 8:
            recommendations.append("📊 Use Advanced Data Cleaning Pipeline with smart imputation for missing values")
        
        if metrics.duplicate_rows > 0:
            recommendations.append("🔄 Remove duplicate records using pipeline.add_step('drop_duplicates', {})")
        
        # Rule-specific recommendations from failed rules
        failed_rules = [r for r in results if not r.passed]
        for result in failed_rules[:5]:  # Top 5 failed rules
            if result.recommendation and result.recommendation not in [r.split(': ', 1)[-1] for r in recommendations]:
                icon = "⚠️" if result.severity == "critical" else "💡"
                recommendations.append(f"{icon} {result.recommendation}")
        
        # Integration recommendations
        if metrics.business_readiness == "critical":
            recommendations.append("🚨 Critical data quality issues detected - postpone analysis until resolved")
        elif metrics.business_readiness == "needs_attention":
            recommendations.append("⚠️ Data needs attention before business analysis - use Business Intelligence module cautiously")
        else:
            recommendations.append("✅ Data is ready for Business Intelligence analysis and ML modeling")
        
        # Memory optimization
        if metrics.memory_usage_mb > 100:
            recommendations.append(f"💾 Large dataset ({metrics.memory_usage_mb:.1f}MB) - use memory optimization: pipeline.add_step('optimize_memory')")
        
        if not recommendations:
            recommendations.append("🎉 Excellent data quality! Continue current data management practices")
        
        # Generate HTML
        rec_html = []
        for rec in recommendations[:8]:  # Limit to 8 recommendations
            icon = rec.split(' ', 1)[0]
            text = rec.split(' ', 1)[1] if len(rec.split(' ', 1)) > 1 else rec
            rec_html.append(f"""
            <div class="recommendation">
                <div class="rec-icon">{icon}</div>
                <div class="rec-text">{html.escape(text)}</div>
            </div>""")
        
        return f'<div class="recommendations">{"".join(rec_html)}</div>'

# ============================
# STREAMLIT INTEGRATION
# ============================

def render_quality_dashboard(df: pd.DataFrame, results: List[QualityResult], metrics: QualityMetrics):
    """Render professional interactive quality dashboard in Streamlit."""
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit not available - cannot render dashboard")
        return None
    
    # Set page configuration
    st.set_page_config(
        page_title="Data Quality Dashboard",
        page_icon="🔍",
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
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .success-result {
        border-left-color: #2ecc71;
    }
    .error-result {
        border-left-color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Data Quality Dashboard</h1>', unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "✅ Results", "💡 Recommendations", "📈 Charts"])
    
    with tab1:
        # Quality overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Overall Score", f"{metrics.overall_score:.1f}/10")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Completeness", f"{metrics.data_completeness:.1f}/10")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Uniqueness", f"{metrics.data_uniqueness:.1f}/10")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Business Ready", metrics.business_readiness.title())
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data overview
        st.markdown("### 📈 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", metrics.total_rows)
        with col2:
            st.metric("Total Columns", metrics.total_columns)
        with col3:
            st.metric("Missing Values", metrics.missing_values)
        with col4:
            st.metric("Duplicate Rows", metrics.duplicate_rows)
        
        # Execution info
        st.markdown("### ⚙️ Execution Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Execution Time", f"{metrics.execution_time:.2f}s")
        with col2:
            st.metric("Memory Usage", f"{metrics.memory_usage_mb:.1f} MB")
    
    with tab2:
        # Results summary
        st.markdown("### 📊 Quality Results Summary")
        
        if results:
            results_data = []
            for result in results:
                results_data.append({
                    "Rule": result.label,
                    "Status": "✅ Pass" if result.passed else "❌ Fail",
                    "Failed Count": result.failed_count,
                    "Pass Rate": f"{result.pass_rate:.1f}%",
                    "Severity": result.severity.title(),
                    "Business Impact": result.business_impact[:50] + "..." if len(result.business_impact) > 50 else result.business_impact
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Failed rules details
            failed_rules = [r for r in results if not r.passed]
            if failed_rules:
                st.markdown("### ⚠️ Failed Rules Details")
                
                selected_rule = st.selectbox("Select failed rule for details:", 
                                           [r.label for r in failed_rules])
                
                if selected_rule:
                    rule = next(r for r in failed_rules if r.label == selected_rule)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Rule Details:**")
                        st.write(f"**Failed Count:** {rule.failed_count:,}")
                        st.write(f"**Pass Rate:** {rule.pass_rate:.1f}%")
                        st.write(f"**Severity:** {rule.severity.title()}")
                        st.write(f"**Execution Time:** {rule.execution_time:.2f}s")
                    
                    with col2:
                        st.markdown("**Recommendations:**")
                        st.info(rule.recommendation)
                        if rule.fix_strategy:
                            st.code(rule.fix_strategy)
        else:
            st.info("No quality results available")
    
    with tab3:
        # Quality recommendations
        st.markdown("### 💡 Quality Recommendations")
        
        recommendations = get_quality_recommendations_enhanced(df, results, metrics)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    with tab4:
        # Quality charts
        st.markdown("### 📈 Quality Charts")
        
        if results and PLOTLY_AVAILABLE:
            # Create pass/fail chart
            pass_fail_data = {
                "Status": ["Passed", "Failed"],
                "Count": [metrics.passed_rules, metrics.failed_rules]
            }
            pass_fail_df = pd.DataFrame(pass_fail_data)
            
            fig1 = px.pie(pass_fail_df, values='Count', names='Status', 
                         title='Rule Pass/Fail Distribution',
                         color='Status', color_discrete_map={'Passed':'green', 'Failed':'red'})
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create severity distribution chart
            severity_counts = {}
            for result in results:
                severity = result.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            severity_df = pd.DataFrame({
                "Severity": list(severity_counts.keys()),
                "Count": list(severity_counts.values())
            })
            
            if not severity_df.empty:
                fig2 = px.bar(severity_df, x='Severity', y='Count', 
                             title='Rule Severity Distribution',
                             color='Severity', 
                             color_discrete_map={
                                 'critical': 'red',
                                 'high': 'orange',
                                 'medium': 'blue',
                                 'low': 'green'
                             })
                st.plotly_chart(fig2, use_container_width=True)
        
        elif not PLOTLY_AVAILABLE:
            st.warning("Plotly not available - charts cannot be displayed")

# ============================
# ENHANCED CONVENIENCE FUNCTIONS
# ============================

def run_rules(df: pd.DataFrame, rules: List[Dict[str, Any]], 
              progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Main function to run quality rules (backward compatible)."""
    
    # Use enhanced engine
    engine = EnhancedQualityEngine()
    results, metrics = engine.run_rules(df, rules, progress_callback)
    
    # Convert to legacy format for backward compatibility
    legacy_results = []
    for result in results:
        legacy_results.append({
            "label": result.label,
            "type": result.rule_id.split('_')[0] if '_' in result.rule_id else 'unknown',
            "params": {},  # Simplified
            "passed": result.passed,
            "failed_count": result.failed_count,
            "failed_indices": result.failed_indices,
            "stats": result.stats
        })
    
    # Create summary
    summary = {
        "pass_rate": (metrics.passed_rules / max(len(results), 1)) * 100,
        "failed_rows": len(set().union(*[r.failed_indices for r in results if r.failed_indices])),
        "total_rules": len(results),
        "passed_rules": metrics.passed_rules,
        "failed_rules": metrics.failed_rules,
        "execution_time": metrics.execution_time,
        "overall_score": metrics.overall_score
    }
    
    return legacy_results, summary

def results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate a results summary table (backward compatible)."""
    return pd.DataFrame([
        {
            "Rule": r["label"],
            "Failed Rows": r["failed_count"],
            "Status": "Passed" if r["passed"] else "Failed",
            "Pass Rate": f"{((r.get('total_count', 0) - r['failed_count']) / max(r.get('total_count', 1), 1) * 100):.1f}%"
        }
        for r in results
    ])

def generate_html_report(df: pd.DataFrame, rules: List[Dict[str, Any]], 
                        results: List[Dict[str, Any]], summary: Dict[str, Any], 
                        meta: Dict[str, Any]) -> str:
    """Generate HTML report (backward compatible)."""
    
    # Convert legacy results to enhanced format
    quality_results = []
    for r in results:
        quality_results.append(QualityResult(
            rule_id=r.get('label', 'unknown'),
            label=r['label'],
            passed=r['passed'],
            failed_count=r['failed_count'],
            total_count=len(df),
            failed_indices=r.get('failed_indices', []),
            stats=r.get('stats', {})
        ))
    
    # Create metrics
    metrics = QualityMetrics()
    metrics.total_rows = len(df)
    metrics.total_columns = len(df.columns)
    metrics.passed_rules = summary.get('passed_rules', 0)
    metrics.failed_rules = summary.get('failed_rules', 0)
    metrics.overall_score = summary.get('overall_score', summary.get('pass_rate', 100) / 10)
    metrics.missing_values = int(df.isnull().sum().sum())
    metrics.duplicate_rows = int(df.duplicated().sum())
    metrics.execution_time = summary.get('execution_time', 0)
    
    # Generate report
    reporter = EnhancedQualityReporter()
    return reporter.generate_html_report(
        df=df, rules=rules, results=quality_results, 
        metrics=metrics, dataset_name=meta.get('dataset_name', 'Dataset')
    )

def get_quality_recommendations_enhanced(df: pd.DataFrame, results: List[QualityResult], metrics: QualityMetrics) -> List[str]:
    """Generate enhanced quality recommendations with Phase 1 integration."""
    recommendations = []
    
    # Data quality recommendations
    missing_pct = (df.isnull().sum().sum() / df.size) * 100 if df.size > 0 else 0
    if missing_pct > 10:
        recommendations.append(f"🔧 High missing data ({missing_pct:.1f}%) - Use Advanced Data Cleaning Pipeline with smart imputation")
    
    duplicate_pct = (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
    if duplicate_pct > 1:
        recommendations.append(f"🔄 Found {duplicate_pct:.1f}% duplicate rows - Use cleaning pipeline to remove duplicates")
    
    # Memory optimization
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024 if not df.empty else 0
    if memory_mb > 100:
        recommendations.append(f"💾 Large dataset ({memory_mb:.1f}MB) - Use dtype optimization in Advanced Cleaning")
    
    # Rule-specific recommendations
    failed_rules = [r for r in results if not r.passed]
    if len(failed_rules) / max(len(results), 1) > 0.3:
        recommendations.append("⚠️ High failure rate - Consider using Business Intelligence module for deeper insights into data issues")
    
    # Business readiness
    if metrics.business_readiness == "critical":
        recommendations.append("🚨 Critical data quality issues - Data is not ready for business use")
    elif metrics.business_readiness == "needs_attention":
        recommendations.append("⚠️ Data needs attention - Review and fix quality issues before business analysis")
    else:
        recommendations.append("✅ Data is ready for Business Intelligence analysis and ML modeling")
    
    if not recommendations:
        recommendations.append("🎉 Excellent data quality! Continue current data management practices")
    
    return recommendations

# ============================
# BUSINESS RULE VALIDATION
# ============================

def validate_business_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate business rules with enhanced reporting."""
    engine = EnhancedQualityEngine()
    results, metrics = engine.run_rules(df, rules)
    
    # Categorize results by business impact
    critical_issues = [r for r in results if r.severity == "critical" and not r.passed]
    high_issues = [r for r in results if r.severity == "high" and not r.passed]
    medium_issues = [r for r in results if r.severity == "medium" and not r.passed]
    low_issues = [r for r in results if r.severity == "low" and not r.passed]
    
    # Calculate business impact score
    impact_score = max(0, 100 - (
        len(critical_issues) * 10 + 
        len(high_issues) * 5 + 
        len(medium_issues) * 2 + 
        len(low_issues) * 1
    ))
    
    return {
        "impact_score": impact_score,
        "critical_issues": len(critical_issues),
        "high_issues": len(high_issues),
        "medium_issues": len(medium_issues),
        "low_issues": len(low_issues),
        "all_issues": [r.to_dict() for r in results if not r.passed],
        "business_readiness": "ready" if impact_score >= 80 else "needs_attention" if impact_score >= 60 else "critical",
        "recommendations": get_quality_recommendations_enhanced(df, results, metrics)
    }

# ============================
# EXPORTS
# ============================

__all__ = [
    'EnhancedQualityEngine',
    'EnhancedQualityReporter', 
    'QualityResult',
    'QualityMetrics',
    'run_rules',
    'results_table',
    'generate_html_report',
    'rule_label',
    'get_quality_recommendations_enhanced',
    'render_quality_dashboard',
    'validate_business_rules'
]

# Print module load status
logger.info("Enhanced Data Quality Module v3.0 - Loaded Successfully!")
print("✅ Enhanced Data Quality Module v3.0 - Loaded Successfully!")
print(f"   📊 SciPy Available: {SCIPY_AVAILABLE}")
print(f"   🤖 Scikit-learn Available: {SKLEARN_AVAILABLE}")
print(f"   🎨 Streamlit Available: {STREAMLIT_AVAILABLE}")
print(f"   📈 Plotly Available: {PLOTLY_AVAILABLE}")
print("   🚀 All functions ready for import!")