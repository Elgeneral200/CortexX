# filename: quality.py
"""
Data Quality Module - Fixed Professional Edition

Clean, reliable data quality validation and reporting with:
- Essential quality rules (null, unique, range, pattern validation)
- Simple quality scoring and metrics
- Professional HTML reporting
- Easy integration with other modules
- Production-ready error handling

Author: CortexX Team
Version: 1.2.0 - Fixed Professional Edition
"""

import pandas as pd
import numpy as np
import json
import re
import html
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================
# CORE DATA STRUCTURES
# ============================

class QualityResult:
    """Simple quality check result."""

    def __init__(self, rule_id: str, label: str, passed: bool, failed_count: int, 
                 total_count: int, failed_indices: List[int] = None, stats: Dict = None):
        self.rule_id = rule_id
        self.label = label
        self.passed = passed
        self.failed_count = failed_count
        self.total_count = total_count
        self.failed_indices = failed_indices or []
        self.stats = stats or {}
        self.pass_rate = ((total_count - failed_count) / total_count * 100) if total_count > 0 else 100.0

class QualityMetrics:
    """Simple quality metrics."""

    def __init__(self):
        self.total_rows = 0
        self.total_columns = 0
        self.passed_rules = 0
        self.failed_rules = 0
        self.overall_score = 0.0
        self.execution_time = 0.0
        self.missing_values = 0
        self.duplicate_rows = 0

# ============================
# RULE LABEL GENERATION
# ============================

def rule_label(rule: Dict[str, Any]) -> str:
    """Generate user-friendly labels for quality rules."""
    rule_type = rule.get("type", "unknown")
    params = rule.get("params", {})

    if rule_type == "not_null":
        return f"Not Null: {params.get('column', 'N/A')}"
    elif rule_type == "not_null_threshold":
        threshold = params.get('threshold', 1.0) * 100
        return f"Completeness ≥{threshold:.0f}%: {params.get('column', 'N/A')}"
    elif rule_type == "unique":
        return f"Unique Values: {params.get('column', 'N/A')}"
    elif rule_type == "unique_multi":
        cols = params.get('columns', [])
        return f"Unique Combination: {', '.join(cols)}"
    elif rule_type == "min":
        return f"Min Value ≥{params.get('min')}: {params.get('column', 'N/A')}"
    elif rule_type == "max":
        return f"Max Value ≤{params.get('max')}: {params.get('column', 'N/A')}"
    elif rule_type == "between":
        return f"Range [{params.get('min')}, {params.get('max')}]: {params.get('column', 'N/A')}"
    elif rule_type == "allowed":
        allowed_count = len(params.get('allowed', []))
        return f"Allowed Values ({allowed_count} items): {params.get('column', 'N/A')}"
    elif rule_type == "regex":
        pattern = params.get('pattern', '')[:15] + ('...' if len(params.get('pattern', '')) > 15 else '')
        return f"Pattern Match '{pattern}': {params.get('column', 'N/A')}"
    elif rule_type == "dtype":
        return f"Data Type = {params.get('dtype')}: {params.get('column', 'N/A')}"
    else:
        return f"Custom Rule: {rule_type}"

# ============================
# QUALITY CHECK FUNCTIONS
# ============================

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
        numeric_series = pd.to_numeric(df[column], errors="coerce")
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
        numeric_series = pd.to_numeric(df[column], errors="coerce")
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
        numeric_series = pd.to_numeric(df[column], errors="coerce")
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
        series = df[column].astype(str).fillna("")
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

# ============================
# RULE EXECUTION ENGINE
# ============================

class QualityEngine:
    """Simple, reliable quality rule execution engine."""

    def __init__(self):
        self.rule_functions = {
            "not_null": check_not_null,
            "not_null_threshold": check_not_null_threshold,
            "unique": check_unique,
            "unique_multi": check_unique_multi,
            "min": check_min,
            "max": check_max,
            "between": check_between,
            "allowed": check_allowed,
            "regex": check_regex,
            "dtype": check_dtype
        }

    def run_rules(
        self, 
        df: pd.DataFrame, 
        rules: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[List[QualityResult], QualityMetrics]:
        """Execute quality rules and return results."""
        start_time = time.time()

        results = []
        failed_indices_all = set()

        for i, rule in enumerate(rules):
            if progress_callback:
                try:
                    progress_callback(i, len(rules), f"Checking: {rule.get('type', 'unknown')}")
                except Exception:
                    pass  # Don't let callback errors stop execution

            try:
                result = self._execute_rule(df, rule)
                results.append(result)

                if not result.passed:
                    failed_indices_all.update(result.failed_indices)

            except Exception as e:
                # Create error result
                error_result = QualityResult(
                    rule_id=rule.get("id", f"error_{i}"),
                    label=f"ERROR: {rule_label(rule)}",
                    passed=False,
                    failed_count=len(df),
                    total_count=len(df),
                    stats={"error": str(e)}
                )
                results.append(error_result)

        # Calculate metrics
        metrics = self._calculate_metrics(df, results, time.time() - start_time)

        return results, metrics

    def _execute_rule(self, df: pd.DataFrame, rule: Dict[str, Any]) -> QualityResult:
        """Execute a single quality rule."""
        rule_type = rule.get("type", "unknown")
        params = rule.get("params", {})
        rule_id = rule.get("id", f"rule_{rule_type}")

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

            # Process results
            if "error" in stats:
                return QualityResult(
                    rule_id=rule_id,
                    label=rule_label(rule),
                    passed=False,
                    failed_count=len(df),
                    total_count=len(df),
                    stats=stats
                )

            failed_mask = (~mask).fillna(True)
            failed_indices = df.index[failed_mask].tolist()

            return QualityResult(
                rule_id=rule_id,
                label=rule_label(rule),
                passed=len(failed_indices) == 0,
                failed_count=len(failed_indices),
                total_count=len(df),
                failed_indices=failed_indices[:100],  # Limit for memory
                stats=stats
            )

        except Exception as e:
            return QualityResult(
                rule_id=rule_id,
                label=rule_label(rule),
                passed=False,
                failed_count=len(df),
                total_count=len(df),
                stats={"error": str(e)}
            )

    def _calculate_metrics(self, df: pd.DataFrame, results: List[QualityResult], execution_time: float) -> QualityMetrics:
        """Calculate quality metrics from results."""
        metrics = QualityMetrics()

        metrics.total_rows = len(df)
        metrics.total_columns = len(df.columns)
        metrics.passed_rules = sum(1 for r in results if r.passed)
        metrics.failed_rules = sum(1 for r in results if not r.passed)
        metrics.execution_time = execution_time

        # Calculate overall score based on pass rates
        if results:
            avg_pass_rate = sum(r.pass_rate for r in results) / len(results)
            metrics.overall_score = round(avg_pass_rate / 10, 1)  # Convert to 0-10 scale
        else:
            metrics.overall_score = 10.0

        # Basic data quality metrics
        if not df.empty:
            metrics.missing_values = int(df.isnull().sum().sum())
            metrics.duplicate_rows = int(df.duplicated().sum())

        return metrics

# ============================
# HTML REPORT GENERATION
# ============================

class QualityReporter:
    """Generate professional HTML quality reports."""

    def generate_html_report(
        self,
        df: pd.DataFrame,
        rules: List[Dict[str, Any]],
        results: List[QualityResult],
        metrics: QualityMetrics,
        dataset_name: str = "Dataset",
        theme: str = "dark"
    ) -> str:
        """Generate a professional HTML quality report."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate CSS
        css = self._get_report_css(theme)

        # Generate summary statistics
        summary_html = self._generate_summary_section(metrics, len(results))

        # Generate rules table
        rules_html = self._generate_rules_table(rules)

        # Generate results table
        results_html = self._generate_results_table(results)

        # Generate recommendations
        recommendations_html = self._generate_recommendations(results, metrics)

        # Generate data overview
        overview_html = self._generate_data_overview(df, metrics)

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

        <!-- Rules Configuration -->
        <section class="card">
            <h2>📋 Quality Rules Configuration</h2>
            {rules_html}
        </section>

        <!-- Results -->
        <section class="card">
            <h2>🔍 Quality Check Results</h2>
            {results_html}
        </section>

        <!-- Recommendations -->
        <section class="card">
            <h2>💡 Recommendations</h2>
            {recommendations_html}
        </section>

        <!-- Footer -->
        <footer class="footer">
            <div class="footer-content">
                <p>Generated by CortexX Data Quality Engine</p>
                <p>Execution time: {metrics.execution_time:.2f} seconds</p>
            </div>
        </footer>
    </div>
</body>
</html>"""

        return html_content

    def _get_report_css(self, theme: str = "dark") -> str:
        """Generate CSS for the report."""
        return """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

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

            .header-meta {
                color: #cbd5e1;
            }

            .dataset-name {
                font-size: 1.2rem;
                font-weight: 600;
                color: #e2e8f0;
            }

            .report-date {
                font-size: 0.9rem;
                color: #94a3b8;
            }

            .quality-score {
                text-align: center;
            }

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

            .score-number {
                font-size: 1.8rem;
                font-weight: 700;
                color: white;
            }

            .score-label {
                font-size: 0.7rem;
                color: rgba(255, 255, 255, 0.9);
            }

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

            .summary-icon {
                font-size: 2rem;
                margin-bottom: 8px;
            }

            .summary-value {
                font-size: 1.5rem;
                font-weight: 600;
                color: #3b82f6;
                margin-bottom: 4px;
            }

            .summary-label {
                color: #cbd5e1;
                font-size: 0.9rem;
            }

            .table-wrapper {
                overflow-x: auto;
                margin-top: 16px;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                background: #334155;
                border-radius: 8px;
                overflow: hidden;
            }

            th, td {
                padding: 12px 16px;
                text-align: left;
                border-bottom: 1px solid #475569;
            }

            th {
                background: #475569;
                font-weight: 600;
                color: #f8fafc;
            }

            td {
                color: #e2e8f0;
            }

            .status-pass {
                color: #10b981;
                font-weight: 600;
            }

            .status-fail {
                color: #ef4444;
                font-weight: 600;
            }

            .recommendations {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 16px;
            }

            .recommendation {
                background: #334155;
                border: 1px solid #475569;
                border-radius: 8px;
                padding: 16px;
                display: flex;
                align-items: start;
                gap: 12px;
            }

            .rec-icon {
                font-size: 1.2rem;
                margin-top: 2px;
            }

            .rec-text {
                color: #cbd5e1;
                font-size: 0.9rem;
            }

            .footer {
                text-align: center;
                padding: 24px;
                color: #94a3b8;
                border-top: 1px solid #334155;
                margin-top: 40px;
            }

            .footer p {
                margin-bottom: 4px;
                font-size: 0.9rem;
            }

            @media (max-width: 768px) {
                .header {
                    flex-direction: column;
                    gap: 20px;
                    text-align: center;
                }

                .summary-grid {
                    grid-template-columns: 1fr;
                }

                .recommendations {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """

    def _generate_summary_section(self, metrics: QualityMetrics, total_rules: int) -> str:
        """Generate executive summary section."""
        success_rate = (metrics.passed_rules / total_rules * 100) if total_rules > 0 else 100

        return f"""
        <section class="card">
            <h2>📈 Executive Summary</h2>
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
                    <div class="summary-icon">📊</div>
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
        </section>
        """

    def _generate_data_overview(self, df: pd.DataFrame, metrics: QualityMetrics) -> str:
        """Generate data overview section."""
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        text_cols = len(df.select_dtypes(include=['object']).columns)

        return f"""
        <section class="card">
            <h2>📊 Data Overview</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-icon">💾</div>
                    <div class="summary-value">{memory_mb:.1f} MB</div>
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
                    <div class="summary-icon">⭐</div>
                    <div class="summary-value">{metrics.overall_score:.1f}/10</div>
                    <div class="summary-label">Quality Score</div>
                </div>
            </div>
        </section>
        """

    def _generate_rules_table(self, rules: List[Dict[str, Any]]) -> str:
        """Generate rules configuration table."""
        rows = []
        for i, rule in enumerate(rules, 1):
            rule_type = html.escape(rule.get("type", "unknown"))
            params = html.escape(str(rule.get("params", {})))
            label = html.escape(rule_label(rule))

            rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td>{label}</td>
                    <td><code>{rule_type}</code></td>
                    <td><small>{params}</small></td>
                </tr>
            """)

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
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _generate_results_table(self, results: List[QualityResult]) -> str:
        """Generate results table."""
        rows = []
        for result in results:
            status_class = "status-pass" if result.passed else "status-fail"
            status_text = "✅ PASS" if result.passed else "❌ FAIL"

            rows.append(f"""
                <tr>
                    <td>{html.escape(result.label)}</td>
                    <td>{result.failed_count:,}</td>
                    <td>{result.pass_rate:.1f}%</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
            """)

        return f"""
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>Rule</th>
                        <th>Failed Rows</th>
                        <th>Pass Rate</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _generate_recommendations(self, results: List[QualityResult], metrics: QualityMetrics) -> str:
        """Generate recommendations based on results."""
        recommendations = []

        # General recommendations based on overall quality
        if metrics.overall_score < 7:
            recommendations.append("🔧 Overall data quality needs improvement - consider implementing data validation at source")

        if metrics.missing_values > metrics.total_rows * 0.1:
            recommendations.append("📝 High number of missing values - review data collection processes")

        if metrics.duplicate_rows > 0:
            recommendations.append("🔄 Remove duplicate rows to improve data quality")

        # Rule-specific recommendations
        failed_rules = [r for r in results if not r.passed]
        if len(failed_rules) > len(results) * 0.5:
            recommendations.append("⚠️ More than 50% of quality rules failed - comprehensive data quality program needed")

        # Add rule-specific suggestions
        for result in failed_rules[:3]:  # Top 3 failed rules
            if "not_null" in result.label.lower():
                recommendations.append(f"💡 {result.label}: Implement mandatory field validation or data imputation")
            elif "unique" in result.label.lower():
                recommendations.append(f"🔑 {result.label}: Add unique constraints or remove duplicates")
            elif "range" in result.label.lower() or "between" in result.label.lower():
                recommendations.append(f"📊 {result.label}: Review business rules and add input validation")

        if not recommendations:
            recommendations.append("✅ Data quality is excellent! Continue current data management practices")

        # Generate HTML
        rec_html = []
        for rec in recommendations[:8]:  # Limit to 8 recommendations
            rec_html.append(f"""
                <div class="recommendation">
                    <div class="rec-icon">💡</div>
                    <div class="rec-text">{html.escape(rec)}</div>
                </div>
            """)

        return f'<div class="recommendations">{"".join(rec_html)}</div>'

# ============================
# CONVENIENCE FUNCTIONS
# ============================

def run_rules(
    df: pd.DataFrame, 
    rules: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main function to run quality rules (backward compatible).

    Returns:
        - List of rule results in dictionary format
        - Summary statistics
    """
    engine = QualityEngine()
    results, metrics = engine.run_rules(df, rules, progress_callback)

    # Convert to legacy format
    legacy_results = []
    for result in results:
        legacy_results.append({
            "label": result.label,
            "type": result.rule_id.split("_")[0] if "_" in result.rule_id else "unknown",
            "params": {},  # Simplified
            "passed": result.passed,
            "failed_count": result.failed_count,
            "failed_indices": result.failed_indices,
            "stats": result.stats
        })

    # Generate summary
    all_failed_indices = set()
    for result in results:
        all_failed_indices.update(result.failed_indices)

    summary = {
        "pass_rate": (metrics.passed_rules / len(results) * 100) if results else 100.0,
        "failed_rows": len(all_failed_indices),
        "total_rules": len(results),
        "passed_rules": metrics.passed_rules,
        "failed_rules": metrics.failed_rules
    }

    return legacy_results, summary

def results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate a results summary table."""
    return pd.DataFrame([
        {
            "Rule": r["label"],
            "Failed Rows": r["failed_count"],
            "Passed": "✅" if r["passed"] else "❌",
            "Pass Rate": f"{((r.get('total_count', r.get('failed_count', 0)) - r.get('failed_count', 0)) / max(r.get('total_count', r.get('failed_count', 0)), 1) * 100):.1f}%"
        }
        for r in results
    ])

def generate_html_report(
    df: pd.DataFrame,
    rules: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """Generate HTML report (backward compatible)."""
    # Convert legacy results back to QualityResult objects
    quality_results = []
    for r in results:
        quality_results.append(QualityResult(
            rule_id=r.get("label", "unknown"),
            label=r["label"],
            passed=r["passed"],
            failed_count=r["failed_count"],
            total_count=len(df),
            failed_indices=r.get("failed_indices", []),
            stats=r.get("stats", {})
        ))

    # Create metrics
    metrics = QualityMetrics()
    metrics.total_rows = len(df)
    metrics.total_columns = len(df.columns)
    metrics.passed_rules = summary.get("passed_rules", 0)
    metrics.failed_rules = summary.get("failed_rules", 0)
    metrics.overall_score = summary.get("pass_rate", 100) / 10
    metrics.missing_values = int(df.isnull().sum().sum())
    metrics.duplicate_rows = int(df.duplicated().sum())

    # Generate report
    reporter = QualityReporter()
    return reporter.generate_html_report(
        df=df,
        rules=rules,
        results=quality_results,
        metrics=metrics,
        dataset_name=meta.get("dataset_name", "Dataset")
    )

# ============================
# EXPORTS
# ============================

__all__ = [
    'QualityEngine',
    'QualityReporter',
    'QualityResult',
    'QualityMetrics',
    'run_rules',
    'results_table',
    'generate_html_report',
    'rule_label'
]
