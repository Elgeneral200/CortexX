"""
Data Preprocessing Module

Comprehensive data cleaning and preprocessing functions including
missing value handling, data type conversion, and intelligent column type detection.

Author: CortexX Team
Version: 1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pandas import DataFrame


def check_missing_values(df: DataFrame, percent: bool = False) -> pd.Series:
    """Check missing values in DataFrame columns."""
    if df.empty:
        return pd.Series(dtype=float)
    return df.isna().mean() * 100 if percent else df.isna().sum()


def process_missing_values(
    df: DataFrame,
    strategy: Literal["drop", "mean", "median", "mode", "constant"],
    columns: Optional[List[str]] = None,
    fill_value: Any = None,
    column_types: Optional[Dict[str, str]] = None,
) -> DataFrame:
    """Process missing values in DataFrame using various strategies."""
    if df.empty:
        return df.copy()

    df_copy = df.copy()
    target_cols = list(columns) if columns else list(df.columns)
    column_types = column_types or {}

    if strategy == "drop":
        return df_copy.dropna(subset=target_cols)

    for col in target_cols:
        if col not in df_copy.columns:
            continue

        col_type = column_types.get(col, "auto")

        if strategy in ["mean", "median"]:
            if pd.api.types.is_numeric_dtype(df_copy[col]) or col_type == "num":
                try:
                    value = df_copy[col].mean() if strategy == "mean" else df_copy[col].median()
                    if not pd.isna(value):
                        df_copy[col] = df_copy[col].fillna(value)
                    else:
                        # Fall back to mode if mean/median is NaN
                        mode_series = df_copy[col].mode()
                        if not mode_series.empty:
                            df_copy[col] = df_copy[col].fillna(mode_series.iloc[0])
                except Exception:
                    mode_series = df_copy[col].mode()
                    if not mode_series.empty:
                        df_copy[col] = df_copy[col].fillna(mode_series.iloc[0])

        elif strategy == "mode":
            mode_series = df_copy[col].mode()
            if not mode_series.empty:
                df_copy[col] = df_copy[col].fillna(mode_series.iloc[0])

        elif strategy == "constant":
            df_copy[col] = df_copy[col].fillna(fill_value)

        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    return df_copy


def convert_data_type(df: DataFrame, column: str, dtype: str) -> DataFrame:
    """Convert a DataFrame column to a specified data type."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    df_copy = df.copy()
    try:
        if dtype == "datetime":
            df_copy[column] = pd.to_datetime(df_copy[column], errors="coerce")
        elif dtype == "int":
            df_copy[column] = pd.to_numeric(df_copy[column], errors="coerce").astype("Int64")
        elif dtype == "float":
            df_copy[column] = pd.to_numeric(df_copy[column], errors="coerce")
        elif dtype == "bool":
            df_copy[column] = df_copy[column].astype("boolean")
        else:
            df_copy[column] = df_copy[column].astype(dtype)
    except Exception as e:
        raise ValueError(f"Conversion error for column '{column}' to {dtype}: {str(e)}") from e

    return df_copy


def detect_column_types(df: DataFrame) -> Dict[str, str]:
    """Automatically detect if columns should be treated as numerical or categorical."""
    if df.empty:
        return {}

    column_types: Dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique(dropna=True)
            unique_ratio = unique_count / len(df[col]) if len(df[col]) else 0.0
            if (unique_ratio < 0.05 and unique_count < 20) or (
                unique_count <= 10 and str(df[col].dtype).startswith("int")
            ):
                column_types[col] = "cat"
            else:
                column_types[col] = "num"
        else:
            column_types[col] = "cat"

    return column_types


def get_column_summary(df: DataFrame, column_types: Dict[str, str]) -> Dict[str, Dict]:
    """Generate comprehensive summary statistics for columns based on their types."""
    if df.empty:
        return {}

    summary: Dict[str, Dict] = {}
    for col, col_type in column_types.items():
        if col not in df.columns:
            continue

        col_summary: Dict[str, Any] = {
            "type": col_type,
            "missing_count": int(df[col].isna().sum()),
            "missing_percent": float((df[col].isna().sum() / len(df)) * 100),
            "unique_count": int(df[col].nunique()),
            "total_count": int(len(df[col])),
        }

        if col_type == "num" and pd.api.types.is_numeric_dtype(df[col]):
            try:
                non_na = ~df[col].isna()
                if non_na.any():
                    col_summary.update(
                        {
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                            "median": float(df[col].median()),
                            "q25": float(df[col].quantile(0.25)),
                            "q75": float(df[col].quantile(0.75)),
                        }
                    )
                else:
                    col_summary.update(
                        {"mean": None, "std": None, "min": None, "max": None, "median": None, "q25": None, "q75": None}
                    )
            except Exception:
                col_summary["type"] = "cat"
                col_summary["top_values"] = df[col].value_counts().head(5).to_dict()
        else:
            try:
                value_counts = df[col].value_counts().head(10)
                col_summary.update(
                    {
                        "top_values": value_counts.to_dict(),
                        "most_frequent": str(value_counts.index[0]) if not value_counts.empty else None,
                        "most_frequent_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                    }
                )
            except Exception:
                col_summary["top_values"] = {}

        summary[col] = col_summary

    return summary


def validate_dataframe(df: DataFrame) -> Dict[str, Any]:
    """Validate DataFrame and return validation results."""
    validation_results: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": [], "info": {}}
    try:
        if df.empty:
            validation_results["errors"].append("DataFrame is empty")
            validation_results["is_valid"] = False

        if len(df.columns) == 0:
            validation_results["errors"].append("DataFrame has no columns")
            validation_results["is_valid"] = False

        if len(df.columns) != len(set(df.columns)):
            validation_results["warnings"].append("DataFrame contains duplicate column names")

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_mb > 500:
            validation_results["warnings"].append(f"Large DataFrame detected ({memory_mb:.1f} MB)")

        validation_results["info"] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "memory_mb": float(memory_mb),
            "missing_values": int(df.isna().sum().sum()),
        }
    except Exception as e:
        validation_results["errors"].append(f"Validation error: {str(e)}")
        validation_results["is_valid"] = False

    return validation_results
