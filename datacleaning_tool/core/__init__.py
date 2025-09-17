"""
Core package for the Data Cleaning Tool.

This package provides:
- file_handler: File reading and validation utilities.
- preprocessing: Data cleaning and type handling utilities.
- visualization: Professional plotting utilities (dark theme).
- pipeline: Pipeline engine for undo/redo and export/import.
- quality: Data quality rule engine and report generation.
"""

from . import file_handler, preprocessing, visualization, pipeline, quality

__all__ = ["file_handler", "preprocessing", "visualization", "pipeline", "quality"]