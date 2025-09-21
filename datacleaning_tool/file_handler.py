# filename: file_handler.py
"""
Enhanced File Handling Module - Professional Platform Edition v3.0

A comprehensive data ingestion and processing system designed for enterprise use.
Features advanced format support, robust error handling, performance optimization,
and seamless integration with data quality and pipeline systems.

Author: CortexX Team  
Version: 3.0.0 - Professional Platform Edition
License: MIT
"""

from __future__ import annotations
import os
import sqlite3
import tempfile
import json
import warnings
import zipfile
import gzip
import logging
import io
import re
from pathlib import Path
from typing import IO, Union, Dict, List, Optional, Any, Tuple, Callable, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from contextlib import contextmanager
from urllib.parse import urlparse
import time
import hashlib
import mimetypes
import chardet
from enum import Enum, auto

import pandas as pd
import numpy as np
from pandas import DataFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    ET = None
    XML_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import xlrd
    XLRD_AVAILABLE = True  
except ImportError:
    XLRD_AVAILABLE = False

try:
    import pyarrow
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import fastavro
    AVRO_AVAILABLE = True
except ImportError:
    AVRO_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ============================
# ENUMS AND CONSTANTS
# ============================

class FileFormat(Enum):
    """Supported file formats."""
    CSV = auto()
    EXCEL = auto()
    JSON = auto()
    JSONL = auto()
    PARQUET = auto()
    SQLITE = auto()
    XML = auto()
    AVRO = auto()
    YAML = auto()
    ZIP = auto()
    GZIP = auto()
    UNKNOWN = auto()

class ProcessingMode(Enum):
    """Processing modes for different scenarios."""
    STANDARD = auto()
    MEMORY_EFFICIENT = auto()
    HIGH_PERFORMANCE = auto()
    STREAMING = auto()

# Global constants
DEFAULT_CHUNK_SIZE = 10000
MAX_FILE_SIZE_MB = 1024  # 1GB default limit
SUPPORTED_ENCODINGS = ['utf-8', 'iso-8859-1', 'latin-1', 'cp1252', 'utf-16', 'ascii']

# ============================
# ENHANCED DATA STRUCTURES
# ============================

@dataclass
class FileMetadata:
    """Comprehensive file metadata structure with enhanced fields."""
    filepath: str
    filename: str
    filesize: int
    filetype: str
    mimetype: str
    encoding: str
    last_modified: str
    checksum: str
    
    # Content metadata
    rows: int = 0
    columns: int = 0
    memory_usage_mb: float = 0.0
    missing_values: int = 0
    data_quality_score: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    processing_errors: List[str] = field(default_factory=list)
    processing_warnings: List[str] = field(default_factory=list)
    
    # Schema information
    column_types: Dict[str, str] = field(default_factory=dict)
    column_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced metadata
    sample_data: Dict[str, List[Any]] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    anomalies: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize metadata to JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

@dataclass
class DataIngestionResult:
    """Result of data ingestion process with enhanced reporting."""
    success: bool
    dataframe: Optional[DataFrame]
    metadata: FileMetadata
    quality_report: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    processing_mode: ProcessingMode = ProcessingMode.STANDARD
    
    def to_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the ingestion result."""
        summary = {
            "success": self.success,
            "data_shape": (self.metadata.rows, self.metadata.columns) if self.dataframe is not None else (0, 0),
            "file_info": {
                "name": self.metadata.filename,
                "size_mb": round(self.metadata.filesize / 1024 / 1024, 2),
                "type": self.metadata.filetype,
                "checksum": self.metadata.checksum
            },
            "quality_score": round(self.metadata.data_quality_score, 2),
            "processing_time": round(self.metadata.processing_time, 2),
            "recommendations_count": len(self.recommendations),
            "has_issues": len(self.metadata.processing_errors) > 0 or len(self.metadata.processing_warnings) > 0,
            "processing_mode": self.processing_mode.name
        }
        
        if self.quality_report:
            summary["quality_metrics"] = {
                "completeness": self.quality_report.get('completeness', {}).get('score', 0),
                "uniqueness": self.quality_report.get('uniqueness', {}).get('score', 0),
                "consistency": self.quality_report.get('consistency', {}).get('score', 0)
            }
            
        return summary

# ============================
# UTILITY FUNCTIONS
# ============================

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance with detailed metrics."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = os.getpid().memory_info().rss if hasattr(os, 'getpid') else 0
        
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Calculate memory usage
            end_memory = os.getpid().memory_info().rss if hasattr(os, 'getpid') else 0
            memory_used_mb = (end_memory - start_memory) / 1024 / 1024
            
            # Add performance data to result if it's a DataIngestionResult
            if isinstance(result, DataIngestionResult):
                result.metadata.processing_time = processing_time
                if not hasattr(result.metadata, 'memory_usage_mb'):
                    result.metadata.memory_usage_mb = memory_used_mb
            
            logger.debug(f"Function {func.__name__} completed in {processing_time:.2f}s, memory: {memory_used_mb:.2f}MB")
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {processing_time:.2f}s: {str(e)}")
            raise
    return wrapper

@contextmanager
def safe_file_processing(filepath: str):
    """Context manager for safe file processing with cleanup and error handling."""
    temp_files = []
    try:
        yield temp_files
    except Exception as e:
        logger.error(f"Error during file processing: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {str(e)}")

def detect_encoding(file_source: Union[str, IO]) -> str:
    """Detect file encoding using chardet with fallback."""
    try:
        if isinstance(file_source, str):
            with open(file_source, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
        else:
            current_pos = file_source.tell()
            file_source.seek(0)
            raw_data = file_source.read(10000)
            file_source.seek(current_pos)
            
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'
    except Exception:
        return 'utf-8'  # Default fallback

def validate_file_path(filepath: str) -> bool:
    """Validate file path for security and correctness."""
    if not isinstance(filepath, str):
        return False
        
    # Check for path traversal attempts
    if '../' in filepath or '..\\' in filepath:
        return False
        
    # Check if file exists and is readable
    if not os.path.exists(filepath):
        return False
        
    if not os.path.isfile(filepath):
        return False
        
    return True

def sanitize_column_names(df: DataFrame) -> DataFrame:
    """Sanitize column names for better compatibility."""
    if df is None or df.empty:
        return df
        
    new_columns = []
    for col in df.columns:
        # Convert to string if not already
        col_str = str(col)
        
        # Replace special characters and spaces
        col_clean = re.sub(r'[^\w]', '_', col_str)
        
        # Ensure it starts with a letter or underscore
        if not col_clean[0].isalpha() and col_clean[0] != '_':
            col_clean = '_' + col_clean
            
        # Ensure uniqueness
        if col_clean in new_columns:
            counter = 1
            while f"{col_clean}_{counter}" in new_columns:
                counter += 1
            col_clean = f"{col_clean}_{counter}"
            
        new_columns.append(col_clean)
    
    df.columns = new_columns
    return df

# ============================
# ENHANCED FILE PROCESSOR
# ============================

class EnhancedFileProcessor:
    """Advanced file processing with comprehensive capabilities for professional use."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize supported formats mapping
        self.supported_formats = self._initialize_supported_formats()
        
        # Configuration
        self.encoding_detection = self.config.get('encoding_detection', True)
        self.chunk_size = self.config.get('chunk_size', DEFAULT_CHUNK_SIZE)
        self.max_file_size_mb = self.config.get('max_file_size_mb', MAX_FILE_SIZE_MB)
        self.auto_sanitize_columns = self.config.get('auto_sanitize_columns', True)
        
        # Performance tracking
        self.processing_stats = {
            'files_processed': 0,
            'total_rows_processed': 0,
            'total_processing_time': 0.0
        }
    
    def _initialize_supported_formats(self) -> Dict[str, Callable]:
        """Initialize the mapping of supported file formats to their processors."""
        formats = {
            '.csv': self._read_csv_advanced,
            '.tsv': self._read_csv_advanced,
            '.txt': self._read_text_file,
            '.xlsx': self._read_excel_advanced,
            '.xls': self._read_excel_advanced,
            '.xlsm': self._read_excel_advanced,
            '.json': self._read_json_advanced,
            '.jsonl': self._read_json_lines,
            '.db': self._read_sqlite_advanced,
            '.sqlite': self._read_sqlite_advanced,
            '.sqlite3': self._read_sqlite_advanced,
            '.xml': self._read_xml,
            '.gz': self._read_compressed,
            '.gzip': self._read_compressed,
            '.zip': self._read_zip_archive,
            '.yaml': self._read_yaml,
            '.yml': self._read_yaml,
        }
        
        # Add parquet support if available
        if PARQUET_AVAILABLE:
            formats['.parquet'] = self._read_parquet
            
        # Add avro support if available
        if AVRO_AVAILABLE:
            formats['.avro'] = self._read_avro
            
        return formats
    
    @performance_monitor
    def process_file(self, filepath_or_buffer: Union[str, IO], filename: Optional[str] = None, 
                    processing_mode: ProcessingMode = ProcessingMode.STANDARD, **kwargs) -> DataIngestionResult:
        """Process any supported file format with comprehensive analysis."""
        start_time = time.time()
        
        # Determine file path and name
        if isinstance(filepath_or_buffer, str):
            filepath = filepath_or_buffer
            filename = filename or os.path.basename(filepath)
            
            # Validate file path
            if not validate_file_path(filepath):
                error_msg = f"Invalid file path: {filepath}"
                logger.error(error_msg)
                metadata = self._create_error_metadata(filepath, filename, error_msg)
                return DataIngestionResult(
                    success=False,
                    dataframe=None,
                    metadata=metadata,
                    processing_mode=processing_mode
                )
        else:
            filepath = getattr(filepath_or_buffer, 'name', 'uploaded_file')
            filename = filename or 'uploaded_file'
        
        # Initialize metadata
        metadata = self._extract_file_metadata(filepath_or_buffer, filename)
        
        try:
            # Determine file type and processor
            file_extension = Path(filename).suffix.lower()
            if file_extension not in self.supported_formats:
                # Try to detect format by content
                file_extension = self._detect_file_format(filepath_or_buffer)
                if file_extension not in self.supported_formats:
                    raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Process the file based on processing mode
            processor = self.supported_formats[file_extension]
            
            if processing_mode == ProcessingMode.MEMORY_EFFICIENT and file_extension in ['.csv', '.txt', '.tsv']:
                dataframe = self._read_csv_chunked(filepath_or_buffer, **kwargs)
            elif processing_mode == ProcessingMode.STREAMING and file_extension in ['.csv', '.jsonl', '.txt']:
                # For streaming, we process in chunks but return the first chunk as sample
                dataframe = self._read_first_chunk(filepath_or_buffer, **kwargs)
                metadata.processing_warnings.append("Streaming mode: Only first chunk returned as sample")
            else:
                dataframe = processor(filepath_or_buffer, **kwargs)
            
            # Sanitize column names if enabled
            if self.auto_sanitize_columns and dataframe is not None:
                dataframe = sanitize_column_names(dataframe)
            
            # Update metadata with data information
            self._update_data_metadata(metadata, dataframe)
            
            # Perform data quality assessment
            quality_report = self._assess_data_quality(dataframe)
            metadata.data_quality_score = quality_report.get('overall_score', 0)
            
            # Extract sample data
            self._extract_sample_data(metadata, dataframe)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metadata, quality_report, processing_mode)
            
            # Update processing statistics
            self._update_processing_stats(metadata, start_time)
            
            return DataIngestionResult(
                success=True,
                dataframe=dataframe,
                metadata=metadata,
                quality_report=quality_report,
                recommendations=recommendations,
                processing_mode=processing_mode
            )
            
        except Exception as e:
            error_msg = f"Failed to process file {filename}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            metadata.processing_errors.append(error_msg)
            
            # Update processing statistics even for failures
            self.processing_stats['files_processed'] += 1
            self.processing_stats['total_processing_time'] += (time.time() - start_time)
            
            return DataIngestionResult(
                success=False,
                dataframe=None,
                metadata=metadata,
                processing_mode=processing_mode,
                recommendations=[error_msg]
            )
    
    def _create_error_metadata(self, filepath: str, filename: str, error: str) -> FileMetadata:
        """Create metadata for error cases."""
        return FileMetadata(
            filepath=filepath,
            filename=filename,
            filesize=0,
            filetype=Path(filename).suffix.lower(),
            mimetype='unknown',
            encoding='unknown',
            last_modified=datetime.now(timezone.utc).isoformat(),
            checksum='unknown',
            processing_errors=[error]
        )
    
    def _extract_file_metadata(self, file_source: Union[str, IO], filename: str) -> FileMetadata:
        """Extract comprehensive file metadata with enhanced detection."""
        try:
            if isinstance(file_source, str):
                # File path
                filepath = file_source
                file_stats = os.stat(filepath)
                filesize = file_stats.st_size
                last_modified = datetime.fromtimestamp(file_stats.st_mtime, timezone.utc).isoformat()
                checksum = self._calculate_checksum(filepath)
                
                # Detect MIME type and encoding
                mimetype, encoding = mimetypes.guess_type(filepath)
                mimetype = mimetype or 'application/octet-stream'
                
                if self.encoding_detection and encoding is None:
                    encoding = detect_encoding(filepath)
            else:
                # File-like object
                filepath = getattr(file_source, 'name', 'uploaded_file')
                
                # Get size
                if hasattr(file_source, 'size'):
                    filesize = file_source.size
                else:
                    # Try to determine size
                    current_pos = file_source.tell() if hasattr(file_source, 'tell') else 0
                    try:
                        file_source.seek(0, 2)  # Seek to end
                        filesize = file_source.tell()
                        file_source.seek(current_pos)  # Seek back
                    except:
                        filesize = 0
                        
                last_modified = datetime.now(timezone.utc).isoformat()
                checksum = 'unknown'
                mimetype = getattr(file_source, 'content_type', 'application/octet-stream')
                encoding = getattr(file_source, 'encoding', 'unknown')
            
            return FileMetadata(
                filepath=filepath,
                filename=filename,
                filesize=filesize,
                filetype=Path(filename).suffix.lower(),
                mimetype=mimetype,
                encoding=encoding or 'unknown',
                last_modified=last_modified,
                checksum=checksum
            )
        except Exception as e:
            logger.warning(f"Error extracting file metadata: {str(e)}")
            return FileMetadata(
                filepath=file_source if isinstance(file_source, str) else 'unknown',
                filename=filename,
                filesize=0,
                filetype='unknown',
                mimetype='unknown',
                encoding='unknown',
                last_modified=datetime.now(timezone.utc).isoformat(),
                checksum='unknown',
                processing_warnings=[f"Metadata extraction incomplete: {str(e)}"]
            )
    
    def _calculate_checksum(self, filepath: str, algorithm: str = 'md5') -> str:
        """Calculate file checksum using specified algorithm."""
        try:
            hash_func = hashlib.new(algorithm)
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating checksum: {str(e)}")
            return 'unknown'
    
    def _detect_file_format(self, file_source: Union[str, IO]) -> str:
        """Detect file format by content analysis with enhanced detection."""
        try:
            if isinstance(file_source, str):
                with open(file_source, 'rb') as f:
                    header = f.read(512)
            else:
                current_pos = file_source.tell() if hasattr(file_source, 'tell') else 0
                header = file_source.read(512)
                if hasattr(file_source, 'seek'):
                    file_source.seek(current_pos)
            
            # Enhanced format detection
            if header.startswith(b'PK'):
                if b'xl/' in header or b'[Content_Types].xml' in header:
                    return '.xlsx'
                elif b'word/' in header:
                    return '.docx'
                else:
                    return '.zip'
            elif header.startswith(b'\x50\x4b\x03\x04'):
                return '.xlsx'
            elif header.startswith((b'{', b'[')):
                return '.json'
            elif header.startswith(b'<?xml'):
                return '.xml'
            elif header.startswith(b'SQLite format 3'):
                return '.db'
            elif header.startswith(b'\x1f\x8b'):
                return '.gz'
            elif header.startswith(b'PAR1'):
                return '.parquet'
            elif header.startswith(b'Obj\x01'):
                return '.avro'
            elif b',' in header[:100] and b'\n' in header[:100]:
                return '.csv'
            elif b'\t' in header[:100] and b'\n' in header[:100]:
                return '.tsv'
            else:
                # Try to decode as text
                try:
                    text_sample = header.decode('utf-8', errors='ignore')
                    if any(char in text_sample for char in [',', ';', '\t']):
                        return '.csv'
                    elif '---' in text_sample and ('yaml' in text_sample.lower() or 'yml' in text_sample.lower()):
                        return '.yaml'
                except:
                    pass
                    
            return '.unknown'
        except Exception as e:
            logger.warning(f"Error detecting file format: {str(e)}")
            return '.unknown'
    
    def _read_csv_advanced(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced CSV reading with encoding detection and error handling."""
        # Default parameters optimized for professional use
        default_params = {
            'na_values': ['', 'NULL', 'null', 'NA', 'na', 'N/A', '#N/A', '#VALUE!', '#REF!'],
            'keep_default_na': True,
            'low_memory': False,
            'on_bad_lines': 'warn',
            'encoding': 'utf-8'
        }
        
        # Update with user-provided parameters
        params = {**default_params, **kwargs}
        
        # Auto-detect encoding if not specified
        if self.encoding_detection and 'encoding' not in kwargs:
            try:
                detected_encoding = detect_encoding(file_source)
                params['encoding'] = detected_encoding
                logger.debug(f"Auto-detected encoding: {detected_encoding}")
            except Exception as e:
                logger.warning(f"Encoding detection failed: {str(e)}")
        
        # Handle large files with chunking
        if isinstance(file_source, str) and os.path.getsize(file_source) > self.max_file_size_mb * 1024 * 1024:
            return self._read_csv_chunked(file_source, params)
        
        try:
            return pd.read_csv(file_source, **params)
        except UnicodeDecodeError as e:
            # Try fallback encodings
            for encoding in SUPPORTED_ENCODINGS:
                if encoding != params.get('encoding'):
                    try:
                        params['encoding'] = encoding
                        return pd.read_csv(file_source, **params)
                    except UnicodeDecodeError:
                        continue
            raise ValueError(f"Failed to read CSV with any encoding: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
    
    def _read_csv_chunked(self, filepath: str, **kwargs) -> DataFrame:
        """Read large CSV files in chunks with memory optimization."""
        chunk_size = kwargs.pop('chunksize', self.chunk_size)
        chunks = []
        processed_rows = 0
        
        try:
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, **kwargs):
                chunks.append(chunk)
                processed_rows += len(chunk)
                logger.debug(f"Processed {processed_rows} rows...")
            
            if chunks:
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise ValueError(f"Error reading chunked CSV: {str(e)}")
    
    def _read_first_chunk(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read only the first chunk of a file for streaming mode."""
        chunk_size = kwargs.pop('chunksize', self.chunk_size)
        
        try:
            if isinstance(file_source, str):
                for chunk in pd.read_csv(file_source, chunksize=chunk_size, **kwargs):
                    return chunk
            else:
                # For file-like objects, we need to read the first chunk differently
                return pd.read_csv(file_source, nrows=chunk_size, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading first chunk: {str(e)}")
    
    def _read_text_file(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read generic text files with automatic delimiter detection."""
        try:
            # Read first few lines to detect structure
            if isinstance(file_source, str):
                with open(file_source, 'r', encoding='utf-8') as f:
                    sample_lines = [f.readline() for _ in range(5)]
            else:
                current_pos = file_source.tell()
                file_source.seek(0)
                sample_lines = [file_source.readline().decode('utf-8') for _ in range(5)]
                file_source.seek(current_pos)
            
            # Detect delimiter
            delimiter = self._detect_delimiter(''.join(sample_lines))
            
            # Read with detected delimiter
            return self._read_csv_advanced(file_source, delimiter=delimiter, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")
    
    def _detect_delimiter(self, sample_text: str) -> str:
        """Detect the most likely delimiter in text data."""
        delimiters = [',', '\t', ';', '|', ':']
        delimiter_counts = {delim: sample_text.count(delim) for delim in delimiters}
        
        if not any(delimiter_counts.values()):
            return ','  # Default to comma
        
        return max(delimiter_counts, key=delimiter_counts.get)
    
    def _read_excel_advanced(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced Excel reading with sheet detection and optimization."""
        default_params = {
            'na_values': ['', 'NULL', 'null', 'NA', 'na', 'N/A', '#N/A'],
            'keep_default_na': True,
        }
        
        params = {**default_params, **kwargs}
        
        try:
            # If no sheet specified, try to read all sheets and pick the most appropriate
            if 'sheet_name' not in params:
                excel_file = pd.ExcelFile(file_source)
                sheet_names = excel_file.sheet_names
                
                if len(sheet_names) == 1:
                    params['sheet_name'] = sheet_names[0]
                else:
                    # Heuristic to find the most data-rich sheet
                    sheet_stats = []
                    for sheet in sheet_names:
                        try:
                            df_sample = pd.read_excel(file_source, sheet_name=sheet, nrows=10)
                            sheet_stats.append({
                                'name': sheet,
                                'rows': len(df_sample),
                                'columns': len(df_sample.columns),
                                'non_empty_cells': df_sample.count().sum()
                            })
                        except:
                            continue
                    
                    if sheet_stats:
                        # Prefer sheets with more data
                        best_sheet = max(sheet_stats, key=lambda x: x['non_empty_cells'])
                        params['sheet_name'] = best_sheet['name']
                    else:
                        params['sheet_name'] = 0  # First sheet as fallback
            
            return pd.read_excel(file_source, **params)
            
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")
    
    def _read_json_advanced(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced JSON reading with multiple format support."""
        try:
            # Try pandas read_json first
            return pd.read_json(file_source, **kwargs)
        except Exception as e:
            # Try manual JSON parsing for complex structures
            try:
                if isinstance(file_source, str):
                    with open(file_source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    content = file_source.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    data = json.loads(content)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Check if it's a records-style JSON
                    if 'records' in data and isinstance(data['records'], list):
                        return pd.DataFrame(data['records'])
                    # Check if it's a data-focused JSON
                    elif 'data' in data and isinstance(data['data'], list):
                        return pd.DataFrame(data['data'])
                    else:
                        # Normalize nested JSON
                        return pd.json_normalize(data)
                else:
                    raise ValueError("Unsupported JSON structure")
                    
            except Exception as inner_e:
                raise ValueError(f"Failed to read JSON file: {str(inner_e)}")
    
    def _read_json_lines(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read JSON Lines format with error handling."""
        try:
            return pd.read_json(file_source, lines=True, **kwargs)
        except Exception as e:
            # Fallback: read line by line
            try:
                lines = []
                if isinstance(file_source, str):
                    with open(file_source, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                lines.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
                else:
                    content = file_source.read().decode('utf-8')
                    for line in content.split('\n'):
                        try:
                            if line.strip():
                                lines.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
                
                return pd.DataFrame(lines)
            except Exception as inner_e:
                raise ValueError(f"Error reading JSON Lines file: {str(inner_e)}")
    
    def _read_sqlite_advanced(self, db_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced SQLite reading with table detection and query support."""
        table_name = kwargs.get('table_name', kwargs.get('table'))
        query = kwargs.get('query')
        
        try:
            if isinstance(db_source, str) and os.path.exists(db_source):
                with sqlite3.connect(db_source) as conn:
                    if query:
                        # Use custom query
                        return pd.read_sql_query(query, conn)
                    elif not table_name:
                        # Auto-detect table
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT name FROM sqlite_master 
                            WHERE type='table' 
                            AND name NOT LIKE 'sqlite_%'
                            ORDER BY name
                        """)
                        tables = [row[0] for row in cursor.fetchall()]
                        
                        if not tables:
                            raise ValueError("No user tables found in SQLite database")
                        
                        # Try to find the main data table
                        preferred_tables = ['data', 'main', 'records', 'table']
                        for preferred in preferred_tables:
                            if preferred in tables:
                                table_name = preferred
                                break
                        else:
                            table_name = tables[0]  # First table as fallback
                    
                    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            else:
                raise ValueError("Invalid SQLite database source")
                
        except Exception as e:
            raise ValueError(f"SQLite reading failed: {str(e)}")
    
    def _read_xml(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read XML files with namespace support."""
        if not XML_AVAILABLE:
            raise ImportError("XML parsing requires xml module")
        
        try:
            return pd.read_xml(file_source, **kwargs)
        except Exception as e:
            # Fallback: manual XML parsing
            try:
                if isinstance(file_source, str):
                    tree = ET.parse(file_source)
                else:
                    content = file_source.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    tree = ET.ElementTree(ET.fromstring(content))
                
                root = tree.getroot()
                
                # Simple XML to DataFrame conversion
                data = []
                for child in root:
                    row = {}
                    for element in child:
                        row[element.tag] = element.text
                    data.append(row)
                
                return pd.DataFrame(data)
            except Exception as inner_e:
                raise ValueError(f"Error reading XML file: {str(inner_e)}")
    
    def _read_parquet(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read Parquet files with pyarrow backend."""
        if not PARQUET_AVAILABLE:
            raise ImportError("Parquet support requires pyarrow library")
        
        try:
            return pd.read_parquet(file_source, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading Parquet file: {str(e)}")
    
    def _read_avro(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read Avro files."""
        if not AVRO_AVAILABLE:
            raise ImportError("Avro support requires fastavro library")
        
        try:
            if isinstance(file_source, str):
                with open(file_source, 'rb') as f:
                    reader = fastavro.reader(f)
                    records = [record for record in reader]
            else:
                records = fastavro.reader(file_source)
            
            return pd.DataFrame(records)
        except Exception as e:
            raise ValueError(f"Error reading Avro file: {str(e)}")
    
    def _read_yaml(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read YAML files."""
        if not YAML_AVAILABLE:
            raise ImportError("YAML support requires pyyaml library")
        
        try:
            if isinstance(file_source, str):
                with open(file_source, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                content = file_source.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                data = yaml.safe_load(content)
            
            # Convert YAML structure to DataFrame
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.json_normalize(data)
            else:
                raise ValueError("Unsupported YAML structure")
        except Exception as e:
            raise ValueError(f"Error reading YAML file: {str(e)}")
    
    def _read_compressed(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read compressed files (gzip) with format detection."""
        try:
            if isinstance(file_source, str):
                with gzip.open(file_source, 'rt', encoding='utf-8') as f:
                    # Detect format of compressed content
                    content_start = f.read(100)
                    f.seek(0)
                    
                    if content_start.startswith(('{', '[')):
                        return pd.read_json(f, **kwargs)
                    elif content_start.startswith('<?xml'):
                        return pd.read_xml(f, **kwargs)
                    else:
                        return pd.read_csv(f, **kwargs)
            else:
                # For file-like objects, decompress first
                decompressed = gzip.decompress(file_source.read())
                file_like = io.BytesIO(decompressed)
                
                # Try to detect format
                try:
                    return pd.read_json(file_like, **kwargs)
                except:
                    file_like.seek(0)
                    try:
                        return pd.read_xml(file_like, **kwargs)
                    except:
                        file_like.seek(0)
                        return pd.read_csv(io.TextIOWrapper(file_like, encoding='utf-8'), **kwargs)
                        
        except Exception as e:
            raise ValueError(f"Error reading compressed file: {str(e)}")
    
    def _read_zip_archive(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read ZIP archives and extract data files with intelligent selection."""
        try:
            with zipfile.ZipFile(file_source) as zip_file:
                file_list = zip_file.namelist()
                
                # Find data files (exclude directories and system files)
                data_files = [
                    f for f in file_list 
                    if not f.endswith('/') 
                    and not f.startswith('__MACOSX/')
                    and not os.path.basename(f).startswith('.')
                    and Path(f).suffix.lower() in self.supported_formats
                ]
                
                if not data_files:
                    raise ValueError("No supported data files found in ZIP archive")
                
                # Prioritize files that look like data
                data_priority = []
                for f in data_files:
                    priority = 0
                    filename = os.path.basename(f).lower()
                    
                    # Higher priority for files that look like main data
                    if any(keyword in filename for keyword in ['data', 'main', 'export', 'sheet']):
                        priority += 10
                    if filename.startswith(('data', 'main')):
                        priority += 5
                    
                    data_priority.append((f, priority))
                
                # Sort by priority and size
                data_priority.sort(key=lambda x: (-x[1], -zip_file.getinfo(x[0]).file_size))
                target_file = data_priority[0][0]
                
                # Extract and process the file
                with zip_file.open(target_file) as extracted_file:
                    file_ext = Path(target_file).suffix.lower()
                    processor = self.supported_formats[file_ext]
                    return processor(extracted_file, **kwargs)
                    
        except Exception as e:
            raise ValueError(f"Error reading ZIP archive: {str(e)}")
    
    def _update_data_metadata(self, metadata: FileMetadata, df: DataFrame) -> None:
        """Update metadata with DataFrame information including enhanced metrics."""
        if df is None:
            return
            
        metadata.rows = len(df)
        metadata.columns = len(df.columns)
        metadata.memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        metadata.missing_values = df.isnull().sum().sum()
        
        # Store column types and basic statistics
        metadata.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        for col in df.columns:
            col_data = df[col]
            
            if pd.api.types.is_numeric_dtype(col_data):
                metadata.column_summary[col] = {
                    'type': 'numeric',
                    'min': float(col_data.min()) if not pd.isna(col_data.min()) else None,
                    'max': float(col_data.max()) if not pd.isna(col_data.max()) else None,
                    'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                    'std': float(col_data.std()) if not pd.isna(col_data.std()) else None,
                    'missing_count': int(col_data.isnull().sum()),
                    'unique_count': int(col_data.nunique())
                }
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                metadata.column_summary[col] = {
                    'type': 'datetime',
                    'min': col_data.min().isoformat() if not pd.isna(col_data.min()) else None,
                    'max': col_data.max().isoformat() if not pd.isna(col_data.max()) else None,
                    'missing_count': int(col_data.isnull().sum()),
                    'unique_count': int(col_data.nunique())
                }
            else:
                metadata.column_summary[col] = {
                    'type': 'categorical',
                    'unique_count': int(col_data.nunique()),
                    'missing_count': int(col_data.isnull().sum()),
                    'most_frequent': str(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                    'max_length': int(col_data.astype(str).str.len().max()) if len(col_data) > 0 else 0
                }
    
    def _extract_sample_data(self, metadata: FileMetadata, df: DataFrame, sample_size: int = 5) -> None:
        """Extract sample data from DataFrame for metadata."""
        if df is None or len(df) == 0:
            return
            
        sample_size = min(sample_size, len(df))
        sample_df = df.head(sample_size)
        
        for col in sample_df.columns:
            metadata.sample_data[col] = sample_df[col].head(sample_size).tolist()
    
    def _assess_data_quality(self, df: DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment."""
        if df is None or len(df) == 0:
            return {
                'overall_score': 0,
                'completeness': {'score': 0, 'missing_count': 0, 'missing_percentage': 100},
                'uniqueness': {'score': 0, 'duplicate_count': 0, 'duplicate_percentage': 0},
                'consistency': {'score': 0, 'issues': []}
            }
        
        total_cells = len(df) * len(df.columns)
        missing_count = df.isnull().sum().sum()
        missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df) * 100) if len(df) > 0 else 0
        
        # Check for consistency issues
        consistency_issues = []
        
        # Check for mixed data types in object columns
        for col in df.select_dtypes(include=['object']).columns:
            type_counts = df[col].apply(type).value_counts()
            if len(type_counts) > 1:
                consistency_issues.append(f"Mixed types in column '{col}': {dict(type_counts.head(3))}")
        
        # Calculate scores (0-10 scale)
        completeness_score = max(0, 10 - (missing_pct / 10))  # Lose 1 point per 10% missing
        uniqueness_score = max(0, 10 - (duplicate_pct / 2))   # Lose 1 point per 2% duplicates
        consistency_score = 10 if not consistency_issues else max(0, 10 - len(consistency_issues))
        
        overall_score = (completeness_score + uniqueness_score + consistency_score) / 3
        
        return {
            'overall_score': round(overall_score, 1),
            'completeness': {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'score': round(completeness_score, 1)
            },
            'uniqueness': {
                'duplicate_count': int(duplicate_count),
                'duplicate_percentage': round(duplicate_pct, 2),
                'score': round(uniqueness_score, 1)
            },
            'consistency': {
                'score': round(consistency_score, 1),
                'issues': consistency_issues
            },
            'structure': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
        }
    
    def _generate_recommendations(self, metadata: FileMetadata, quality_report: Dict[str, Any], 
                                 processing_mode: ProcessingMode) -> List[str]:
        """Generate actionable recommendations based on comprehensive analysis."""
        recommendations = []
        
        # Data quality recommendations
        completeness = quality_report.get('completeness', {})
        if completeness.get('missing_percentage', 0) > 20:
            recommendations.append(f"Critical: High missing data ({completeness['missing_percentage']:.1f}%) - Investigate data source")
        elif completeness.get('missing_percentage', 0) > 5:
            recommendations.append(f"Warning: Moderate missing data ({completeness['missing_percentage']:.1f}%) - Consider imputation")
        
        uniqueness = quality_report.get('uniqueness', {})
        if uniqueness.get('duplicate_percentage', 0) > 5:
            recommendations.append(f"Critical: High duplicate rate ({uniqueness['duplicate_percentage']:.1f}%) - Review ETL process")
        elif uniqueness.get('duplicate_percentage', 0) > 1:
            recommendations.append(f"Warning: Moderate duplicate rate ({uniqueness['duplicate_percentage']:.1f}%) - Consider deduplication")
        
        consistency = quality_report.get('consistency', {})
        if consistency.get('issues'):
            issues = consistency['issues']
            if len(issues) > 3:
                recommendations.append(f"Found {len(issues)} consistency issues - Review data validation")
            else:
                for issue in issues[:2]:
                    recommendations.append(f"Consistency: {issue}")
        
        # Performance recommendations
        if metadata.memory_usage_mb > 500:
            recommendations.append("Large dataset - Consider using memory-efficient processing or database storage")
        elif metadata.memory_usage_mb > 100:
            recommendations.append("Moderate dataset - Optimize data types for better performance")
        
        # Format-specific recommendations
        if metadata.filetype == '.xlsx' and metadata.rows > 100000:
            recommendations.append("Very large Excel file - Convert to Parquet or CSV for better performance")
        
        if metadata.filetype == '.csv' and processing_mode == ProcessingMode.STANDARD:
            recommendations.append("CSV file - Consider chunked processing for memory efficiency")
        
        # Add processing mode recommendation
        if processing_mode == ProcessingMode.STREAMING and metadata.rows > self.chunk_size:
            recommendations.append("Streaming mode enabled - For full processing, use standard or memory-efficient mode")
        
        if not recommendations:
            recommendations.append("Data quality looks good! Ready for analysis")
        
        return recommendations
    
    def _update_processing_stats(self, metadata: FileMetadata, start_time: float) -> None:
        """Update global processing statistics."""
        self.processing_stats['files_processed'] += 1
        self.processing_stats['total_rows_processed'] += metadata.rows
        self.processing_stats['total_processing_time'] += (time.time() - start_time)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def reset_processing_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'files_processed': 0,
            'total_rows_processed': 0,
            'total_processing_time': 0.0
        }

# ============================
# ENHANCED API DATA INGESTION
# ============================

class APIDataIngestion:
    """Professional-grade API data ingestion with retry logic and comprehensive error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not REQUESTS_AVAILABLE:
            raise ImportError("API ingestion requires requests library")
        
        self.config = config or {}
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_backoff = self.config.get('retry_backoff', 0.5)
        
        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'CortexX-DataIngestion/3.0',
            'Accept': 'application/json, text/csv, */*',
            'Accept-Encoding': 'gzip, deflate'
        })
    
    @performance_monitor
    def fetch_from_api(self, url: str, method: str = 'GET', 
                      headers: Optional[Dict[str, str]] = None,
                      params: Optional[Dict[str, Any]] = None,
                      data: Optional[Any] = None,
                      json_data: Optional[Any] = None,
                      **kwargs) -> DataIngestionResult:
        """Fetch data from API endpoint with comprehensive error handling and parsing."""
        start_time = time.time()
        
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL scheme: {url}")
            
            # Make request
            response = self.session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            
            # Parse response based on content type
            content_type = response.headers.get('content-type', '').lower()
            df = self._parse_api_response(response, content_type)
            
            # Create comprehensive metadata
            metadata = self._create_api_metadata(url, response, df, start_time)
            
            # Assess data quality
            processor = EnhancedFileProcessor()
            quality_report = processor._assess_data_quality(df)
            metadata.data_quality_score = quality_report.get('overall_score', 0)
            
            return DataIngestionResult(
                success=True,
                dataframe=df,
                metadata=metadata,
                quality_report=quality_report,
                recommendations=["API data successfully ingested"]
            )
            
        except Exception as e:
            logger.error(f"API request failed: {str(e)}", exc_info=True)
            
            metadata = FileMetadata(
                filepath=url,
                filename=f"api_error_{int(time.time())}",
                filesize=0,
                filetype='api',
                mimetype='unknown',
                encoding='unknown',
                last_modified=datetime.now(timezone.utc).isoformat(),
                checksum='unknown',
                processing_time=time.time() - start_time,
                processing_errors=[str(e)]
            )
            
            return DataIngestionResult(
                success=False,
                dataframe=None,
                metadata=metadata,
                recommendations=[f"API ingestion failed: {str(e)}"]
            )
    
    def _parse_api_response(self, response: requests.Response, content_type: str) -> DataFrame:
        """Parse API response based on content type with fallback logic."""
        try:
            if 'application/json' in content_type:
                data = response.json()
                
                if isinstance(data, dict):
                    # Handle various JSON structures
                    if 'data' in data and isinstance(data['data'], list):
                        return pd.DataFrame(data['data'])
                    elif 'results' in data and isinstance(data['results'], list):
                        return pd.DataFrame(data['results'])
                    elif 'items' in data and isinstance(data['items'], list):
                        return pd.DataFrame(data['items'])
                    elif 'records' in data and isinstance(data['records'], list):
                        return pd.DataFrame(data['records'])
                    else:
                        # Flatten nested JSON
                        return pd.json_normalize(data)
                elif isinstance(data, list):
                    return pd.DataFrame(data)
                else:
                    raise ValueError(f"Unsupported JSON structure: {type(data)}")
                    
            elif 'text/csv' in content_type or 'application/csv' in content_type:
                return pd.read_csv(io.StringIO(response.text))
                
            elif 'application/xml' in content_type or 'text/xml' in content_type:
                return pd.read_xml(io.StringIO(response.text))
                
            elif 'text/plain' in content_type:
                # Try to parse as JSON first, then as CSV
                try:
                    data = response.json()
                    return pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
                except:
                    return pd.read_csv(io.StringIO(response.text))
                    
            else:
                # Try to auto-detect format
                try:
                    data = response.json()
                    return pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
                except:
                    try:
                        return pd.read_csv(io.StringIO(response.text))
                    except:
                        raise ValueError(f"Unsupported content type: {content_type}")
                        
        except Exception as e:
            raise ValueError(f"Failed to parse API response: {str(e)}")
    
    def _create_api_metadata(self, url: str, response: requests.Response, 
                            df: DataFrame, start_time: float) -> FileMetadata:
        """Create comprehensive metadata for API responses."""
        processing_time = time.time() - start_time
        
        return FileMetadata(
            filepath=url,
            filename=f"api_data_{int(time.time())}",
            filesize=len(response.content),
            filetype='api',
            mimetype=response.headers.get('content-type', 'unknown'),
            encoding=response.encoding or 'utf-8',
            last_modified=datetime.now(timezone.utc).isoformat(),
            checksum=hashlib.md5(response.content).hexdigest(),
            rows=len(df) if df is not None else 0,
            columns=len(df.columns) if df is not None else 0,
            memory_usage_mb=df.memory_usage(deep=True).sum() / 1024 / 1024 if df is not None else 0,
            missing_values=df.isnull().sum().sum() if df is not None else 0,
            processing_time=processing_time
        )

# ============================
# PROFESSIONAL BATCH PROCESSING
# ============================

class BatchProcessor:
    """Enterprise-grade batch processing with progress tracking and result aggregation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.processor = EnhancedFileProcessor(config)
        self.progress_callback = None
        
    def set_progress_callback(self, callback: Callable[[str, int, int], None]) -> None:
        """Set a callback function for progress reporting."""
        self.progress_callback = callback
    
    def process_directory(self, directory_path: str, pattern: Optional[str] = None, 
                         recursive: bool = False, 
                         processing_mode: ProcessingMode = ProcessingMode.STANDARD) -> Dict[str, DataIngestionResult]:
        """Process all supported files in a directory with progress tracking."""
        results = {}
        
        try:
            directory = Path(directory_path)
            
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Invalid directory: {directory_path}")
            
            if recursive:
                files = list(directory.rglob(pattern or '*'))
            else:
                files = list(directory.glob(pattern or '*'))
            
            # Filter for supported files
            supported_files = [
                f for f in files 
                if f.is_file() and f.suffix.lower() in self.processor.supported_formats
            ]
            
            total_files = len(supported_files)
            
            for i, filepath in enumerate(supported_files):
                try:
                    if self.progress_callback:
                        self.progress_callback(str(filepath), i + 1, total_files)
                    
                    result = self.processor.process_file(str(filepath), processing_mode=processing_mode)
                    results[str(filepath)] = result
                    
                    logger.info(f"Processed {filepath.name} ({i+1}/{total_files}): {result.success}")
                    
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {str(e)}")
                    error_metadata = self.processor._create_error_metadata(
                        str(filepath), filepath.name, str(e)
                    )
                    results[str(filepath)] = DataIngestionResult(
                        success=False,
                        dataframe=None,
                        metadata=error_metadata,
                        processing_mode=processing_mode
                    )
                    
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise
        
        return results
    
    def combine_results(self, results: Dict[str, DataIngestionResult], 
                       how: str = 'concat', **kwargs) -> Optional[DataFrame]:
        """Combine multiple processing results into a single DataFrame with schema alignment."""
        successful_results = [r for r in results.values() if r.success and r.dataframe is not None]
        
        if not successful_results:
            return None
        
        dataframes = [r.dataframe for r in successful_results]
        
        if how == 'concat':
            try:
                return pd.concat(dataframes, ignore_index=True, sort=False, **kwargs)
            except Exception as e:
                logger.warning(f"Direct concat failed, aligning schemas: {str(e)}")
                return self._concat_with_schema_alignment(dataframes, **kwargs)
                
        elif how == 'merge':
            if not dataframes:
                return None
                
            result_df = dataframes[0]
            for df in dataframes[1:]:
                # Find common columns for merge
                common_cols = list(set(result_df.columns) & set(df.columns))
                if common_cols:
                    result_df = pd.merge(result_df, df, on=common_cols, how='outer', **kwargs)
                else:
                    # No common columns, concat instead
                    result_df = pd.concat([result_df, df], ignore_index=True, sort=False, **kwargs)
            return result_df
        
        return None
    
    def _concat_with_schema_alignment(self, dataframes: List[DataFrame], **kwargs) -> DataFrame:
        """Concatenate DataFrames with schema alignment for incompatible schemas."""
        if not dataframes:
            return pd.DataFrame()
        
        # Get all unique columns across all DataFrames
        all_columns = set()
        for df in dataframes:
            all_columns.update(df.columns)
        
        # Align each DataFrame to have all columns
        aligned_dfs = []
        for df in dataframes:
            aligned_df = df.copy()
            for col in all_columns:
                if col not in aligned_df.columns:
                    aligned_df[col] = None  # Add missing columns with null values
            aligned_dfs.append(aligned_df[list(all_columns)])  # Ensure consistent column order
        
        return pd.concat(aligned_dfs, ignore_index=True, **kwargs)
    
    def generate_batch_report(self, results: Dict[str, DataIngestionResult]) -> Dict[str, Any]:
        """Generate a comprehensive batch processing report."""
        successful = [r for r in results.values() if r.success]
        failed = [r for r in results.values() if not r.success]
        
        total_files = len(results)
        total_rows = sum(r.metadata.rows for r in successful)
        total_size_mb = sum(r.metadata.filesize for r in results) / 1024 / 1024
        
        quality_scores = [r.metadata.data_quality_score for r in successful]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'summary': {
                'total_files': total_files,
                'successful_files': len(successful),
                'failed_files': len(failed),
                'success_rate': (len(successful) / total_files * 100) if total_files > 0 else 0,
                'total_rows_processed': total_rows,
                'total_size_mb': round(total_size_mb, 2),
                'average_quality_score': round(avg_quality, 2)
            },
            'file_details': {
                'successful': [r.metadata.filename for r in successful],
                'failed': [{
                    'filename': r.metadata.filename,
                    'error': r.metadata.processing_errors[0] if r.metadata.processing_errors else 'Unknown error'
                } for r in failed]
            },
            'quality_distribution': {
                'excellent': len([s for s in quality_scores if s >= 9]),
                'good': len([s for s in quality_scores if 7 <= s < 9]),
                'fair': len([s for s in quality_scores if 5 <= s < 7]),
                'poor': len([s for s in quality_scores if s < 5])
            }
        }

# ============================
# GLOBAL INSTANCES AND FUNCTIONS
# ============================

# Initialize global processor instances with default config
global_processor = EnhancedFileProcessor()
global_batch_processor = BatchProcessor()

# Initialize API processor only if requests is available
global_api_processor = None
if REQUESTS_AVAILABLE:
    try:
        global_api_processor = APIDataIngestion()
    except ImportError:
        pass

# ============================
# CONVENIENCE FUNCTIONS
# ============================

def process_file_advanced(file_source: Union[str, IO], filename: Optional[str] = None, 
                         processing_mode: ProcessingMode = ProcessingMode.STANDARD, **kwargs) -> DataIngestionResult:
    """Advanced file processing with comprehensive analysis."""
    return global_processor.process_file(file_source, filename, processing_mode, **kwargs)

def fetch_data_from_api(url: str, method: str = 'GET', **kwargs) -> DataIngestionResult:
    """Fetch data from API endpoint."""
    if global_api_processor is None:
        raise ImportError("API functionality requires requests library")
    return global_api_processor.fetch_from_api(url, method, **kwargs)

def process_directory_batch(directory_path: str, pattern: Optional[str] = None, 
                           recursive: bool = False, 
                           processing_mode: ProcessingMode = ProcessingMode.STANDARD) -> Dict[str, DataIngestionResult]:
    """Process all files in a directory."""
    return global_batch_processor.process_directory(directory_path, pattern, recursive, processing_mode)

def combine_datasets(results: Dict[str, DataIngestionResult], how: str = 'concat') -> Optional[DataFrame]:
    """Combine multiple datasets."""
    return global_batch_processor.combine_results(results, how)

def generate_batch_report(results: Dict[str, DataIngestionResult]) -> Dict[str, Any]:
    """Generate a batch processing report."""
    return global_batch_processor.generate_batch_report(results)

# ============================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================

def read_csv(path_or_buffer: Union[str, IO], **kwargs) -> DataFrame:
    """Enhanced CSV reading with backward compatibility."""
    try:
        result = global_processor.process_file(path_or_buffer, **kwargs)
        if result.success:
            return result.dataframe
        else:
            raise ValueError(f"Failed to read CSV: {result.metadata.processing_errors}")
    except Exception:
        # Fallback to original pandas method
        return pd.read_csv(path_or_buffer, **kwargs)

def read_excel(path_or_buffer: Union[str, IO], **kwargs) -> DataFrame:
    """Enhanced Excel reading with backward compatibility."""
    try:
        result = global_processor.process_file(path_or_buffer, **kwargs)
        if result.success:
            return result.dataframe
        else:
            raise ValueError(f"Failed to read Excel: {result.metadata.processing_errors}")
    except Exception:
        # Fallback to original pandas method
        return pd.read_excel(path_or_buffer, **kwargs)

def read_json(path_or_buffer: Union[str, IO], **kwargs) -> DataFrame:
    """Enhanced JSON reading with backward compatibility."""
    try:
        result = global_processor.process_file(path_or_buffer, **kwargs)
        if result.success:
            return result.dataframe
        else:
            raise ValueError(f"Failed to read JSON: {result.metadata.processing_errors}")
    except Exception:
        # Fallback to original pandas method
        return pd.read_json(path_or_buffer, **kwargs)

def read_sqlite(db_source: Union[str, IO], table_name: str, **kwargs) -> DataFrame:
    """Enhanced SQLite reading with backward compatibility."""
    try:
        result = global_processor.process_file(db_source, table_name=table_name, **kwargs)
        if result.success:
            return result.dataframe
        else:
            raise ValueError(f"Failed to read SQLite: {result.metadata.processing_errors}")
    except Exception as e:
        # Fallback to original method
        if isinstance(db_source, str) and os.path.exists(db_source):
            with sqlite3.connect(db_source) as conn:
                return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        else:
            raise ValueError(f"SQLite reading failed: {str(e)}")

def validate_file_format(filepath: str) -> bool:
    """Enhanced file format validation."""
    supported_extensions = list(global_processor.supported_formats.keys())
    return any(filepath.lower().endswith(ext) for ext in supported_extensions)

def get_file_info(df: DataFrame) -> dict:
    """Enhanced file information with additional metrics."""
    if df is None:
        return {
            'rows': 0,
            'columns': 0,
            'memory_usage_mb': 0.0,
            'missing_values': 0,
            'data_types': {}
        }
    
    basic_info = {
        'rows': int(len(df)),
        'columns': int(len(df.columns)),
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        'missing_values': int(df.isna().sum().sum()),
        'data_types': {k: str(v) for k, v in df.dtypes.to_dict().items()},
    }
    
    return basic_info

# ============================
# EXPORTS
# ============================

__all__ = [
    # Enhanced classes
    'EnhancedFileProcessor',
    'APIDataIngestion', 
    'BatchProcessor',
    'FileMetadata',
    'DataIngestionResult',
    'ProcessingMode',
    'FileFormat',
    
    # Main functions
    'process_file_advanced',
    'fetch_data_from_api',
    'process_directory_batch',
    'combine_datasets',
    'generate_batch_report',
    
    # Utility functions
    'detect_encoding',
    'validate_file_path',
    'sanitize_column_names',
    
    # Backward compatibility
    'read_csv',
    'read_excel', 
    'read_json',
    'read_sqlite',
    'validate_file_format',
    'get_file_info'
]

# ============================
# MODULE INITIALIZATION
# ============================

def _check_dependencies():
    """Check and report on optional dependencies."""
    dependency_status = {
        'requests': REQUESTS_AVAILABLE,
        'pyarrow': PARQUET_AVAILABLE,
        'fastavro': AVRO_AVAILABLE,
        'yaml': YAML_AVAILABLE,
        'openpyxl': OPENPYXL_AVAILABLE,
        'xlrd': XLRD_AVAILABLE,
        'xml': XML_AVAILABLE
    }
    
    missing_deps = [name for name, available in dependency_status.items() if not available]
    
    if missing_deps:
        logger.warning(f"Optional dependencies missing: {', '.join(missing_deps)}. Some features may be limited.")

# Run dependency check on import
_check_dependencies()

print("✅ Enhanced File Handler Module v3.0 - Professional Platform Edition")
print(f"   📁 Supported Formats: {len(global_processor.supported_formats)}")
print(f"   🌐 API Support: {REQUESTS_AVAILABLE}")
print(f"   📊 Excel Support: {OPENPYXL_AVAILABLE or XLRD_AVAILABLE}")
print(f"   🗂️ XML Support: {XML_AVAILABLE}")
print(f"   🏹 Parquet Support: {PARQUET_AVAILABLE}")
print(f"   🔄 Avro Support: {AVRO_AVAILABLE}")
print(f"   📝 YAML Support: {YAML_AVAILABLE}")
print("   🚀 All functions ready for import!")