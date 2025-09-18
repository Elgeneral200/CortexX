# filename: file_handler.py
"""
File Handling Module - Enhanced Professional Edition (Fixed)

Comprehensive file processing and data ingestion system with:
- Multi-format support (CSV, Excel, JSON, SQLite, Parquet, XML, API)
- Advanced error handling and validation
- Performance optimization for large files
- Data quality assessment during ingestion
- Metadata extraction and cataloging
- Streaming and batch processing capabilities
- Integration with quality and pipeline modules

Author: CortexX Team
Version: 1.1.1 - Enhanced Professional Edition (Fixed)
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
from pathlib import Path
from typing import IO, Union, Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager
from urllib.parse import urlparse
import time
import hashlib
import mimetypes

import pandas as pd
import numpy as np
from pandas import DataFrame

# Optional dependencies with graceful handling
try:
    import requests
except ImportError:
    requests = None

try:
    import xml.etree.ElementTree as ET
except ImportError:
    ET = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ============================
# ENHANCED DATA STRUCTURES
# ============================

@dataclass
class FileMetadata:
    """Comprehensive file metadata structure."""
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    mime_type: str
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

    # Data types and schema
    column_types: Dict[str, str] = field(default_factory=dict)
    column_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'file_info': {
                'path': self.file_path,
                'name': self.file_name,
                'size_bytes': self.file_size,
                'type': self.file_type,
                'mime_type': self.mime_type,
                'encoding': self.encoding,
                'last_modified': self.last_modified,
                'checksum': self.checksum
            },
            'data_info': {
                'rows': self.rows,
                'columns': self.columns,
                'memory_usage_mb': self.memory_usage_mb,
                'missing_values': self.missing_values,
                'data_quality_score': self.data_quality_score
            },
            'processing_info': {
                'processing_time': self.processing_time,
                'errors': self.processing_errors,
                'warnings': self.processing_warnings
            },
            'schema_info': {
                'column_types': self.column_types,
                'column_summary': self.column_summary
            }
        }

@dataclass 
class DataIngestionResult:
    """Result of data ingestion process."""
    success: bool
    dataframe: Optional[DataFrame]
    metadata: FileMetadata
    quality_report: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary of the ingestion result."""
        return {
            'success': self.success,
            'data_shape': (self.metadata.rows, self.metadata.columns) if self.dataframe is not None else (0, 0),
            'file_info': {
                'name': self.metadata.file_name,
                'size_mb': self.metadata.file_size / 1024 / 1024,
                'type': self.metadata.file_type
            },
            'quality_score': self.metadata.data_quality_score,
            'processing_time': self.metadata.processing_time,
            'recommendations_count': len(self.recommendations),
            'has_issues': len(self.metadata.processing_errors) > 0 or len(self.metadata.processing_warnings) > 0
        }

# ============================
# PERFORMANCE UTILITIES
# ============================

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor file processing performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time

            # Add performance data to result if it's a DataIngestionResult
            if isinstance(result, DataIngestionResult):
                result.metadata.processing_time = processing_time

            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Function {func.__name__} failed after {processing_time:.2f}s: {str(e)}")
            raise
    return wrapper

@contextmanager
def safe_file_processing(file_path: str):
    """Context manager for safe file processing with cleanup."""
    temp_files = []
    try:
        yield temp_files
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass

# ============================
# ENHANCED FILE READERS
# ============================

class EnhancedFileProcessor:
    """Advanced file processing with comprehensive capabilities."""

    def __init__(self):
        self.supported_formats = {
            '.csv': self._read_csv_advanced,
            '.tsv': self._read_csv_advanced,
            '.txt': self._read_csv_advanced,
            '.xlsx': self._read_excel_advanced,
            '.xls': self._read_excel_advanced,
            '.json': self._read_json_advanced,
            '.jsonl': self._read_jsonlines,
            '.db': self._read_sqlite_advanced,
            '.sqlite': self._read_sqlite_advanced,
            '.sqlite3': self._read_sqlite_advanced,
            '.xml': self._read_xml,
            '.gz': self._read_compressed,
            '.zip': self._read_zip_archive
        }

        # Add parquet support if available
        try:
            import pyarrow
            self.supported_formats['.parquet'] = self._read_parquet
        except ImportError:
            pass

        self.encoding_detection = True
        self.chunk_size = 50000  # For large file processing
        self.max_file_size_mb = 500  # Maximum file size for direct loading

    @performance_monitor
    def process_file(
        self, 
        file_path_or_buffer: Union[str, IO], 
        file_name: Optional[str] = None,
        **kwargs
    ) -> DataIngestionResult:
        """Process any supported file format with comprehensive analysis."""

        start_time = time.time()

        # Initialize metadata
        if isinstance(file_path_or_buffer, str):
            file_path = file_path_or_buffer
            file_name = file_name or os.path.basename(file_path)
        else:
            file_path = getattr(file_path_or_buffer, 'name', 'uploaded_file')
            file_name = file_name or 'uploaded_file'

        metadata = self._extract_file_metadata(file_path_or_buffer, file_name)

        try:
            # Determine file type and processor
            file_extension = Path(file_name).suffix.lower()

            if file_extension not in self.supported_formats:
                # Try to detect format by content
                file_extension = self._detect_file_format(file_path_or_buffer)

            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Process the file
            processor = self.supported_formats[file_extension]
            dataframe = processor(file_path_or_buffer, **kwargs)

            # Update metadata with data information
            self._update_data_metadata(metadata, dataframe)

            # Perform data quality assessment
            quality_report = self._assess_data_quality(dataframe)
            metadata.data_quality_score = quality_report.get('overall_score', 0)

            # Generate recommendations
            recommendations = self._generate_recommendations(metadata, quality_report)

            return DataIngestionResult(
                success=True,
                dataframe=dataframe,
                metadata=metadata,
                quality_report=quality_report,
                recommendations=recommendations
            )

        except Exception as e:
            metadata.processing_errors.append(str(e))
            return DataIngestionResult(
                success=False,
                dataframe=None,
                metadata=metadata,
                recommendations=[f"Failed to process file: {str(e)}"]
            )

    def _extract_file_metadata(self, file_source: Union[str, IO], file_name: str) -> FileMetadata:
        """Extract comprehensive file metadata."""

        if isinstance(file_source, str):
            # File path
            file_path = file_source
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            last_modified = datetime.fromtimestamp(file_stats.st_mtime, timezone.utc).isoformat()

            # Calculate checksum
            checksum = self._calculate_checksum(file_path)

            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or 'application/octet-stream'

        else:
            # File-like object
            file_path = getattr(file_source, 'name', 'uploaded_file')

            # Get size
            if hasattr(file_source, 'size'):
                file_size = file_source.size
            else:
                # Try to determine size
                current_pos = file_source.tell() if hasattr(file_source, 'tell') else 0
                try:
                    file_source.seek(0, 2)  # Seek to end
                    file_size = file_source.tell()
                    file_source.seek(current_pos)  # Seek back
                except:
                    file_size = 0

            last_modified = datetime.now(timezone.utc).isoformat()
            checksum = "unknown"
            mime_type = getattr(file_source, 'type', 'application/octet-stream')

        return FileMetadata(
            file_path=file_path,
            file_name=file_name,
            file_size=file_size,
            file_type=Path(file_name).suffix.lower(),
            mime_type=mime_type,
            encoding="unknown",
            last_modified=last_modified,
            checksum=checksum
        )

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return "unknown"

    def _detect_file_format(self, file_source: Union[str, IO]) -> str:
        """Detect file format by content analysis."""

        try:
            # Read first few bytes to detect format
            if isinstance(file_source, str):
                with open(file_source, 'rb') as f:
                    header = f.read(512)
            else:
                current_pos = file_source.tell() if hasattr(file_source, 'tell') else 0
                header = file_source.read(512)
                if hasattr(file_source, 'seek'):
                    file_source.seek(current_pos)

            # Detect format by header
            if header.startswith(b'PK'):
                return '.zip' if b'xl/' in header else '.xlsx'
            elif header.startswith(b'\x50\x4b\x03\x04'):
                return '.xlsx'
            elif header.startswith(b'{') or header.startswith(b'['):
                return '.json'
            elif header.startswith(b'<?xml'):
                return '.xml'
            elif header.startswith(b'SQLite format 3'):
                return '.db'
            elif header.startswith(b'\x1f\x8b'):
                return '.gz'
            elif b',' in header[:100] and b'\n' in header[:100]:
                return '.csv'
            else:
                # Default to CSV for text files
                try:
                    header.decode('utf-8')
                    return '.csv'
                except:
                    return '.unknown'

        except:
            return '.unknown'

    def _read_csv_advanced(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced CSV reading with encoding detection and error handling."""

        # Default parameters
        default_params = {
            'encoding': 'utf-8',
            'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NA', '#N/A'],
            'keep_default_na': True,
        }

        # Remove dtype_backend if pandas version doesn't support it
        try:
            import pandas
            if hasattr(pandas, '__version__') and pandas.__version__ >= '2.0.0':
                default_params['dtype_backend'] = 'numpy_nullable'
        except:
            pass  # Ignore if not supported

        # Update with provided parameters
        params = {**default_params, **kwargs}

        # Try different encodings if detection is enabled
        encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16']

        for encoding in encodings_to_try:
            try:
                params['encoding'] = encoding

                # Handle large files with chunking
                if isinstance(file_source, str) and os.path.getsize(file_source) > self.max_file_size_mb * 1024 * 1024:
                    return self._read_csv_chunked(file_source, **params)
                else:
                    return pd.read_csv(file_source, **params)

            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings_to_try[-1]:  # Last encoding
                    raise ValueError(f"Failed to read CSV file with all attempted encodings: {str(e)}")
                continue

        raise ValueError("Unable to read CSV file with any supported encoding")

    def _read_csv_chunked(self, file_path: str, **kwargs) -> DataFrame:
        """Read large CSV files in chunks."""

        chunk_size = kwargs.pop('chunksize', self.chunk_size)
        chunks = []

        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, **kwargs):
                chunks.append(chunk)

            return pd.concat(chunks, ignore_index=True)

        except Exception as e:
            raise ValueError(f"Error reading chunked CSV: {str(e)}")

    def _read_excel_advanced(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced Excel reading with sheet detection and optimization."""

        default_params = {
            'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NA', '#N/A'],
            'keep_default_na': True,
        }

        # Remove dtype_backend if pandas version doesn't support it
        try:
            import pandas
            if hasattr(pandas, '__version__') and pandas.__version__ >= '2.0.0':
                default_params['dtype_backend'] = 'numpy_nullable'
        except:
            pass  # Ignore if not supported

        params = {**default_params, **kwargs}

        try:
            # If no sheet specified, try to read all sheets and pick the largest
            if 'sheet_name' not in params:
                excel_file = pd.ExcelFile(file_source)

                if len(excel_file.sheet_names) == 1:
                    # Single sheet
                    params['sheet_name'] = excel_file.sheet_names[0]
                else:
                    # Multiple sheets - find the one with most data
                    sheet_sizes = {}
                    for sheet in excel_file.sheet_names:
                        try:
                            df_temp = pd.read_excel(file_source, sheet_name=sheet, nrows=0)
                            sheet_sizes[sheet] = len(df_temp.columns)
                        except:
                            sheet_sizes[sheet] = 0

                    # Pick sheet with most columns (likely main data)
                    if sheet_sizes:
                        params['sheet_name'] = max(sheet_sizes, key=sheet_sizes.get)

            return pd.read_excel(file_source, **params)

        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")

    def _read_json_advanced(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced JSON reading with multiple format support."""

        try:
            # Try pandas read_json first
            return pd.read_json(file_source, **kwargs)

        except Exception:
            # Try manual JSON parsing for complex structures
            try:
                if isinstance(file_source, str):
                    with open(file_source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    if hasattr(file_source, 'read'):
                        content = file_source.read()
                        if isinstance(content, bytes):
                            content = content.decode('utf-8')
                        data = json.loads(content)
                    else:
                        raise ValueError("Invalid JSON source")

                # Handle different JSON structures
                if isinstance(data, dict):
                    if 'data' in data:
                        # JSON with data wrapper
                        return pd.DataFrame(data['data'])
                    elif 'records' in data:
                        # JSON with records wrapper
                        return pd.DataFrame(data['records'])
                    else:
                        # Try to convert dict to DataFrame
                        try:
                            return pd.DataFrame([data])
                        except:
                            # Handle nested dictionaries
                            return pd.json_normalize(data)

                elif isinstance(data, list):
                    return pd.DataFrame(data)
                else:
                    raise ValueError("Unsupported JSON structure")

            except Exception as e:
                raise ValueError(f"Error reading JSON file: {str(e)}")

    def _read_jsonlines(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read JSON Lines format."""

        try:
            records = []

            if isinstance(file_source, str):
                with open(file_source, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
            else:
                content = file_source.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')

                for line in content.split('\n'):
                    if line.strip():
                        records.append(json.loads(line))

            return pd.DataFrame(records)

        except Exception as e:
            raise ValueError(f"Error reading JSON Lines file: {str(e)}")

    def _read_sqlite_advanced(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced SQLite reading with table detection and optimization."""

        table_name = kwargs.get('table_name', None)
        query = kwargs.get('query', None)

        with safe_file_processing("sqlite_processing") as temp_files:
            try:
                # Handle file-like objects
                if not isinstance(file_source, str):
                    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
                    temp_files.append(temp_file.name)

                    if hasattr(file_source, 'read'):
                        file_source.seek(0)
                        temp_file.write(file_source.read())
                        temp_file.close()
                        db_path = temp_file.name
                    else:
                        raise ValueError("Invalid SQLite source")
                else:
                    db_path = file_source

                with sqlite3.connect(db_path) as conn:
                    # If no table specified, find tables
                    if not table_name and not query:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()

                        if not tables:
                            raise ValueError("No tables found in SQLite database")

                        # Use the first table or the largest one
                        if len(tables) == 1:
                            table_name = tables[0][0]
                        else:
                            # Find table with most rows
                            table_sizes = {}
                            for (table,) in tables:
                                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                                table_sizes[table] = cursor.fetchone()[0]
                            table_name = max(table_sizes, key=table_sizes.get)

                    # Execute query
                    if query:
                        return pd.read_sql_query(query, conn)
                    else:
                        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

            except Exception as e:
                raise ValueError(f"Error reading SQLite database: {str(e)}")

    def _read_parquet(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read Parquet files."""

        try:
            return pd.read_parquet(file_source, **kwargs)
        except ImportError:
            raise ValueError("Parquet support requires 'pyarrow' or 'fastparquet' library")
        except Exception as e:
            raise ValueError(f"Error reading Parquet file: {str(e)}")

    def _read_xml(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read XML files with structure detection."""

        if ET is None:
            raise ValueError("XML support not available")

        try:
            # Try pandas read_xml first (requires lxml)
            return pd.read_xml(file_source, **kwargs)

        except ImportError:
            # Manual XML parsing
            try:
                if isinstance(file_source, str):
                    tree = ET.parse(file_source)
                else:
                    content = file_source.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    tree = ET.ElementTree(ET.fromstring(content))

                root = tree.getroot()

                # Try to detect XML structure and convert to DataFrame
                records = []

                # Look for repeating elements
                for child in root:
                    if len(list(child)) > 0:  # Has sub-elements
                        record = {}
                        for subchild in child:
                            record[subchild.tag] = subchild.text
                        records.append(record)
                    else:
                        # Simple structure
                        records.append({child.tag: child.text})

                if records:
                    return pd.DataFrame(records)
                else:
                    raise ValueError("Could not parse XML structure into tabular format")

            except Exception as e:
                raise ValueError(f"Error reading XML file: {str(e)}")

        except Exception as e:
            raise ValueError(f"Error reading XML file: {str(e)}")

    def _read_compressed(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read compressed files (gzip, etc.)."""

        try:
            if isinstance(file_source, str):
                if file_source.endswith('.gz'):
                    # Detect the underlying format
                    base_name = file_source[:-3]  # Remove .gz
                    base_ext = Path(base_name).suffix.lower()

                    if base_ext in self.supported_formats:
                        with gzip.open(file_source, 'rt', encoding='utf-8') as f:
                            processor = self.supported_formats[base_ext]
                            return processor(f, **kwargs)
                    else:
                        # Default to CSV
                        with gzip.open(file_source, 'rt', encoding='utf-8') as f:
                            return self._read_csv_advanced(f, **kwargs)
            else:
                # File-like object
                if hasattr(file_source, 'read'):
                    content = gzip.decompress(file_source.read())
                    # Try to detect format from decompressed content
                    return self._read_csv_advanced(io.StringIO(content.decode('utf-8')), **kwargs)

        except Exception as e:
            raise ValueError(f"Error reading compressed file: {str(e)}")

    def _read_zip_archive(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read ZIP archives containing data files."""

        try:
            with zipfile.ZipFile(file_source, 'r') as zip_file:
                # List files in archive
                file_list = zip_file.namelist()

                # Find data files (exclude directories and system files)
                data_files = [f for f in file_list 
                             if not f.endswith('/') and not f.startswith('__MACOSX/') 
                             and Path(f).suffix.lower() in self.supported_formats]

                if not data_files:
                    raise ValueError("No supported data files found in ZIP archive")

                # Use the first data file or largest one
                if len(data_files) == 1:
                    target_file = data_files[0]
                else:
                    # Pick largest file
                    file_sizes = {f: zip_file.getinfo(f).file_size for f in data_files}
                    target_file = max(file_sizes, key=file_sizes.get)

                # Extract and process the file
                with zip_file.open(target_file) as extracted_file:
                    file_ext = Path(target_file).suffix.lower()
                    processor = self.supported_formats[file_ext]
                    return processor(extracted_file, **kwargs)

        except Exception as e:
            raise ValueError(f"Error reading ZIP archive: {str(e)}")

    def _update_data_metadata(self, metadata: FileMetadata, df: DataFrame) -> None:
        """Update metadata with DataFrame information."""

        metadata.rows = len(df)
        metadata.columns = len(df.columns)
        metadata.memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        metadata.missing_values = df.isnull().sum().sum()

        # Column types
        metadata.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Column summary
        metadata.column_summary = {}
        for col in df.columns:
            col_summary = {
                'type': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'memory_mb': df[col].memory_usage(deep=True) / 1024 / 1024
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                col_summary.update({
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None
                })

            metadata.column_summary[col] = col_summary

    def _assess_data_quality(self, df: DataFrame) -> Dict[str, Any]:
        """Perform basic data quality assessment."""

        total_cells = len(df) * len(df.columns) if len(df.columns) > 0 else 0
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()

        # Calculate quality dimensions
        completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 100
        uniqueness = ((len(df) - duplicate_rows) / len(df) * 100) if len(df) > 0 else 100

        # Simple quality score (0-10)
        overall_score = (completeness + uniqueness) / 20

        return {
            'overall_score': round(overall_score, 2),
            'completeness_pct': round(completeness, 2),
            'uniqueness_pct': round(uniqueness, 2),
            'missing_values': int(missing_cells),
            'duplicate_rows': int(duplicate_rows),
            'total_cells': int(total_cells),
            'data_types': df.dtypes.value_counts().to_dict()
        }

    def _generate_recommendations(self, metadata: FileMetadata, quality_report: Dict[str, Any]) -> List[str]:
        """Generate processing and improvement recommendations."""

        recommendations = []

        # File size recommendations
        if metadata.file_size > 100 * 1024 * 1024:  # > 100MB
            recommendations.append("Consider using chunked processing for this large file to optimize memory usage")

        # Data quality recommendations
        if quality_report['completeness_pct'] < 90:
            recommendations.append("Data has significant missing values - consider data cleaning or imputation strategies")

        if quality_report['uniqueness_pct'] < 95:
            recommendations.append("Data contains duplicate records - consider deduplication")

        # Performance recommendations
        if metadata.memory_usage_mb > 500:
            recommendations.append("Dataset uses significant memory - consider data type optimization")

        # Column-specific recommendations
        if metadata.columns > 100:
            recommendations.append("Dataset has many columns - consider feature selection for better performance")

        # Data type recommendations
        object_columns = sum(1 for dtype in metadata.column_types.values() if 'object' in dtype)
        if object_columns > metadata.columns * 0.5:
            recommendations.append("Many columns have object dtype - consider categorical conversion for better performance")

        return recommendations

# ============================
# API AND URL DATA INGESTION
# ============================

class APIDataIngestion:
    """Handle data ingestion from APIs and web sources."""

    def __init__(self):
        if requests is None:
            raise ImportError("API ingestion requires 'requests' library")
        
        self.session = requests.Session()
        self.timeout = 30
        self.max_retries = 3

    def fetch_from_api(
        self, 
        url: str, 
        method: str = 'GET',
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> DataIngestionResult:
        """Fetch data from API endpoint."""

        start_time = time.time()

        try:
            # Make API request with retries
            for attempt in range(self.max_retries):
                try:
                    response = self.session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        timeout=self.timeout,
                        **kwargs
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt)  # Exponential backoff

            # Parse response based on content type
            content_type = response.headers.get('content-type', '').lower()

            if 'application/json' in content_type:
                data = response.json()
                if isinstance(data, dict):
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                    else:
                        df = pd.json_normalize(data)
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    raise ValueError("Unsupported JSON structure from API")

            elif 'text/csv' in content_type:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))

            elif 'application/xml' in content_type or 'text/xml' in content_type:
                from io import StringIO
                df = pd.read_xml(StringIO(response.text))

            else:
                # Try to parse as JSON by default
                try:
                    data = response.json()
                    df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
                except:
                    raise ValueError(f"Unsupported content type: {content_type}")

            # Create metadata
            metadata = FileMetadata(
                file_path=url,
                file_name=f"api_data_{int(time.time())}",
                file_size=len(response.content),
                file_type='api',
                mime_type=content_type,
                encoding=response.encoding or 'utf-8',
                last_modified=datetime.now(timezone.utc).isoformat(),
                checksum=hashlib.md5(response.content).hexdigest(),
                rows=len(df),
                columns=len(df.columns),
                memory_usage_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
                missing_values=df.isnull().sum().sum(),
                processing_time=time.time() - start_time
            )

            return DataIngestionResult(
                success=True,
                dataframe=df,
                metadata=metadata,
                recommendations=["API data successfully ingested"]
            )

        except Exception as e:
            metadata = FileMetadata(
                file_path=url,
                file_name="api_error",
                file_size=0,
                file_type='api',
                mime_type='unknown',
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

# ============================
# BATCH PROCESSING
# ============================

class BatchProcessor:
    """Handle batch processing of multiple files."""

    def __init__(self):
        self.processor = EnhancedFileProcessor()

    def process_directory(
        self, 
        directory_path: str, 
        pattern: Optional[str] = None,
        recursive: bool = False
    ) -> Dict[str, DataIngestionResult]:
        """Process all supported files in a directory."""

        results = {}

        try:
            directory = Path(directory_path)

            if recursive:
                files = directory.rglob(pattern or '*')
            else:
                files = directory.glob(pattern or '*')

            for file_path in files:
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    if file_ext in self.processor.supported_formats:
                        try:
                            result = self.processor.process_file(str(file_path))
                            results[str(file_path)] = result
                        except Exception as e:
                            # Create error result
                            error_metadata = FileMetadata(
                                file_path=str(file_path),
                                file_name=file_path.name,
                                file_size=file_path.stat().st_size if file_path.exists() else 0,
                                file_type=file_ext,
                                mime_type='unknown',
                                encoding='unknown',
                                last_modified=datetime.now(timezone.utc).isoformat(),
                                checksum='unknown',
                                processing_errors=[str(e)]
                            )

                            results[str(file_path)] = DataIngestionResult(
                                success=False,
                                dataframe=None,
                                metadata=error_metadata,
                                recommendations=[f"Processing failed: {str(e)}"]
                            )

        except Exception as e:
            logging.error(f"Error processing directory {directory_path}: {str(e)}")

        return results

    def combine_results(
        self, 
        results: Dict[str, DataIngestionResult], 
        how: str = 'concat'
    ) -> Optional[DataFrame]:
        """Combine multiple processing results into a single DataFrame."""

        successful_results = [r for r in results.values() if r.success and r.dataframe is not None]

        if not successful_results:
            return None

        dataframes = [r.dataframe for r in successful_results]

        if how == 'concat':
            try:
                return pd.concat(dataframes, ignore_index=True, sort=False)
            except Exception:
                # If concat fails, try to align columns
                all_columns = set()
                for df in dataframes:
                    all_columns.update(df.columns)

                aligned_dfs = []
                for df in dataframes:
                    aligned_df = df.copy()
                    for col in all_columns:
                        if col not in aligned_df.columns:
                            aligned_df[col] = None
                    aligned_dfs.append(aligned_df[sorted(all_columns)])

                return pd.concat(aligned_dfs, ignore_index=True)

        elif how == 'merge':
            result_df = dataframes[0]
            for df in dataframes[1:]:
                # Find common columns for merging
                common_cols = list(set(result_df.columns).intersection(set(df.columns)))
                if common_cols:
                    result_df = result_df.merge(df, on=common_cols, how='outer')
                else:
                    # No common columns, concat instead
                    result_df = pd.concat([result_df, df], ignore_index=True, sort=False)

            return result_df

        else:
            raise ValueError(f"Unsupported combination method: {how}")

# ============================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================

# Initialize global processor instance
_global_processor = EnhancedFileProcessor()
_global_batch_processor = BatchProcessor()

# Initialize API processor only if requests is available
_global_api_processor = None
if requests is not None:
    try:
        _global_api_processor = APIDataIngestion()
    except ImportError:
        pass

def read_csv(path_or_buffer: Union[str, IO], **kwargs) -> DataFrame:
    """Enhanced CSV reading with backward compatibility."""
    try:
        result = _global_processor.process_file(path_or_buffer, **kwargs)
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
        result = _global_processor.process_file(path_or_buffer, **kwargs)
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
        result = _global_processor.process_file(path_or_buffer, **kwargs)
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
        result = _global_processor.process_file(db_source, table_name=table_name, **kwargs)
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

def validate_file_format(file_path: str) -> bool:
    """Enhanced file format validation."""
    supported_extensions = list(_global_processor.supported_formats.keys())
    return any(file_path.lower().endswith(ext) for ext in supported_extensions)

def get_file_info(df: DataFrame) -> dict:
    """Enhanced file information with additional metrics."""
    basic_info = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        "missing_values": int(df.isna().sum().sum()),
        "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
    }

    # Additional enhanced metrics
    enhanced_info = {
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(df.select_dtypes(include=[np.number]).shape[1]),
        "categorical_columns": int(df.select_dtypes(include=['object', 'category']).shape[1]),
        "datetime_columns": int(df.select_dtypes(include=['datetime']).shape[1]),
        "completeness_pct": float(((len(df) * len(df.columns) - df.isna().sum().sum()) / (len(df) * len(df.columns))) * 100) if len(df) > 0 and len(df.columns) > 0 else 100,
        "uniqueness_pct": float(((len(df) - df.duplicated().sum()) / len(df)) * 100) if len(df) > 0 else 100
    }

    return {**basic_info, **enhanced_info}

# ============================
# ENHANCED API FUNCTIONS
# ============================

def process_file_advanced(
    file_source: Union[str, IO], 
    file_name: Optional[str] = None,
    **kwargs
) -> DataIngestionResult:
    """Advanced file processing with comprehensive analysis."""
    return _global_processor.process_file(file_source, file_name, **kwargs)

def fetch_data_from_api(
    url: str,
    method: str = 'GET',
    **kwargs
) -> DataIngestionResult:
    """Fetch data from API endpoint."""
    if _global_api_processor is None:
        raise ImportError("API functionality requires 'requests' library")
    return _global_api_processor.fetch_from_api(url, method, **kwargs)

def process_directory_batch(
    directory_path: str,
    pattern: Optional[str] = None,
    recursive: bool = False
) -> Dict[str, DataIngestionResult]:
    """Process all files in a directory."""
    return _global_batch_processor.process_directory(directory_path, pattern, recursive)

def combine_datasets(
    results: Dict[str, DataIngestionResult],
    how: str = 'concat'
) -> Optional[DataFrame]:
    """Combine multiple datasets."""
    return _global_batch_processor.combine_results(results, how)

# Export all main functions
__all__ = [
    # Enhanced classes
    'EnhancedFileProcessor',
    'APIDataIngestion', 
    'BatchProcessor',
    'FileMetadata',
    'DataIngestionResult',

    # Enhanced functions
    'process_file_advanced',
    'fetch_data_from_api',
    'process_directory_batch',
    'combine_datasets',

    # Backward compatible functions
    'read_csv',
    'read_excel', 
    'read_json',
    'read_sqlite',
    'validate_file_format',
    'get_file_info'
]
