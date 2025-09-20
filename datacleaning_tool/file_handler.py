# filename: file_handler.py
"""
File Handling Module - Enhanced Professional Edition v2.2

Comprehensive file processing and data ingestion system with:
- Multi-format support (CSV, Excel, JSON, SQLite, Parquet, XML, API)
- Advanced error handling and validation
- Performance optimization for large files  
- Data quality assessment during ingestion
- Metadata extraction and cataloging
- Streaming and batch processing capabilities
- Integration with quality and pipeline modules
- Fixed imports and proper error handling

Author: CortexX Team
Version: 2.2.0 - Enhanced Professional Edition with Phase 1 Integration
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

# Try to import optional dependencies
try:
    import requests
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ============================
# ENHANCED DATA STRUCTURES
# ============================

@dataclass
class FileMetadata:
    """Comprehensive file metadata structure."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "file_info": {
                "path": self.filepath,
                "name": self.filename,
                "size_bytes": self.filesize,
                "type": self.filetype,
                "mimetype": self.mimetype,
                "encoding": self.encoding,
                "last_modified": self.last_modified,
                "checksum": self.checksum
            },
            "data_info": {
                "rows": self.rows,
                "columns": self.columns,
                "memory_usage_mb": self.memory_usage_mb,
                "missing_values": self.missing_values,
                "data_quality_score": self.data_quality_score
            },
            "processing_info": {
                "processing_time": self.processing_time,
                "errors": self.processing_errors,
                "warnings": self.processing_warnings
            },
            "schema_info": {
                "column_types": self.column_types,
                "column_summary": self.column_summary
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
            "success": self.success,
            "data_shape": (self.metadata.rows, self.metadata.columns) if self.dataframe is not None else (0, 0),
            "file_info": {
                "name": self.metadata.filename,
                "size_mb": self.metadata.filesize / 1024 / 1024,
                "type": self.metadata.filetype
            },
            "quality_score": self.metadata.data_quality_score,
            "processing_time": self.metadata.processing_time,
            "recommendations_count": len(self.recommendations),
            "has_issues": len(self.metadata.processing_errors) > 0 or len(self.metadata.processing_warnings) > 0
        }

# ============================
# UTILITY FUNCTIONS
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
def safe_file_processing(filepath: str):
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
# ENHANCED FILE PROCESSOR
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
            '.jsonl': self._read_json_lines,
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
            PARQUET_AVAILABLE = True
        except ImportError:
            PARQUET_AVAILABLE = False
        
        self.encoding_detection = True
        self.chunk_size = 50000  # For large file processing
        self.max_file_size_mb = 500  # Maximum file size for direct loading
    
    @performance_monitor
    def process_file(self, filepath_or_buffer: Union[str, IO], filename: Optional[str] = None, **kwargs) -> DataIngestionResult:
        """Process any supported file format with comprehensive analysis."""
        start_time = time.time()
        
        # Determine file path and name
        if isinstance(filepath_or_buffer, str):
            filepath = filepath_or_buffer
            filename = filename or os.path.basename(filepath)
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
            
            # Process the file
            processor = self.supported_formats[file_extension]
            dataframe = processor(filepath_or_buffer, **kwargs)
            
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
    
    def _extract_file_metadata(self, file_source: Union[str, IO], filename: str) -> FileMetadata:
        """Extract comprehensive file metadata."""
        if isinstance(file_source, str):
            # File path
            filepath = file_source
            file_stats = os.stat(filepath)
            filesize = file_stats.st_size
            last_modified = datetime.fromtimestamp(file_stats.st_mtime, timezone.utc).isoformat()
            checksum = self._calculate_checksum(filepath)
            
            # Detect MIME type
            mimetype, _ = mimetypes.guess_type(filepath)
            mimetype = mimetype or 'application/octet-stream'
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
            mimetype = getattr(file_source, 'type', 'application/octet-stream')
        
        return FileMetadata(
            filepath=filepath,
            filename=filename,
            filesize=filesize,
            filetype=Path(filename).suffix.lower(),
            mimetype=mimetype,
            encoding='unknown',
            last_modified=last_modified,
            checksum=checksum
        )
    
    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of file."""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return 'unknown'
    
    def _detect_file_format(self, file_source: Union[str, IO]) -> str:
        """Detect file format by content analysis."""
        try:
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
                return '.zip' if b'xl/' not in header else '.xlsx'
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
            'na_values': ['', 'NULL', 'null', 'NA', 'na', 'N/A', '#N/A'],
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
        
        # Try different encodings if detection is enabled
        encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                params['encoding'] = encoding
                
                # Handle large files with chunking
                if isinstance(file_source, str) and os.path.getsize(file_source) > self.max_file_size_mb * 1024 * 1024:
                    return self._read_csv_chunked(file_source, params)
                else:
                    return pd.read_csv(file_source, **params)
                    
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings_to_try[-1]:  # Last encoding
                    raise ValueError(f"Failed to read CSV file with all attempted encodings: {str(e)}")
                continue
        
        raise ValueError("Unable to read CSV file with any supported encoding")
    
    def _read_csv_chunked(self, filepath: str, **kwargs) -> DataFrame:
        """Read large CSV files in chunks."""
        chunk_size = kwargs.pop('chunksize', self.chunk_size)
        chunks = []
        
        try:
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, **kwargs):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        except Exception as e:
            raise ValueError(f"Error reading chunked CSV: {str(e)}")
    
    def _read_excel_advanced(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced Excel reading with sheet detection and optimization."""
        default_params = {
            'na_values': ['', 'NULL', 'null', 'NA', 'na', 'N/A', '#N/A'],
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
            # Try manual JSON parsing
            try:
                if isinstance(file_source, str):
                    with open(file_source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    content = file_source.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    data = json.loads(content)
                
                # Convert to DataFrame
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to find the data array
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            return pd.DataFrame(value)
                    # If no array found, normalize the dict
                    return pd.json_normalize(data)
                else:
                    raise ValueError("Unsupported JSON structure")
                    
            except Exception as e:
                raise ValueError(f"Failed to read JSON file: {str(e)}")
    
    def _read_json_lines(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read JSON Lines format."""
        try:
            return pd.read_json(file_source, lines=True, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading JSON Lines file: {str(e)}")
    
    def _read_sqlite_advanced(self, db_source: Union[str, IO], **kwargs) -> DataFrame:
        """Advanced SQLite reading with table detection."""
        table_name = kwargs.get('table_name', kwargs.get('table'))
        
        try:
            if isinstance(db_source, str) and os.path.exists(db_source):
                with sqlite3.connect(db_source) as conn:
                    if not table_name:
                        # Get list of tables
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = [row[0] for row in cursor.fetchall()]
                        
                        if not tables:
                            raise ValueError("No tables found in SQLite database")
                        
                        # Use first table or largest table
                        table_name = tables[0]
                    
                    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            else:
                raise ValueError(f"SQLite reading failed: {str(e)}")
                
        except Exception as e:
            raise ValueError(f"SQLite reading failed: {str(e)}")
    
    def _read_xml(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read XML files."""
        if not XML_AVAILABLE:
            raise ImportError("XML parsing requires xml module")
        
        try:
            return pd.read_xml(file_source, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading XML file: {str(e)}")
    
    def _read_parquet(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read Parquet files."""
        try:
            return pd.read_parquet(file_source, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading Parquet file: {str(e)}")
    
    def _read_compressed(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read compressed files (gzip)."""
        try:
            if isinstance(file_source, str):
                with gzip.open(file_source, 'rt', encoding='utf-8') as f:
                    # Detect format of compressed content
                    content_start = f.read(100)
                    f.seek(0)
                    
                    if content_start.startswith(('{', '[')):
                        return pd.read_json(f, **kwargs)
                    else:
                        return pd.read_csv(f, **kwargs)
            else:
                raise ValueError("Compressed file reading from buffer not supported")
                
        except Exception as e:
            raise ValueError(f"Error reading compressed file: {str(e)}")
    
    def _read_zip_archive(self, file_source: Union[str, IO], **kwargs) -> DataFrame:
        """Read ZIP archives and extract data files."""
        try:
            with zipfile.ZipFile(file_source) as zip_file:
                file_list = zip_file.namelist()
                
                # Find data files (exclude directories and system files)
                data_files = [f for f in file_list if not f.endswith('/') 
                             and not f.startswith('__MACOSX/') 
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
        
        # Store column types and basic statistics
        metadata.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                metadata.column_summary[col] = {
                    'type': 'numeric',
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    'missing_count': int(df[col].isnull().sum())
                }
            else:
                metadata.column_summary[col] = {
                    'type': 'categorical',
                    'unique_count': int(df[col].nunique()),
                    'missing_count': int(df[col].isnull().sum()),
                    'most_frequent': str(df[col].mode().iloc[0]) if not df[col].mode().empty else None
                }
    
    def _assess_data_quality(self, df: DataFrame) -> Dict[str, Any]:
        """Perform basic data quality assessment."""
        total_cells = len(df) * len(df.columns) if len(df.columns) > 0 else 0
        missing_count = df.isnull().sum().sum()
        missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df) * 100) if len(df) > 0 else 0
        
        # Calculate overall quality score (0-10)
        completeness_score = max(0, 10 - (missing_pct / 5))  # Lose 1 point per 5% missing
        uniqueness_score = max(0, 10 - (duplicate_pct / 2))  # Lose 1 point per 2% duplicates
        
        overall_score = (completeness_score + uniqueness_score) / 2
        
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
            'structure': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
        }
    
    def _generate_recommendations(self, metadata: FileMetadata, quality_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Data quality recommendations
        completeness = quality_report.get('completeness', {})
        if completeness.get('missing_percentage', 0) > 10:
            recommendations.append(f"High missing data ({completeness['missing_percentage']:.1f}%) - Consider data validation at source")
        
        uniqueness = quality_report.get('uniqueness', {})
        if uniqueness.get('duplicate_percentage', 0) > 1:
            recommendations.append(f"Found {uniqueness['duplicate_percentage']:.1f}% duplicate records - Review data collection process")
        
        # Performance recommendations
        if metadata.memory_usage_mb > 100:
            recommendations.append(f"Large dataset ({metadata.memory_usage_mb:.1f}MB) - Consider data type optimization")
        
        # File-specific recommendations
        if metadata.filetype == '.xlsx' and metadata.rows > 50000:
            recommendations.append("Large Excel file - Consider using CSV format for better performance")
        
        if not recommendations:
            recommendations.append("Data quality looks good! Ready for analysis")
        
        return recommendations

# ============================
# API DATA INGESTION
# ============================

class APIDataIngestion:
    """Handle data ingestion from APIs and web sources."""
    
    def __init__(self):
        if not REQUESTS_AVAILABLE:
            raise ImportError("API ingestion requires requests library")
        
        self.session = requests.Session()
        self.timeout = 30
        self.max_retries = 3
    
    def fetch_from_api(self, url: str, method: str = 'GET', 
                      headers: Optional[Dict[str, str]] = None,
                      params: Optional[Dict[str, Any]] = None, **kwargs) -> DataIngestionResult:
        """Fetch data from API endpoint."""
        start_time = time.time()
        
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
                filepath=url,
                filename=f"api_data_{int(time.time())}",
                filesize=len(response.content),
                filetype='api',
                mimetype=content_type,
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
                filepath=url,
                filename="api_error",
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

# ============================
# BATCH PROCESSING
# ============================

class BatchProcessor:
    """Handle batch processing of multiple files."""
    
    def __init__(self):
        self.processor = EnhancedFileProcessor()
    
    def process_directory(self, directory_path: str, pattern: Optional[str] = None, 
                         recursive: bool = False) -> Dict[str, DataIngestionResult]:
        """Process all supported files in a directory."""
        results = {}
        
        try:
            directory = Path(directory_path)
            
            if recursive:
                files = directory.rglob(pattern or '*')
            else:
                files = directory.glob(pattern or '*')
            
            for filepath in files:
                if filepath.is_file():
                    file_ext = filepath.suffix.lower()
                    if file_ext in self.processor.supported_formats:
                        try:
                            result = self.processor.process_file(str(filepath))
                            results[str(filepath)] = result
                        except Exception as e:
                            # Create error result
                            error_metadata = FileMetadata(
                                filepath=str(filepath),
                                filename=filepath.name,
                                filesize=filepath.stat().st_size if filepath.exists() else 0,
                                filetype=file_ext,
                                mimetype='unknown',
                                encoding='unknown',
                                last_modified=datetime.now(timezone.utc).isoformat(),
                                checksum='unknown',
                                processing_errors=[str(e)]
                            )
                            results[str(filepath)] = DataIngestionResult(
                                success=False,
                                dataframe=None,
                                metadata=error_metadata,
                                recommendations=[f"Processing failed: {str(e)}"]
                            )
                            
        except Exception as e:
            logging.error(f"Error processing directory {directory_path}: {str(e)}")
        
        return results
    
    def combine_results(self, results: Dict[str, DataIngestionResult], 
                       how: str = 'concat') -> Optional[DataFrame]:
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
                # Find common columns for merge
                common_cols = list(set(result_df.columns) & set(df.columns))
                if common_cols:
                    result_df = pd.merge(result_df, df, on=common_cols, how='outer')
                else:
                    # No common columns, concat instead
                    result_df = pd.concat([result_df, df], ignore_index=True, sort=False)
            return result_df
        
        return None

# ============================
# GLOBAL INSTANCES AND FUNCTIONS
# ============================

# Initialize global processor instances
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

def process_file_advanced(file_source: Union[str, IO], filename: Optional[str] = None, **kwargs) -> DataIngestionResult:
    """Advanced file processing with comprehensive analysis."""
    return global_processor.process_file(file_source, filename, **kwargs)

def fetch_data_from_api(url: str, method: str = 'GET', **kwargs) -> DataIngestionResult:
    """Fetch data from API endpoint."""
    if global_api_processor is None:
        raise ImportError("API functionality requires requests library")
    return global_api_processor.fetch_from_api(url, method, **kwargs)

def process_directory_batch(directory_path: str, pattern: Optional[str] = None, 
                           recursive: bool = False) -> Dict[str, DataIngestionResult]:
    """Process all files in a directory."""
    return global_batch_processor.process_directory(directory_path, pattern, recursive)

def combine_datasets(results: Dict[str, DataIngestionResult], how: str = 'concat') -> Optional[DataFrame]:
    """Combine multiple datasets."""
    return global_batch_processor.combine_results(results, how)

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
    
    # Main functions
    'process_file_advanced',
    'fetch_data_from_api',
    'process_directory_batch',
    'combine_datasets',
    
    # Backward compatibility
    'read_csv',
    'read_excel', 
    'read_json',
    'read_sqlite',
    'validate_file_format',
    'get_file_info'
]

print("✅ Enhanced File Handler Module v2.2 - Loaded Successfully!")
print(f"   📁 Supported Formats: {len(global_processor.supported_formats)}")
print(f"   🌐 API Support: {REQUESTS_AVAILABLE}")
print(f"   📊 Excel Support: {OPENPYXL_AVAILABLE or XLRD_AVAILABLE}")
print(f"   🗂️ XML Support: {XML_AVAILABLE}")
print("   🚀 All functions ready for import!")