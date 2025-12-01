"""
Export Manager for CortexX Forecasting Platform

PHASE 3 - SESSION 2: Data export capabilities
- Excel export (multi-sheet)
- CSV export
- Filtered data export
"""

import pandas as pd
import io
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExportManager:
    """
    Manages data export operations.
    
    ✅ NEW: Phase 3 - Session 2
    """
    
    @staticmethod
    def export_to_excel(df: pd.DataFrame,
                       stats_df: Optional[pd.DataFrame] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       filename: str = None) -> bytes:
        """
        Export data to Excel with multiple sheets.
        
        Args:
            df: Main DataFrame to export
            stats_df: Optional statistics DataFrame
            metadata: Optional metadata dictionary
            filename: Optional filename (default: auto-generated)
            
        Returns:
            Excel file as bytes
        """
        try:
            # Create BytesIO object
            output = io.BytesIO()
            
            # Create Excel writer
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Sheet 1: Main Data
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Sheet 2: Statistics (if provided)
                if stats_df is not None:
                    stats_df.to_excel(writer, sheet_name='Statistics', index=True)
                
                # Sheet 3: Metadata (if provided)
                if metadata:
                    metadata_df = pd.DataFrame([metadata]).T
                    metadata_df.columns = ['Value']
                    metadata_df.to_excel(writer, sheet_name='Info', index=True)
                
                # Sheet 4: Summary
                summary_data = {
                    'Metric': [
                        'Total Records',
                        'Total Columns',
                        'Export Date',
                        'Export Time'
                    ],
                    'Value': [
                        len(df),
                        len(df.columns),
                        datetime.now().strftime('%Y-%m-%d'),
                        datetime.now().strftime('%H:%M:%S')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Get bytes
            excel_data = output.getvalue()
            logger.info(f"Excel export created: {len(df)} records, {len(df.columns)} columns")
            
            return excel_data
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            raise
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame) -> bytes:
        """
        Export data to CSV.
        
        Args:
            df: DataFrame to export
            
        Returns:
            CSV file as bytes
        """
        try:
            # Convert to CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            logger.info(f"CSV export created: {len(df)} records")
            return csv_data
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise
    
    @staticmethod
    def create_export_metadata(filter_summary: Dict[str, Any],
                              data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for export.
        
        Args:
            filter_summary: Filter information
            data_summary: Data summary information
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Total Records': data_summary.get('total_records', 0),
            'Filtered Records': data_summary.get('filtered_records', 0),
            'Filters Applied': 'Yes' if filter_summary.get('has_filters') else 'No',
        }
        
        # Add filter details if active
        if filter_summary.get('has_filters'):
            if filter_summary.get('start_date'):
                metadata['Filter Start Date'] = str(filter_summary['start_date'])
            if filter_summary.get('end_date'):
                metadata['Filter End Date'] = str(filter_summary['end_date'])
            if filter_summary.get('products'):
                metadata['Products Filtered'] = ', '.join(filter_summary['products'])
            if filter_summary.get('categories'):
                metadata['Categories Filtered'] = ', '.join(filter_summary['categories'])
        
        return metadata


def generate_filename(prefix: str = "cortexx_export", extension: str = "xlsx") -> str:
    """
    Generate filename with timestamp.
    
    Args:
        prefix: Filename prefix
        extension: File extension (without dot)
        
    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}"
