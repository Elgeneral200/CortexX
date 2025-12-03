"""
PDF Report Generator for CortexX Forecasting Platform
PHASE 3 - SESSION 4: Professional PDF Reports with WeasyPrint
✅ Beautiful, enterprise-grade PDF reports
"""

from io import BytesIO
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import os

# ✅ Set DLL directory for Windows
if os.name == 'nt':  # Windows
    os.environ.setdefault('WEASYPRINT_DLL_DIRECTORIES', r'C:\msys64\mingw64\bin')

# Try WeasyPrint first (better quality)
PDF_ENGINE = None
weasyprint_available = False
xhtml2pdf_available = False

try:
    from weasyprint import HTML, CSS
    weasyprint_available = True
    PDF_ENGINE = 'weasyprint'
    print("✅ Using WeasyPrint for PDF generation (HIGH QUALITY)")
except ImportError:
    print("⚠️ WeasyPrint not available, falling back to xhtml2pdf")

try:
    from xhtml2pdf import pisa
    xhtml2pdf_available = True
    if PDF_ENGINE is None:
        PDF_ENGINE = 'xhtml2pdf'
        print("✅ Using xhtml2pdf for PDF generation (BASIC QUALITY)")
except ImportError:
    print("❌ No PDF engine available")

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except:
        return default


class PDFReportGenerator:
    """Generate professional PDF reports for business intelligence."""
    
    @staticmethod
    def generate_dashboard_report(
        filtered_df: pd.DataFrame,
        full_df: pd.DataFrame,
        filter_summary: Dict[str, Any],
        comparison_data: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Generate comprehensive executive dashboard PDF report."""
        if PDF_ENGINE is None:
            raise ImportError(
                "No PDF engine available. Install WeasyPrint:\n"
                "pip install weasyprint"
            )
        
        try:
            html_content = PDFReportGenerator._build_professional_report(
                filtered_df, full_df, filter_summary, comparison_data
            )
            
            # Use WeasyPrint if available (better quality)
            if PDF_ENGINE == 'weasyprint' and weasyprint_available:
                pdf_bytes = PDFReportGenerator._weasyprint_convert(html_content)
                logger.info("✅ PDF generated using WeasyPrint (HIGH QUALITY)")
            elif PDF_ENGINE == 'xhtml2pdf' and xhtml2pdf_available:
                pdf_bytes = PDFReportGenerator._xhtml2pdf_convert(html_content)
                logger.info("✅ PDF generated using xhtml2pdf (BASIC QUALITY)")
            else:
                raise ImportError(f"PDF engine '{PDF_ENGINE}' not available")
            
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise
    
    @staticmethod
    def _build_professional_report(
        filtered_df: pd.DataFrame,
        full_df: pd.DataFrame,
        filter_summary: Dict[str, Any],
        comparison_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build professional HTML report with beautiful styling."""
        
        # Calculate metrics
        total_records = len(filtered_df)
        total_full = len(full_df)
        retention_pct = (total_records / total_full * 100) if total_full > 0 else 0
        
        missing_count = int(filtered_df.isnull().sum().sum())
        total_cells = len(filtered_df) * len(filtered_df.columns)
        completeness = ((total_cells - missing_count) / total_cells * 100) if total_cells > 0 else 0
        
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        numeric_count = len(numeric_cols)
        
        primary_metric = numeric_cols[0] if numeric_cols else None
        
        # Calculate statistics
        metric_sum = 0.0
        metric_mean = 0.0
        metric_median = 0.0
        metric_std = 0.0
        metric_min = 0.0
        metric_max = 0.0
        
        if primary_metric and primary_metric in filtered_df.columns:
            try:
                clean_series = filtered_df[primary_metric].dropna()
                if len(clean_series) > 0:
                    metric_sum = safe_float(clean_series.sum())
                    metric_mean = safe_float(clean_series.mean())
                    metric_median = safe_float(clean_series.median())
                    metric_std = safe_float(clean_series.std())
                    metric_min = safe_float(clean_series.min())
                    metric_max = safe_float(clean_series.max())
            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")
                primary_metric = None
        
        # Date range
        date_cols = filtered_df.select_dtypes(include=['datetime64']).columns
        date_range_text = "Full dataset"
        
        if len(date_cols) > 0:
            try:
                date_col = date_cols[0]
                date_series = filtered_df[date_col].dropna()
                if len(date_series) > 0:
                    min_date = date_series.min()
                    max_date = date_series.max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        date_range_text = f"{min_date.strftime('%B %d, %Y')} to {max_date.strftime('%B %d, %Y')}"
            except Exception as e:
                logger.warning(f"Error with date range: {e}")
        
        # Executive insights
        insights = []
        
        if retention_pct < 50:
            insights.append(f"⚠️ <strong>Data Filtering:</strong> Analysis includes {retention_pct:.1f}% of total dataset ({total_records:,} of {total_full:,} records)")
        else:
            insights.append(f"✓ <strong>Data Coverage:</strong> {retention_pct:.1f}% of dataset ({total_records:,} records)")
        
        if completeness >= 95:
            insights.append(f"✓ <strong>Excellent Data Quality:</strong> {completeness:.1f}% complete")
        elif completeness >= 80:
            insights.append(f"⚠️ <strong>Good Data Quality:</strong> {completeness:.1f}% complete")
        else:
            insights.append(f"❌ <strong>Data Quality Concern:</strong> {completeness:.1f}% complete")
        
        if comparison_data:
            variance = safe_float(comparison_data.get('variance_pct', 0))
            if variance > 10:
                insights.append(f"📈 <strong>Strong Growth:</strong> {variance:+.1f}% increase vs previous period")
            elif variance > 0:
                insights.append(f"↗️ <strong>Positive Trend:</strong> {variance:+.1f}% growth")
            elif variance > -10:
                insights.append(f"↘️ <strong>Slight Decline:</strong> {variance:+.1f}% vs previous")
            else:
                insights.append(f"📉 <strong>Significant Decline:</strong> {variance:+.1f}% vs previous")
        
        insights_html = "".join([f"<li>{insight}</li>" for insight in insights])
        
        # Build beautiful HTML (WeasyPrint supports modern CSS!)
        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CortexX Executive Report</title>
<style>
@page {{
    size: A4;
    margin: 2cm;
}}

body {{
    font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
    color: #2c3e50;
    line-height: 1.6;
    font-size: 11pt;
}}

.header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    text-align: center;
    border-radius: 10px;
    margin-bottom: 30px;
}}

.header h1 {{
    margin: 0;
    font-size: 32pt;
    font-weight: 700;
}}

.header p {{
    margin: 10px 0 0 0;
    font-size: 12pt;
    opacity: 0.95;
}}

.metadata {{
    background: #f8f9fa;
    padding: 15px 20px;
    border-radius: 8px;
    margin-bottom: 25px;
    border-left: 4px solid #667eea;
}}

.section {{
    margin-bottom: 30px;
    page-break-inside: avoid;
}}

.section h2 {{
    color: #2c3e50;
    border-bottom: 3px solid #667eea;
    padding-bottom: 8px;
    font-size: 16pt;
    font-weight: 600;
    margin-top: 25px;
}}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin: 20px 0;
}}

.kpi-card {{
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.kpi-card h3 {{
    margin: 0 0 10px 0;
    font-size: 10pt;
    color: #6c757d;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

.kpi-card p {{
    margin: 0;
    font-size: 24pt;
    font-weight: 700;
    color: #667eea;
}}

.kpi-card .label {{
    font-size: 9pt;
    color: #95a5a6;
    margin-top: 5px;
}}

.insight-box {{
    background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%);
    border-left: 5px solid #28a745;
    padding: 20px 25px;
    border-radius: 8px;
    margin: 15px 0;
}}

.insight-box ul {{
    margin: 10px 0;
    padding-left: 20px;
    list-style: none;
}}

.insight-box li {{
    margin-bottom: 12px;
    line-height: 1.7;
}}

.comparison-box {{
    background: #fff3cd;
    border: 2px solid #ffc107;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
}}

.comparison-box.positive {{
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-color: #28a745;
}}

.comparison-box.negative {{
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-color: #dc3545;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

th {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px;
    text-align: left;
    font-weight: 600;
}}

td {{
    padding: 10px 12px;
    border-bottom: 1px solid #dee2e6;
}}

tr:nth-child(even) {{
    background: #f8f9fa;
}}

tr:hover {{
    background: #e9ecef;
}}

.footer {{
    margin-top: 50px;
    padding-top: 20px;
    border-top: 2px solid #dee2e6;
    text-align: center;
    font-size: 9pt;
    color: #6c757d;
}}

.footer strong {{
    color: #495057;
}}
</style>
</head>
<body>

<div class="header">
    <h1>📊 Executive Dashboard Report</h1>
    <p>CortexX Forecasting Platform - Business Intelligence Analytics</p>
</div>

<div class="metadata">
    <strong>Report Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
    <strong>Time Period:</strong> {date_range_text}<br/>
    <strong>Records Analyzed:</strong> {total_records:,}
</div>

<div class="section">
    <h2>📋 Executive Summary</h2>
    <div class="insight-box">
        <ul>
{insights_html}
        </ul>
    </div>
</div>

<div class="section">
    <h2>📊 Key Performance Indicators</h2>
    <div class="kpi-grid">
        <div class="kpi-card">
            <h3>Records</h3>
            <p>{total_records:,}</p>
            <div class="label">of {total_full:,} total</div>
        </div>
        <div class="kpi-card">
            <h3>Data Quality</h3>
            <p>{completeness:.1f}%</p>
            <div class="label">completeness</div>
        </div>
        <div class="kpi-card">
            <h3>Features</h3>
            <p>{len(filtered_df.columns)}</p>
            <div class="label">{numeric_count} numeric</div>
        </div>
        <div class="kpi-card">
            <h3>Coverage</h3>
            <p>{retention_pct:.1f}%</p>
            <div class="label">of dataset</div>
        </div>
    </div>
</div>
"""
        
        # Add filters
        if filter_summary.get('has_filters', False):
            html += '<div class="section"><h2>🔍 Active Filters</h2>\n<div class="insight-box" style="background: #e3f2fd; border-left-color: #2196f3;">\n'
            if filter_summary.get('start_date'):
                html += f"<strong>Date Range:</strong> {filter_summary['start_date']} to {filter_summary['end_date']}<br/>\n"
            if filter_summary.get('products'):
                products = filter_summary['products'][:5]
                html += f"<strong>Products:</strong> {', '.join(str(p) for p in products)}"
                if len(filter_summary['products']) > 5:
                    html += f" <em>(+{len(filter_summary['products'])-5} more)</em>"
                html += "<br/>\n"
            if filter_summary.get('categories'):
                categories = filter_summary['categories'][:5]
                html += f"<strong>Categories:</strong> {', '.join(str(c) for c in categories)}"
                if len(filter_summary['categories']) > 5:
                    html += f" <em>(+{len(filter_summary['categories'])-5} more)</em>"
                html += "<br/>\n"
            html += "</div></div>\n"
        
        # Add comparison
        if comparison_data:
            category = comparison_data.get('category', 'Neutral')
            variance_pct = safe_float(comparison_data.get('variance_pct', 0))
            daily_growth = safe_float(comparison_data.get('daily_growth', 0))
            box_class = "positive" if variance_pct > 0 else ("negative" if variance_pct < 0 else "")
            
            html += f"""
<div class="section">
    <h2>⚖️ Period Comparison Analysis</h2>
    <div class="comparison-box {box_class}">
        <strong>Performance Status:</strong> {comparison_data.get('emoji', '📊')} {category}<br/>
        <strong>Overall Change:</strong> <span style="font-size: 20pt; font-weight: bold; color: {'#28a745' if variance_pct > 0 else '#dc3545' if variance_pct < 0 else '#495057'};">{variance_pct:+.1f}%</span><br/>
        <strong>Daily Avg Growth:</strong> {daily_growth:+.2f}%
    </div>
</div>
"""
        
        # Add metrics table
        if primary_metric:
            html += f"""
<div class="section">
    <h2>📊 Key Performance Metrics: {primary_metric}</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Interpretation</th>
        </tr>
        <tr>
            <td><strong>Total</strong></td>
            <td>{metric_sum:,.2f}</td>
            <td>Aggregate sum across all records</td>
        </tr>
        <tr>
            <td><strong>Average</strong></td>
            <td>{metric_mean:,.2f}</td>
            <td>Mean value per record</td>
        </tr>
        <tr>
            <td><strong>Median</strong></td>
            <td>{metric_median:,.2f}</td>
            <td>Typical middle value</td>
        </tr>
        <tr>
            <td><strong>Std Deviation</strong></td>
            <td>{metric_std:,.2f}</td>
            <td>Measure of variability</td>
        </tr>
        <tr>
            <td><strong>Range</strong></td>
            <td>{metric_min:,.2f} - {metric_max:,.2f}</td>
            <td>Min to max observed</td>
        </tr>
    </table>
</div>
"""
        
        # Add data preview
        if not filtered_df.empty:
            html += '<div class="section"><h2>📋 Data Preview (First 10 Records)</h2>\n<table>\n<tr>\n'
            
            preview_cols = list(filtered_df.columns[:6])
            for col in preview_cols:
                html += f"<th>{col}</th>\n"
            html += "</tr>\n"
            
            for idx, row in filtered_df.head(10).iterrows():
                html += "<tr>\n"
                for col in preview_cols:
                    value = row[col]
                    if pd.isna(value):
                        html += "<td style='color: #999;'>—</td>\n"
                    elif isinstance(value, (int, float)):
                        html += f"<td>{value:,.2f}</td>\n"
                    else:
                        html += f"<td>{str(value)[:40]}</td>\n"
                html += "</tr>\n"
            
            html += "</table>\n</div>\n"
        
        # Footer
        html += f"""
<div class="footer">
    <p><strong>CortexX Forecasting Platform</strong> | Enterprise Analytics & Business Intelligence</p>
    <p>This report is confidential and intended for business decision-making purposes.</p>
    <p style="margin-top: 10px;">Generated automatically by CortexX Dashboard | © {datetime.now().year} | All Rights Reserved</p>
</div>

</body>
</html>
"""
        
        return html
    
    @staticmethod
    def _weasyprint_convert(html_content: str) -> bytes:
        """Convert HTML to PDF using WeasyPrint (high quality)."""
        if not weasyprint_available:
            raise ImportError("WeasyPrint not available")
        
        try:
            pdf_bytes = BytesIO()
            HTML(string=html_content).write_pdf(pdf_bytes)
            return pdf_bytes.getvalue()
        except Exception as e:
            logger.error(f"WeasyPrint conversion error: {str(e)}")
            raise
    
    @staticmethod
    def _xhtml2pdf_convert(html_content: str) -> bytes:
        """Convert HTML to PDF using xhtml2pdf (fallback, basic quality)."""
        if not xhtml2pdf_available:
            raise ImportError("xhtml2pdf not available")
        
        try:
            pdf_bytes = BytesIO()
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_bytes)
            
            if pisa_status.err:
                raise Exception(f"PDF generation error: {pisa_status.err}")
            
            return pdf_bytes.getvalue()
        except Exception as e:
            logger.error(f"xhtml2pdf conversion error: {str(e)}")
            raise


def generate_filename_pdf(prefix: str = "cortexx_executive_report") -> str:
    """Generate professional PDF filename with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.pdf"
