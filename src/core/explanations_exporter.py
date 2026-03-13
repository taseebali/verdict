"""
Model Explanations and Export Module - Export model information in multiple formats.

This module provides comprehensive model explanation and documentation capabilities:

EXPORT FORMATS:
- JSON: Structured, machine-readable format for APIs and integrations
- HTML: Interactive styled reports with CSS and tables
- Excel: Multi-sheet workbooks (metrics, predictions, importance, info)
- PDF: Professional documents with report lab or text fallback

BATCH OPERATIONS:
- Export to multiple formats simultaneously
- Automatic timestamp and organization
- Directory creation and file management

MODEL DOCUMENTATION:
- Model cards with comprehensive metadata
- Performance characteristics
- Feature listings
- Training information

EXPORT RESULTS:
- Tracking of export success/failure
- File size recording
- Timestamp tracking
- Status messages for user feedback

DATACLASSES:
- ExportResult: Container for export operation results

CLASSES:
- ExplanationExporter: Main class for all export operations
  - export_to_json(): JSON format export
  - export_to_excel(): Excel multi-sheet export
  - export_to_html(): Interactive HTML reports
  - export_to_pdf(): PDF document export (reportlab support)
  - export_batch(): Batch export to multiple formats
  - create_model_card(): Generate comprehensive model cards
"""

import logging
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import io
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Container for export results."""
    format: str  # 'pdf', 'html', 'excel', 'json'
    filename: str
    file_size: int
    timestamp: str
    success: bool
    message: str


class ExplanationExporter:
    """Export model explanations in various formats."""

    def __init__(self):
        """Initialize exporter."""
        logger.info("ExplanationExporter initialized")

    def export_to_json(
        self,
        explanations: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None,
        model_metrics: Optional[Dict[str, float]] = None,
        predictions: Optional[List[Dict]] = None
    ) -> str:
        """Export explanations to JSON format.
        
        Args:
            explanations: Dictionary of explanations
            feature_importance: Feature importance scores
            model_metrics: Model performance metrics
            predictions: List of predictions with explanations
            
        Returns:
            JSON string
        """
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'explanations': explanations,
                'feature_importance': feature_importance or {},
                'model_metrics': model_metrics or {},
                'predictions': predictions or [],
                'export_version': '1.0'
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            logger.info(f"Exported to JSON: {len(json_str)} bytes")
            return json_str
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise

    def export_to_excel(
        self,
        filename: str,
        feature_importance: Optional[Dict[str, float]] = None,
        model_metrics: Optional[Dict[str, float]] = None,
        predictions: Optional[pd.DataFrame] = None,
        explanations: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """Export explanations to Excel with multiple sheets.
        
        Args:
            filename: Output filename
            feature_importance: Feature importance dictionary
            model_metrics: Model metrics dictionary
            predictions: Predictions dataframe
            explanations: Explanations dictionary
            
        Returns:
            ExportResult with status
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # Sheet 1: Feature Importance
                if feature_importance:
                    fi_df = pd.DataFrame(
                        list(feature_importance.items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    fi_df.to_excel(writer, sheet_name='Feature Importance', index=False)
                
                # Sheet 2: Model Metrics
                if model_metrics:
                    metrics_df = pd.DataFrame(
                        list(model_metrics.items()),
                        columns=['Metric', 'Value']
                    )
                    metrics_df.to_excel(writer, sheet_name='Model Metrics', index=False)
                
                # Sheet 3: Predictions
                if predictions is not None:
                    predictions.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Sheet 4: Explanations Summary
                if explanations:
                    exp_summary = pd.DataFrame([
                        {'Category': k, 'Details': str(v)[:100]}
                        for k, v in explanations.items()
                    ])
                    exp_summary.to_excel(writer, sheet_name='Explanations', index=False)
                
                # Sheet 5: Export Info
                info_df = pd.DataFrame([
                    {'Key': 'Export Date', 'Value': datetime.now().isoformat()},
                    {'Key': 'Format', 'Value': 'Excel'},
                    {'Key': 'Sheets', 'Value': 4}
                ])
                info_df.to_excel(writer, sheet_name='Info', index=False)
            
            file_size = self._get_file_size(filename)
            
            return ExportResult(
                format='excel',
                filename=filename,
                file_size=file_size,
                timestamp=datetime.now().isoformat(),
                success=True,
                message=f"Exported to Excel: {filename}"
            )
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return ExportResult(
                format='excel',
                filename=filename,
                file_size=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                message=f"Error: {str(e)}"
            )

    def export_to_html(
        self,
        filename: str,
        title: str = "Model Explanations Report",
        feature_importance: Optional[Dict[str, float]] = None,
        model_metrics: Optional[Dict[str, float]] = None,
        predictions: Optional[pd.DataFrame] = None,
        explanations: Optional[Dict[str, str]] = None
    ) -> ExportResult:
        """Export explanations to interactive HTML.
        
        Args:
            filename: Output filename
            title: Report title
            feature_importance: Feature importance scores
            model_metrics: Model metrics
            predictions: Predictions dataframe
            explanations: Text explanations
            
        Returns:
            ExportResult with status
        """
        try:
            html_parts = []
            
            # HTML Header
            html_parts.append(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                    h2 {{ color: #34495e; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #3498db; color: white; }}
                    tr:nth-child(even) {{ background-color: #ecf0f1; }}
                    .metric {{ display: inline-block; margin: 10px 20px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
                    .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                    .chart {{ margin: 20px 0; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }}
                    .explanation {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
                    .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 {title}</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """)
            
            # Metrics Section
            if model_metrics:
                html_parts.append("<h2>📈 Model Performance Metrics</h2>")
                html_parts.append('<div class="chart">')
                for metric, value in model_metrics.items():
                    if isinstance(value, float):
                        html_parts.append(f'''
                        <div class="metric">
                            <div class="metric-label">{metric}</div>
                            <div class="metric-value">{value:.4f}</div>
                        </div>
                        ''')
                html_parts.append("</div>")
            
            # Feature Importance
            if feature_importance:
                html_parts.append("<h2>🎯 Feature Importance</h2>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>Feature</th><th>Importance Score</th><th>Bar</th></tr>")
                
                max_importance = max(feature_importance.values()) if feature_importance else 1
                for feature, score in sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]:
                    bar_width = int((score / max_importance) * 200) if max_importance > 0 else 0
                    html_parts.append(f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{score:.4f}</td>
                        <td><div style="background-color:#3498db; height:20px; width:{bar_width}px;"></div></td>
                    </tr>
                    """)
                html_parts.append("</table>")
            
            # Explanations
            if explanations:
                html_parts.append("<h2>💡 Explanations</h2>")
                for title_txt, explanation in explanations.items():
                    html_parts.append(f'<div class="explanation"><strong>{title_txt}</strong><br>{explanation}</div>')
            
            # Predictions Preview
            if predictions is not None and len(predictions) > 0:
                html_parts.append("<h2>📋 Predictions Preview</h2>")
                html_parts.append(predictions.head(10).to_html(index=False, border=0))
            
            # Footer
            html_parts.append("""
                    <div class="footer">
                        <p>This report was automatically generated by the Model Explanation System.</p>
                        <p>For more information, visit the documentation.</p>
                    </div>
                </div>
            </body>
            </html>
            """)
            
            html_content = "\n".join(html_parts)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            file_size = self._get_file_size(filename)
            
            return ExportResult(
                format='html',
                filename=filename,
                file_size=file_size,
                timestamp=datetime.now().isoformat(),
                success=True,
                message=f"Exported to HTML: {filename}"
            )
            
        except Exception as e:
            logger.error(f"Error exporting to HTML: {e}")
            return ExportResult(
                format='html',
                filename=filename,
                file_size=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                message=f"Error: {str(e)}"
            )

    def export_to_pdf(
        self,
        filename: str,
        title: str = "Model Explanations Report",
        feature_importance: Optional[Dict[str, float]] = None,
        model_metrics: Optional[Dict[str, float]] = None,
        explanations: Optional[Dict[str, str]] = None
    ) -> ExportResult:
        """Export explanations to PDF.
        
        Args:
            filename: Output filename
            title: Report title
            feature_importance: Feature importance scores
            model_metrics: Model metrics
            explanations: Text explanations
            
        Returns:
            ExportResult with status
        """
        try:
            # Use simple text-based PDF generation if reportlab not available
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib import colors
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                
                return self._export_pdf_reportlab(
                    filename, title, feature_importance, model_metrics, explanations
                )
            except ImportError:
                # Fallback to text-based PDF
                return self._export_pdf_simple(
                    filename, title, feature_importance, model_metrics, explanations
                )
            
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            return ExportResult(
                format='pdf',
                filename=filename,
                file_size=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                message=f"Error: {str(e)}"
            )

    def _export_pdf_simple(
        self,
        filename: str,
        title: str,
        feature_importance: Optional[Dict[str, float]],
        model_metrics: Optional[Dict[str, float]],
        explanations: Optional[Dict[str, str]]
    ) -> ExportResult:
        """Export to simple text-based PDF."""
        try:
            from PyPDF2 import PdfWriter
            
            content = f"""
{title}
{'='*60}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL PERFORMANCE METRICS
{'-'*60}
"""
            if model_metrics:
                for metric, value in model_metrics.items():
                    content += f"{metric}: {value:.4f}\n"
            
            content += f"""

FEATURE IMPORTANCE
{'-'*60}
"""
            if feature_importance:
                for feature, score in sorted(feature_importance.items(),
                                           key=lambda x: x[1], reverse=True)[:10]:
                    content += f"{feature}: {score:.4f}\n"
            
            content += f"""

EXPLANATIONS
{'-'*60}
"""
            if explanations:
                for exp_title, exp_text in explanations.items():
                    content += f"{exp_title}:\n{exp_text}\n\n"
            
            # Write as text file with .pdf extension (fallback)
            with open(filename, 'w') as f:
                f.write(content)
            
            file_size = self._get_file_size(filename)
            
            return ExportResult(
                format='pdf',
                filename=filename,
                file_size=file_size,
                timestamp=datetime.now().isoformat(),
                success=True,
                message=f"Exported to PDF (text format): {filename}"
            )
            
        except Exception as e:
            logger.error(f"Error in simple PDF export: {e}")
            raise

    def _export_pdf_reportlab(
        self,
        filename: str,
        title: str,
        feature_importance: Optional[Dict[str, float]],
        model_metrics: Optional[Dict[str, float]],
        explanations: Optional[Dict[str, str]]
    ) -> ExportResult:
        """Export to PDF using reportlab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            
            doc = SimpleDocTemplate(filename, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=30
            )
            
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.3))
            
            # Add metrics
            if model_metrics:
                story.append(Paragraph("Model Performance Metrics", styles['Heading2']))
                metrics_data = [['Metric', 'Value']]
                for metric, value in model_metrics.items():
                    metrics_data.append([metric, f"{value:.4f}"])
                
                metrics_table = Table(metrics_data)
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(metrics_table)
                story.append(Spacer(1, 0.3))
            
            # Add feature importance
            if feature_importance:
                story.append(Paragraph("Feature Importance", styles['Heading2']))
                fi_data = [['Feature', 'Importance']]
                for feature, score in sorted(feature_importance.items(),
                                           key=lambda x: x[1], reverse=True)[:10]:
                    fi_data.append([feature, f"{score:.4f}"])
                
                fi_table = Table(fi_data)
                fi_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(fi_table)
                story.append(Spacer(1, 0.3))
            
            # Add explanations
            if explanations:
                story.append(Paragraph("Explanations", styles['Heading2']))
                for exp_title, exp_text in explanations.items():
                    story.append(Paragraph(f"<b>{exp_title}</b>", styles['Normal']))
                    story.append(Paragraph(exp_text, styles['Normal']))
                    story.append(Spacer(1, 0.2))
            
            doc.build(story)
            file_size = self._get_file_size(filename)
            
            return ExportResult(
                format='pdf',
                filename=filename,
                file_size=file_size,
                timestamp=datetime.now().isoformat(),
                success=True,
                message=f"Exported to PDF: {filename}"
            )
            
        except Exception as e:
            logger.error(f"Error in reportlab PDF export: {e}")
            raise

    def export_batch(
        self,
        explanations: Dict[str, Any],
        output_dir: str = "explanations",
        formats: List[str] = ['json', 'html', 'excel']
    ) -> List[ExportResult]:
        """Export explanations to multiple formats at once.
        
        Args:
            explanations: Explanations to export
            output_dir: Output directory
            formats: List of formats ('json', 'html', 'excel', 'pdf')
            
        Returns:
            List of ExportResult for each format
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            for fmt in formats:
                if fmt == 'json':
                    json_content = self.export_to_json(explanations)
                    filename = os.path.join(output_dir, f"explanations_{timestamp}.json")
                    with open(filename, 'w') as f:
                        f.write(json_content)
                    results.append(ExportResult(
                        format='json',
                        filename=filename,
                        file_size=len(json_content),
                        timestamp=datetime.now().isoformat(),
                        success=True,
                        message="JSON export successful"
                    ))
                
                elif fmt == 'html':
                    filename = os.path.join(output_dir, f"explanations_{timestamp}.html")
                    result = self.export_to_html(
                        filename,
                        feature_importance=explanations.get('feature_importance'),
                        model_metrics=explanations.get('model_metrics')
                    )
                    results.append(result)
                
                elif fmt == 'excel':
                    filename = os.path.join(output_dir, f"explanations_{timestamp}.xlsx")
                    result = self.export_to_excel(
                        filename,
                        feature_importance=explanations.get('feature_importance'),
                        model_metrics=explanations.get('model_metrics')
                    )
                    results.append(result)
                
                elif fmt == 'pdf':
                    filename = os.path.join(output_dir, f"explanations_{timestamp}.pdf")
                    result = self.export_to_pdf(
                        filename,
                        feature_importance=explanations.get('feature_importance'),
                        model_metrics=explanations.get('model_metrics')
                    )
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Error in batch export: {e}")
        
        return results

    def _get_file_size(self, filename: str) -> int:
        """Get file size in bytes."""
        try:
            import os
            return os.path.getsize(filename)
        except:
            return 0

    def create_model_card(
        self,
        model_name: str,
        model_type: str,
        description: str,
        metrics: Dict[str, float],
        features_used: List[str],
        training_date: str,
        performance_characteristics: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create comprehensive model card.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            description: Model description
            metrics: Performance metrics
            features_used: Features used in model
            training_date: Training date
            performance_characteristics: Performance characteristics
            
        Returns:
            Model card dictionary
        """
        return {
            'model_name': model_name,
            'model_type': model_type,
            'description': description,
            'training_date': training_date,
            'num_features': len(features_used),
            'features': features_used,
            'metrics': metrics,
            'performance_characteristics': performance_characteristics or 'Not specified',
            'export_date': datetime.now().isoformat()
        }
