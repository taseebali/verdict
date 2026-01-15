"""HTML report generation for model results."""

from datetime import datetime
from typing import Dict, List, Any
import json


class ReportGenerator:
    """Generates HTML reports of model results."""

    def __init__(self, title: str = "Verdict Report"):
        """Initialize report generator."""
        self.title = title

    def generate_html_report(self, pipeline, eval_results: Dict[str, Dict], 
                            data_info: Dict[str, Any], output_file: str = "report.html") -> str:
        """Generate a comprehensive HTML report."""
        
        html_content = self._create_html_template(pipeline, eval_results, data_info)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file

    def _create_html_template(self, pipeline, eval_results: Dict[str, Dict], 
                              data_info: Dict[str, Any]) -> str:
        """Create HTML template for report."""
        
        models_html = self._generate_models_section(eval_results)
        data_html = self._generate_data_section(data_info)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{self.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            font-size: 1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.3em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .metric {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 20px;
            margin: 10px 10px 10px 0;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .task-type {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1> {self.title}</h1>
            <p class="subtitle">Machine Learning Model Report</p>
            <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="section">
            <h2> Dataset Overview</h2>
            {data_html}
        </div>

        <div class="section">
            <h2> Model Performance</h2>
            <p><strong>Task Type:</strong> <span class="task-type">{pipeline.model_manager.task_type.upper()}</span></p>
            <p><strong>Models Trained:</strong> {len(eval_results)}</p>
            {models_html}
        </div>

        <footer>
            <p>Verdict v1.0 | Powered by AutoML & Explainable AI</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _generate_data_section(self, data_info: Dict[str, Any]) -> str:
        """Generate data overview section."""
        html = f"""
            <div class="metric">
                <div class="metric-value">{data_info.get('rows', 'N/A')}</div>
                <div class="metric-label">Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data_info.get('columns', 'N/A')}</div>
                <div class="metric-label">Features</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data_info.get('missing', 'N/A')}</div>
                <div class="metric-label">Missing Values</div>
            </div>
"""
        return html

    def _generate_models_section(self, eval_results: Dict[str, Dict]) -> str:
        """Generate models comparison section."""
        html = "<table><thead><tr><th>Model</th>"
        
        # Get all metric names
        all_metrics = set()
        for metrics in eval_results.values():
            if isinstance(metrics, dict):
                all_metrics.update(metrics.keys())
        
        for metric in sorted(all_metrics):
            html += f"<th>{metric.upper()}</th>"
        
        html += "</tr></thead><tbody>"
        
        # Add rows for each model
        for model_name, metrics in eval_results.items():
            html += f"<tr><td><strong>{model_name.replace('_', ' ').title()}</strong></td>"
            
            if isinstance(metrics, dict):
                for metric in sorted(all_metrics):
                    value = metrics.get(metric, "N/A")
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    html += f"<td>{value}</td>"
            else:
                html += f"<td colspan='{len(all_metrics)}'>Error: {metrics}</td>"
            
            html += "</tr>"
        
        html += "</tbody></table>"
        return html
