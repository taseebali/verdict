"""Auto-generated model cards for documentation and governance."""

from datetime import datetime
from typing import Dict, List, Any
import json


class ModelCardGenerator:
    """
    Generates standardized model cards documenting model usage,
    performance, limitations, and ethical considerations.
    Follows best practices for responsible AI documentation.
    """

    def __init__(self, model_name: str, version: str = "1.0"):
        """
        Initialize model card generator.

        Args:
            model_name: Name of the model
            version: Model version
        """
        self.model_name = model_name
        self.version = version
        self.card_data = {}

    def add_model_details(
        self,
        model_type: str,
        framework: str,
        task_type: str,
        created_date: str = None,
        last_updated: str = None,
    ) -> None:
        """
        Add model technical details.

        Args:
            model_type: Type of model (e.g., "Logistic Regression", "Random Forest")
            framework: Framework used (e.g., "scikit-learn")
            task_type: Task type (e.g., "binary_classification")
            created_date: Model creation date (ISO format)
            last_updated: Last update date (ISO format)
        """
        self.card_data["model_details"] = {
            "name": self.model_name,
            "version": self.version,
            "type": model_type,
            "framework": framework,
            "task_type": task_type,
            "created_date": created_date or datetime.now().isoformat(),
            "last_updated": last_updated or datetime.now().isoformat(),
        }

    def add_intended_use(
        self,
        primary_use: str,
        primary_users: List[str],
        out_of_scope_uses: List[str] = None,
    ) -> None:
        """
        Document intended use and users.

        Args:
            primary_use: Primary use case
            primary_users: List of intended users
            out_of_scope_uses: List of prohibited uses
        """
        self.card_data["intended_use"] = {
            "primary_use": primary_use,
            "primary_users": primary_users,
            "out_of_scope_uses": out_of_scope_uses or [],
        }

    def add_training_data(
        self,
        dataset_name: str,
        dataset_size: int,
        features: List[str],
        target_variable: str,
        data_preprocessing: List[str] = None,
        data_splits: Dict[str, float] = None,
    ) -> None:
        """
        Document training data.

        Args:
            dataset_name: Name of training dataset
            dataset_size: Number of samples
            features: List of feature names
            target_variable: Name of target variable
            data_preprocessing: List of preprocessing steps
            data_splits: Train/val/test splits
        """
        self.card_data["training_data"] = {
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "features": features,
            "target_variable": target_variable,
            "preprocessing": data_preprocessing or [],
            "splits": data_splits or {"train": 0.7, "test": 0.3},
        }

    def add_performance_metrics(
        self,
        metrics: Dict[str, float],
        performance_by_subgroup: Dict[str, Dict[str, float]] = None,
    ) -> None:
        """
        Document model performance.

        Args:
            metrics: Dictionary of metric names and values
            performance_by_subgroup: Performance breakdown by demographic groups
        """
        self.card_data["performance"] = {
            "metrics": metrics,
            "performance_by_subgroup": performance_by_subgroup or {},
        }

    def add_limitations(self, limitations: List[str]) -> None:
        """
        Document model limitations.

        Args:
            limitations: List of known limitations
        """
        self.card_data["limitations"] = {"known_limitations": limitations}

    def add_ethical_considerations(
        self,
        fairness_considerations: List[str] = None,
        bias_mitigation: List[str] = None,
        privacy_measures: List[str] = None,
        other_considerations: List[str] = None,
    ) -> None:
        """
        Document ethical considerations.

        Args:
            fairness_considerations: Fairness-related considerations
            bias_mitigation: Bias mitigation strategies
            privacy_measures: Privacy protection measures
            other_considerations: Other ethical considerations
        """
        self.card_data["ethical_considerations"] = {
            "fairness": fairness_considerations or [],
            "bias_mitigation": bias_mitigation or [],
            "privacy": privacy_measures or [],
            "other": other_considerations or [],
        }

    def add_recommendations(
        self,
        recommended_actions: List[str] = None,
        monitoring_recommendations: List[str] = None,
    ) -> None:
        """
        Document recommendations.

        Args:
            recommended_actions: Recommended deployment actions
            monitoring_recommendations: Monitoring recommendations
        """
        self.card_data["recommendations"] = {
            "actions": recommended_actions or [],
            "monitoring": monitoring_recommendations or [],
        }

    def generate_html_card(self) -> str:
        """
        Generate HTML model card.

        Returns:
            HTML string for the model card
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Model Card - {self.model_name}</title>
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
            background: #ecf0f1;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 30px;
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
            margin-top: 15px;
            margin-bottom: 10px;
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
        ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📋 Model Card</h1>
            <p class="subtitle">Standardized Model Documentation</p>
            <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
"""

        # Model Details
        if "model_details" in self.card_data:
            md = self.card_data["model_details"]
            html += f"""
        <div class="section">
            <h2>📊 Model Details</h2>
            <div class="metric">
                <div class="metric-value">{md.get('name', 'N/A')}</div>
                <div class="metric-label">Model Name</div>
            </div>
            <div class="metric">
                <div class="metric-value">{md.get('version', 'N/A')}</div>
                <div class="metric-label">Version</div>
            </div>
            <div class="metric">
                <div class="metric-value">{md.get('type', 'N/A')}</div>
                <div class="metric-label">Type</div>
            </div>
            <div class="metric">
                <div class="metric-value">{md.get('framework', 'N/A')}</div>
                <div class="metric-label">Framework</div>
            </div>
            <h3>Details</h3>
            <ul>
                <li><strong>Task Type:</strong> {md.get('task_type', 'N/A')}</li>
                <li><strong>Created:</strong> {md.get('created_date', 'N/A')}</li>
                <li><strong>Updated:</strong> {md.get('last_updated', 'N/A')}</li>
            </ul>
        </div>
"""

        # Intended Use
        if "intended_use" in self.card_data:
            iu = self.card_data["intended_use"]
            html += f"""
        <div class="section">
            <h2>🎯 Intended Use</h2>
            <h3>Primary Use</h3>
            <p>{iu.get('primary_use', 'N/A')}</p>
            <h3>Primary Users</h3>
            <ul>
"""
            for user in iu.get("primary_users", []):
                html += f"                <li>{user}</li>\n"

            if iu.get("out_of_scope_uses"):
                html += """            </ul>
            <h3>Out-of-Scope Uses</h3>
            <div class="warning">
                <ul>
"""
                for use in iu.get("out_of_scope_uses", []):
                    html += f"                    <li>{use}</li>\n"
                html += """                </ul>
            </div>
"""
            else:
                html += "            </ul>\n"
            html += "        </div>\n"

        # Training Data
        if "training_data" in self.card_data:
            td = self.card_data["training_data"]
            html += f"""
        <div class="section">
            <h2>📚 Training Data</h2>
            <div class="metric">
                <div class="metric-value">{td.get('dataset_size', 'N/A')}</div>
                <div class="metric-label">Samples</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(td.get('features', []))}</div>
                <div class="metric-label">Features</div>
            </div>
            <h3>Dataset</h3>
            <ul>
                <li><strong>Name:</strong> {td.get('dataset_name', 'N/A')}</li>
                <li><strong>Target:</strong> {td.get('target_variable', 'N/A')}</li>
            </ul>
            <h3>Data Splits</h3>
            <table>
                <tr>
"""
            for split_name, split_pct in td.get("splits", {}).items():
                html += f"                    <th>{split_name.title()}</th>\n"
            html += "                </tr>\n                <tr>\n"
            for split_name, split_pct in td.get("splits", {}).items():
                html += f"                    <td>{split_pct*100:.0f}%</td>\n"
            html += """                </tr>
            </table>
        </div>
"""

        # Performance
        if "performance" in self.card_data:
            perf = self.card_data["performance"]
            html += """
        <div class="section">
            <h2>📈 Performance</h2>
            <h3>Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
"""
            for metric_name, metric_value in perf.get("metrics", {}).items():
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.4f}"
                html += f"                <tr><td>{metric_name}</td><td>{metric_value}</td></tr>\n"
            html += """            </table>
        </div>
"""

        # Limitations
        if "limitations" in self.card_data:
            html += """
        <div class="section">
            <h2>⚠️ Limitations</h2>
            <ul>
"""
            for limitation in self.card_data["limitations"].get("known_limitations", []):
                html += f"                <li>{limitation}</li>\n"
            html += """            </ul>
        </div>
"""

        # Ethical Considerations
        if "ethical_considerations" in self.card_data:
            ec = self.card_data["ethical_considerations"]
            html += """
        <div class="section">
            <h2>⚖️ Ethical Considerations</h2>
"""
            if ec.get("fairness"):
                html += """            <h3>Fairness</h3>
            <ul>
"""
                for item in ec.get("fairness", []):
                    html += f"                <li>{item}</li>\n"
                html += "            </ul>\n"

            if ec.get("bias_mitigation"):
                html += """            <h3>Bias Mitigation</h3>
            <ul>
"""
                for item in ec.get("bias_mitigation", []):
                    html += f"                <li>{item}</li>\n"
                html += "            </ul>\n"

            if ec.get("privacy"):
                html += """            <h3>Privacy Measures</h3>
            <ul>
"""
                for item in ec.get("privacy", []):
                    html += f"                <li>{item}</li>\n"
                html += "            </ul>\n"

            html += "        </div>\n"

        # Recommendations
        if "recommendations" in self.card_data:
            rec = self.card_data["recommendations"]
            html += """
        <div class="section">
            <h2>✅ Recommendations</h2>
"""
            if rec.get("actions"):
                html += """            <h3>Recommended Actions</h3>
            <ul>
"""
                for action in rec.get("actions", []):
                    html += f"                <li>{action}</li>\n"
                html += "            </ul>\n"

            if rec.get("monitoring"):
                html += """            <h3>Monitoring</h3>
            <ul>
"""
                for item in rec.get("monitoring", []):
                    html += f"                <li>{item}</li>\n"
                html += "            </ul>\n"

            html += "        </div>\n"

        html += """
        <footer>
            <p>Verdict v1.0 | AI Decision Copilot | Model Card v1.0</p>
        </footer>
    </div>
</body>
</html>
"""

        return html

    def save_card(self, filepath: str) -> str:
        """
        Save model card to HTML file.

        Args:
            filepath: Output filepath

        Returns:
            Path to saved file
        """
        html = self.generate_html_card()
        with open(filepath, "w") as f:
            f.write(html)

        return filepath

    def export_json(self, filepath: str) -> str:
        """
        Export card data as JSON.

        Args:
            filepath: Output filepath

        Returns:
            Path to saved file
        """
        with open(filepath, "w") as f:
            json.dump(self.card_data, f, indent=2)

        return filepath

    def export_html(self, filepath: str) -> str:
        """
        Export card data as HTML for easy viewing.
        
        Args:
            filepath: Output filepath (should end with .html)
        
        Returns:
            Path to saved file
        """
        html_content = self._generate_html()
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return filepath

    def _generate_html(self) -> str:
        """Generate HTML representation of model card."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Card - {model_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ 
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .section {{ 
            margin-bottom: 40px;
            border-left: 4px solid #667eea;
            padding-left: 20px;
        }}
        .section h2 {{ 
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }}
        .section h2::before {{
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .subsection {{ margin-top: 20px; margin-bottom: 20px; }}
        .subsection h3 {{ color: #555; font-size: 1.2em; margin-bottom: 10px; }}
        .metric {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 15px 0;
        }}
        .metric-item {{
            background: #f7f7f7;
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }}
        .metric-label {{ color: #666; font-size: 0.9em; font-weight: 600; }}
        .metric-value {{ color: #333; font-size: 1.3em; font-weight: bold; margin-top: 5px; }}
        .badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-right: 8px;
            margin-bottom: 8px;
        }}
        .badge.warning {{ background: #f59e0b; }}
        .badge.success {{ background: #10b981; }}
        .badge.danger {{ background: #ef4444; }}
        ul, ol {{ margin-left: 20px; margin-top: 10px; }}
        li {{ margin-bottom: 8px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th {{ 
            background: #f0f0f0;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #667eea;
        }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f9f9f9; }}
        .footer {{
            text-align: center;
            padding: 20px;
            background: #f7f7f7;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #666;
        }}
        .alert {{
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .alert.warning {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            color: #92400e;
        }}
        .alert.info {{
            background: #dbeafe;
            border-left: 4px solid #3b82f6;
            color: #1e40af;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚖️ {model_name}</h1>
            <p>Model Card & Documentation</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Version {version} | Generated {generated_date}</p>
        </div>
        
        <div class="content">
            {sections_html}
        </div>
        
        <div class="footer">
            <p>Generated by Verdict ML Platform</p>
            <p style="margin-top: 10px; font-size: 0.85em;">This model card documents the model's purpose, performance, limitations, and ethical considerations.</p>
        </div>
    </div>
</body>
</html>"""

        sections_html = ""
        
        # Model Details Section
        if "model_details" in self.card_data:
            md = self.card_data["model_details"]
            sections_html += f"""
        <div class="section">
            <h2>📋 Model Details</h2>
            <div class="metric">
                <div class="metric-item">
                    <div class="metric-label">Model Type</div>
                    <div class="metric-value">{md.get('type', 'N/A')}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Framework</div>
                    <div class="metric-value">{md.get('framework', 'N/A')}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Task Type</div>
                    <div class="metric-value">{md.get('task_type', 'N/A')}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Version</div>
                    <div class="metric-value">{md.get('version', 'N/A')}</div>
                </div>
            </div>
            <div class="subsection">
                <h3>Dates</h3>
                <p>Created: {md.get('created_date', 'N/A')}</p>
                <p>Last Updated: {md.get('last_updated', 'N/A')}</p>
            </div>
        </div>
"""
        
        # Intended Use Section
        if "intended_use" in self.card_data:
            iu = self.card_data["intended_use"]
            sections_html += f"""
        <div class="section">
            <h2>🎯 Intended Use</h2>
            <div class="subsection">
                <h3>Primary Use Case</h3>
                <p>{iu.get('primary_use_case', 'N/A')}</p>
            </div>
            <div class="subsection">
                <h3>Intended Users</h3>
                <p>{iu.get('intended_users', 'N/A')}</p>
            </div>
            <div class="subsection">
                <h3>Out-of-Scope Uses</h3>
                <ul>
                    {''.join(f"<li>{use}</li>" for use in iu.get('out_of_scope_uses', []))}
                </ul>
            </div>
        </div>
"""
        
        # Training Data Section
        if "training_data" in self.card_data:
            td = self.card_data["training_data"]
            sections_html += f"""
        <div class="section">
            <h2>📊 Training Data</h2>
            <div class="metric">
                <div class="metric-item">
                    <div class="metric-label">Dataset Name</div>
                    <div class="metric-value">{td.get('dataset_name', 'N/A')}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Dataset Size</div>
                    <div class="metric-value">{td.get('dataset_size', 'N/A')} samples</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Target Variable</div>
                    <div class="metric-value">{td.get('target_variable', 'N/A')}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Features</div>
                    <div class="metric-value">{len(td.get('features', []))} features</div>
                </div>
            </div>
            <div class="subsection">
                <h3>Data Splits</h3>
                <ul>
                    {''.join(f"<li>{k}: {v}</li>" for k,v in td.get('data_splits', {}).items())}
                </ul>
            </div>
            <div class="subsection">
                <h3>Preprocessing</h3>
                <ul>
                    {''.join(f"<li>{p}</li>" for p in td.get('data_preprocessing', []))}
                </ul>
            </div>
        </div>
"""
        
        # Performance Section
        if "performance" in self.card_data:
            perf = self.card_data["performance"]
            sections_html += f"""
        <div class="section">
            <h2>📈 Performance Metrics</h2>
            <div class="metric">
                {''.join(f'<div class="metric-item"><div class="metric-label">{k.replace("_", " ").title()}</div><div class="metric-value">{v:.4f if isinstance(v, float) else v}</div></div>' for k, v in perf.get('metrics', {}).items())}
            </div>
        </div>
"""
        
        # Limitations Section
        if "limitations" in self.card_data:
            lim = self.card_data["limitations"]
            sections_html += f"""
        <div class="section">
            <h2>⚠️ Limitations</h2>
            <div class="alert warning">
                <strong>Important:</strong> This model has important limitations users should be aware of.
            </div>
            <div class="subsection">
                <h3>Known Limitations</h3>
                <ul>
                    {''.join(f"<li>{l}</li>" for l in lim.get('known_limitations', []))}
                </ul>
            </div>
        </div>
"""
        
        # Ethical Considerations Section
        if "ethical_considerations" in self.card_data:
            eth = self.card_data["ethical_considerations"]
            sections_html += f"""
        <div class="section">
            <h2>⚖️ Ethical Considerations</h2>
            <div class="subsection">
                <h3>Bias & Fairness</h3>
                <p>{eth.get('bias_and_fairness', 'N/A')}</p>
            </div>
            <div class="subsection">
                <h3>Privacy</h3>
                <p>{eth.get('privacy_considerations', 'N/A')}</p>
            </div>
            <div class="subsection">
                <h3>Usage Guidelines</h3>
                <ul>
                    {''.join(f"<li>{g}</li>" for g in eth.get('usage_guidelines', []))}
                </ul>
            </div>
        </div>
"""
        
        # Recommendations Section
        if "recommendations" in self.card_data:
            rec = self.card_data["recommendations"]
            sections_html += f"""
        <div class="section">
            <h2>✨ Recommendations</h2>
            <div class="subsection">
                <h3>When to Use This Model</h3>
                <ul>
                    {''.join(f"<li>{r}</li>" for r in rec.get('when_to_use', []))}
                </ul>
            </div>
            <div class="subsection">
                <h3>When NOT to Use This Model</h3>
                <ul>
                    {''.join(f"<li>{r}</li>" for r in rec.get('when_not_to_use', []))}
                </ul>
            </div>
        </div>
"""
        
        return html.format(
            model_name=self.model_name,
            version=self.version,
            generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sections_html=sections_html
        )

    def get_card_summary(self) -> Dict[str, Any]:
        """
        Get summary of card data.

        Returns:
            Dictionary with card summary
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "sections_completed": list(self.card_data.keys()),
            "sections_missing": [
                s
                for s in [
                    "model_details",
                    "intended_use",
                    "training_data",
                    "performance",
                    "limitations",
                    "ethical_considerations",
                    "recommendations",
                ]
                if s not in self.card_data
            ],
        }
