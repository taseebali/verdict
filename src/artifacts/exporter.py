"""Model export and artifact management."""

import joblib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ModelExporter:
    """Exports trained models and metadata."""

    def __init__(self, output_dir: str = "models"):
        """Initialize exporter."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def export_model(self, model, model_name: str, metadata: Optional[Dict] = None) -> Dict[str, str]:
        """Export a trained model using joblib."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Path(self.output_dir) / f"{model_name}_{timestamp}.joblib"
        
        joblib.dump(model, str(model_path))
        
        result = {
            "model_path": str(model_path),
            "model_name": model_name,
            "timestamp": timestamp,
            "file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2)
        }
        
        if metadata:
            metadata_path = Path(self.output_dir) / f"{model_name}_{timestamp}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            result["metadata_path"] = str(metadata_path)
        
        return result

    def export_all_models(self, pipeline, eval_results: Dict[str, Dict]) -> Dict[str, Dict[str, str]]:
        """Export all trained models from pipeline."""
        exports = {}
        
        for model_name, model in pipeline.model_manager.get_models().items():
            metadata = {
                "model_type": type(model).__name__,
                "task_type": pipeline.model_manager.task_type,
                "training_date": datetime.now().isoformat(),
                "metrics": eval_results.get(model_name, {}),
                "feature_names": pipeline.preprocessor.get_feature_names(),
                "feature_count": len(pipeline.preprocessor.get_feature_names()),
            }
            
            exports[model_name] = self.export_model(model, model_name, metadata)
        
        return exports

    def load_model(self, model_path: str):
        """Load a saved model."""
        return joblib.load(model_path)
