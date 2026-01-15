"""Decision audit logging for compliance and traceability."""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np


class DecisionAuditLogger:
    """Logs all prediction decisions, thresholds, confidence scores, and recommended actions for compliance and traceability."""

    def __init__(self, log_dir: str = "audit_logs"):
        """Initialize audit logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.audit_records = []

    def log_prediction(
        self,
        prediction: int,
        probability: float,
        confidence: float,
        model_name: str,
        threshold: float = 0.5,
        recommended_action: str = None,
        feature_values: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Log a single prediction decision."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "prediction": int(prediction),
            "probability": float(probability),
            "confidence": float(confidence),
            "confidence_level": (
                "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
            ),
            "model": model_name,
            "threshold": float(threshold),
            "recommended_action": recommended_action,
            "record_id": len(self.audit_records),
        }

        if feature_values:
            record["features"] = feature_values

        if metadata:
            record["metadata"] = metadata

        self.audit_records.append(record)
        return record

    def log_batch_predictions(
        self,
        predictions: List[int],
        probabilities: List[float],
        confidences: List[float],
        model_name: str,
        threshold: float = 0.5,
        batch_metadata: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Log a batch of predictions."""
        batch_id = str(datetime.now().timestamp())
        records = []

        for i, (pred, prob, conf) in enumerate(zip(predictions, probabilities, confidences)):
            metadata = batch_metadata or {}
            metadata["batch_id"] = batch_id
            metadata["batch_index"] = i

            record = self.log_prediction(
                pred,
                prob,
                conf,
                model_name,
                threshold,
                metadata=metadata,
            )
            records.append(record)

        return records

    def log_threshold_change(
        self,
        model_name: str,
        old_threshold: float,
        new_threshold: float,
        reason: str = None,
        changed_by: str = None,
    ) -> Dict[str, Any]:
        """Log a threshold change event."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "threshold_change",
            "model": model_name,
            "old_threshold": float(old_threshold),
            "new_threshold": float(new_threshold),
            "reason": reason,
            "changed_by": changed_by,
        }

        self.audit_records.append(record)
        return record

    def log_action_taken(
        self,
        prediction_id: int,
        action: str,
        action_result: str = None,
        actioned_by: str = None,
    ) -> Dict[str, Any]:
        """Log an action taken based on prediction."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "action_taken",
            "prediction_id": int(prediction_id),
            "action": action,
            "action_result": action_result,
            "actioned_by": actioned_by,
        }

        self.audit_records.append(record)
        return record

    def save_audit_log(self, filepath: str = None) -> str:
        """Save audit log to JSON file."""
        if filepath is None:
            filepath = self.log_file

        with open(filepath, "w") as f:
            json.dump(self.audit_records, f, indent=2, default=str)

        return str(filepath)

    def export_as_csv(self, filepath: str = None) -> str:
        """Export audit log to CSV."""
        if filepath is None:
            filepath = self.log_dir / f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        flat_records = []
        for record in self.audit_records:
            flat_record = record.copy()
            if "features" in flat_record:
                flat_record["features"] = json.dumps(flat_record["features"])
            if "metadata" in flat_record:
                flat_record["metadata"] = json.dumps(flat_record["metadata"])
            flat_records.append(flat_record)

        df = pd.DataFrame(flat_records)
        df.to_csv(filepath, index=False)

        return str(filepath)

    def get_audit_trail(self, prediction_id: int = None) -> List[Dict[str, Any]]:
        """Get audit trail for a specific prediction or all records."""
        if prediction_id is None:
            return self.audit_records

        return [
            r for r in self.audit_records
            if r.get("record_id") == prediction_id or r.get("prediction_id") == prediction_id
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged decisions."""
        predictions = [r for r in self.audit_records if "prediction" in r]

        if not predictions:
            return {"total_records": 0}

        confidences = [r["confidence"] for r in predictions]
        predictions_list = [r["prediction"] for r in predictions]

        return {
            "total_records": len(self.audit_records),
            "total_predictions": len(predictions),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0,
            "min_confidence": float(np.min(confidences)) if confidences else 0,
            "max_confidence": float(np.max(confidences)) if confidences else 0,
            "high_confidence_count": sum(1 for c in confidences if c > 0.8),
            "medium_confidence_count": sum(1 for c in confidences if 0.6 <= c <= 0.8),
            "low_confidence_count": sum(1 for c in confidences if c < 0.6),
            "positive_predictions": sum(1 for p in predictions_list if p == 1),
            "negative_predictions": sum(1 for p in predictions_list if p == 0),
            "first_record_time": predictions[0]["timestamp"] if predictions else None,
            "last_record_time": predictions[-1]["timestamp"] if predictions else None,
        }

    def get_audit_report(self) -> str:
        """Generate human-readable audit report."""
        stats = self.get_statistics()

        if stats["total_records"] == 0:
            return "No audit records found."

        report = f"""
=== AUDIT LOG REPORT ===
Generated: {datetime.now().isoformat()}

STATISTICS:
- Total Records: {stats['total_records']}
- Total Predictions: {stats['total_predictions']}
- Avg Confidence: {stats['avg_confidence']:.3f}
- High Confidence: {stats['high_confidence_count']} ({stats['high_confidence_count']/stats['total_predictions']*100:.1f}%)
- Medium Confidence: {stats['medium_confidence_count']} ({stats['medium_confidence_count']/stats['total_predictions']*100:.1f}%)
- Low Confidence: {stats['low_confidence_count']} ({stats['low_confidence_count']/stats['total_predictions']*100:.1f}%)
- Positive Predictions: {stats['positive_predictions']}
- Negative Predictions: {stats['negative_predictions']}
- Time Range: {stats['first_record_time']} to {stats['last_record_time']}
"""

        return report

    def clear_logs(self) -> None:
        """Clear all audit records from memory."""
        self.audit_records = []

    def load_audit_log(self, filepath: str) -> List[Dict[str, Any]]:
        """Load audit log from file."""
        with open(filepath, "r") as f:
            self.audit_records = json.load(f)

        return self.audit_records
