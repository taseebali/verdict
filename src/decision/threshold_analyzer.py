"""Threshold control and tradeoff analysis for classification models."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc


class ThresholdAnalyzer:
    """
    Analyzes classification metrics across different decision thresholds.
    Enables users to see precision-recall tradeoffs and select optimal thresholds.
    """

    def __init__(self):
        """Initialize threshold analyzer."""
        self.threshold_cache: Dict[str, Dict] = {}

    def analyze_thresholds(
        self, y_true: np.ndarray, y_proba: np.ndarray, step: float = 0.01
    ) -> pd.DataFrame:
        """
        Calculate metrics at multiple thresholds.

        Args:
            y_true: True labels (0 or 1)
            y_proba: Predicted probabilities (float 0-1)
            step: Threshold step size (default 0.01 for 100 thresholds)

        Returns:
            DataFrame with metrics at each threshold
        """
        thresholds = np.arange(0, 1 + step, step)
        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate metrics
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            results.append(
                {
                    "threshold": round(threshold, 2),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "specificity": round(specificity, 4),
                    "accuracy": round(accuracy, 4),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                }
            )

        return pd.DataFrame(results)

    def get_metrics_at_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray, threshold: float
    ) -> Dict[str, float]:
        """
        Calculate metrics at a specific threshold.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            threshold: Decision threshold

        Returns:
            Dictionary with metrics
        """
        y_pred = (y_proba >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "accuracy": accuracy,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    def get_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f1",
        step: float = 0.01,
    ) -> float:
        """
        Find the optimal threshold based on a metric.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ("f1", "precision", "recall", "accuracy")
            step: Threshold step size

        Returns:
            Optimal threshold value
        """
        thresholds_df = self.analyze_thresholds(y_true, y_proba, step)
        best_idx = thresholds_df[metric].idxmax()
        return float(thresholds_df.loc[best_idx, "threshold"])

    def get_precision_recall_curve_data(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, List]:
        """
        Get precision-recall curve data for visualization.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary with precision, recall, and thresholds for plotting
        """
        # Determine the positive label if not numeric
        pos_label = None
        if hasattr(y_true, 'dtype') and y_true.dtype == 'object':
            # For string labels, use the first unique value
            unique_vals = pd.Series(y_true).unique()
            if len(unique_vals) == 2:
                pos_label = unique_vals[0]
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba, pos_label=pos_label)

        return {
            "precision": precisions.tolist(),
            "recall": recalls.tolist(),
            "thresholds": thresholds.tolist(),
        }

    def get_roc_curve_data(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, List]:
        """
        Get ROC curve data for visualization.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary with FPR, TPR, and thresholds for plotting
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(roc_auc),
        }

    def compare_thresholds(
        self, y_true: np.ndarray, y_proba: np.ndarray, thresholds: List[float]
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple user-specified thresholds.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: List of thresholds to compare

        Returns:
            DataFrame with metrics for each threshold
        """
        results = []
        for threshold in thresholds:
            metrics = self.get_metrics_at_threshold(y_true, y_proba, threshold)
            results.append(metrics)

        return pd.DataFrame(results)
