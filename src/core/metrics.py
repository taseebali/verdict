"""Evaluation metrics module."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
    confusion_matrix, roc_curve, auc
)
from typing import Dict, Tuple
from config.settings import CLASSIFICATION_METRICS, REGRESSION_METRICS


class MetricsCalculator:
    """Calculates evaluation metrics for both classification and regression."""

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        # Calculate ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                # Handle binary and multi-class cases
                if len(np.unique(y_true)) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")
            except Exception:
                metrics["roc_auc"] = None
        
        return metrics

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "mape": mean_absolute_percentage_error(y_true, y_pred),
        }
        
        return metrics

    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix for classification."""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get ROC curve data for binary classification.
        
        Returns:
            Tuple of (fpr, tpr, auc_score)
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score
