"""Evaluation metrics module."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from typing import Dict, Tuple, List, Optional
from config.settings import CLASSIFICATION_METRICS, REGRESSION_METRICS
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates evaluation metrics for both classification and regression with multiclass support."""

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
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
                metrics["roc_auc"] = None
        
        return metrics

    @staticmethod
    def calculate_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_pred_proba: np.ndarray = None) -> Dict[str, any]:
        """Calculate comprehensive multiclass metrics including macro and weighted averages.
        
        Args:
            y_true: True labels (multiclass)
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (NxK matrix where K=n_classes)
        
        Returns:
            Dictionary with overall, macro, weighted, and per-class metrics
        """
        n_classes = len(np.unique(y_true))
        classes = np.unique(y_true)
        
        logger.info(f"Calculating multiclass metrics for {n_classes} classes")
        
        metrics = {
            "overall": {
                "accuracy": accuracy_score(y_true, y_pred),
            },
            "macro": {
                "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            },
            "weighted": {
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            },
            "per_class": {}
        }
        
        # Per-class metrics
        for class_label in classes:
            y_true_binary = (y_true == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)
            
            metrics["per_class"][class_label] = {
                "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
                "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
                "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
                "support": np.sum(y_true == class_label),
            }
        
        # ROC-AUC for multiclass OvR
        if y_pred_proba is not None:
            try:
                metrics["overall"]["roc_auc_ovr"] = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")
                metrics["overall"]["roc_auc_ovo"] = roc_auc_score(y_true, y_pred_proba, multi_class="ovo", average="weighted")
            except Exception as e:
                logger.warning(f"Could not calculate multiclass ROC-AUC: {str(e)}")
        
        return metrics

    @staticmethod
    def get_metric_summary(metrics: Dict) -> Dict:
        """Create a summary of multiclass metrics in tabular format.
        
        Args:
            metrics: Output from calculate_multiclass_metrics()
        
        Returns:
            Flattened dictionary suitable for display
        """
        summary = {
            "accuracy": metrics["overall"]["accuracy"],
            "precision_macro": metrics["macro"]["precision"],
            "precision_weighted": metrics["weighted"]["precision"],
            "recall_macro": metrics["macro"]["recall"],
            "recall_weighted": metrics["weighted"]["recall"],
            "f1_macro": metrics["macro"]["f1"],
            "f1_weighted": metrics["weighted"]["f1"],
        }
        
        if "roc_auc_ovr" in metrics["overall"]:
            summary["roc_auc_ovr"] = metrics["overall"]["roc_auc_ovr"]
        if "roc_auc_ovo" in metrics["overall"]:
            summary["roc_auc_ovo"] = metrics["overall"]["roc_auc_ovo"]
        
        return summary

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
