import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class CrossValidationEngine:
    """K-fold cross-validation orchestration for model evaluation."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize cross-validation engine.
        
        Args:
            n_splits: Number of folds (default: 5)
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.fold_results = []

    def run_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        model_name: str,
        task_type: str = "binary"
    ) -> Dict[str, Any]:
        """Execute k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Sklearn model (must have fit/predict methods)
            model_name: Name of the model (for logging)
            task_type: 'binary' or 'multiclass'
            
        Returns:
            Dict with fold results and aggregated metrics (mean ± std)
        """
        logger.info(f"Starting {self.n_splits}-fold CV for {model_name}")
        self.fold_results = []
        
        fold_idx = 0
        for train_idx, test_idx in self.skf.split(X, y):
            fold_idx += 1
            
            # Split data
            X_train_fold = X.iloc[train_idx].reset_index(drop=True)
            X_test_fold = X.iloc[test_idx].reset_index(drop=True)
            y_train_fold = y.iloc[train_idx].reset_index(drop=True)
            y_test_fold = y.iloc[test_idx].reset_index(drop=True)
            
            # Train model on this fold
            try:
                model.fit(X_train_fold, y_train_fold)
            except Exception as e:
                logger.error(f"Fold {fold_idx} training failed: {str(e)}")
                raise
            
            # Calculate metrics for this fold
            fold_metrics = self._calculate_metrics(
                y_test_fold,
                model,
                X_test_fold,
                task_type,
                fold_idx
            )
            self.fold_results.append(fold_metrics)
        
        # Aggregate results across folds
        aggregated = self._aggregate_metrics(self.fold_results)
        
        logger.info(f"CV complete. Mean F1: {aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}")
        
        return {
            "model_name": model_name,
            "n_folds": self.n_splits,
            "fold_results": self.fold_results,
            "aggregated_metrics": aggregated
        }

    def _calculate_metrics(
        self,
        y_test: pd.Series,
        model,
        X_test: pd.DataFrame,
        task_type: str,
        fold_idx: int
    ) -> Dict[str, float]:
        """Calculate metrics for a single fold.
        
        Args:
            y_test: Test target values
            model: Trained model
            X_test: Test features
            task_type: 'binary' or 'multiclass'
            fold_idx: Fold number (for logging)
            
        Returns:
            Dict with accuracy, precision, recall, f1, roc_auc
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        
        # ROC-AUC only for binary or with probability estimates
        try:
            if task_type == "binary":
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            else:
                # Multi-class OvR
                y_proba = model.predict_proba(X_test)
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception as e:
            logger.warning(f"ROC-AUC calculation failed for fold {fold_idx}: {str(e)}")
            metrics["roc_auc"] = None
        
        logger.info(
            f"Fold {fold_idx}: Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}"
        )
        
        return metrics

    def _aggregate_metrics(self, fold_results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across all folds.
        
        Args:
            fold_results: List of metric dicts from each fold
            
        Returns:
            Dict with mean and std for each metric
        """
        aggregated = {}
        
        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            values = [fold[metric_name] for fold in fold_results if fold.get(metric_name) is not None]
            
            if values:
                aggregated[f"{metric_name}_mean"] = np.mean(values)
                aggregated[f"{metric_name}_std"] = np.std(values)
            else:
                aggregated[f"{metric_name}_mean"] = None
                aggregated[f"{metric_name}_std"] = None
        
        return aggregated

    def get_fold_results(self) -> List[Dict]:
        """Return results from last CV run."""
        return self.fold_results
