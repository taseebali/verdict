"""Comprehensive tests for multiclass metrics calculation."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.core.metrics import MetricsCalculator


class TestMulticlassMetrics:
    """Test multiclass metric calculations."""
    
    @pytest.fixture
    def multiclass_predictions_3class(self):
        """Generate 3-class test data."""
        np.random.seed(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 5)
        y_pred = np.array([0, 1, 2, 0, 0, 2, 1, 1, 2, 0] * 5)  # Some errors
        
        # Generate reasonable probabilities
        n_samples = len(y_true)
        y_pred_proba = np.random.dirichlet([1, 1, 1], n_samples)
        
        # Make predictions match better for correct predictions
        for i in range(n_samples):
            true_class = y_true[i]
            y_pred_proba[i, true_class] *= 1.5  # Boost true class probability
            y_pred_proba[i] /= y_pred_proba[i].sum()  # Normalize
        
        return y_true, y_pred, y_pred_proba
    
    @pytest.fixture
    def multiclass_predictions_5class(self):
        """Generate 5-class test data."""
        X, y_true = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=5,
            n_informative=4,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Create predictions with some errors
        y_pred = y_true.copy()
        error_indices = np.random.choice(len(y_true), 15, replace=False)
        for idx in error_indices:
            wrong_class = (y_true[idx] + 1) % 5
            y_pred[idx] = wrong_class
        
        # Generate probabilities
        n_samples = len(y_true)
        y_pred_proba = np.random.dirichlet([1]*5, n_samples)
        
        # Boost correct class
        for i in range(n_samples):
            y_pred_proba[i, y_true[i]] *= 2
            y_pred_proba[i] /= y_pred_proba[i].sum()
        
        return y_true, y_pred, y_pred_proba
    
    def test_multiclass_metrics_structure(self, multiclass_predictions_3class):
        """Test that multiclass metrics return correct structure."""
        y_true, y_pred, y_pred_proba = multiclass_predictions_3class
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        assert "overall" in metrics
        assert "macro" in metrics
        assert "weighted" in metrics
        assert "per_class" in metrics
    
    def test_multiclass_overall_accuracy(self, multiclass_predictions_3class):
        """Test overall accuracy calculation."""
        y_true, y_pred, y_pred_proba = multiclass_predictions_3class
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        assert "accuracy" in metrics["overall"]
        assert 0 <= metrics["overall"]["accuracy"] <= 1
    
    def test_multiclass_macro_metrics(self, multiclass_predictions_3class):
        """Test macro-averaged metrics."""
        y_true, y_pred, y_pred_proba = multiclass_predictions_3class
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        assert "precision" in metrics["macro"]
        assert "recall" in metrics["macro"]
        assert "f1" in metrics["macro"]
        
        for metric in ["precision", "recall", "f1"]:
            assert 0 <= metrics["macro"][metric] <= 1
    
    def test_multiclass_weighted_metrics(self, multiclass_predictions_3class):
        """Test weighted-averaged metrics."""
        y_true, y_pred, y_pred_proba = multiclass_predictions_3class
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        assert "precision" in metrics["weighted"]
        assert "recall" in metrics["weighted"]
        assert "f1" in metrics["weighted"]
        
        for metric in ["precision", "recall", "f1"]:
            assert 0 <= metrics["weighted"][metric] <= 1
    
    def test_multiclass_per_class_metrics(self, multiclass_predictions_3class):
        """Test per-class metrics."""
        y_true, y_pred, y_pred_proba = multiclass_predictions_3class
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        assert len(metrics["per_class"]) == 3  # 3 classes
        
        for class_label in [0, 1, 2]:
            assert class_label in metrics["per_class"]
            class_metrics = metrics["per_class"][class_label]
            
            assert "precision" in class_metrics
            assert "recall" in class_metrics
            assert "f1" in class_metrics
            assert "support" in class_metrics
            
            assert 0 <= class_metrics["precision"] <= 1
            assert 0 <= class_metrics["recall"] <= 1
            assert 0 <= class_metrics["f1"] <= 1
            assert class_metrics["support"] > 0
    
    def test_multiclass_roc_auc(self, multiclass_predictions_3class):
        """Test ROC-AUC calculation for multiclass."""
        y_true, y_pred, y_pred_proba = multiclass_predictions_3class
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        assert "roc_auc_ovr" in metrics["overall"]
        assert "roc_auc_ovo" in metrics["overall"]
        assert 0 <= metrics["overall"]["roc_auc_ovr"] <= 1
        assert 0 <= metrics["overall"]["roc_auc_ovo"] <= 1
    
    def test_multiclass_5class_metrics(self, multiclass_predictions_5class):
        """Test multiclass metrics on 5-class problem."""
        y_true, y_pred, y_pred_proba = multiclass_predictions_5class
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        assert len(metrics["per_class"]) == 5
        assert metrics["overall"]["accuracy"] > 0.5  # Should have decent accuracy
    
    def test_metric_summary(self, multiclass_predictions_3class):
        """Test metric summary flattening."""
        y_true, y_pred, y_pred_proba = multiclass_predictions_3class
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        summary = MetricsCalculator.get_metric_summary(metrics)
        
        expected_keys = [
            "accuracy", "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted", "f1_macro", "f1_weighted"
        ]
        
        for key in expected_keys:
            assert key in summary
            assert 0 <= summary[key] <= 1
    
    def test_perfect_predictions_3class(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        
        # Perfect probabilities
        y_pred_proba = np.eye(3)  # Identity matrix for 3 classes
        y_pred_proba = np.vstack([y_pred_proba] * 3)
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        assert metrics["overall"]["accuracy"] == 1.0
        assert metrics["macro"]["precision"] == 1.0
        assert metrics["macro"]["recall"] == 1.0
        assert metrics["macro"]["f1"] == 1.0
    
    def test_imbalanced_multiclass(self):
        """Test metrics with imbalanced classes."""
        y_true = np.array([0]*50 + [1]*30 + [2]*20)
        y_pred = y_true.copy()
        y_pred[np.random.choice(len(y_pred), 10)] = (y_pred[np.random.choice(len(y_pred), 10)] + 1) % 3
        
        y_pred_proba = np.random.dirichlet([1, 1, 1], len(y_true))
        
        metrics = MetricsCalculator.calculate_multiclass_metrics(y_true, y_pred, y_pred_proba)
        
        # Macro and weighted should differ for imbalanced
        assert metrics["macro"]["f1"] != metrics["weighted"]["f1"]
        
        # Support should reflect class imbalance
        assert metrics["per_class"][0]["support"] == 50
        assert metrics["per_class"][1]["support"] == 30
        assert metrics["per_class"][2]["support"] == 20


class TestClassificationMetricsWithProba:
    """Test classification metrics with probability predictions."""
    
    def test_binary_classification_with_proba(self):
        """Test binary classification metrics with probabilities."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1])
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.1, 0.9]
        ])
        
        metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] is not None
    
    def test_multiclass_classification_with_proba(self):
        """Test multiclass classification metrics with probabilities."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_pred_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7]
        ])
        
        metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1


class TestRegressionMetrics:
    """Test regression metrics (existing functionality)."""
    
    def test_regression_metrics_perfect(self):
        """Test regression metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        assert metrics["r2"] == 1.0
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
    
    def test_regression_metrics_with_errors(self):
        """Test regression metrics with errors."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        assert metrics["r2"] < 1.0
        assert metrics["mae"] > 0.0
        assert metrics["rmse"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
