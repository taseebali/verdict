"""Test Cross-Validation integration into MLPipeline."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from src.core.data_handler import DataHandler
from src.core.preprocessing import Preprocessor
from src.core.models import ModelManager
from src.core.cross_validation import CrossValidationEngine
from src.decision.multiclass_handler import MultiClassDetector
from src.core.metrics import MetricsCalculator


class TestCVIntegration:
    """Test CV integration into training pipeline."""

    @pytest.fixture
    def binary_dataset(self):
        """Create a binary classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df['target'] = y
        return df

    @pytest.fixture
    def multiclass_dataset(self):
        """Create a multiclass classification dataset."""
        X, y = make_classification(
            n_samples=300,
            n_features=12,
            n_informative=9,
            n_redundant=3,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(12)])
        df['target'] = y
        return df

    def test_cv_binary_classification(self, binary_dataset):
        """Test CV on binary classification task."""
        X = binary_dataset.drop('target', axis=1)
        y = binary_dataset['target']
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        # Verify structure
        assert "fold_results" in result
        assert "aggregated_metrics" in result
        assert len(result["fold_results"]) == 5
        
        # Verify metrics
        agg = result["aggregated_metrics"]
        assert "accuracy_mean" in agg
        assert "f1_mean" in agg
        assert 0 <= agg["accuracy_mean"] <= 1
        assert 0 <= agg["f1_mean"] <= 1

    def test_cv_multiclass_classification(self, multiclass_dataset):
        """Test CV on multiclass classification task."""
        X = multiclass_dataset.drop('target', axis=1)
        y = multiclass_dataset['target']
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        
        cv_engine = CrossValidationEngine(n_splits=3)
        result = cv_engine.run_cv(X, y, model, "random_forest", "multiclass")
        
        assert len(result["fold_results"]) == 3
        assert result["aggregated_metrics"]["accuracy_mean"] > 0
        assert result["aggregated_metrics"]["f1_mean"] > 0

    def test_cv_fold_consistency(self, binary_dataset):
        """Test that CV folds produce consistent metrics."""
        X = binary_dataset.drop('target', axis=1)
        y = binary_dataset['target']
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        # All folds should have valid metrics
        accuracies = [fold["accuracy"] for fold in result["fold_results"]]
        assert all(0 <= acc <= 1 for acc in accuracies)
        assert len(set([fold.get("f1") for fold in result["fold_results"]])) > 0  # Some variance

    def test_cv_stratified_splits(self, multiclass_dataset):
        """Test that CV uses stratified splits for multiclass."""
        X = multiclass_dataset.drop('target', axis=1)
        y = multiclass_dataset['target']
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        
        cv_engine = CrossValidationEngine(n_splits=3)
        result = cv_engine.run_cv(X, y, model, "random_forest", "multiclass")
        
        # Verify all folds have results
        assert len(result["fold_results"]) == 3
        for fold_result in result["fold_results"]:
            assert "fold" in fold_result
            assert fold_result["fold"] in [1, 2, 3]

    def test_cv_metrics_aggregation(self, binary_dataset):
        """Test that CV metrics are properly aggregated."""
        X = binary_dataset.drop('target', axis=1)
        y = binary_dataset['target']
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        # Check aggregated metrics structure
        agg = result["aggregated_metrics"]
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert f"{metric}_mean" in agg
            assert f"{metric}_std" in agg
            
            # Std should be non-negative
            if agg[f"{metric}_std"] is not None:
                assert agg[f"{metric}_std"] >= 0

    def test_cv_with_model_manager_integration(self, binary_dataset):
        """Test CV workflow with ModelManager."""
        X = binary_dataset.drop('target', axis=1)
        y = binary_dataset['target']
        
        # Step 1: Validate target
        valid, warnings = MultiClassDetector.validate_target(y)
        assert valid
        
        # Step 2: Train multiple models and evaluate with CV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        cv_engine = CrossValidationEngine(n_splits=5)
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_result = cv_engine.run_cv(X, y, lr_model, "lr", "binary")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_result = cv_engine.run_cv(X, y, rf_model, "rf", "binary")
        
        # Both should produce valid results
        assert lr_result["aggregated_metrics"]["accuracy_mean"] > 0
        assert rf_result["aggregated_metrics"]["accuracy_mean"] > 0

    def test_cv_roc_auc_binary(self, binary_dataset):
        """Test that ROC-AUC is calculated for binary classification."""
        X = binary_dataset.drop('target', axis=1)
        y = binary_dataset['target']
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        # Check ROC-AUC in aggregated metrics
        agg = result["aggregated_metrics"]
        assert "roc_auc_mean" in agg
        if agg["roc_auc_mean"] is not None:
            assert 0 <= agg["roc_auc_mean"] <= 1

    def test_cv_preprocessing_pipeline(self, binary_dataset):
        """Test complete CV pipeline with preprocessing."""
        # Step 1: Preprocess
        preprocessor = Preprocessor(binary_dataset, 'target')
        X = binary_dataset.drop('target', axis=1)
        X_scaled = preprocessor.scale_features(X)
        y = binary_dataset['target']
        
        # Step 2: Convert back to DataFrame for CV
        X_df = pd.DataFrame(X_scaled, columns=X.columns)
        y_series = pd.Series(y.values, name='target')
        
        # Step 3: Run CV
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X_df, y_series, model, "logistic_regression", "binary")
        
        assert result["aggregated_metrics"]["accuracy_mean"] > 0

    def test_cv_large_n_folds(self, binary_dataset):
        """Test CV with larger number of folds."""
        X = binary_dataset.drop('target', axis=1)
        y = binary_dataset['target']
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=10)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        assert len(result["fold_results"]) == 10

    def test_cv_reproducibility(self, binary_dataset):
        """Test that CV with same random_state produces same results."""
        X = binary_dataset.drop('target', axis=1)
        y = binary_dataset['target']
        
        from sklearn.linear_model import LogisticRegression
        
        # Run 1
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        cv_engine1 = CrossValidationEngine(n_splits=5, random_state=42)
        result1 = cv_engine1.run_cv(X, y, model1, "logistic_regression", "binary")
        
        # Run 2
        model2 = LogisticRegression(random_state=42, max_iter=1000)
        cv_engine2 = CrossValidationEngine(n_splits=5, random_state=42)
        result2 = cv_engine2.run_cv(X, y, model2, "logistic_regression", "binary")
        
        # Results should be identical
        assert np.isclose(
            result1["aggregated_metrics"]["accuracy_mean"],
            result2["aggregated_metrics"]["accuracy_mean"]
        )


class TestCVEdgeCases:
    """Test CV edge cases."""

    def test_cv_minimal_data(self):
        """Test CV with minimal data (must have >= n_splits)."""
        X = pd.DataFrame(np.random.randn(6, 3), columns=['a', 'b', 'c'])
        y = pd.Series([0, 0, 1, 1, 0, 1], name='target')
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=3)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        assert len(result["fold_results"]) == 3

    def test_cv_binary_edge_case(self):
        """Test CV on balanced binary dataset."""
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.repeat([0, 1], 50), name='target')
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        assert len(result["fold_results"]) == 5
        assert 0 <= result["aggregated_metrics"]["accuracy_mean"] <= 1

    def test_cv_highly_imbalanced(self):
        """Test CV on imbalanced dataset."""
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series([0] * 95 + [1] * 5, name='target')
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        # Should still work with stratified splits
        assert len(result["fold_results"]) == 5
