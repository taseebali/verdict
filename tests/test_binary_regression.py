"""End-to-end binary classification regression tests."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from src.core.data_handler import DataHandler
from src.core.preprocessing import Preprocessor
from src.core.models import ModelManager
from src.core.cross_validation import CrossValidationEngine
from src.decision.multiclass_handler import MultiClassDetector
from src.core.metrics import MetricsCalculator


class TestBinaryClassificationPipeline:
    """End-to-end tests for binary classification pipeline."""
    
    @pytest.fixture
    def binary_dataset(self):
        """Create a binary classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=4,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        df['target'] = y
        return df
    
    def test_binary_classification_full_pipeline(self, binary_dataset):
        """Test full pipeline: Load → Preprocess → Train → Predict."""
        # Step 1: Load and validate data
        handler = DataHandler(binary_dataset)
        is_valid, message = handler.validate_data()
        assert is_valid, f"Data validation failed: {message}"
        
        # Step 2: Detect problem type
        target = binary_dataset['target']
        problem_type = MultiClassDetector.detect_problem_type(target)
        assert problem_type == "binary", f"Expected binary, got {problem_type}"
        
        # Step 3: Preprocess
        preprocessor = Preprocessor(binary_dataset, 'target')
        X = binary_dataset.drop('target', axis=1)
        X_processed = preprocessor.scale_features(X)
        y = binary_dataset['target'].values
        
        # Step 4: Split data
        split_idx = int(0.8 * len(X_processed))
        X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Step 5: Train models
        manager = ModelManager(task_type="classification", strategy="binary")
        
        result_lr = manager.train("logistic_regression", X_train, y_train)
        assert result_lr["status"] == "success"
        
        result_rf = manager.train("random_forest", X_train, y_train)
        assert result_rf["status"] == "success"
        
        # Step 6: Make predictions
        pred_lr = manager.predict("logistic_regression", X_test)
        pred_rf = manager.predict("random_forest", X_test)
        
        assert len(pred_lr) == len(X_test)
        assert len(pred_rf) == len(X_test)
        assert all(p in [0, 1] for p in pred_lr)
        assert all(p in [0, 1] for p in pred_rf)
        
        # Step 7: Evaluate
        metrics_lr = MetricsCalculator.calculate_classification_metrics(y_test, pred_lr)
        metrics_rf = MetricsCalculator.calculate_classification_metrics(y_test, pred_rf)
        
        assert 0 <= metrics_lr["accuracy"] <= 1
        assert 0 <= metrics_rf["accuracy"] <= 1
    
    def test_binary_classification_with_cv(self, binary_dataset):
        """Test binary classification with cross-validation."""
        X = binary_dataset.drop('target', axis=1)
        y = binary_dataset['target']
        
        # Validate target
        valid, warnings = MultiClassDetector.validate_target(y)
        assert valid, f"Target validation failed: {warnings}"
        
        # Use sklearn model directly for CV
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Train with CV using sklearn model
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X, y, model, "logistic_regression", "binary")
        
        assert "fold_results" in result
        assert "aggregated_metrics" in result
        assert "f1_mean" in result["aggregated_metrics"]
        assert len(result["fold_results"]) == 5
        
        # Check metrics
        for fold_result in result["fold_results"]:
            assert 0 <= fold_result["accuracy"] <= 1
    
    def test_binary_backward_compatibility(self):
        """Test that binary classification still works with original code."""
        # Create simple binary dataset
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                      [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        # Simple train-test split
        X_train, X_test = X[:8], X[8:]
        y_train, y_test = y[:8], y[8:]
        
        # Train
        manager = ModelManager(task_type="classification", strategy="binary")
        result = manager.train("logistic_regression", X_train, y_train)
        
        assert result["status"] == "success"
        assert result["train_score"] > 0
        
        # Predict
        predictions = manager.predict("logistic_regression", X_test)
        assert len(predictions) == 2
        assert all(p in [0, 1] for p in predictions)
    
    def test_binary_prediction_confidence(self, binary_dataset):
        """Test that prediction confidence is reasonable."""
        X = binary_dataset.drop('target', axis=1).values
        y = binary_dataset['target'].values
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train = y[:split_idx]
        
        manager = ModelManager(task_type="classification", strategy="binary")
        manager.train("logistic_regression", X_train, y_train)
        
        # Get probabilities
        proba = manager.predict_proba("logistic_regression", X_test)
        
        # Check shapes and values
        assert proba.shape[0] == len(X_test)
        assert proba.shape[1] == 2
        assert all(0 <= p <= 1 for row in proba for p in row)
        assert all(abs(row.sum() - 1.0) < 1e-6 for row in proba)  # Sum to 1


class TestDataValidationRegression:
    """Test data validation for regression scenarios."""
    
    def test_data_quality_binary_dataset(self):
        """Test data quality checks on binary dataset."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df['target'] = y
        
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        assert "null_rows" in report
        assert "duplicates" in report
        assert report["null_rows"]["null_row_count"] == 0  # No nulls in synthetic data
    
    def test_edge_case_minimal_binary_data(self):
        """Test edge case with minimal binary dataset."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8],
                      [2, 3], [4, 5], [6, 7], [8, 9],
                      [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        
        df = pd.DataFrame(X, columns=['f1', 'f2'])
        df['target'] = y
        
        handler = DataHandler(df)
        is_valid, message = handler.validate_data()
        assert is_valid


class TestMulticlassIntegration:
    """Test multiclass problem detection and routing."""
    
    def test_multiclass_problem_detection(self):
        """Test that multiclass problems are correctly detected."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_classes=3,
            n_informative=4,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42
        )
        
        problem_type = MultiClassDetector.detect_problem_type(pd.Series(y))
        assert problem_type == "multiclass"
        
        strategy = MultiClassDetector.get_training_strategy(problem_type)
        assert strategy == "ovr"
    
    def test_multiclass_data_quality(self):
        """Test data quality for multiclass."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_classes=3,
            n_informative=4,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df['target'] = y
        
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        assert isinstance(passed, bool)
        assert isinstance(warnings, list)


class TestCVExecution:
    """Test cross-validation execution for binary and multiclass."""
    
    def test_cv_binary_execution(self):
        """Test CV on binary dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_series = pd.Series(y, name='target')
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=5)
        result = cv_engine.run_cv(X_df, y_series, model, "logistic_regression", "binary")
        
        assert result is not None
        assert "fold_results" in result
        assert len(result["fold_results"]) == 5
    
    def test_cv_fold_results_valid(self):
        """Test that CV fold results are valid."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_series = pd.Series(y, name='target')
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_engine = CrossValidationEngine(n_splits=3)
        result = cv_engine.run_cv(X_df, y_series, model, "logistic_regression", "binary")
        
        for fold_result in result["fold_results"]:
            assert "accuracy" in fold_result
            assert "precision" in fold_result or "recall" in fold_result
            assert 0 <= fold_result["accuracy"] <= 1


class TestModelPersistence:
    """Test that trained models can be retrieved and used."""
    
    def test_model_retrieval(self):
        """Test getting trained model."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        manager = ModelManager(task_type="classification", strategy="binary")
        manager.train("logistic_regression", X_train, y_train)
        
        # Get model
        model = manager.get_model("logistic_regression")
        assert model is not None
        assert hasattr(model, "predict")
        
        # Use retrieved model
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_list_trained_models(self):
        """Test listing trained models."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        X_train = X[:80]
        y_train = y[:80]
        
        manager = ModelManager(task_type="classification", strategy="binary")
        manager.train("logistic_regression", X_train, y_train)
        manager.train("random_forest", X_train, y_train)
        
        models = manager.get_models()
        assert "logistic_regression" in models
        assert "random_forest" in models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
