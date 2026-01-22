"""Comprehensive tests for One-vs-Rest (OvR) multiclass training."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.core.models import ModelManager


class TestOvRTraining:
    """Test OvR model training."""
    
    @pytest.fixture
    def multiclass_data_3classes(self):
        """Create 3-class multiclass dataset."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_informative=4,
            n_redundant=1,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def multiclass_data_5classes(self):
        """Create 5-class multiclass dataset."""
        X, y = make_classification(
            n_samples=250,
            n_features=8,
            n_informative=6,
            n_redundant=2,
            n_classes=5,
            n_clusters_per_class=1,
            random_state=42
        )
        return X, y
    
    def test_ovr_initialization(self):
        """Test OvR model manager initialization."""
        manager = ModelManager(task_type="classification", strategy="ovr")
        assert manager.strategy == "ovr"
        assert manager.classes_ is None
        assert len(manager.ovr_models) == 0
    
    def test_ovr_train_3class_logistic(self, multiclass_data_3classes):
        """Test OvR training with logistic regression on 3-class problem."""
        X, y = multiclass_data_3classes
        manager = ModelManager(task_type="classification", strategy="ovr")
        
        result = manager.train_ovr("logistic_regression", X, y)
        
        assert result["status"] == "success"
        assert result["strategy"] == "ovr"
        assert result["n_classes"] == 3
        assert "train_scores" in result
        assert "average_score" in result
        assert len(result["train_scores"]) == 3
    
    def test_ovr_train_5class_random_forest(self, multiclass_data_5classes):
        """Test OvR training with random forest on 5-class problem."""
        X, y = multiclass_data_5classes
        manager = ModelManager(task_type="classification", strategy="ovr")
        
        result = manager.train_ovr("random_forest", X, y)
        
        assert result["status"] == "success"
        assert result["n_classes"] == 5
        assert len(result["train_scores"]) == 5
    
    def test_ovr_classes_stored(self, multiclass_data_3classes):
        """Test that classes are correctly stored."""
        X, y = multiclass_data_3classes
        manager = ModelManager(task_type="classification", strategy="ovr")
        manager.train_ovr("logistic_regression", X, y)
        
        assert manager.classes_ is not None
        assert len(manager.classes_) == 3
        assert np.array_equal(manager.classes_, np.array([0, 1, 2]))
    
    def test_ovr_binary_models_stored(self, multiclass_data_3classes):
        """Test that binary models are stored for each class."""
        X, y = multiclass_data_3classes
        manager = ModelManager(task_type="classification", strategy="ovr")
        manager.train_ovr("logistic_regression", X, y)
        
        assert "logistic_regression" in manager.ovr_models
        assert len(manager.ovr_models["logistic_regression"]) == 3
        
        for class_label in [0, 1, 2]:
            assert class_label in manager.ovr_models["logistic_regression"]
            assert hasattr(manager.ovr_models["logistic_regression"][class_label], "predict")
    
    def test_ovr_invalid_strategy_error(self, multiclass_data_3classes):
        """Test that OvR fails with non-OvR strategy."""
        X, y = multiclass_data_3classes
        manager = ModelManager(task_type="classification", strategy="binary")
        
        with pytest.raises(ValueError):
            manager.train_ovr("logistic_regression", X, y)
    
    def test_ovr_too_few_classes_error(self):
        """Test that OvR fails with <2 classes."""
        X = np.random.randn(50, 5)
        y = np.zeros(50)  # Only 1 class
        manager = ModelManager(task_type="classification", strategy="ovr")
        
        with pytest.raises(ValueError, match="at least 2 classes"):
            manager.train_ovr("logistic_regression", X, y)


class TestOvRPrediction:
    """Test OvR model prediction."""
    
    @pytest.fixture
    def trained_ovr_manager(self):
        """Create a trained OvR model."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        manager = ModelManager(task_type="classification", strategy="ovr")
        manager.train_ovr("logistic_regression", X, y)
        
        return manager, X, y
    
    def test_ovr_predict_shape(self, trained_ovr_manager):
        """Test OvR predictions have correct shape."""
        manager, X, y = trained_ovr_manager
        X_test = X[:10]
        
        predictions = manager.predict_ovr("logistic_regression", X_test)
        
        assert predictions.shape == (10,)
    
    def test_ovr_predict_valid_classes(self, trained_ovr_manager):
        """Test OvR predictions contain valid class labels."""
        manager, X, y = trained_ovr_manager
        X_test = X[:20]
        
        predictions = manager.predict_ovr("logistic_regression", X_test)
        
        unique_preds = np.unique(predictions)
        assert all(pred in [0, 1, 2] for pred in unique_preds)
    
    def test_ovr_predict_not_trained_error(self):
        """Test error when predicting with untrained model."""
        manager = ModelManager(task_type="classification", strategy="ovr")
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="not a trained OvR model"):
            manager.predict_ovr("logistic_regression", X)


class TestOvRProbabilities:
    """Test OvR probability predictions."""
    
    @pytest.fixture
    def trained_ovr_manager(self):
        """Create a trained OvR model."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        manager = ModelManager(task_type="classification", strategy="ovr")
        manager.train_ovr("logistic_regression", X, y)
        
        return manager, X, y
    
    def test_ovr_proba_shape(self, trained_ovr_manager):
        """Test OvR probability predictions have correct shape."""
        manager, X, y = trained_ovr_manager
        X_test = X[:10]
        
        predictions, probas = manager.predict_proba_ovr("logistic_regression", X_test)
        
        assert predictions.shape == (10,)
        assert probas.shape == (10, 3)  # 10 samples, 3 classes
    
    def test_ovr_proba_sum_to_one(self, trained_ovr_manager):
        """Test that probabilities sum to 1."""
        manager, X, y = trained_ovr_manager
        X_test = X[:20]
        
        _, probas = manager.predict_proba_ovr("logistic_regression", X_test)
        
        sums = probas.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(20))
    
    def test_ovr_proba_in_range(self, trained_ovr_manager):
        """Test that probabilities are in [0, 1] range."""
        manager, X, y = trained_ovr_manager
        X_test = X[:20]
        
        _, probas = manager.predict_proba_ovr("logistic_regression", X_test)
        
        assert probas.min() >= 0
        assert probas.max() <= 1
    
    def test_ovr_proba_predictions_consistent(self, trained_ovr_manager):
        """Test that max probability corresponds to prediction."""
        manager, X, y = trained_ovr_manager
        X_test = X[:20]
        
        predictions, probas = manager.predict_proba_ovr("logistic_regression", X_test)
        
        predicted_from_proba = np.argmax(probas, axis=1)
        
        # Map indices to class labels
        predicted_from_proba = manager.classes_[predicted_from_proba]
        
        np.testing.assert_array_equal(predictions, predicted_from_proba)
    
    def test_ovr_proba_not_trained_error(self):
        """Test error when getting probabilities from untrained model."""
        manager = ModelManager(task_type="classification", strategy="ovr")
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="not a trained OvR model"):
            manager.predict_proba_ovr("logistic_regression", X)


class TestOvRIntegration:
    """Integration tests for OvR pipeline."""
    
    def test_ovr_train_predict_pipeline(self):
        """Test full train-predict pipeline with OvR."""
        # Generate data
        X, y = make_classification(
            n_samples=200,
            n_features=6,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Split
        split_idx = 150
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train
        manager = ModelManager(task_type="classification", strategy="ovr")
        train_result = manager.train_ovr("random_forest", X_train, y_train)
        
        assert train_result["status"] == "success"
        assert train_result["n_classes"] == 4
        
        # Predict
        predictions = manager.predict_ovr("random_forest", X_test)
        
        assert predictions.shape == (50,)
        assert len(np.unique(predictions)) <= 4
    
    def test_ovr_multiple_models(self):
        """Test training multiple OvR models."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        manager = ModelManager(task_type="classification", strategy="ovr")
        
        # Train two different models
        result1 = manager.train_ovr("logistic_regression", X, y)
        result2 = manager.train_ovr("random_forest", X, y)
        
        assert result1["status"] == "success"
        assert result2["status"] == "success"
        
        # Both should have their own models
        assert "logistic_regression" in manager.ovr_models
        assert "random_forest" in manager.ovr_models
    
    def test_ovr_large_multiclass(self):
        """Test OvR with 10-class problem."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_classes=10,
            n_informative=8,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42
        )
        
        manager = ModelManager(task_type="classification", strategy="ovr")
        result = manager.train_ovr("logistic_regression", X, y)
        
        assert result["n_classes"] == 10
        assert len(result["train_scores"]) == 10
        
        predictions = manager.predict_ovr("logistic_regression", X[:20])
        assert predictions.shape == (20,)


class TestOvREdgeCases:
    """Test edge cases for OvR."""
    
    def test_ovr_imbalanced_classes(self):
        """Test OvR with highly imbalanced classes."""
        X = np.random.randn(200, 5)
        y = np.concatenate([np.zeros(180), np.ones(15), np.full(5, 2)])
        
        manager = ModelManager(task_type="classification", strategy="ovr")
        result = manager.train_ovr("logistic_regression", X, y)
        
        assert result["status"] == "success"
        assert result["n_classes"] == 3
    
    def test_ovr_single_sample_per_class(self):
        """Test OvR with minimum samples per class."""
        X = np.random.randn(3, 5)
        y = np.array([0, 1, 2])
        
        manager = ModelManager(task_type="classification", strategy="ovr")
        result = manager.train_ovr("logistic_regression", X, y)
        
        assert result["status"] == "success"
        assert result["n_classes"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
