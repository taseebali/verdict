"""Unit tests for core modules."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
from src.data_handler import DataHandler
from src.preprocessing import Preprocessor
from src.models import ModelManager
from src.metrics import MetricsCalculator
from src.pipeline import MLPipeline


# Test fixtures
@pytest.fixture
def sample_df():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    }
    
    return pd.DataFrame(data)


class TestDataHandler:
    """Test DataHandler class."""

    def test_initialization(self, sample_df):
        """Test DataHandler initialization."""
        dh = DataHandler(sample_df)
        assert dh.df.shape == sample_df.shape
        assert len(dh.numeric_cols) > 0
        assert len(dh.categorical_cols) > 0

    def test_data_summary(self, sample_df):
        """Test data summary generation."""
        dh = DataHandler(sample_df)
        summary = dh.get_data_summary()
        
        assert "shape" in summary
        assert "dtypes" in summary
        assert "missing_values" in summary
        assert summary["shape"] == (100, 4)

    def test_validate_data_with_valid_data(self, sample_df):
        """Test validation with valid data."""
        dh = DataHandler(sample_df)
        is_valid, message = dh.validate_data()
        assert is_valid is True

    def test_validate_data_with_empty_data(self):
        """Test validation with empty data."""
        df = pd.DataFrame()
        dh = DataHandler(df)
        is_valid, message = dh.validate_data()
        assert is_valid is False

    def test_get_columns(self, sample_df):
        """Test column retrieval."""
        dh = DataHandler(sample_df)
        cols = dh.get_columns()
        assert len(cols) == 4
        assert "target" in cols


class TestPreprocessor:
    """Test Preprocessor class."""

    def test_initialization(self, sample_df):
        """Test Preprocessor initialization."""
        prep = Preprocessor(sample_df, "target")
        assert prep.target_col == "target"
        assert len(prep.numeric_cols) > 0

    def test_encode_categorical(self, sample_df):
        """Test categorical encoding."""
        prep = Preprocessor(sample_df, "target")
        prep.encode_categorical()
        
        # Check that all categorical columns are now numeric
        for col in prep.categorical_cols:
            assert prep.df[col].dtype in ["int64", "int32"]

    def test_prepare_data(self, sample_df):
        """Test data preparation."""
        prep = Preprocessor(sample_df, "target")
        X_train, X_test, y_train, y_test = prep.prepare_data(test_size=0.2)
        
        assert len(X_train) > len(X_test)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_task_type_detection(self, sample_df):
        """Test task type detection."""
        prep = Preprocessor(sample_df, "target")
        prep.encode_categorical()
        task_type = prep.get_task_type()
        
        assert task_type in ["classification", "regression"]


class TestModelManager:
    """Test ModelManager class."""

    def test_initialization(self):
        """Test ModelManager initialization."""
        mm = ModelManager(task_type="classification")
        assert mm.task_type == "classification"
        assert len(mm.models) == 0

    def test_train_logistic_regression(self, sample_df):
        """Test training Logistic Regression."""
        prep = Preprocessor(sample_df, "target")
        X_train, X_test, y_train, y_test = prep.prepare_data(test_size=0.2)
        
        mm = ModelManager(task_type="classification")
        result = mm.train("logistic_regression", X_train, y_train)
        
        assert result["status"] == "success"
        assert "logistic_regression" in mm.models

    def test_predict(self, sample_df):
        """Test prediction."""
        prep = Preprocessor(sample_df, "target")
        X_train, X_test, y_train, y_test = prep.prepare_data(test_size=0.2)
        
        mm = ModelManager(task_type="classification")
        mm.train("logistic_regression", X_train, y_train)
        predictions = mm.predict("logistic_regression", X_test)
        
        assert len(predictions) == len(X_test)


class TestMetricsCalculator:
    """Test MetricsCalculator class."""

    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics


class TestMLPipeline:
    """Test MLPipeline class."""

    def test_pipeline_validation(self, sample_df):
        """Test pipeline validation."""
        pipeline = MLPipeline(sample_df, "target")
        is_valid, message = pipeline.validate()
        
        assert is_valid is True

    def test_full_pipeline(self, sample_df):
        """Test complete pipeline execution."""
        pipeline = MLPipeline(sample_df, "target")
        results = pipeline.run_full_pipeline()
        
        assert results["status"] == "success"
        assert "eval_results" in results
        assert len(results["eval_results"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
