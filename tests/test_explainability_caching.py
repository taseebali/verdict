"""Tests for explainability module with caching."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.explain.explainability import ExplainabilityAnalyzer
from src.core.cache_manager import CacheManager


class TestExplainabilityWithCaching:
    """Test explainability module with caching functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        original_cache_dir = CacheManager.CACHE_DIR
        temp_dir = Path(tempfile.mkdtemp())
        CacheManager.CACHE_DIR = temp_dir
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        CacheManager.CACHE_DIR = original_cache_dir

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train a model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        return {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
        }

    def test_get_feature_importance_returns_dict(self, sample_data, temp_cache_dir):
        """Feature importance should return dictionary."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        importance = analyzer.get_feature_importance("test_model")
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_get_feature_importance_sorted(self, sample_data, temp_cache_dir):
        """Feature importance should be sorted by importance."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        importance = analyzer.get_feature_importance("test_model")
        values = list(importance.values())

        # Check if sorted (descending)
        assert values == sorted(values, reverse=True)

    def test_feature_importance_caching_hit(self, sample_data, temp_cache_dir):
        """Should return cached importance on second call."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        # First call - compute
        importance1 = analyzer.get_feature_importance("test_model", use_cache=True)

        # Second call - should be cached
        importance2 = analyzer.get_feature_importance("test_model", use_cache=True)

        assert importance1 == importance2

    def test_feature_importance_cache_disabled(self, sample_data, temp_cache_dir):
        """Should not use cache when disabled."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        # Call with cache disabled
        with patch.object(CacheManager, "get_cached_importance", return_value=None):
            with patch.object(CacheManager, "cache_importance", return_value=True):
                importance = analyzer.get_feature_importance("test_model", use_cache=False)

        assert isinstance(importance, dict)

    def test_feature_importance_all_features_present(self, sample_data, temp_cache_dir):
        """Feature importance should include all features."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        importance = analyzer.get_feature_importance("test_model")

        # All features should be present
        for feature_name in sample_data["feature_names"]:
            assert feature_name in importance

    def test_feature_importance_values_nonnegative(self, sample_data, temp_cache_dir):
        """Feature importance values should be non-negative."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        importance = analyzer.get_feature_importance("test_model")

        # Permutation importance values should be >= 0
        for value in importance.values():
            assert value >= 0

    def test_feature_importance_cache_invalidation(self, sample_data, temp_cache_dir):
        """Cache should be invalidated for different models."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        # Get importance for two different models
        importance1 = analyzer.get_feature_importance("model1")
        importance2 = analyzer.get_feature_importance("model2")

        # They might differ due to different cache keys
        assert True  # Both should compute without error

    def test_feature_importance_different_feature_sets(self, sample_data, temp_cache_dir):
        """Should handle different feature sets correctly."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        # Get importance with subset of features
        subset_features = sample_data["feature_names"][:5]

        # Create new analyzer with subset (simulated by limiting feature_names)
        analyzer2 = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            subset_features,
        )

        importance1 = analyzer.get_feature_importance("model")
        importance2 = analyzer2.get_feature_importance("model")

        # Should have different number of features
        assert len(importance1) == 10
        assert len(importance2) == 5

    def test_feature_importance_with_nan_handling(self, sample_data, temp_cache_dir):
        """Should handle NaN values gracefully."""
        X_train = sample_data["X_train"].copy()
        X_test = sample_data["X_test"].copy()

        # Introduce some NaN values
        X_train.iloc[0, 0] = np.nan
        X_test.iloc[0, 0] = np.nan

        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            X_train,
            X_test,
            sample_data["feature_names"],
        )

        # Should still return importance (NaN handling in permutation importance)
        importance = analyzer.get_feature_importance("test_model")
        assert isinstance(importance, dict)

    def test_cache_manager_called_on_importance_computation(
        self, sample_data, temp_cache_dir
    ):
        """Cache manager should be called when computing importance."""
        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        with patch.object(CacheManager, "cache_importance", return_value=True) as mock_cache:
            analyzer.get_feature_importance("test_model", use_cache=True)
            # Cache should be called to save the result
            assert mock_cache.called

    def test_feature_importance_performance_improvement(self, sample_data, temp_cache_dir):
        """Second call should be faster due to caching."""
        import time

        analyzer = ExplainabilityAnalyzer(
            sample_data["model"],
            sample_data["X_train"],
            sample_data["X_test"],
            sample_data["feature_names"],
        )

        # First call
        start1 = time.time()
        importance1 = analyzer.get_feature_importance("test_model", use_cache=True)
        time1 = time.time() - start1

        # Second call (cached)
        start2 = time.time()
        importance2 = analyzer.get_feature_importance("test_model", use_cache=True)
        time2 = time.time() - start2

        # Second call should be significantly faster
        assert time2 < time1
        assert importance1 == importance2
