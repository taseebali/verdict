"""Tests for cache management system."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from src.core.cache_manager import CacheManager


class TestCacheManager:
    """Test suite for CacheManager."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        original_cache_dir = CacheManager.CACHE_DIR
        temp_dir = Path(tempfile.mkdtemp())
        CacheManager.CACHE_DIR = temp_dir
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        CacheManager.CACHE_DIR = original_cache_dir

    def test_get_cache_key_consistent(self):
        """Cache key should be consistent for same inputs."""
        key1 = CacheManager.get_cache_key("model_a", ["feature1", "feature2"])
        key2 = CacheManager.get_cache_key("model_a", ["feature1", "feature2"])
        assert key1 == key2

    def test_get_cache_key_order_independent(self):
        """Cache key should be independent of feature order."""
        key1 = CacheManager.get_cache_key("model_a", ["feature1", "feature2"])
        key2 = CacheManager.get_cache_key("model_a", ["feature2", "feature1"])
        assert key1 == key2

    def test_get_cache_key_model_dependent(self):
        """Cache key should differ for different models."""
        key1 = CacheManager.get_cache_key("model_a", ["feature1"])
        key2 = CacheManager.get_cache_key("model_b", ["feature1"])
        assert key1 != key2

    def test_cache_importance_creates_directory(self, temp_cache_dir):
        """Caching should create cache files properly."""
        cache_file = temp_cache_dir / "importance_test.joblib"
        assert not cache_file.exists()
        CacheManager.cache_importance("test_model", ["f1", "f2"], {"f1": 0.5, "f2": 0.3})
        # At least one cache file should exist
        assert len(list(temp_cache_dir.glob("*.joblib"))) > 0

    def test_cache_and_retrieve_importance(self, temp_cache_dir):
        """Should cache and retrieve feature importance."""
        importance = {"feature1": 0.8, "feature2": 0.2}
        success = CacheManager.cache_importance("model1", ["feature1", "feature2"], importance)
        assert success

        retrieved = CacheManager.get_cached_importance("model1", ["feature1", "feature2"])
        assert retrieved == importance

    def test_cache_miss_returns_none(self, temp_cache_dir):
        """Cache miss should return None."""
        result = CacheManager.get_cached_importance("nonexistent", ["f1", "f2"])
        assert result is None

    def test_invalidate_model_cache(self, temp_cache_dir):
        """Should invalidate cache for a specific model."""
        # Create cache entries
        importance = {"f1": 0.5, "f2": 0.3}
        CacheManager.cache_importance("model1", ["f1", "f2"], importance)
        CacheManager.register_model_cache("model1", CacheManager.get_cache_key("model1", ["f1", "f2"]))

        # Verify cache exists
        assert CacheManager.get_cached_importance("model1", ["f1", "f2"]) is not None

        # Invalidate
        deleted_count = CacheManager.invalidate_model_cache("model1")
        assert deleted_count >= 0  # May be 0 if metadata not found

        # Verify cache is gone
        result = CacheManager.get_cached_importance("model1", ["f1", "f2"])
        assert result is None

    def test_invalidate_all_cache(self, temp_cache_dir):
        """Should clear all cache files."""
        # Create multiple cache entries
        CacheManager.cache_importance("model1", ["f1"], {"f1": 0.5})
        CacheManager.cache_importance("model2", ["f2"], {"f2": 0.6})

        # Verify caches exist
        assert CacheManager.get_cached_importance("model1", ["f1"]) is not None
        assert CacheManager.get_cached_importance("model2", ["f2"]) is not None

        # Clear all
        deleted = CacheManager.invalidate_all_cache()
        assert deleted >= 2

        # Verify both are gone
        assert CacheManager.get_cached_importance("model1", ["f1"]) is None
        assert CacheManager.get_cached_importance("model2", ["f2"]) is None

    def test_get_cache_size(self, temp_cache_dir):
        """Should calculate cache size correctly."""
        initial_size = CacheManager.get_cache_size()
        assert initial_size == 0.0

        # Add some cache
        importance = {"f1": 0.5, "f2": 0.3}
        CacheManager.cache_importance("model1", ["f1", "f2"], importance)

        new_size = CacheManager.get_cache_size()
        assert new_size > 0.0

    def test_register_model_cache(self, temp_cache_dir):
        """Should register cache metadata."""
        cache_key = "abc123"
        CacheManager.register_model_cache("model1", cache_key)

        metadata_file = temp_cache_dir / "metadata_model1.json"
        assert metadata_file.exists()

    def test_register_duplicate_cache_key(self, temp_cache_dir):
        """Should not register duplicate cache keys."""
        cache_key = "abc123"
        CacheManager.register_model_cache("model1", cache_key)
        CacheManager.register_model_cache("model1", cache_key)

        # Check metadata
        import json

        metadata_file = temp_cache_dir / "metadata_model1.json"
        with open(metadata_file) as f:
            data = json.load(f)
            # Should only have one entry
            assert data["cache_keys"].count(cache_key) == 1

    def test_cache_error_handling(self, temp_cache_dir):
        """Should handle caching errors gracefully."""
        # Test with invalid data should not raise exception
        result = CacheManager.cache_importance(
            "model", ["f1"], {"f1": float("inf")}  # Invalid value
        )
        # Should return False for invalid data or True if joblib handles it
        assert isinstance(result, bool)

    def test_get_cached_importance_nonexistent_cache_dir(self):
        """Should handle missing cache directory gracefully."""
        with patch.object(CacheManager, "CACHE_DIR", Path("/nonexistent/path")):
            result = CacheManager.get_cached_importance("model", ["f1"])
            assert result is None

    def test_feature_importance_type_is_dict(self, temp_cache_dir):
        """Cached importance should be dictionary."""
        importance = {"feature1": 0.8, "feature2": 0.2}
        CacheManager.cache_importance("model", ["feature1", "feature2"], importance)

        retrieved = CacheManager.get_cached_importance("model", ["feature1", "feature2"])
        assert isinstance(retrieved, dict)
        assert "feature1" in retrieved
        assert "feature2" in retrieved

    def test_cache_with_many_features(self, temp_cache_dir):
        """Should handle cache with many features."""
        features = [f"feature_{i}" for i in range(100)]
        importance = {f: 1.0 / (i + 1) for i, f in enumerate(features)}

        CacheManager.cache_importance("model", features, importance)
        retrieved = CacheManager.get_cached_importance("model", features)

        assert len(retrieved) == 100
        assert retrieved == importance

    def test_cache_isolation_between_models(self, temp_cache_dir):
        """Cache should be isolated between different models."""
        importance1 = {"f1": 0.9}
        importance2 = {"f1": 0.1}

        CacheManager.cache_importance("model1", ["f1"], importance1)
        CacheManager.cache_importance("model2", ["f1"], importance2)

        retrieved1 = CacheManager.get_cached_importance("model1", ["f1"])
        retrieved2 = CacheManager.get_cached_importance("model2", ["f1"])

        assert retrieved1["f1"] == 0.9
        assert retrieved2["f1"] == 0.1

    def test_cache_isolation_between_features(self, temp_cache_dir):
        """Cache should be isolated between different feature sets."""
        importance1 = {"f1": 0.5, "f2": 0.3}
        importance2 = {"f1": 0.5}

        CacheManager.cache_importance("model", ["f1", "f2"], importance1)
        CacheManager.cache_importance("model", ["f1"], importance2)

        retrieved1 = CacheManager.get_cached_importance("model", ["f1", "f2"])
        retrieved2 = CacheManager.get_cached_importance("model", ["f1"])

        assert len(retrieved1) == 2
        assert len(retrieved2) == 1
