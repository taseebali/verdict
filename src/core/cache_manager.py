"""Cache management for expensive computations like feature importance."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import logging

# Setup logging
logger = logging.getLogger(__name__)


class CacheManager:
    """Manage caching of expensive computations (feature importance, SHAP values)."""

    CACHE_DIR = Path("cache")
    MAX_CACHE_SIZE_MB = 100  # Maximum cache size in MB

    @staticmethod
    def _ensure_cache_dir() -> None:
        """Ensure cache directory exists."""
        CacheManager.CACHE_DIR.mkdir(exist_ok=True)

    @staticmethod
    def get_cache_key(model_name: str, feature_set: List[str]) -> str:
        """
        Generate unique cache key from model name and features.

        Args:
            model_name: Name of the model
            feature_set: List of feature names

        Returns:
            MD5 hash of model_name + sorted features
        """
        sorted_features = sorted(feature_set)
        content = f"{model_name}_{'_'.join(sorted_features)}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def get_cached_importance(
        model_name: str, features: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Load cached feature importance if it exists.

        Args:
            model_name: Name of the model
            features: List of feature names

        Returns:
            Dictionary of feature importance scores, or None if not cached
        """
        try:
            key = CacheManager.get_cache_key(model_name, features)
            cache_file = CacheManager.CACHE_DIR / f"importance_{key}.joblib"

            if cache_file.exists():
                logger.info(f"Cache hit for model '{model_name}' (key: {key[:8]}...)")
                return joblib.load(str(cache_file))
            else:
                logger.debug(f"Cache miss for model '{model_name}' (key: {key[:8]}...)")
                return None
        except Exception as e:
            logger.warning(f"Failed to load cached importance: {e}")
            return None

    @staticmethod
    def cache_importance(
        model_name: str, features: List[str], importance: Dict[str, float]
    ) -> bool:
        """
        Cache feature importance scores.

        Args:
            model_name: Name of the model
            features: List of feature names
            importance: Dictionary of feature importance scores

        Returns:
            True if cached successfully, False otherwise
        """
        try:
            CacheManager._ensure_cache_dir()
            key = CacheManager.get_cache_key(model_name, features)
            cache_file = CacheManager.CACHE_DIR / f"importance_{key}.joblib"

            joblib.dump(importance, str(cache_file), compress=3)
            logger.info(
                f"Cached importance for model '{model_name}' "
                f"with {len(features)} features (key: {key[:8]}...)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache importance: {e}")
            return False

    @staticmethod
    def invalidate_model_cache(model_name: str) -> int:
        """
        Invalidate all cache entries for a specific model.

        Args:
            model_name: Name of the model to invalidate

        Returns:
            Number of cache files deleted
        """
        try:
            if not CacheManager.CACHE_DIR.exists():
                return 0

            # Get all files for this model by checking metadata
            prefix = "importance_"
            cache_files = list(CacheManager.CACHE_DIR.glob(f"{prefix}*.joblib"))

            # We delete all importance caches for this model by checking metadata file
            metadata_file = CacheManager.CACHE_DIR / f"metadata_{model_name}.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        data = json.load(f)
                        cache_keys = data.get("cache_keys", [])

                    deleted_count = 0
                    for key in cache_keys:
                        cache_file = CacheManager.CACHE_DIR / f"importance_{key}.joblib"
                        if cache_file.exists():
                            cache_file.unlink()
                            deleted_count += 1

                    metadata_file.unlink()
                    logger.info(
                        f"Invalidated {deleted_count} cache entries for model '{model_name}'"
                    )
                    return deleted_count
                except Exception as e:
                    logger.warning(f"Failed to invalidate cache for {model_name}: {e}")

            return 0
        except Exception as e:
            logger.error(f"Error during cache invalidation: {e}")
            return 0

    @staticmethod
    def invalidate_all_cache() -> int:
        """
        Clear all cache files.

        Returns:
            Number of cache files deleted
        """
        try:
            if not CacheManager.CACHE_DIR.exists():
                return 0

            cache_files = list(CacheManager.CACHE_DIR.glob("*.joblib"))
            for cache_file in cache_files:
                cache_file.unlink()

            # Clean up metadata files
            metadata_files = list(CacheManager.CACHE_DIR.glob("metadata_*.json"))
            for metadata_file in metadata_files:
                metadata_file.unlink()

            logger.info(f"Cleared all cache ({len(cache_files)} files)")
            return len(cache_files)
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return 0

    @staticmethod
    def get_cache_size() -> float:
        """
        Get total cache size in MB.

        Returns:
            Total size of cache directory in MB
        """
        try:
            if not CacheManager.CACHE_DIR.exists():
                return 0.0

            total_size = sum(
                f.stat().st_size for f in CacheManager.CACHE_DIR.glob("**/*")
                if f.is_file()
            )
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to calculate cache size: {e}")
            return 0.0

    @staticmethod
    def register_model_cache(model_name: str, cache_key: str) -> None:
        """
        Register a cache key for a model (for cleanup purposes).

        Args:
            model_name: Name of the model
            cache_key: Cache key to register
        """
        try:
            CacheManager._ensure_cache_dir()
            metadata_file = CacheManager.CACHE_DIR / f"metadata_{model_name}.json"

            # Load existing metadata or create new
            metadata = {"cache_keys": []}
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                except Exception:
                    pass

            # Add cache key if not already present
            if cache_key not in metadata["cache_keys"]:
                metadata["cache_keys"].append(cache_key)

            # Write metadata
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.debug(f"Failed to register cache metadata: {e}")
