"""Tests for model version compatibility and migration."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.core.model_migration import ModelMigration
from src.artifacts.model_serializer import ModelSerializer


class TestModelMigration:
    """Test model version compatibility and migration."""

    def test_current_format_version_is_defined(self):
        """Should have a defined current format version."""
        assert ModelMigration.CURRENT_FORMAT_VERSION >= 0
        assert isinstance(ModelMigration.CURRENT_FORMAT_VERSION, int)

    def test_current_verdict_version_is_defined(self):
        """Should have a defined current verdict version."""
        assert ModelMigration.CURRENT_VERDICT_VERSION
        assert isinstance(ModelMigration.CURRENT_VERDICT_VERSION, str)

    def test_version_mapping_covers_all_versions(self):
        """Version mapping should include all supported versions."""
        for version in range(ModelMigration.CURRENT_FORMAT_VERSION + 1):
            assert version in ModelMigration.VERSION_MAPPING

    def test_add_version_metadata_to_empty(self):
        """Should add version metadata to empty dict."""
        metadata = {}
        result = ModelMigration.add_version_metadata(metadata)

        assert "model_format_version" in result
        assert "verdict_version" in result
        assert (
            result["model_format_version"]
            == ModelMigration.CURRENT_FORMAT_VERSION
        )

    def test_add_version_metadata_preserves_existing(self):
        """Should preserve existing metadata."""
        metadata = {"model_name": "test", "saved_at": "2024-01-01"}
        result = ModelMigration.add_version_metadata(metadata)

        assert result["model_name"] == "test"
        assert result["saved_at"] == "2024-01-01"
        assert "model_format_version" in result

    def test_add_version_metadata_idempotent(self):
        """Should be idempotent - same result when called twice."""
        metadata = {"model_name": "test"}
        result1 = ModelMigration.add_version_metadata(metadata.copy())
        result2 = ModelMigration.add_version_metadata(result1.copy())

        assert result1["model_format_version"] == result2["model_format_version"]
        assert result1["verdict_version"] == result2["verdict_version"]

    def test_get_format_version_default_zero(self):
        """Should return 0 for metadata without version."""
        metadata = {"model_name": "test"}
        version = ModelMigration.get_format_version(metadata)
        assert version == 0

    def test_get_format_version_from_metadata(self):
        """Should extract version from metadata."""
        metadata = {"model_format_version": 1}
        version = ModelMigration.get_format_version(metadata)
        assert version == 1

    def test_is_compatible_with_current_version(self):
        """Should accept current format version."""
        metadata = {
            "model_format_version": ModelMigration.CURRENT_FORMAT_VERSION
        }
        assert ModelMigration.is_compatible(metadata) is True

    def test_is_compatible_with_old_version(self):
        """Should accept old format versions."""
        metadata = {"model_format_version": 0}
        assert ModelMigration.is_compatible(metadata) is True

    def test_is_compatible_with_future_version(self):
        """Should reject future format versions."""
        metadata = {"model_format_version": 999}
        assert ModelMigration.is_compatible(metadata) is False

    def test_migrate_if_needed_current_version(self):
        """Should not modify metadata already at current version."""
        metadata = {
            "model_name": "test",
            "model_format_version": ModelMigration.CURRENT_FORMAT_VERSION,
        }
        result = ModelMigration.migrate_if_needed(metadata.copy())

        assert result["model_name"] == "test"
        assert (
            result["model_format_version"]
            == ModelMigration.CURRENT_FORMAT_VERSION
        )

    def test_migrate_if_needed_old_version(self):
        """Should migrate metadata from old version."""
        metadata = {"model_name": "test", "model_format_version": 0}
        result = ModelMigration.migrate_if_needed(metadata)

        assert result["model_format_version"] == ModelMigration.CURRENT_FORMAT_VERSION
        assert result["model_name"] == "test"

    def test_migrate_v0_to_v1_adds_required_fields(self):
        """Migration v0->v1 should add required v1 fields."""
        old_metadata = {"model_name": "test"}
        migrated = ModelMigration._migrate_v0_to_v1(old_metadata)

        assert "model_format_version" in migrated
        assert "verdict_version" in migrated
        assert "saved_at" in migrated
        assert "model_type" in migrated
        assert "user_metadata" in migrated

    def test_migrate_v0_to_v1_preserves_data(self):
        """Migration v0->v1 should preserve existing data."""
        old_metadata = {
            "model_name": "my_model",
            "saved_at": "2024-01-01",
            "model_type": "LogisticRegression",
        }
        migrated = ModelMigration._migrate_v0_to_v1(old_metadata)

        assert migrated["model_name"] == "my_model"
        assert migrated["saved_at"] == "2024-01-01"
        assert migrated["model_type"] == "LogisticRegression"

    def test_get_version_info_returns_dict(self):
        """Should return structured version information."""
        metadata = {"model_format_version": 1}
        info = ModelMigration.get_version_info(metadata)

        assert isinstance(info, dict)
        assert "format_version" in info
        assert "verdict_version" in info
        assert "format_name" in info
        assert "is_compatible" in info
        assert "requires_migration" in info

    def test_get_version_info_compatible_model(self):
        """Version info should show compatible model."""
        metadata = {
            "model_format_version": ModelMigration.CURRENT_FORMAT_VERSION
        }
        info = ModelMigration.get_version_info(metadata)

        assert info["is_compatible"] is True
        assert info["requires_migration"] is False

    def test_get_version_info_old_model(self):
        """Version info should show old model requires migration."""
        metadata = {"model_format_version": 0}
        info = ModelMigration.get_version_info(metadata)

        assert info["requires_migration"] is True

    def test_validate_metadata_valid(self):
        """Should validate correct metadata."""
        metadata = {
            "model_name": "test",
            "saved_at": "2024-01-01",
            "model_type": "LogisticRegression",
        }
        is_valid, msg = ModelMigration.validate_metadata(metadata)

        assert is_valid is True

    def test_validate_metadata_missing_field(self):
        """Should reject metadata missing required fields."""
        metadata = {"model_name": "test", "saved_at": "2024-01-01"}
        is_valid, msg = ModelMigration.validate_metadata(metadata)

        assert is_valid is False
        assert "required field" in msg.lower()

    def test_validate_metadata_not_dict(self):
        """Should reject non-dict metadata."""
        is_valid, msg = ModelMigration.validate_metadata("not a dict")

        assert is_valid is False

    def test_get_migration_info_complete(self):
        """Should return complete migration information."""
        info = ModelMigration.get_migration_info()

        assert "current_format_version" in info
        assert "current_verdict_version" in info
        assert "supported_versions" in info
        assert "version_descriptions" in info
        assert "migration_paths" in info

    def test_get_migration_info_current_version_valid(self):
        """Migration info current version should be consistent."""
        info = ModelMigration.get_migration_info()

        assert info["current_format_version"] == ModelMigration.CURRENT_FORMAT_VERSION
        assert (
            info["current_verdict_version"]
            == ModelMigration.CURRENT_VERDICT_VERSION
        )


class TestModelSerializerWithMigration:
    """Test ModelSerializer integration with migration layer."""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory."""
        original_dir = ModelSerializer.MODELS_DIR
        temp_dir = Path(tempfile.mkdtemp())
        ModelSerializer.MODELS_DIR = temp_dir
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        ModelSerializer.MODELS_DIR = original_dir

    def test_save_model_includes_version_metadata(self, temp_models_dir):
        """Saved model should include version metadata."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        ModelSerializer.save_model(model, "test_model")

        _, metadata = ModelSerializer.load_model("test_model", load_metadata=True)

        assert "model_format_version" in metadata
        assert "verdict_version" in metadata
        assert metadata["model_format_version"] > 0

    def test_load_model_applies_migration(self, temp_models_dir):
        """Loading model should apply migrations automatically."""
        from sklearn.linear_model import LogisticRegression
        import joblib

        # Create old format model (without version)
        model = LogisticRegression()
        model_path = temp_models_dir / "old_model.joblib"
        metadata_path = temp_models_dir / "old_model_metadata.joblib"

        joblib.dump(model, str(model_path))
        old_metadata = {
            "model_name": "old_model",
            "saved_at": "2024-01-01",
            "model_type": "LogisticRegression",
        }
        joblib.dump(old_metadata, str(metadata_path))

        # Load should migrate
        _, loaded_metadata = ModelSerializer.load_model("old_model", load_metadata=True)

        assert "model_format_version" in loaded_metadata
        assert "verdict_version" in loaded_metadata

    def test_list_models_with_metadata(self, temp_models_dir):
        """List models should include version information."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        ModelSerializer.save_model(model, "model1")
        ModelSerializer.save_model(model, "model2")

        models = ModelSerializer.list_models(include_metadata=True)

        assert len(models) == 2
        for model_info in models:
            assert "name" in model_info
            assert "type" in model_info
            assert "saved_at" in model_info

    def test_get_model_info_includes_version(self, temp_models_dir):
        """Model info should include version details."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        ModelSerializer.save_model(model, "versioned_model")

        info = ModelSerializer.get_model_info("versioned_model")

        assert "name" in info
        assert "type" in info
        assert "saved_at" in info
        assert "user_metadata" in info
