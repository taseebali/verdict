"""Model version compatibility and migration layer."""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelMigration:
    """Handle loading and migrating models between different format versions."""

    # Current model format version
    CURRENT_FORMAT_VERSION = 1
    CURRENT_VERDICT_VERSION = "2.0.0"

    # Define version metadata
    VERSION_MAPPING = {
        0: {"name": "initial", "description": "Original model format without version tracking"},
        1: {
            "name": "versioned",
            "description": "Models with version metadata and migration support",
        },
    }

    @staticmethod
    def add_version_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add version metadata to model metadata dict if not present.

        Args:
            metadata: Existing metadata dictionary

        Returns:
            Updated metadata with version information
        """
        if "model_format_version" not in metadata:
            metadata["model_format_version"] = ModelMigration.CURRENT_FORMAT_VERSION

        if "verdict_version" not in metadata:
            metadata["verdict_version"] = ModelMigration.CURRENT_VERDICT_VERSION

        logger.debug(
            f"Added version metadata: format_v{metadata['model_format_version']}, "
            f"verdict_v{metadata['verdict_version']}"
        )

        return metadata

    @staticmethod
    def get_format_version(metadata: Dict[str, Any]) -> int:
        """
        Get the model format version from metadata.

        Args:
            metadata: Model metadata dictionary

        Returns:
            Format version number (defaults to 0 for old models)
        """
        return metadata.get("model_format_version", 0)

    @staticmethod
    def is_compatible(metadata: Dict[str, Any]) -> bool:
        """
        Check if model is compatible with current VERDICT version.

        Args:
            metadata: Model metadata dictionary

        Returns:
            True if compatible, False otherwise
        """
        version = ModelMigration.get_format_version(metadata)

        # Currently compatible with versions 0 and 1
        compatible_versions = [0, 1]

        if version not in compatible_versions:
            logger.warning(
                f"Model format version {version} may not be compatible. "
                f"Current supported versions: {compatible_versions}"
            )
            return False

        return True

    @staticmethod
    def migrate_if_needed(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply migrations if model format version is older than current.

        Args:
            metadata: Model metadata dictionary

        Returns:
            Migrated metadata dictionary
        """
        version = ModelMigration.get_format_version(metadata)

        logger.info(f"Checking migrations for model format v{version}")

        if version < 0:
            logger.error(f"Invalid model format version: {version}")
            return metadata

        # Apply migrations in sequence
        if version < 1:
            metadata = ModelMigration._migrate_v0_to_v1(metadata)

        # Ensure current version metadata
        metadata = ModelMigration.add_version_metadata(metadata)

        return metadata

    @staticmethod
    def _migrate_v0_to_v1(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate model metadata from format v0 to v1.

        Format v0: Basic metadata without version info
        Format v1: Metadata with format version and verdict version

        Args:
            metadata: Old format metadata

        Returns:
            Migrated metadata with v1 structure
        """
        logger.info("Migrating model metadata from v0 to v1")

        # Ensure required v1 fields exist
        if "model_name" not in metadata:
            metadata["model_name"] = "unknown"

        if "saved_at" not in metadata:
            metadata["saved_at"] = "unknown"

        if "model_type" not in metadata:
            metadata["model_type"] = "unknown"

        if "user_metadata" not in metadata:
            metadata["user_metadata"] = {}

        # Add v1 specific fields
        metadata["model_format_version"] = 1
        metadata["verdict_version"] = ModelMigration.CURRENT_VERDICT_VERSION

        logger.debug("Successfully migrated metadata to v1")

        return metadata

    @staticmethod
    def get_version_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract version information from metadata.

        Args:
            metadata: Model metadata dictionary

        Returns:
            Dictionary with version information
        """
        format_version = ModelMigration.get_format_version(metadata)
        verdict_version = metadata.get(
            "verdict_version", "unknown"
        )

        return {
            "format_version": format_version,
            "verdict_version": verdict_version,
            "format_name": ModelMigration.VERSION_MAPPING.get(
                format_version, {}
            ).get("name", "unknown"),
            "is_compatible": ModelMigration.is_compatible(metadata),
            "requires_migration": format_version < ModelMigration.CURRENT_FORMAT_VERSION,
        }

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate model metadata structure.

        Args:
            metadata: Model metadata dictionary

        Returns:
            Tuple of (is_valid, message)
        """
        if not isinstance(metadata, dict):
            return False, "Metadata is not a dictionary"

        required_fields = ["model_name", "saved_at", "model_type"]

        for field in required_fields:
            if field not in metadata:
                return False, f"Missing required field: {field}"

        if "user_metadata" not in metadata:
            metadata["user_metadata"] = {}

        return True, "Metadata is valid"

    @staticmethod
    def get_migration_info() -> Dict[str, Any]:
        """
        Get information about available model format versions and migrations.

        Returns:
            Dictionary with version information and migration paths
        """
        return {
            "current_format_version": ModelMigration.CURRENT_FORMAT_VERSION,
            "current_verdict_version": ModelMigration.CURRENT_VERDICT_VERSION,
            "supported_versions": list(ModelMigration.VERSION_MAPPING.keys()),
            "version_descriptions": ModelMigration.VERSION_MAPPING,
            "migration_paths": {
                "0_to_1": "Add version metadata to model",
            },
        }
