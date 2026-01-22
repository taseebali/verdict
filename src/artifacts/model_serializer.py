"""
Model Serialization and Persistence Module

Handles saving and loading trained models with metadata for production deployment.
"""

import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ModelSerializer:
    """Serialize and persist ML models to disk with metadata tracking."""
    
    MODELS_DIR = Path("models")
    METADATA_SUFFIX = "_metadata"
    
    @classmethod
    def setup_directory(cls) -> Path:
        """Create models directory if it doesn't exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODELS_DIR
    
    @classmethod
    def save_model(
        cls, 
        model: Any, 
        model_name: str, 
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Save a trained model to disk with metadata.
        
        Args:
            model: Trained model object (sklearn, custom, etc.)
            model_name: Unique name for the model
            metadata: Optional metadata (e.g., hyperparameters, training info)
            overwrite: Whether to overwrite existing model
            
        Returns:
            Dictionary with save status and paths
            
        Raises:
            FileExistsError: If model exists and overwrite=False
            ValueError: If model_name is invalid
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be non-empty string")
        
        cls.setup_directory()
        model_path = cls.MODELS_DIR / f"{model_name}.joblib"
        
        # Check if exists
        if model_path.exists() and not overwrite:
            raise FileExistsError(f"Model '{model_name}' already exists. Use overwrite=True to replace.")
        
        try:
            # Save model
            joblib.dump(model, str(model_path), compress=3)
            
            # Save metadata
            full_metadata = {
                "model_name": model_name,
                "saved_at": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "user_metadata": metadata or {}
            }
            
            joblib.dump(full_metadata, str(cls.MODELS_DIR / f"{model_name}{cls.METADATA_SUFFIX}.joblib"))
            
            logger.info(f"Model '{model_name}' saved to {model_path}")
            return {
                "status": "success",
                "model_path": str(model_path),
                "metadata_path": str(cls.MODELS_DIR / f"{model_name}{cls.METADATA_SUFFIX}.joblib"),
                "saved_at": full_metadata["saved_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to save model '{model_name}': {e}")
            raise
    
    @classmethod
    def load_model(cls, model_name: str, load_metadata: bool = False) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
            load_metadata: Whether to also return metadata
            
        Returns:
            Loaded model (or tuple of model and metadata if load_metadata=True)
            
        Raises:
            FileNotFoundError: If model doesn't exist
        """
        model_path = cls.MODELS_DIR / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}")
        
        try:
            model = joblib.load(str(model_path))
            logger.info(f"Model '{model_name}' loaded from {model_path}")
            
            if load_metadata:
                metadata_path = cls.MODELS_DIR / f"{model_name}{cls.METADATA_SUFFIX}.joblib"
                if metadata_path.exists():
                    metadata = joblib.load(str(metadata_path))
                    return model, metadata
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
    
    @classmethod
    def list_models(cls, include_metadata: bool = False):
        """
        List all saved models.
        
        Args:
            include_metadata: Whether to include metadata for each model
            
        Returns:
            List of model names or list of dicts with metadata
        """
        if not cls.MODELS_DIR.exists():
            return []
        
        model_files = list(cls.MODELS_DIR.glob("*.joblib"))
        models = [f.stem for f in model_files if not f.stem.endswith(cls.METADATA_SUFFIX)]
        
        if include_metadata:
            result = []
            for model_name in models:
                try:
                    _, metadata = cls.load_model(model_name, load_metadata=True)
                    result.append({
                        "name": model_name,
                        "type": metadata.get("model_type"),
                        "saved_at": metadata.get("saved_at")
                    })
                except:
                    result.append({"name": model_name, "type": "unknown", "saved_at": None})
            return result
        
        return sorted(models)
    
    @classmethod
    def delete_model(cls, model_name: str) -> Dict[str, Any]:
        """
        Delete a saved model and its metadata.
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            Deletion status
            
        Raises:
            FileNotFoundError: If model doesn't exist
        """
        model_path = cls.MODELS_DIR / f"{model_name}.joblib"
        metadata_path = cls.MODELS_DIR / f"{model_name}{cls.METADATA_SUFFIX}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found")
        
        try:
            model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Model '{model_name}' deleted")
            return {"status": "success", "deleted": model_name}
            
        except Exception as e:
            logger.error(f"Failed to delete model '{model_name}': {e}")
            raise
    
    @classmethod
    def model_exists(cls, model_name: str) -> bool:
        """Check if a model exists."""
        model_path = cls.MODELS_DIR / f"{model_name}.joblib"
        return model_path.exists()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get metadata and file info for a model."""
        model_path = cls.MODELS_DIR / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found")
        
        try:
            _, metadata = cls.load_model(model_name, load_metadata=True)
            
            return {
                "name": model_name,
                "type": metadata.get("model_type"),
                "saved_at": metadata.get("saved_at"),
                "file_size_kb": model_path.stat().st_size / 1024,
                "user_metadata": metadata.get("user_metadata", {})
            }
        except Exception as e:
            logger.error(f"Failed to get info for model '{model_name}': {e}")
            raise
