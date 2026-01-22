"""
Tests for Model Serialization (P2.1)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from src.artifacts.model_serializer import ModelSerializer


@pytest.fixture
def temp_models_dir():
    """Create temporary models directory"""
    temp_dir = tempfile.mkdtemp()
    original_dir = ModelSerializer.MODELS_DIR
    ModelSerializer.MODELS_DIR = Path(temp_dir)
    yield Path(temp_dir)
    ModelSerializer.MODELS_DIR = original_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def trained_model():
    """Create a simple trained model"""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


class TestModelSerializer:
    """Test model serialization and persistence"""
    
    def test_setup_directory(self, temp_models_dir):
        """Test directory creation"""
        ModelSerializer.setup_directory()
        assert ModelSerializer.MODELS_DIR.exists()
    
    def test_save_model_basic(self, temp_models_dir, trained_model):
        """Test basic model saving"""
        result = ModelSerializer.save_model(trained_model, "test_model")
        
        assert result["status"] == "success"
        assert "test_model" in result["model_path"]
        assert Path(result["model_path"]).exists()
    
    def test_save_model_with_metadata(self, temp_models_dir, trained_model):
        """Test model saving with metadata"""
        metadata = {
            "accuracy": 0.92,
            "training_samples": 100,
            "features": 10
        }
        result = ModelSerializer.save_model(trained_model, "test_model", metadata=metadata)
        
        assert result["status"] == "success"
        assert "metadata_path" in result
    
    def test_save_model_invalid_name(self, temp_models_dir, trained_model):
        """Test saving with invalid model name"""
        with pytest.raises(ValueError):
            ModelSerializer.save_model(trained_model, "")
        
        with pytest.raises(ValueError):
            ModelSerializer.save_model(trained_model, None)
    
    def test_save_model_duplicate_error(self, temp_models_dir, trained_model):
        """Test error when saving duplicate model"""
        ModelSerializer.save_model(trained_model, "test_model")
        
        with pytest.raises(FileExistsError):
            ModelSerializer.save_model(trained_model, "test_model", overwrite=False)
    
    def test_save_model_overwrite(self, temp_models_dir, trained_model):
        """Test overwriting existing model"""
        ModelSerializer.save_model(trained_model, "test_model")
        result = ModelSerializer.save_model(trained_model, "test_model", overwrite=True)
        
        assert result["status"] == "success"
    
    def test_load_model(self, temp_models_dir, trained_model):
        """Test loading saved model"""
        ModelSerializer.save_model(trained_model, "test_model")
        loaded_model = ModelSerializer.load_model("test_model")
        
        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")
    
    def test_load_model_with_metadata(self, temp_models_dir, trained_model):
        """Test loading model with metadata"""
        metadata = {"accuracy": 0.92}
        ModelSerializer.save_model(trained_model, "test_model", metadata=metadata)
        
        loaded_model, loaded_metadata = ModelSerializer.load_model("test_model", load_metadata=True)
        
        assert loaded_model is not None
        assert loaded_metadata["user_metadata"]["accuracy"] == 0.92
        assert loaded_metadata["model_type"] == "RandomForestClassifier"
    
    def test_load_nonexistent_model(self, temp_models_dir):
        """Test loading nonexistent model"""
        with pytest.raises(FileNotFoundError):
            ModelSerializer.load_model("nonexistent_model")
    
    def test_list_models_empty(self, temp_models_dir):
        """Test listing models when none saved"""
        models = ModelSerializer.list_models()
        assert models == []
    
    def test_list_models(self, temp_models_dir, trained_model):
        """Test listing saved models"""
        ModelSerializer.save_model(trained_model, "model1")
        ModelSerializer.save_model(trained_model, "model2")
        
        models = ModelSerializer.list_models()
        assert "model1" in models
        assert "model2" in models
        assert len(models) == 2
    
    def test_list_models_with_metadata(self, temp_models_dir, trained_model):
        """Test listing models with metadata"""
        ModelSerializer.save_model(trained_model, "model1", metadata={"v": 1})
        ModelSerializer.save_model(trained_model, "model2", metadata={"v": 2})
        
        models = ModelSerializer.list_models(include_metadata=True)
        assert len(models) == 2
        assert models[0]["name"] in ["model1", "model2"]
        assert "type" in models[0]
        assert "saved_at" in models[0]
    
    def test_delete_model(self, temp_models_dir, trained_model):
        """Test deleting a model"""
        ModelSerializer.save_model(trained_model, "test_model")
        result = ModelSerializer.delete_model("test_model")
        
        assert result["status"] == "success"
        assert not ModelSerializer.model_exists("test_model")
    
    def test_delete_nonexistent_model(self, temp_models_dir):
        """Test deleting nonexistent model"""
        with pytest.raises(FileNotFoundError):
            ModelSerializer.delete_model("nonexistent")
    
    def test_model_exists(self, temp_models_dir, trained_model):
        """Test checking if model exists"""
        assert not ModelSerializer.model_exists("test_model")
        
        ModelSerializer.save_model(trained_model, "test_model")
        assert ModelSerializer.model_exists("test_model")
    
    def test_get_model_info(self, temp_models_dir, trained_model):
        """Test getting model information"""
        metadata = {"accuracy": 0.92, "features": 10}
        ModelSerializer.save_model(trained_model, "test_model", metadata=metadata)
        
        info = ModelSerializer.get_model_info("test_model")
        
        assert info["name"] == "test_model"
        assert info["type"] == "RandomForestClassifier"
        assert "saved_at" in info
        assert info["file_size_kb"] > 0
        assert info["user_metadata"]["accuracy"] == 0.92
    
    def test_model_predictions_after_load(self, temp_models_dir, trained_model):
        """Test that loaded model makes same predictions"""
        from sklearn.datasets import make_classification
        
        X_train, y_train = make_classification(n_samples=100, n_features=10, 
                                               n_classes=2, random_state=42)
        trained_model.fit(X_train, y_train)
        
        # Get predictions before saving
        pred_before = trained_model.predict(X_train[:5])
        
        # Save and load
        ModelSerializer.save_model(trained_model, "test_model")
        loaded_model = ModelSerializer.load_model("test_model")
        
        # Get predictions after loading
        pred_after = loaded_model.predict(X_train[:5])
        
        assert all(pred_before == pred_after)
    
    def test_save_multiple_models(self, temp_models_dir, trained_model):
        """Test saving multiple models"""
        for i in range(5):
            ModelSerializer.save_model(trained_model, f"model_{i}")
        
        models = ModelSerializer.list_models()
        assert len(models) == 5
        
        # All should be loadable
        for i in range(5):
            loaded = ModelSerializer.load_model(f"model_{i}")
            assert loaded is not None


class TestModelSerializerIntegration:
    """Integration tests for model serialization"""
    
    def test_save_train_save_cycle(self, temp_models_dir):
        """Test complete save-train-save cycle"""
        # Train first model
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model1 = RandomForestClassifier(n_estimators=5, random_state=42)
        model1.fit(X, y)
        
        ModelSerializer.save_model(model1, "v1", metadata={"version": 1})
        
        # Train improved model
        X2, y2 = make_classification(n_samples=150, n_features=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2.fit(X2, y2)
        
        ModelSerializer.save_model(model2, "v2", metadata={"version": 2})
        
        # Verify both exist
        assert ModelSerializer.model_exists("v1")
        assert ModelSerializer.model_exists("v2")
        
        # Load and compare
        loaded_v1, meta_v1 = ModelSerializer.load_model("v1", load_metadata=True)
        loaded_v2, meta_v2 = ModelSerializer.load_model("v2", load_metadata=True)
        
        assert meta_v1["user_metadata"]["version"] == 1
        assert meta_v2["user_metadata"]["version"] == 2
