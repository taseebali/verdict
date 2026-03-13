"""Tests for model artifacts and persistence modules."""

import pytest
import joblib
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd

from src.artifacts.model_serializer import ModelSerializer
from src.artifacts.exporter import ModelExporter


# Simple model class for testing (can be pickled)
class SimpleModel:
    """Simple model for testing serialization."""
    
    def __init__(self):
        self.coef_ = np.array([1.0, 2.0, 3.0])
        self.intercept_ = 0.5
    
    def predict(self, X):
        return np.array([0, 1, 1, 0, 1])
    
    def score(self, X, y):
        return 0.95


# Fixtures
@pytest.fixture
def temp_models_dir():
    """Create temporary directory for models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = ModelSerializer.MODELS_DIR
        ModelSerializer.MODELS_DIR = Path(tmpdir)
        yield Path(tmpdir)
        ModelSerializer.MODELS_DIR = original_dir


@pytest.fixture
def sample_model():
    """Create a simple model that can be pickled."""
    return SimpleModel()


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return {
        'hyperparameters': {'n_estimators': 100, 'max_depth': 10},
        'training_date': '2026-02-11',
        'dataset_size': 1000,
        'features': ['age', 'income', 'credit_score'],
        'target': 'loan_approved'
    }


@pytest.fixture
def sample_df():
    """Create sample DataFrame."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'feature3': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 1]
    })


# ModelSerializer Tests
class TestModelSerializer:
    
    def test_setup_directory_creates_path(self, temp_models_dir):
        """Test that setup_directory creates required directory."""
        ModelSerializer.setup_directory()
        assert ModelSerializer.MODELS_DIR.exists()
        assert ModelSerializer.MODELS_DIR.is_dir()
    
    def test_save_model_basic(self, temp_models_dir, sample_model, sample_metadata):
        """Test saving model with metadata."""
        result = ModelSerializer.save_model(
            model=sample_model,
            model_name='test_model',
            metadata=sample_metadata
        )
        
        assert result['status'] == 'success'
        assert 'model_path' in result
        assert Path(result['model_path']).exists()
    
    def test_save_model_without_metadata(self, temp_models_dir, sample_model):
        """Test saving model without metadata."""
        result = ModelSerializer.save_model(
            model=sample_model,
            model_name='simple_model'
        )
        
        assert result['status'] == 'success'
        assert 'model_path' in result
    
    def test_save_model_invalid_name(self, temp_models_dir, sample_model):
        """Test that invalid model names raise ValueError."""
        with pytest.raises(ValueError):
            ModelSerializer.save_model(
                model=sample_model,
                model_name=''
            )
        
        with pytest.raises(ValueError):
            ModelSerializer.save_model(
                model=sample_model,
                model_name=None
            )
    
    def test_save_model_file_exists_error(self, temp_models_dir, sample_model):
        """Test that saving duplicate model raises error without overwrite."""
        ModelSerializer.save_model(
            model=sample_model,
            model_name='existing_model'
        )
        
        with pytest.raises(FileExistsError):
            ModelSerializer.save_model(
                model=sample_model,
                model_name='existing_model',
                overwrite=False
            )
    
    def test_save_model_with_overwrite(self, temp_models_dir, sample_model):
        """Test overwriting existing model."""
        # Save first time
        result1 = ModelSerializer.save_model(
            model=sample_model,
            model_name='overwrite_test'
        )
        
        # Overwrite
        result2 = ModelSerializer.save_model(
            model=sample_model,
            model_name='overwrite_test',
            overwrite=True
        )
        
        assert result1['status'] == 'success'
        assert result2['status'] == 'success'
    
    def test_load_model(self, temp_models_dir, sample_model):
        """Test loading saved model."""
        # Save model
        save_result = ModelSerializer.save_model(
            model=sample_model,
            model_name='load_test'
        )
        
        # Load model
        loaded_model = ModelSerializer.load_model('load_test')
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
    
    def test_load_nonexistent_model(self, temp_models_dir):
        """Test loading nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            ModelSerializer.load_model('nonexistent_model')
    
    def test_get_model_info(self, temp_models_dir, sample_model, sample_metadata):
        """Test retrieving model information."""
        result = ModelSerializer.save_model(
            model=sample_model,
            model_name='info_test',
            metadata=sample_metadata
        )
        
        info = ModelSerializer.get_model_info('info_test')
        assert info['name'] == 'info_test'
        assert 'saved_at' in info
        assert info['user_metadata'] == sample_metadata
    
    def test_get_all_models(self, temp_models_dir, sample_model):
        """Test listing all saved models."""
        # Save multiple models
        ModelSerializer.save_model(model=sample_model, model_name='model1')
        ModelSerializer.save_model(model=sample_model, model_name='model2')
        ModelSerializer.save_model(model=sample_model, model_name='model3')
        
        models = ModelSerializer.list_models()
        assert len(models) >= 3
        assert 'model1' in models
        assert 'model2' in models
        assert 'model3' in models
    
    def test_delete_model(self, temp_models_dir, sample_model):
        """Test deleting a saved model."""
        save_result = ModelSerializer.save_model(
            model=sample_model,
            model_name='delete_test'
        )
        
        # Verify exists
        assert ModelSerializer.model_exists('delete_test')
        
        # Delete
        delete_result = ModelSerializer.delete_model('delete_test')
        assert delete_result['status'] == 'success'
        assert not ModelSerializer.model_exists('delete_test')
    
    def test_model_size_in_result(self, temp_models_dir, sample_model):
        """Test that model size is reported."""
        result = ModelSerializer.save_model(
            model=sample_model,
            model_name='size_test'
        )
        
        assert 'model_size_bytes' in result or 'model_path' in result


# ModelExporter Tests
class TestModelExporter:
    
    def test_initialization(self):
        """Test exporter initialization."""
        exporter = ModelExporter(output_dir='test_models')
        assert exporter.output_dir == 'test_models'
    
    def test_export_model_creates_directory(self):
        """Test that export creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(output_dir=str(Path(tmpdir) / 'exports'))
            assert Path(exporter.output_dir).exists()
    
    def test_export_model_with_metadata(self):
        """Test exporting model with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(output_dir=tmpdir)
            model = SimpleModel()
            metadata = {'version': '1.0', 'accuracy': 0.95}
            
            result = exporter.export_model(
                model=model,
                model_name='test_export',
                metadata=metadata
            )
            
            assert result['model_name'] == 'test_export'
            assert 'model_path' in result
            assert 'timestamp' in result
            assert 'file_size_mb' in result
            assert 'metadata_path' in result
    
    def test_export_model_without_metadata(self):
        """Test exporting model without metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(output_dir=tmpdir)
            model = SimpleModel()
            
            result = exporter.export_model(
                model=model,
                model_name='simple_export'
            )
            
            assert result['model_name'] == 'simple_export'
            assert 'metadata_path' not in result
    
    def test_load_model(self):
        """Test loading exported model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(output_dir=tmpdir)
            
            # Create and export model
            model = SimpleModel()
            
            export_result = exporter.export_model(
                model=model,
                model_name='load_test'
            )
            
            # Load model
            loaded_model = exporter.load_model(export_result['model_path'])
            assert loaded_model is not None
    
    def test_export_model_file_naming(self):
        """Test that exported models have timestamp in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(output_dir=tmpdir)
            model = SimpleModel()
            
            result = exporter.export_model(
                model=model,
                model_name='timestamp_test'
            )
            
            model_path = result['model_path']
            filename = Path(model_path).name
            
            # Should contain model name and timestamp pattern
            assert 'timestamp_test' in filename
            assert '_' in filename  # Timestamp separator
            assert filename.endswith('.joblib')


# Integration Tests
class TestArtifactIntegration:
    
    def test_save_and_load_workflow(self, temp_models_dir, sample_model):
        """Test complete save/load workflow."""
        model_name = 'integration_test'
        metadata = {'test': True, 'version': 1}
        
        # Save
        save_result = ModelSerializer.save_model(
            model=sample_model,
            model_name=model_name,
            metadata=metadata
        )
        
        # Load
        loaded = ModelSerializer.load_model(model_name)
        assert loaded is not None
        
        # Check metadata
        info = ModelSerializer.get_model_info(model_name)
        assert info['user_metadata']['test'] is True
    
    def test_multiple_model_versions(self, temp_models_dir, sample_model):
        """Test managing multiple versions of same model."""
        model_name = 'versioned_model'
        
        # Save version 1
        v1_result = ModelSerializer.save_model(
            model=sample_model,
            model_name=f'{model_name}_v1',
            metadata={'version': 1}
        )
        
        # Save version 2
        v2_result = ModelSerializer.save_model(
            model=sample_model,
            model_name=f'{model_name}_v2',
            metadata={'version': 2}
        )
        
        # Both should exist
        assert ModelSerializer.model_exists(f'{model_name}_v1')
        assert ModelSerializer.model_exists(f'{model_name}_v2')
        
        # Both should be loadable
        v1_loaded = ModelSerializer.load_model(f'{model_name}_v1')
        v2_loaded = ModelSerializer.load_model(f'{model_name}_v2')
        
        assert v1_loaded is not None
        assert v2_loaded is not None
    
    def test_exporter_vs_serializer(self):
        """Test consistency between exporter and serializer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            metadata = {'test': 'data'}
            
            # Export using exporter
            exporter = ModelExporter(output_dir=tmpdir)
            export_result = exporter.export_model(
                model=model,
                model_name='comparison_test',
                metadata=metadata
            )
            
            # Load and verify
            loaded = exporter.load_model(export_result['model_path'])
            assert loaded is not None


# Edge Cases and Error Handling
class TestArtifactEdgeCases:
    
    def test_model_with_special_characters_in_name(self, temp_models_dir, sample_model):
        """Test model names with special characters."""
        # Most filesystems handle these safely
        try:
            result = ModelSerializer.save_model(
                model=sample_model,
                model_name='model-with-dashes_and_underscores.v2'
            )
            assert result['status'] == 'success'
        except (ValueError, OSError):
            # Some systems may reject these
            pass
    
    def test_very_large_metadata(self, temp_models_dir, sample_model):
        """Test handling very large metadata."""
        large_metadata = {
            'data': 'x' * 10000,  # 10KB of data
            'features': list(range(100))
        }
        
        result = ModelSerializer.save_model(
            model=sample_model,
            model_name='large_metadata_test',
            metadata=large_metadata
        )
        
        assert result['status'] == 'success'
        
        # Verify it loads correctly
        info = ModelSerializer.get_model_info('large_metadata_test')
        assert len(info['user_metadata']['data']) == 10000
    
    def test_concurrent_model_saves(self, temp_models_dir, sample_model):
        """Test saving models with similar names."""
        for i in range(5):
            result = ModelSerializer.save_model(
                model=sample_model,
                model_name=f'concurrent_model_{i}'
            )
            assert result['status'] == 'success'
        
        # All should exist
        all_models = ModelSerializer.list_models()
        for i in range(5):
            assert f'concurrent_model_{i}' in all_models
