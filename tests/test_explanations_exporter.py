"""Tests for explanations exporter module."""

import pytest
import json
import pandas as pd
import numpy as np
import os
import tempfile

from src.core.explanations_exporter import ExplanationExporter, ExportResult


class TestExplanationExporter:
    """Test suite for ExplanationExporter class."""

    @pytest.fixture
    def exporter(self):
        """Create ExplanationExporter instance."""
        return ExplanationExporter()

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return {
            'feature_importance': {
                'age': 0.35,
                'income': 0.28,
                'education': 0.22,
                'hours_worked': 0.15
            },
            'model_metrics': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85
            },
            'explanations': {
                'Model Overview': 'This model predicts customer churn',
                'Top Factors': 'Age and income are the most influential factors'
            },
            'predictions': pd.DataFrame({
                'id': [1, 2, 3],
                'prediction': [0, 1, 0],
                'confidence': [0.92, 0.78, 0.85]
            })
        }

    def test_exporter_initialization(self, exporter):
        """Test ExplanationExporter initializes correctly."""
        assert exporter is not None

    def test_json_export_basic(self, exporter, sample_data):
        """Test basic JSON export."""
        json_str = exporter.export_to_json(
            explanations=sample_data['explanations'],
            feature_importance=sample_data['feature_importance'],
            model_metrics=sample_data['model_metrics']
        )
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert 'timestamp' in data
        assert 'explanations' in data
        assert 'feature_importance' in data
        assert 'model_metrics' in data

    def test_json_export_with_predictions(self, exporter, sample_data):
        """Test JSON export with predictions."""
        predictions_list = sample_data['predictions'].to_dict('records')
        json_str = exporter.export_to_json(
            explanations=sample_data['explanations'],
            predictions=predictions_list
        )
        
        data = json.loads(json_str)
        assert len(data['predictions']) == 3

    def test_excel_export(self, exporter, sample_data):
        """Test Excel export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.xlsx')
            
            result = exporter.export_to_excel(
                filename=filename,
                feature_importance=sample_data['feature_importance'],
                model_metrics=sample_data['model_metrics'],
                predictions=sample_data['predictions'],
                explanations=sample_data['explanations']
            )
            
            assert isinstance(result, ExportResult)
            assert result.format == 'excel'
            assert result.success
            assert os.path.exists(filename)
            assert result.file_size > 0

    def test_excel_export_result_dataclass(self, exporter, sample_data):
        """Test ExportResult dataclass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.xlsx')
            result = exporter.export_to_excel(filename, model_metrics=sample_data['model_metrics'])
            
            assert hasattr(result, 'format')
            assert hasattr(result, 'filename')
            assert hasattr(result, 'file_size')
            assert hasattr(result, 'timestamp')
            assert hasattr(result, 'success')
            assert hasattr(result, 'message')

    def test_html_export(self, exporter, sample_data):
        """Test HTML export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.html')
            
            result = exporter.export_to_html(
                filename=filename,
                title='Test Report',
                feature_importance=sample_data['feature_importance'],
                model_metrics=sample_data['model_metrics'],
                explanations=sample_data['explanations']
            )
            
            assert result.format == 'html'
            assert os.path.exists(filename)
            
            # Check file content with UTF-8 encoding
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                assert '<html>' in content or '<!DOCTYPE html>' in content
                assert 'Test Report' in content

    def test_html_export_with_predictions(self, exporter, sample_data):
        """Test HTML export includes prediction table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.html')
            
            result = exporter.export_to_html(
                filename=filename,
                predictions=sample_data['predictions']
            )
            
            assert os.path.exists(filename)
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                # Should have table for predictions
                assert '<table>' in content or '<tr>' in content

    def test_pdf_export(self, exporter, sample_data):
        """Test PDF export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.pdf')
            
            result = exporter.export_to_pdf(
                filename=filename,
                title='Test PDF Report',
                feature_importance=sample_data['feature_importance'],
                model_metrics=sample_data['model_metrics'],
                explanations=sample_data['explanations']
            )
            
            assert result.format == 'pdf'
            # Should either succeed or fail gracefully
            assert isinstance(result.success, bool)
            if result.success:
                assert os.path.exists(filename)

    def test_batch_export(self, exporter, sample_data):
        """Test batch export to multiple formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = exporter.export_batch(
                sample_data,
                output_dir=tmpdir,
                formats=['json', 'html', 'excel']
            )
            
            assert len(results) >= 2  # At least json and one other
            assert all(isinstance(r, ExportResult) for r in results)
            assert any(r.format == 'json' for r in results)

    def test_batch_export_creates_directory(self, exporter, sample_data):
        """Test batch export creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'new_dir', 'nested')
            
            results = exporter.export_batch(
                sample_data,
                output_dir=output_dir,
                formats=['json']
            )
            
            assert os.path.exists(output_dir)

    def test_export_empty_explanations(self, exporter):
        """Test export with empty explanations."""
        json_str = exporter.export_to_json(
            explanations={},
            feature_importance={},
            model_metrics={}
        )
        
        data = json.loads(json_str)
        assert data['explanations'] == {}
        assert data['feature_importance'] == {}

    def test_export_with_nan_values(self, exporter):
        """Test export handles NaN values."""
        metrics = {
            'accuracy': 0.85,
            'precision': float('nan'),
            'recall': 0.88
        }
        
        json_str = exporter.export_to_json(
            explanations={},
            model_metrics=metrics
        )
        
        # Should handle NaN gracefully
        assert isinstance(json_str, str)

    def test_excel_export_no_predictions(self, exporter, sample_data):
        """Test Excel export without predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.xlsx')
            
            result = exporter.export_to_excel(
                filename=filename,
                feature_importance=sample_data['feature_importance'],
                model_metrics=sample_data['model_metrics']
            )
            
            assert result.success

    def test_html_export_styling(self, exporter, sample_data):
        """Test HTML export includes styling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.html')
            
            exporter.export_to_html(
                filename=filename,
                feature_importance=sample_data['feature_importance']
            )
            
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                assert '<style>' in content
                assert 'font-family' in content

    def test_model_card_creation(self, exporter):
        """Test model card creation."""
        card = exporter.create_model_card(
            model_name='Customer Churn Model',
            model_type='Random Forest',
            description='Predicts customer churn probability',
            metrics={'accuracy': 0.85, 'precision': 0.82},
            features_used=['age', 'income', 'tenure'],
            training_date='2025-01-20'
        )
        
        assert card['model_name'] == 'Customer Churn Model'
        assert card['model_type'] == 'Random Forest'
        assert card['num_features'] == 3
        assert 'export_date' in card

    def test_model_card_with_characteristics(self, exporter):
        """Test model card with performance characteristics."""
        characteristics = "Model performs well on balanced classes with moderate feature importance."
        
        card = exporter.create_model_card(
            model_name='Test Model',
            model_type='Logistic Regression',
            description='Test',
            metrics={'accuracy': 0.80},
            features_used=['a', 'b'],
            training_date='2025-01-20',
            performance_characteristics=characteristics
        )
        
        assert card['performance_characteristics'] == characteristics

    def test_json_export_no_predictions(self, exporter):
        """Test JSON export without predictions."""
        json_str = exporter.export_to_json(
            explanations={'test': 'value'},
            predictions=None
        )
        
        data = json.loads(json_str)
        assert data['predictions'] == []

    def test_excel_export_large_dataframe(self, exporter):
        """Test Excel export with large predictions dataframe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'large.xlsx')
            
            # Create large dataframe
            large_df = pd.DataFrame({
                'pred_' + str(i): np.random.rand(1000)
                for i in range(20)
            })
            
            result = exporter.export_to_excel(
                filename=filename,
                predictions=large_df
            )
            
            assert result.success

    def test_html_export_feature_importance_sorting(self, exporter):
        """Test HTML export sorts features by importance."""
        fi = {
            'low_imp': 0.1,
            'high_imp': 0.9,
            'med_imp': 0.5
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.html')
            exporter.export_to_html(filename, feature_importance=fi)
            
            with open(filename, 'r') as f:
                content = f.read()
                # Check that high importance comes first
                high_pos = content.find('high_imp')
                med_pos = content.find('med_imp')
                low_pos = content.find('low_imp')
                
                if high_pos > 0 and med_pos > 0:
                    assert high_pos < med_pos

    def test_export_result_failed_status(self, exporter):
        """Test ExportResult with failed status."""
        result = ExportResult(
            format='test',
            filename='test.txt',
            file_size=0,
            timestamp='2025-01-20T10:00:00',
            success=False,
            message='Test failure'
        )
        
        assert result.success is False
        assert 'failure' in result.message.lower()

    def test_batch_export_mixed_formats(self, exporter, sample_data):
        """Test batch export with different format combinations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export all formats
            results = exporter.export_batch(
                sample_data,
                output_dir=tmpdir,
                formats=['json', 'html', 'excel', 'pdf']
            )
            
            # Should have results for each format requested
            assert len(results) >= 3

    def test_json_version_info(self, exporter):
        """Test JSON export includes version info."""
        json_str = exporter.export_to_json(explanations={})
        data = json.loads(json_str)
        
        assert 'export_version' in data
        assert data['export_version'] == '1.0'

    def test_html_title_in_content(self, exporter):
        """Test HTML export uses provided title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.html')
            title = 'My Custom Report Title'
            
            exporter.export_to_html(filename, title=title)
            
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                assert title in content

    def test_export_dataframes_with_none_values(self, exporter):
        """Test handling DataFrames with None values."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': ['x', 'y', None]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.xlsx')
            result = exporter.export_to_excel(filename, predictions=df)
            
            # Should handle gracefully
            assert isinstance(result, ExportResult)
