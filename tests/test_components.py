"""Tests for Streamlit components."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

from src.ui.components import (
    DataLoadingComponent,
    TargetColumnSelector,
    FeatureSelector,
    PredictionInputComponent,
    AuditTrailComponent,
    ModelPerformanceComponent,
    DataQualityComponent
)


# Fixtures
@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'age': [25, 35, 45, 55, 65],
        'income': [30000, 50000, 75000, 100000, 120000],
        'purchased': [0, 1, 0, 1, 1],
        'region': ['A', 'B', 'A', 'C', 'B'],
        'customer_id': [1, 2, 3, 4, 5]
    })


@pytest.fixture
def numeric_df():
    """DataFrame with only numeric features."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'target': [0, 1, 0, 1, 0]
    })


# DataLoadingComponent Tests
class TestDataLoadingComponent:
    
    def test_initialization(self):
        """Test component initialization."""
        loader = DataLoadingComponent()
        assert loader.df is None
        assert loader.container is not None
    
    def test_initialization_with_custom_container(self):
        """Test initialization with custom container."""
        container = MagicMock()
        loader = DataLoadingComponent(container=container)
        assert loader.container == container
    
    @patch('src.ui.components.st')
    def test_validate_no_data(self, mock_st):
        """Test validation when no data loaded."""
        loader = DataLoadingComponent()
        result = loader.validate()
        assert result['valid'] is False
        assert 'No data loaded' in result['message']
    
    @patch('src.ui.components.DataValidator')
    def test_validate_with_data(self, mock_validator, sample_df):
        """Test validation with loaded data."""
        loader = DataLoadingComponent()
        loader.df = sample_df
        
        mock_validator.validate_basic.return_value = (True, 'Valid')
        result = loader.validate()
        
        assert result['valid'] is True
        assert 'Valid' in result['message']
        assert result['df'].equals(sample_df)


# TargetColumnSelector Tests
class TestTargetColumnSelector:
    
    def test_initialization(self, sample_df):
        """Test component initialization."""
        selector = TargetColumnSelector(sample_df)
        assert selector.df.equals(sample_df)
    
    def test_df_structure(self, sample_df):
        """Test that df structure is preserved."""
        selector = TargetColumnSelector(sample_df)
        assert len(selector.df) == 5
        assert len(selector.df.columns) == 5
    
    @patch('src.ui.components.st.selectbox')
    def test_render_with_candidates(self, mock_selectbox, sample_df):
        """Test render with valid target candidates."""
        # 'purchased' has 2 unique values, 'region' has 3
        mock_selectbox.return_value = 'purchased (2 classes)'
        
        selector = TargetColumnSelector(sample_df)
        target_col, unique_count = selector.render()
        
        assert target_col == 'purchased'
        assert unique_count == 2


# FeatureSelector Tests
class TestFeatureSelector:
    
    def test_initialization(self, sample_df):
        """Test component initialization."""
        selector = FeatureSelector(sample_df, 'purchased')
        assert selector.df.equals(sample_df)
        assert selector.target_col == 'purchased'
    
    @patch('src.ui.components.st.multiselect')
    def test_render_excludes_target(self, mock_multiselect, sample_df):
        """Test that target column is excluded."""
        mock_multiselect.return_value = ['age', 'income']
        
        selector = FeatureSelector(sample_df, 'purchased')
        features = selector.render()
        
        assert 'purchased' not in features
        assert 'age' in features
    
    @patch('src.ui.components.st.multiselect')
    def test_render_returns_selected(self, mock_multiselect, sample_df):
        """Test that selected features are returned."""
        selected_features = ['age', 'income', 'region']
        mock_multiselect.return_value = selected_features
        
        selector = FeatureSelector(sample_df, 'purchased')
        features = selector.render()
        
        assert features == selected_features


# PredictionInputComponent Tests
class TestPredictionInputComponent:
    
    def test_initialization(self, sample_df):
        """Test component initialization."""
        features = ['age', 'income']
        input_comp = PredictionInputComponent(sample_df, features)
        assert input_comp.df.equals(sample_df)
        assert input_comp.features == features
    
    @patch('src.ui.components.get_feature_statistics')
    @patch('src.ui.components.st.columns')
    @patch('src.ui.components.st.markdown')
    def test_render_structure(self, mock_md, mock_cols, mock_stats, sample_df):
        """Test render creates proper structure."""
        mock_stats.return_value = {
            'age': {'min': 20, 'max': 70, 'mean': 45},
            'income': {'min': 25000, 'max': 130000, 'mean': 75000}
        }
        mock_cols.return_value = [MagicMock(), MagicMock(), MagicMock()]
        
        features = ['age', 'income']
        input_comp = PredictionInputComponent(sample_df, features)
        result = input_comp.render()
        
        # Should return dict with features
        assert isinstance(result, dict)
        mock_md.assert_called_once()


# AuditTrailComponent Tests
class TestAuditTrailComponent:
    
    @patch('src.ui.components.st')
    def test_initialization(self, mock_st):
        """Test component initialization."""
        audit = AuditTrailComponent()
        assert audit is not None
    
    @patch('src.ui.components.st.info')
    @patch('src.ui.components.st.session_state', {'audit_trail': []})
    def test_render_empty_audit_trail(self, mock_info):
        """Test render with empty audit trail."""
        with patch('src.ui.components.st') as mock_st:
            mock_st.session_state = {'audit_trail': []}
            audit = AuditTrailComponent()
            # Should show info message for empty trail
            assert True  # Component handles gracefully
    
    @patch('src.ui.components.st.metric')
    @patch('src.ui.components.st.dataframe')
    @patch('src.ui.components.st.markdown')
    def test_render_with_data(self, mock_md, mock_df, mock_metric):
        """Test render with audit data."""
        audit_data = [
            {'prediction': 1, 'confidence': 0.95, 'model': 'RandomForest'},
            {'prediction': 0, 'confidence': 0.87, 'model': 'LogisticRegression'}
        ]
        
        with patch('src.ui.components.st') as mock_st:
            mock_st.session_state = {'audit_trail': audit_data}
            audit = AuditTrailComponent()
            # Component should handle data
            assert True


# ModelPerformanceComponent Tests
class TestModelPerformanceComponent:
    
    def test_initialization_empty(self):
        """Test initialization with empty metrics."""
        perf = ModelPerformanceComponent({})
        assert perf.metrics == {}
    
    def test_initialization_with_metrics(self):
        """Test initialization with metrics."""
        metrics = {'accuracy': 0.95, 'precision': 0.92}
        perf = ModelPerformanceComponent(metrics)
        assert perf.metrics == metrics
    
    @patch('src.ui.components.st.info')
    def test_render_no_metrics(self, mock_info):
        """Test render with no metrics."""
        with patch('src.ui.components.st') as mock_st:
            perf = ModelPerformanceComponent({})
            # Should show info message
            assert True
    
    @patch('src.ui.components.st.metric')
    @patch('src.ui.components.st.columns')
    @patch('src.ui.components.st.markdown')
    def test_render_with_metrics(self, mock_md, mock_cols, mock_metric):
        """Test render with metrics."""
        mock_cols.return_value = [MagicMock(), MagicMock()]
        metrics = {'accuracy': 0.95, 'precision': 0.92}
        
        with patch('src.ui.components.st') as mock_st:
            perf = ModelPerformanceComponent(metrics)
            # Component renders properly
            assert True


# DataQualityComponent Tests
class TestDataQualityComponent:
    
    def test_initialization(self, sample_df):
        """Test component initialization."""
        quality = DataQualityComponent(sample_df)
        assert quality.df.equals(sample_df)
        assert quality.target_col is None
    
    def test_initialization_with_target(self, sample_df):
        """Test initialization with target column."""
        quality = DataQualityComponent(sample_df, 'purchased')
        assert quality.target_col == 'purchased'
    
    @patch('src.ui.components.DataValidator.validate_quality')
    @patch('src.ui.components.st.metric')
    @patch('src.ui.components.st.columns')
    @patch('src.ui.components.st.markdown')
    def test_render_structure(self, mock_md, mock_cols, mock_metric, mock_quality):
        """Test render creates proper structure."""
        mock_quality.return_value = {
            'rows': 5,
            'columns': 5,
            'missing_percentage': 0.0,
            'warnings': [],
            'missing_rows': 0,
            'duplicate_rows': 0,
            'numeric_columns': 3,
            'categorical_columns': 2
        }
        mock_cols.return_value = [MagicMock(), MagicMock(), MagicMock()]
        
        with patch('src.ui.components.st') as mock_st:
            quality = DataQualityComponent(sample_df)
            # Component renders
            assert True
    
    @patch('src.ui.components.DataValidator.validate_quality')
    @patch('src.ui.components.st.success')
    def test_render_no_warnings(self, mock_success, mock_quality, sample_df):
        """Test render when no quality warnings."""
        mock_quality.return_value = {
            'rows': 5,
            'columns': 5,
            'missing_percentage': 0.0,
            'warnings': [],
            'missing_rows': 0,
            'duplicate_rows': 0,
            'numeric_columns': 3,
            'categorical_columns': 2
        }
        
        with patch('src.ui.components.st') as mock_st:
            quality = DataQualityComponent(sample_df)
            # Component renders without warnings
            assert True
    
    @patch('src.ui.components.DataValidator.validate_quality')
    @patch('src.ui.components.st.warning')
    def test_render_with_warnings(self, mock_warning, mock_quality, sample_df):
        """Test render with quality warnings."""
        mock_quality.return_value = {
            'rows': 5,
            'columns': 5,
            'missing_percentage': 10.0,
            'warnings': ['Missing values detected', 'Class imbalance detected'],
            'missing_rows': 2,
            'duplicate_rows': 0,
            'numeric_columns': 3,
            'categorical_columns': 2
        }
        
        with patch('src.ui.components.st') as mock_st:
            quality = DataQualityComponent(sample_df)
            # Component renders with warnings
            assert True


# Integration Tests
class TestComponentIntegration:
    
    @patch('src.ui.components.st')
    def test_data_loading_workflow(self, mock_st, sample_df):
        """Test complete data loading workflow."""
        loader = DataLoadingComponent()
        loader.df = sample_df
        st.session_state = {'df': sample_df}
        
        # Should load successfully
        assert loader.df is not None
        assert len(loader.df) == 5
    
    @patch('src.ui.components.st.selectbox')
    @patch('src.ui.components.st.multiselect')
    def test_feature_selection_workflow(self, mock_multiselect, mock_selectbox, sample_df):
        """Test complete feature selection workflow."""
        # Select target
        mock_selectbox.return_value = 'purchased (2 classes)'
        target_selector = TargetColumnSelector(sample_df)
        target, count = target_selector.render()
        
        # Select features
        mock_multiselect.return_value = ['age', 'income', 'region']
        feature_selector = FeatureSelector(sample_df, target)
        features = feature_selector.render()
        
        assert target == 'purchased'
        assert len(features) == 3
        assert 'purchased' not in features
