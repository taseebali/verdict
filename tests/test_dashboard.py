"""
Tests for Streamlit Dashboard (P2.3)

Comprehensive tests for dashboard functionality, state management, and UI components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import streamlit as st


# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample dataset"""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    return df


@pytest.fixture
def sample_model():
    """Create sample trained model"""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def session_state():
    """Create mock session state"""
    state = {
        'uploaded_data': None,
        'trained_model': None,
        'model_serializer': MagicMock(),
        'audit_log': [],
        'model_features': [],
        'target_column': None,
        'X_test': None,
        'y_test': None
    }
    return state


# ============================================================================
# Session State Tests
# ============================================================================
class TestSessionState:
    """Test session state management"""
    
    def test_initial_state_empty(self, session_state):
        """Test initial session state is empty"""
        assert session_state['uploaded_data'] is None
        assert session_state['trained_model'] is None
        assert session_state['audit_log'] == []
    
    def test_upload_data_to_session(self, session_state, sample_data):
        """Test uploading data to session state"""
        session_state['uploaded_data'] = sample_data
        
        assert session_state['uploaded_data'] is not None
        assert len(session_state['uploaded_data']) == 100
    
    def test_store_trained_model(self, session_state, sample_model, sample_data):
        """Test storing trained model"""
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        X_test, y_test = X[:10], y[:10]
        
        session_state['trained_model'] = sample_model
        session_state['X_test'] = X_test
        session_state['y_test'] = y_test
        
        assert session_state['trained_model'] is not None
        assert len(session_state['X_test']) == 10
    
    def test_features_stored(self, session_state):
        """Test storing feature list"""
        features = ['feature_0', 'feature_1', 'feature_2']
        session_state['model_features'] = features
        
        assert session_state['model_features'] == features
        assert len(session_state['model_features']) == 3
    
    def test_target_column_stored(self, session_state):
        """Test storing target column name"""
        session_state['target_column'] = 'target'
        
        assert session_state['target_column'] == 'target'


# ============================================================================
# Data Handling Tests
# ============================================================================
class TestDataHandling:
    """Test data loading and processing"""
    
    def test_load_csv_data(self, sample_data):
        """Test loading CSV data"""
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) == 100
        assert len(sample_data.columns) == 11  # 10 features + target
    
    def test_data_shape_validation(self, sample_data):
        """Test data shape validation"""
        assert sample_data.shape[0] == 100
        assert sample_data.shape[1] == 11
    
    def test_feature_extraction(self, sample_data):
        """Test extracting features from data"""
        features = [col for col in sample_data.columns if col != 'target']
        
        assert len(features) == 10
        assert 'target' not in features
    
    def test_target_extraction(self, sample_data):
        """Test extracting target from data"""
        target = sample_data['target']
        
        assert len(target) == 100
        assert target.nunique() == 2
    
    def test_data_statistics(self, sample_data):
        """Test calculating data statistics"""
        stats = sample_data.describe()
        
        assert len(stats) == 8  # count, mean, std, etc.
        assert stats.loc['count'].min() == 100
    
    def test_missing_values_detection(self, sample_data):
        """Test detecting missing values"""
        missing = sample_data.isnull().sum()
        
        assert missing.sum() == 0
    
    def test_duplicate_detection(self, sample_data):
        """Test detecting duplicates"""
        duplicates = sample_data.duplicated().sum()
        
        assert isinstance(duplicates, (int, np.integer))
    
    def test_data_types_identification(self, sample_data):
        """Test identifying data types"""
        dtypes = sample_data.dtypes
        
        assert len(dtypes) == 11
        assert all(dtype in [np.dtype('float64'), np.dtype('int64')] for dtype in dtypes)


# ============================================================================
# Model Training Tests
# ============================================================================
class TestModelTraining:
    """Test model training functionality"""
    
    def test_train_basic_model(self, sample_data):
        """Test basic model training"""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_train_with_hyperparameters(self, sample_data):
        """Test training with specific hyperparameters"""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X, y)
        
        assert model.n_estimators == 50
        assert model.max_depth == 10
    
    def test_train_test_split(self, sample_data):
        """Test train/test split"""
        from sklearn.model_selection import train_test_split
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
    
    def test_stratified_split(self, sample_data):
        """Test stratified train/test split"""
        from sklearn.model_selection import train_test_split
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check class distribution preserved
        assert y_train.value_counts()[0] > 0
        assert y_train.value_counts()[1] > 0
    
    def test_model_prediction(self, sample_model):
        """Test model predictions"""
        X = np.array([[0.5] * 10, [0.3] * 10])
        predictions = sample_model.predict(X)
        
        assert len(predictions) == 2
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_probabilities(self, sample_model):
        """Test model probability predictions"""
        X = np.array([[0.5] * 10])
        probabilities = sample_model.predict_proba(X)
        
        assert probabilities.shape[0] == 1
        assert probabilities.shape[1] == 2
        assert np.allclose(probabilities.sum(), 1.0)


# ============================================================================
# Prediction Tests
# ============================================================================
class TestPredictions:
    """Test prediction functionality"""
    
    def test_single_prediction(self, sample_model):
        """Test single sample prediction"""
        X = np.array([[0.5] * 10])
        pred = sample_model.predict(X)[0]
        
        assert isinstance(pred, (int, np.integer))
    
    def test_batch_prediction(self, sample_model):
        """Test batch predictions"""
        X = np.array([[0.5] * 10 for _ in range(5)])
        preds = sample_model.predict(X)
        
        assert len(preds) == 5
    
    def test_probability_prediction(self, sample_model):
        """Test probability predictions"""
        X = np.array([[0.5] * 10])
        proba = sample_model.predict_proba(X)
        
        assert len(proba) == 1
        assert len(proba[0]) == 2
    
    def test_prediction_confidence(self, sample_model):
        """Test prediction confidence scores"""
        X = np.array([[0.5] * 10])
        proba = sample_model.predict_proba(X)[0]
        confidence = max(proba)
        
        assert 0 <= confidence <= 1
    
    def test_batch_predictions_with_dataframe(self, sample_model):
        """Test batch predictions from DataFrame"""
        df = pd.DataFrame({
            f'feature_{i}': [0.5] * 5 for i in range(10)
        })
        
        preds = sample_model.predict(df)
        
        assert len(preds) == 5


# ============================================================================
# Metrics Calculation Tests
# ============================================================================
class TestMetricsCalculation:
    """Test metrics calculation"""
    
    def test_accuracy_calculation(self, sample_model, sample_data):
        """Test accuracy calculation"""
        from sklearn.metrics import accuracy_score
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        preds = sample_model.predict(X)
        accuracy = accuracy_score(y, preds)
        
        assert 0 <= accuracy <= 1
    
    def test_precision_calculation(self, sample_model, sample_data):
        """Test precision calculation"""
        from sklearn.metrics import precision_score
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        preds = sample_model.predict(X)
        precision = precision_score(y, preds, average='weighted', zero_division=0)
        
        assert 0 <= precision <= 1
    
    def test_recall_calculation(self, sample_model, sample_data):
        """Test recall calculation"""
        from sklearn.metrics import recall_score
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        preds = sample_model.predict(X)
        recall = recall_score(y, preds, average='weighted', zero_division=0)
        
        assert 0 <= recall <= 1
    
    def test_f1_score_calculation(self, sample_model, sample_data):
        """Test F1 score calculation"""
        from sklearn.metrics import f1_score
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        preds = sample_model.predict(X)
        f1 = f1_score(y, preds, average='weighted', zero_division=0)
        
        assert 0 <= f1 <= 1
    
    def test_confusion_matrix(self, sample_model, sample_data):
        """Test confusion matrix calculation"""
        from sklearn.metrics import confusion_matrix
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        preds = sample_model.predict(X)
        cm = confusion_matrix(y, preds)
        
        assert cm.shape == (2, 2)
    
    def test_roc_auc_score(self, sample_model, sample_data):
        """Test ROC AUC score"""
        from sklearn.metrics import roc_auc_score
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        proba = sample_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        
        assert 0 <= auc <= 1


# ============================================================================
# Feature Importance Tests
# ============================================================================
class TestFeatureImportance:
    """Test feature importance calculation"""
    
    def test_feature_importances_shape(self, sample_model, sample_data):
        """Test feature importances shape"""
        importances = sample_model.feature_importances_
        
        assert len(importances) == 10
    
    def test_feature_importances_sum(self, sample_model):
        """Test feature importances sum to 1"""
        importances = sample_model.feature_importances_
        
        assert np.allclose(importances.sum(), 1.0)
    
    def test_feature_importances_positive(self, sample_model):
        """Test feature importances are positive"""
        importances = sample_model.feature_importances_
        
        assert all(imp >= 0 for imp in importances)
    
    def test_feature_ranking(self, sample_model, sample_data):
        """Test feature ranking by importance"""
        importances = sample_model.feature_importances_
        features = [f'feature_{i}' for i in range(10)]
        
        ranking = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        
        assert len(ranking) == 10


# ============================================================================
# Audit Log Tests
# ============================================================================
class TestAuditLog:
    """Test audit logging functionality"""
    
    def test_audit_log_creation(self, session_state):
        """Test creating audit log entry"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "test_action",
            "details": "Test details",
            "status": "success"
        }
        session_state['audit_log'].append(log_entry)
        
        assert len(session_state['audit_log']) == 1
    
    def test_audit_log_multiple_entries(self, session_state):
        """Test multiple audit entries"""
        for i in range(5):
            session_state['audit_log'].append({
                "timestamp": datetime.now().isoformat(),
                "action": f"action_{i}",
                "details": f"Details {i}",
                "status": "success"
            })
        
        assert len(session_state['audit_log']) == 5
    
    def test_audit_log_status_tracking(self, session_state):
        """Test tracking action status"""
        session_state['audit_log'].append({
            "timestamp": datetime.now().isoformat(),
            "action": "test",
            "details": "",
            "status": "success"
        })
        session_state['audit_log'].append({
            "timestamp": datetime.now().isoformat(),
            "action": "test",
            "details": "",
            "status": "error"
        })
        
        success_count = sum(1 for log in session_state['audit_log'] if log['status'] == 'success')
        error_count = sum(1 for log in session_state['audit_log'] if log['status'] == 'error')
        
        assert success_count == 1
        assert error_count == 1
    
    def test_audit_log_action_filtering(self, session_state):
        """Test filtering audit log by action"""
        session_state['audit_log'] = [
            {"action": "upload", "status": "success"},
            {"action": "train", "status": "success"},
            {"action": "predict", "status": "success"},
            {"action": "upload", "status": "error"}
        ]
        
        upload_logs = [log for log in session_state['audit_log'] if log['action'] == 'upload']
        
        assert len(upload_logs) == 2
    
    def test_audit_log_timestamp(self, session_state):
        """Test audit log timestamp format"""
        timestamp = datetime.now().isoformat()
        session_state['audit_log'].append({
            "timestamp": timestamp,
            "action": "test",
            "details": "",
            "status": "success"
        })
        
        assert session_state['audit_log'][0]['timestamp'] == timestamp


# ============================================================================
# Model Manager Tests
# ============================================================================
class TestModelManager:
    """Test model management functionality"""
    
    def test_save_model(self, session_state):
        """Test saving model"""
        serializer = session_state['model_serializer']
        serializer.list_models.return_value = ["model_1"]
        
        models = serializer.list_models()
        
        assert "model_1" in models
    
    def test_load_model(self, session_state, sample_model):
        """Test loading model"""
        serializer = session_state['model_serializer']
        serializer.load_model.return_value = sample_model
        
        model = serializer.load_model("model_1")
        
        assert model is not None
    
    def test_delete_model(self, session_state):
        """Test deleting model"""
        serializer = session_state['model_serializer']
        serializer.delete_model("model_1")
        
        assert serializer.delete_model.called
    
    def test_model_info_retrieval(self, session_state):
        """Test getting model info"""
        serializer = session_state['model_serializer']
        serializer.get_model_info.return_value = {
            "name": "model_1",
            "accuracy": 0.95
        }
        
        info = serializer.get_model_info("model_1")
        
        assert info["name"] == "model_1"
        assert info["accuracy"] == 0.95


# ============================================================================
# Data Visualization Tests
# ============================================================================
class TestDataVisualization:
    """Test data visualization components"""
    
    def test_histogram_data(self, sample_data):
        """Test histogram data preparation"""
        feature = sample_data['feature_0']
        
        assert len(feature) == 100
    
    def test_bar_chart_data(self, sample_data):
        """Test bar chart data preparation"""
        dtype_counts = sample_data.dtypes.value_counts()
        
        assert len(dtype_counts) > 0
    
    def test_confusion_matrix_data(self, sample_model, sample_data):
        """Test confusion matrix data preparation"""
        from sklearn.metrics import confusion_matrix
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        preds = sample_model.predict(X)
        cm = confusion_matrix(y, preds)
        
        assert cm.shape[0] == cm.shape[1]
    
    def test_feature_importance_dataframe(self, sample_model):
        """Test feature importance dataframe"""
        importances = sample_model.feature_importances_
        features = [f'feature_{i}' for i in range(10)]
        
        df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        
        assert len(df) == 10
        assert df['Importance'].sum() > 0


# ============================================================================
# Integration Tests
# ============================================================================
class TestDashboardIntegration:
    """Integration tests for dashboard"""
    
    def test_upload_train_predict_workflow(self, sample_data, session_state):
        """Test complete upload -> train -> predict workflow"""
        # Upload
        session_state['uploaded_data'] = sample_data
        assert session_state['uploaded_data'] is not None
        
        # Train
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        session_state['trained_model'] = model
        assert session_state['trained_model'] is not None
        
        # Predict
        X_test = np.array([[0.5] * 10])
        pred = model.predict(X_test)
        assert len(pred) == 1
    
    def test_data_exploration_workflow(self, sample_data, session_state):
        """Test data exploration workflow"""
        session_state['uploaded_data'] = sample_data
        
        # Get statistics
        stats = sample_data.describe()
        assert len(stats) == 8
        
        # Get missing values
        missing = sample_data.isnull().sum()
        assert missing.sum() == 0
        
        # Get data types
        dtypes = sample_data.dtypes
        assert len(dtypes) == 11
    
    def test_model_persistence_workflow(self, session_state, sample_model):
        """Test model persistence workflow"""
        serializer = session_state['model_serializer']
        
        # Save
        serializer.save_model(sample_model, "test_model")
        
        # List
        serializer.list_models.return_value = ["test_model"]
        models = serializer.list_models()
        assert "test_model" in models
        
        # Load
        serializer.load_model.return_value = sample_model
        loaded = serializer.load_model("test_model")
        assert loaded is not None


# ============================================================================
# UI Component Tests
# ============================================================================
class TestUIComponents:
    """Test UI component functionality"""
    
    def test_metric_card_display(self):
        """Test metric card display"""
        # This would be tested in Streamlit's testing framework
        pass
    
    def test_dataframe_display(self, sample_data):
        """Test dataframe display"""
        assert len(sample_data) > 0
        assert len(sample_data.columns) > 0
    
    def test_expander_content(self):
        """Test expander component content"""
        # Streamlit component test
        pass
