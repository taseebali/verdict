"""Simplified multiclass integration tests - Days 5-6 Focus."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.core.data_handler import DataHandler
from src.core.preprocessing import Preprocessor
from src.core.models import ModelManager
from src.core.cross_validation import CrossValidationEngine
from src.decision.multiclass_handler import MultiClassDetector
from src.core.metrics import MetricsCalculator


class TestMulticlassIntegrationSimplified:
    """Simplified multiclass integration tests focusing on core functionality."""

    @pytest.fixture
    def mc_data(self):
        """Create test multiclass dataset."""
        X, y = make_classification(
            n_samples=300,
            n_features=12,
            n_informative=9,
            n_redundant=3,
            n_classes=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
        df['target'] = y
        return df

    # Core pipeline tests
    def test_mc_load_validate(self, mc_data):
        """Load and validate multiclass data."""
        handler = DataHandler(mc_data)
        assert handler.validate_data()[0]

    def test_mc_detect_problem(self, mc_data):
        """Detect multiclass problem type."""
        assert MultiClassDetector.detect_problem_type(mc_data['target']) == "multiclass"

    def test_mc_validate_target(self, mc_data):
        """Validate target variable."""
        assert MultiClassDetector.validate_target(mc_data['target'])[0]

    def test_mc_preprocess(self, mc_data):
        """Preprocess multiclass data."""
        prep = Preprocessor(mc_data, 'target')
        X_scaled = prep.scale_features(mc_data.drop('target', axis=1))
        assert X_scaled.shape == mc_data.drop('target', axis=1).shape

    def test_mc_train_predict(self, mc_data):
        """Train OvR and make predictions."""
        X = mc_data.drop('target', axis=1)
        y = mc_data['target'].values

        manager = ModelManager(task_type="classification", strategy="ovr")
        manager.train("random_forest", X, y)
        pred = manager.predict("random_forest", X)

        assert len(pred) == len(X)
        assert all(p in [0, 1, 2] for p in pred)

    def test_mc_probabilities(self, mc_data):
        """Get probability predictions."""
        X = mc_data.drop('target', axis=1)
        y = mc_data['target'].values

        manager = ModelManager(task_type="classification", strategy="ovr")
        manager.train("random_forest", X, y)
        proba = manager.predict_proba("random_forest", X)

        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_mc_metrics(self, mc_data):
        """Calculate multiclass metrics."""
        X = mc_data.drop('target', axis=1)
        y = mc_data['target'].values

        manager = ModelManager(task_type="classification", strategy="ovr")
        manager.train("random_forest", X, y)
        pred = manager.predict("random_forest", X)
        proba = manager.predict_proba("random_forest", X)

        metrics = MetricsCalculator.calculate_multiclass_metrics(y, pred, proba)
        assert isinstance(metrics, dict)

    def test_mc_cv_pipeline(self, mc_data):
        """Full CV pipeline."""
        X = mc_data.drop('target', axis=1)
        y = mc_data['target']

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        cv_engine = CrossValidationEngine(n_splits=3)
        result = cv_engine.run_cv(X, y, model, "rf", "multiclass")

        assert len(result["fold_results"]) == 3
        assert "aggregated_metrics" in result

    def test_mc_classes(self, mc_data):
        """Test class distribution."""
        target = mc_data['target']
        classes = MultiClassDetector.get_unique_classes(target)
        assert len(classes) == 3

    def test_mc_consistency(self, mc_data):
        """Test prediction consistency."""
        X = mc_data.drop('target', axis=1)
        y = mc_data['target'].values

        manager = ModelManager(task_type="classification", strategy="ovr")
        manager.train("random_forest", X, y)

        pred = manager.predict("random_forest", X)
        proba = manager.predict_proba("random_forest", X)
        pred_proba = np.argmax(proba, axis=1)

        assert np.array_equal(pred, pred_proba)

    def test_mc_high_dim(self):
        """Test with high-dimensional data."""
        X, y = make_classification(
            n_samples=150,
            n_features=50,
            n_informative=30,
            n_redundant=20,
            n_classes=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(50)])
        df['target'] = y

        handler = DataHandler(df)
        assert handler.validate_data()[0]

    def test_mc_imbalanced(self):
        """Test with imbalanced classes."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=3,
            weights=[0.6, 0.3, 0.1],
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df['target'] = y

        classes = MultiClassDetector.get_unique_classes(df['target'])
        assert len(classes) == 3
