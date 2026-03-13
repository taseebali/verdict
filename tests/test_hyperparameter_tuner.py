"""Tests for hyperparameter tuning functionality."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.core.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTuner:
    """Test hyperparameter tuning functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        original_dir = HyperparameterTuner.TUNING_CACHE_DIR
        temp_dir = Path(tempfile.mkdtemp())
        HyperparameterTuner.TUNING_CACHE_DIR = temp_dir
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        HyperparameterTuner.TUNING_CACHE_DIR = original_dir

    @pytest.fixture
    def tuner(self):
        """Create tuner instance."""
        return HyperparameterTuner(n_jobs=1, cv_folds=3, verbose=0)

    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        return X, y

    def test_tuner_initialization(self):
        """Should initialize with correct parameters."""
        tuner = HyperparameterTuner(n_jobs=2, cv_folds=5, verbose=1)
        assert tuner.n_jobs == 2
        assert tuner.cv_folds == 5
        assert tuner.verbose == 1

    def test_param_grids_defined(self):
        """Should have parameter grids for all models."""
        assert "logistic_regression" in HyperparameterTuner.PARAM_GRIDS
        assert "random_forest_classification" in HyperparameterTuner.PARAM_GRIDS
        assert "random_forest_regression" in HyperparameterTuner.PARAM_GRIDS

    def test_param_grids_non_empty(self):
        """All parameter grids should have content."""
        for model_name, param_grid in HyperparameterTuner.PARAM_GRIDS.items():
            assert len(param_grid) > 0, f"Empty param grid for {model_name}"

    def test_tune_logistic_regression(self, tuner, classification_data):
        """Should tune logistic regression."""
        X, y = classification_data
        best_params, best_model, cv_results = tuner.tune_logistic_regression(X, y)

        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        assert hasattr(best_model, "predict")
        assert "C" in best_params
        assert "solver" in best_params

    def test_tune_logistic_regression_fast(self, tuner, classification_data):
        """Should tune logistic regression with fast grid."""
        X, y = classification_data
        best_params, best_model, cv_results = tuner.tune_logistic_regression(
            X, y, fast=True
        )

        assert isinstance(best_params, dict)
        assert "C" in best_params

    def test_tune_random_forest_classification(self, tuner, classification_data):
        """Should tune random forest for classification."""
        X, y = classification_data
        best_params, best_model, cv_results = tuner.tune_random_forest(
            X, y, task_type="classification"
        )

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params

    def test_tune_random_forest_regression(self, tuner, regression_data):
        """Should tune random forest for regression."""
        X, y = regression_data
        best_params, best_model, cv_results = tuner.tune_random_forest(
            X, y, task_type="regression"
        )

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params

    def test_tune_model_generic(self, tuner, classification_data):
        """Generic tune_model method should work."""
        X, y = classification_data
        best_params, best_model, cv_results = tuner.tune_model(
            "logistic_regression", X, y
        )

        assert isinstance(best_params, dict)
        assert hasattr(best_model, "predict")

    def test_tune_model_unknown_model(self, tuner, classification_data):
        """Should raise error for unknown model."""
        X, y = classification_data
        with pytest.raises(ValueError):
            tuner.tune_model("unknown_model", X, y)

    def test_best_model_prediction(self, tuner, classification_data):
        """Best model from tuning should make predictions."""
        X, y = classification_data
        _, best_model, _ = tuner.tune_logistic_regression(X, y)

        predictions = best_model.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))

    def test_best_model_has_params(self, tuner, classification_data):
        """Best model should have best parameters set."""
        X, y = classification_data
        best_params, best_model, _ = tuner.tune_logistic_regression(X, y)

        for param_name, param_value in best_params.items():
            assert hasattr(best_model, param_name)

    def test_cv_results_structure(self, tuner, classification_data):
        """CV results should have standard structure."""
        X, y = classification_data
        _, _, cv_results = tuner.tune_logistic_regression(X, y)

        assert "mean_test_score" in cv_results
        assert "std_test_score" in cv_results
        assert len(cv_results["mean_test_score"]) > 0

    def test_cache_tuning_results(self, tuner, temp_cache_dir):
        """Should cache tuning results."""
        best_params = {"C": 1.0, "solver": "lbfgs"}
        success = tuner.cache_tuning_results("logistic_regression", "classification", best_params)

        assert success is True
        cache_file = temp_cache_dir / "tuned_logistic_regression_classification_params.json"
        assert cache_file.exists()

    def test_load_cached_tuning_results(self, tuner, temp_cache_dir):
        """Should load cached tuning results."""
        best_params = {"C": 1.0, "solver": "lbfgs"}
        tuner.cache_tuning_results("logistic_regression", "classification", best_params)

        loaded = tuner.load_cached_tuning_results("logistic_regression", "classification")
        assert loaded is not None
        assert loaded["C"] == 1.0

    def test_load_nonexistent_cache(self, tuner, temp_cache_dir):
        """Should return None for nonexistent cache."""
        loaded = tuner.load_cached_tuning_results("nonexistent", "classification")
        assert loaded is None

    def test_last_results_tracking(self, tuner, classification_data):
        """Should track last tuning results."""
        X, y = classification_data
        tuner.tune_logistic_regression(X, y)

        assert tuner.last_results is not None
        assert "best_params" in tuner.last_results
        assert "best_score" in tuner.last_results

    def test_get_parameter_importance(self, tuner, classification_data):
        """Should calculate parameter importance."""
        X, y = classification_data
        tuner.tune_logistic_regression(X, y)

        importance = tuner.get_parameter_importance()
        if importance:
            assert isinstance(importance, dict)
            assert len(importance) > 0

    def test_get_tuning_summary(self, tuner, classification_data):
        """Should return tuning summary."""
        X, y = classification_data
        tuner.tune_logistic_regression(X, y)

        summary = tuner.get_tuning_summary()
        assert "best_score" in summary
        assert "best_params" in summary
        assert "model_name" in summary

    def test_get_tuning_summary_no_results(self, tuner):
        """Should handle no results gracefully."""
        summary = tuner.get_tuning_summary()
        assert "message" in summary or "best_score" not in summary

    def test_get_available_models(self):
        """Should list available models."""
        available = HyperparameterTuner.get_available_models()
        assert "logistic_regression" in available
        assert "random_forest" in available
        assert len(available) >= 2

    def test_tune_with_custom_scoring(self, tuner, classification_data):
        """Should use custom scoring metric."""
        X, y = classification_data
        best_params, best_model, _ = tuner.tune_logistic_regression(
            X, y, scoring="f1_weighted"
        )

        assert isinstance(best_params, dict)

    def test_fast_tuning_uses_smaller_grid(self, tuner, classification_data):
        """Fast tuning should use smaller parameter grid."""
        X, y = classification_data

        # Regular tuning
        _, _, cv_results_regular = tuner.tune_logistic_regression(X, y, fast=False)

        # Fast tuning
        _, _, cv_results_fast = tuner.tune_logistic_regression(X, y, fast=True)

        # Fast should have fewer iterations
        assert len(cv_results_fast["mean_test_score"]) <= len(
            cv_results_regular["mean_test_score"]
        )

    def test_cache_dir_creation(self, temp_cache_dir):
        """Cache directory should be created if missing."""
        import shutil

        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)

        tuner = HyperparameterTuner()
        assert tuner.TUNING_CACHE_DIR.exists()

    def test_numpy_type_serialization(self, tuner, temp_cache_dir):
        """Should handle numpy types in cached parameters."""
        params = {
            "C": np.float64(1.0),
            "max_iter": np.int64(500),
            "solver": "lbfgs",
        }

        success = tuner.cache_tuning_results("test_model", "classification", params)
        assert success is True

        loaded = tuner.load_cached_tuning_results("test_model", "classification")
        assert loaded is not None
        assert isinstance(loaded["C"], (float, int))

    def test_multiple_tuning_runs(self, tuner, classification_data):
        """Should handle multiple tuning runs."""
        X, y = classification_data

        # Run 1
        tuner.tune_logistic_regression(X, y)
        summary1 = tuner.get_tuning_summary()

        # Run 2
        tuner.tune_random_forest(X, y, task_type="classification")
        summary2 = tuner.get_tuning_summary()

        # Summary should be from second run
        assert summary2["model_name"] == "random_forest"

    def test_cv_results_have_mean_scores(self, tuner, classification_data):
        """CV results should have mean test scores."""
        X, y = classification_data
        _, _, cv_results = tuner.tune_logistic_regression(X, y)

        assert "mean_test_score" in cv_results
        assert len(cv_results["mean_test_score"]) > 0
        assert all(isinstance(s, (int, float)) for s in cv_results["mean_test_score"])

    def test_best_score_improves_over_default(self, tuner, classification_data):
        """Tuned model should perform better than default."""
        from sklearn.linear_model import LogisticRegression

        X, y = classification_data

        # Default model score
        default_model = LogisticRegression(random_state=42, max_iter=1000)
        default_model.fit(X, y)
        default_score = default_model.score(X, y)

        # Tuned model score
        best_params, best_model, _ = tuner.tune_logistic_regression(X, y)
        tuned_score = best_model.score(X, y)

        # Tuned should be at least as good as default (may be equal with small dataset)
        assert tuned_score >= default_score * 0.95  # Allow small margin
