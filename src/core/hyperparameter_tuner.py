"""Hyperparameter tuning for ML models using grid search."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Grid search-based hyperparameter optimization for ML models."""

    TUNING_CACHE_DIR = Path("models/tuning_cache")

    # Predefined parameter grids for each model type
    PARAM_GRIDS = {
        "logistic_regression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear", "lbfgs"],
            "max_iter": [200, 500, 1000],
            "penalty": ["l2"],
        },
        "logistic_regression_fast": {
            # Smaller grid for quick tuning
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"],
            "max_iter": [500],
        },
        "random_forest_classification": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "random_forest_classification_fast": {
            # Smaller grid for quick tuning
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [5],
            "min_samples_leaf": [2],
        },
        "random_forest_regression": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "random_forest_regression_fast": {
            # Smaller grid for quick tuning
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [5],
        },
    }

    def __init__(self, n_jobs: int = -1, cv_folds: int = 5, verbose: int = 1):
        """
        Initialize hyperparameter tuner.

        Args:
            n_jobs: Number of parallel jobs (-1 = use all cores)
            cv_folds: Number of cross-validation folds
            verbose: Verbosity level (0, 1, 2)
        """
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.last_results = {}

        # Create cache directory if needed
        self.TUNING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def tune_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fast: bool = False,
        scoring: str = "accuracy",
    ) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
        """
        Optimize LogisticRegression hyperparameters.

        Args:
            X_train: Training features
            y_train: Training target
            fast: Use faster (smaller) parameter grid
            scoring: Scoring metric

        Returns:
            Tuple of (best_params, best_model, cv_results)
        """
        param_grid_key = "logistic_regression_fast" if fast else "logistic_regression"
        param_grid = self.PARAM_GRIDS[param_grid_key]

        model = LogisticRegression(random_state=42, max_iter=1000)

        logger.info(
            f"Tuning LogisticRegression with {len(param_grid)} parameter combinations"
        )

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring=scoring,
        )

        grid_search.fit(X_train, y_train)

        self.last_results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
            "model_name": "logistic_regression",
        }

        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")

        return (
            grid_search.best_params_,
            grid_search.best_estimator_,
            grid_search.cv_results_,
        )

    def tune_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str = "classification",
        fast: bool = False,
        scoring: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
        """
        Optimize RandomForest hyperparameters.

        Args:
            X_train: Training features
            y_train: Training target
            task_type: "classification" or "regression"
            fast: Use faster (smaller) parameter grid
            scoring: Scoring metric

        Returns:
            Tuple of (best_params, best_model, cv_results)
        """
        if task_type == "classification":
            param_grid_key = (
                "random_forest_classification_fast"
                if fast
                else "random_forest_classification"
            )
            model = RandomForestClassifier(random_state=42, n_jobs=1)
            if scoring is None:
                scoring = "accuracy"
        else:
            param_grid_key = "random_forest_regression_fast" if fast else "random_forest_regression"
            model = RandomForestRegressor(random_state=42, n_jobs=1)
            if scoring is None:
                scoring = "neg_mean_squared_error"

        param_grid = self.PARAM_GRIDS[param_grid_key]

        logger.info(
            f"Tuning RandomForest ({task_type}) with {len(param_grid)} parameter combinations"
        )

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring=scoring,
        )

        grid_search.fit(X_train, y_train)

        self.last_results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
            "model_name": "random_forest",
            "task_type": task_type,
        }

        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")

        return (
            grid_search.best_params_,
            grid_search.best_estimator_,
            grid_search.cv_results_,
        )

    def tune_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str = "classification",
        fast: bool = False,
        scoring: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
        """
        Generic method to tune any supported model.

        Args:
            model_name: Model name ("logistic_regression", "random_forest")
            X_train: Training features
            y_train: Training target
            task_type: "classification" or "regression"
            fast: Use faster tuning grid
            scoring: Scoring metric

        Returns:
            Tuple of (best_params, best_model, cv_results)
        """
        if model_name == "logistic_regression":
            return self.tune_logistic_regression(X_train, y_train, fast=fast, scoring=scoring or "accuracy")
        elif model_name == "random_forest":
            return self.tune_random_forest(
                X_train, y_train, task_type=task_type, fast=fast, scoring=scoring
            )
        else:
            raise ValueError(f"Tuning not available for model: {model_name}")

    def cache_tuning_results(
        self, model_name: str, task_type: str, best_params: Dict[str, Any]
    ) -> bool:
        """
        Save tuned hyperparameters to cache.

        Args:
            model_name: Model name
            task_type: Task type ("classification" or "regression")
            best_params: Best hyperparameters found

        Returns:
            True if saved successfully
        """
        try:
            cache_file = (
                self.TUNING_CACHE_DIR
                / f"tuned_{model_name}_{task_type}_params.json"
            )

            # Convert numpy types to serializable Python types
            serializable_params = {}
            for key, value in best_params.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_params[key] = float(value)
                else:
                    serializable_params[key] = value

            with open(cache_file, "w") as f:
                json.dump(serializable_params, f, indent=2)

            logger.info(f"Cached tuning results to {cache_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache tuning results: {e}")
            return False

    def load_cached_tuning_results(
        self, model_name: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load cached tuning results if available.

        Args:
            model_name: Model name
            task_type: Task type ("classification" or "regression")

        Returns:
            Cached best parameters or None if not found
        """
        try:
            cache_file = (
                self.TUNING_CACHE_DIR
                / f"tuned_{model_name}_{task_type}_params.json"
            )

            if not cache_file.exists():
                return None

            with open(cache_file, "r") as f:
                params = json.load(f)

            logger.info(f"Loaded cached tuning results from {cache_file}")
            return params

        except Exception as e:
            logger.warning(f"Failed to load cached tuning results: {e}")
            return None

    def get_parameter_importance(self) -> Optional[Dict[str, float]]:
        """
        Get parameter importance from last tuning run.

        Returns:
            Dictionary of parameter importance scores or None
        """
        if not self.last_results or "cv_results" not in self.last_results:
            return None

        try:
            cv_results = self.last_results["cv_results"]
            best_score = self.last_results["best_score"]

            # Calculate parameter importance based on score variance
            param_importance = {}

            for param_name in self.last_results.get("best_params", {}).keys():
                param_key = f"param_{param_name}"
                if param_key in cv_results:
                    scores = cv_results["mean_test_score"]
                    # Simple importance: normalized variance across parameter values
                    param_importance[param_name] = float(np.std(scores))

            return (
                param_importance if param_importance else None
            )

        except Exception as e:
            logger.debug(f"Failed to calculate parameter importance: {e}")
            return None

    def get_tuning_summary(self) -> Dict[str, Any]:
        """
        Get summary of last tuning run.

        Returns:
            Summary dictionary with key tuning statistics
        """
        if not self.last_results:
            return {"message": "No tuning results available"}

        return {
            "model_name": self.last_results.get("model_name"),
            "best_score": self.last_results.get("best_score"),
            "best_params": self.last_results.get("best_params"),
            "task_type": self.last_results.get("task_type"),
            "num_cv_folds": self.cv_folds,
        }

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Get list of models that support tuning."""
        return {
            "logistic_regression": "Logistic Regression (Classification)",
            "random_forest": "Random Forest (Classification & Regression)",
        }
