"""Model training and management module."""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Tuple
from config.settings import MODEL_CONFIGS, RANDOM_SEED


class ModelManager:
    """Manages model training, evaluation, and prediction."""

    def __init__(self, task_type: str = "classification"):
        """Initialize model manager.
        
        Args:
            task_type: "classification" or "regression"
        """
        self.task_type = task_type
        self.models = {}
        self.model_history = {}

    def _get_model_instance(self, model_name: str):
        """Get model instance based on name and task type."""
        if model_name == "logistic_regression":
            if self.task_type == "classification":
                return LogisticRegression(**MODEL_CONFIGS["logistic_regression"]["params"])
            else:
                return LinearRegression()
        
        elif model_name == "random_forest":
            if self.task_type == "classification":
                return RandomForestClassifier(**MODEL_CONFIGS["random_forest"]["params"])
            else:
                return RandomForestRegressor(**MODEL_CONFIGS["random_forest"]["params"])
        
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train a model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
        
        Returns:
            Training info dictionary
        """
        model = self._get_model_instance(model_name)
        model.fit(X_train, y_train)
        
        self.models[model_name] = model
        train_score = model.score(X_train, y_train)
        
        self.model_history[model_name] = {
            "model": model,
            "train_score": train_score,
        }
        
        return {
            "model_name": model_name,
            "train_score": train_score,
            "status": "success",
        }

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model.
        
        Args:
            model_name: Name of the trained model
            X: Features to predict on
        
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        
        return self.models[model_name].predict(X)

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only).
        
        Args:
            model_name: Name of the trained model
            X: Features to predict on
        
        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        
        if not hasattr(self.models[model_name], "predict_proba"):
            raise ValueError(f"Model '{model_name}' does not support predict_proba.")
        
        return self.models[model_name].predict_proba(X)

    def get_model(self, model_name: str):
        """Get a trained model instance."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        return self.models[model_name]

    def get_models(self) -> Dict[str, Any]:
        """Get all trained models."""
        return self.models

    def list_available_models(self) -> list:
        """List available model types for this task."""
        if self.task_type == "classification":
            return list(MODEL_CONFIGS.keys())
        else:
            return list(MODEL_CONFIGS.keys())
