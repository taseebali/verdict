"""Model training and management module."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from typing import Dict, Any, Tuple, Optional, List
from config.settings import MODEL_CONFIGS, RANDOM_SEED
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model training, evaluation, and prediction with multiclass OvR support."""

    def __init__(self, task_type: str = "classification", strategy: str = "binary"):
        """Initialize model manager.
        
        Args:
            task_type: "classification" or "regression"
            strategy: "binary", "ovr" (One-vs-Rest), or "regression"
        """
        self.task_type = task_type
        self.strategy = strategy
        self.models = {}
        self.model_history = {}
        self.classes_ = None  # For multiclass
        self.ovr_models = {}  # For OvR binary models per class

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

    def train_ovr(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train One-vs-Rest multiclass model.
        
        Args:
            model_name: Base model name (e.g., "logistic_regression", "random_forest")
            X_train: Training features
            y_train: Training target (multiclass labels)
        
        Returns:
            Training info dictionary with OvR details
        """
        if self.strategy != "ovr":
            raise ValueError(f"OvR training only valid with strategy='ovr', got {self.strategy}")
        
        # Get unique classes
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        
        logger.info(f"Training OvR model '{model_name}' with {n_classes} classes: {self.classes_}")
        
        if n_classes < 2:
            raise ValueError(f"OvR requires at least 2 classes, got {n_classes}")
        
        # Train binary classifier for each class (one-vs-rest)
        self.ovr_models[model_name] = {}
        train_scores = {}
        
        for class_idx, class_label in enumerate(self.classes_):
            # Create binary target (current class vs rest)
            y_binary = (y_train == class_label).astype(int)
            
            # Train binary model
            binary_model = self._get_model_instance(model_name)
            binary_model.fit(X_train, y_binary)
            
            # Store model
            self.ovr_models[model_name][class_label] = binary_model
            
            # Evaluate
            train_score = binary_model.score(X_train, y_binary)
            train_scores[f"class_{class_label}"] = train_score
            
            logger.debug(f"Binary classifier for class '{class_label}': train_score={train_score:.3f}")
        
        # Store model reference
        self.models[model_name] = {
            "type": "ovr",
            "classes": self.classes_,
            "base_model": model_name,
            "binary_models": self.ovr_models[model_name]
        }
        
        self.model_history[model_name] = {
            "strategy": "ovr",
            "model": model_name,
            "train_scores": train_scores,
            "classes": self.classes_.tolist(),
        }
        
        avg_score = np.mean(list(train_scores.values()))
        logger.info(f"OvR model '{model_name}' trained. Average binary score: {avg_score:.3f}")
        
        return {
            "model_name": model_name,
            "strategy": "ovr",
            "n_classes": n_classes,
            "train_scores": train_scores,
            "average_score": avg_score,
            "status": "success",
        }

    def predict_ovr(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using OvR model (multiclass).
        
        Args:
            model_name: Name of the trained OvR model
            X: Features to predict on
        
        Returns:
            Predicted class labels
        """
        if model_name not in self.models or self.models[model_name].get("type") != "ovr":
            raise ValueError(f"Model '{model_name}' is not a trained OvR model")
        
        binary_models = self.models[model_name]["binary_models"]
        classes = self.models[model_name]["classes"]
        
        # Get probabilities from each binary classifier
        probabilities = []
        for class_label in classes:
            class_proba = binary_models[class_label].predict_proba(X)[:, 1]
            probabilities.append(class_proba)
        
        probabilities = np.column_stack(probabilities)
        
        # Predict class with highest probability
        predicted_indices = np.argmax(probabilities, axis=1)
        predictions = classes[predicted_indices]
        
        return predictions

    def predict_proba_ovr(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction probabilities for OvR model.
        
        Args:
            model_name: Name of the trained OvR model
            X: Features to predict on
        
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        if model_name not in self.models or self.models[model_name].get("type") != "ovr":
            raise ValueError(f"Model '{model_name}' is not a trained OvR model")
        
        binary_models = self.models[model_name]["binary_models"]
        classes = self.models[model_name]["classes"]
        
        # Get probabilities from each binary classifier
        probabilities = []
        for class_label in classes:
            class_proba = binary_models[class_label].predict_proba(X)[:, 1]
            probabilities.append(class_proba)
        
        probabilities = np.column_stack(probabilities)
        
        # Normalize probabilities to sum to 1
        probabilities = probabilities / (probabilities.sum(axis=1, keepdims=True) + 1e-10)
        
        # Get predictions
        predicted_indices = np.argmax(probabilities, axis=1)
        predictions = classes[predicted_indices]
        
        return predictions, probabilities
