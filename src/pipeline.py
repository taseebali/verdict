"""Main ML pipeline orchestration."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from .data_handler import DataHandler
from .preprocessing import Preprocessor
from .models import ModelManager
from .metrics import MetricsCalculator


class MLPipeline:
    """Main pipeline for the entire ML workflow."""

    def __init__(self, df: pd.DataFrame, target_col: str):
        """Initialize pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
        """
        self.df = df
        self.target_col = target_col
        self.data_handler = DataHandler(df)
        self.preprocessor = None
        self.model_manager = None
        self.metrics_calculator = MetricsCalculator()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}

    def validate(self) -> Tuple[bool, str]:
        """Validate input data."""
        is_valid, message = self.data_handler.validate_data()
        if not is_valid:
            return is_valid, message
        
        if self.target_col not in self.data_handler.get_columns():
            return False, f"Target column '{self.target_col}' not found in data."
        
        return True, "Validation passed."

    def preprocess(self) -> None:
        """Run data preprocessing."""
        # Handle missing values
        self.data_handler.handle_missing_values()
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor(self.data_handler.df, self.target_col)
        
        # Prepare data
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.prepare_data()
        
        # Initialize model manager
        task_type = self.preprocessor.get_task_type()
        self.model_manager = ModelManager(task_type=task_type)

    def train(self, model_names: list = None) -> Dict[str, Any]:
        """Train models.
        
        Args:
            model_names: List of model names to train. If None, train all available.
        
        Returns:
            Training results
        """
        if self.model_manager is None:
            raise RuntimeError("Must call preprocess() before train().")
        
        if model_names is None:
            model_names = self.model_manager.list_available_models()
        
        train_results = {}
        
        for model_name in model_names:
            try:
                result = self.model_manager.train(model_name, self.X_train, self.y_train)
                train_results[model_name] = result
            except Exception as e:
                train_results[model_name] = {"status": "failed", "error": str(e)}
        
        return train_results

    def evaluate(self, model_names: list = None) -> Dict[str, Dict[str, float]]:
        """Evaluate trained models on test set.
        
        Args:
            model_names: List of model names to evaluate. If None, evaluate all trained.
        
        Returns:
            Evaluation results
        """
        if self.model_manager is None:
            raise RuntimeError("Must call preprocess() before evaluate().")
        
        if model_names is None:
            model_names = list(self.model_manager.get_models().keys())
        
        eval_results = {}
        task_type = self.model_manager.task_type
        
        for model_name in model_names:
            try:
                y_pred = self.model_manager.predict(model_name, self.X_test)
                
                if task_type == "classification":
                    y_pred_proba = self.model_manager.predict_proba(model_name, self.X_test)
                    metrics = self.metrics_calculator.calculate_classification_metrics(
                        self.y_test, y_pred, y_pred_proba
                    )
                else:
                    metrics = self.metrics_calculator.calculate_regression_metrics(self.y_test, y_pred)
                
                eval_results[model_name] = metrics
            except Exception as e:
                eval_results[model_name] = {"error": str(e)}
        
        self.results = eval_results
        return eval_results

    def run_full_pipeline(self, model_names: list = None) -> Dict[str, Any]:
        """Run complete pipeline: validate -> preprocess -> train -> evaluate.
        
        Args:
            model_names: List of model names to train. If None, train all available.
        
        Returns:
            Full results dictionary
        """
        # Validate
        is_valid, message = self.validate()
        if not is_valid:
            return {"status": "failed", "error": message}
        
        # Preprocess
        try:
            self.preprocess()
        except Exception as e:
            return {"status": "failed", "error": f"Preprocessing failed: {str(e)}"}
        
        # Train
        try:
            train_results = self.train(model_names)
        except Exception as e:
            return {"status": "failed", "error": f"Training failed: {str(e)}"}
        
        # Evaluate
        try:
            eval_results = self.evaluate()
        except Exception as e:
            return {"status": "failed", "error": f"Evaluation failed: {str(e)}"}
        
        return {
            "status": "success",
            "task_type": self.model_manager.task_type,
            "train_results": train_results,
            "eval_results": eval_results,
            "feature_names": self.preprocessor.get_feature_names(),
        }

    def get_model_predictions(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get predictions from a trained model."""
        return self.model_manager.predict(model_name, X)

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data."""
        return self.X_test, self.y_test

    def get_numeric_columns(self) -> list:
        """Get list of numeric columns."""
        return self.data_handler.numeric_cols

    def get_categorical_columns(self) -> list:
        """Get list of categorical columns."""
        return self.data_handler.categorical_cols

    def get_label_encoders(self) -> dict:
        """Get label encoders from preprocessor."""
        return self.preprocessor.label_encoders if self.preprocessor else {}

    def get_target_encoder(self):
        """Get target label encoder."""
        return self.preprocessor.target_encoder if self.preprocessor else None
