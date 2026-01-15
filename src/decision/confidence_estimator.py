"""Confidence and reliability estimation for model predictions."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ConfidenceEstimator:
    """
    Estimates confidence and uncertainty in predictions.
    Distinguishes between probability and actual confidence.
    """

    def __init__(self):
        """Initialize confidence estimator."""
        self.confidence_cache = {}

    def estimate_probability_confidence(self, y_proba: np.ndarray) -> np.ndarray:
        """Extract confidence from probabilities (highest class probability)."""
        return np.max(y_proba, axis=1)

    def estimate_margin_confidence(self, y_proba: np.ndarray) -> np.ndarray:
        """Estimate confidence as margin between top two classes."""
        if y_proba.shape[1] < 2:
            return self.estimate_probability_confidence(y_proba)
        sorted_proba = np.sort(y_proba, axis=1)[:, ::-1]
        margin = sorted_proba[:, 0] - sorted_proba[:, 1]
        margin_normalized = (margin + 1) / 2
        return margin_normalized

    def estimate_uncertainty(self, y_proba: np.ndarray) -> np.ndarray:
        """Estimate uncertainty using entropy of probability distribution."""
        entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
        max_entropy = np.log(y_proba.shape[1])
        uncertainty = entropy / max_entropy
        return uncertainty

    def estimate_ensemble_confidence(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Estimate confidence from multiple model predictions (ensemble agreement)."""
        if len(predictions_list) < 2:
            return np.ones(len(predictions_list[0]))
        predictions_array = np.array(predictions_list)
        n_models = len(predictions_list)
        agreement_scores = []
        for sample_idx in range(predictions_array.shape[1]):
            predictions_sample = predictions_array[:, sample_idx]
            unique, counts = np.unique(predictions_sample, return_counts=True)
            agreement = np.max(counts) / n_models
            agreement_scores.append(agreement)
        return np.array(agreement_scores)

    def get_confidence_levels(
        self, y_proba: np.ndarray, low_threshold: float = 0.6, high_threshold: float = 0.8
    ) -> List[str]:
        """Categorize confidence into levels (Low, Medium, High)."""
        confidence = self.estimate_probability_confidence(y_proba)
        levels = []
        for conf in confidence:
            if conf < low_threshold:
                levels.append("Low")
            elif conf < high_threshold:
                levels.append("Medium")
            else:
                levels.append("High")
        return levels

    def get_reliability_indicators(
        self, y_proba: np.ndarray, y_pred: np.ndarray = None
    ) -> pd.DataFrame:
        """Get comprehensive reliability indicators for each prediction."""
        n_samples = y_proba.shape[0]
        results = {
            "prediction_id": list(range(n_samples)),
            "probability_confidence": self.estimate_probability_confidence(y_proba),
            "margin_confidence": self.estimate_margin_confidence(y_proba),
            "uncertainty": self.estimate_uncertainty(y_proba),
            "confidence_level": self.get_confidence_levels(y_proba),
        }
        df = pd.DataFrame(results)
        if y_pred is not None:
            df["predicted_class"] = y_pred
        df["reliability_score"] = (
            df["probability_confidence"] * 0.5 + (1 - df["uncertainty"]) * 0.5
        )
        return df

    def flag_uncertain_predictions(
        self, y_proba: np.ndarray, uncertainty_threshold: float = 0.5
    ) -> np.ndarray:
        """Flag predictions with high uncertainty for manual review."""
        uncertainty = self.estimate_uncertainty(y_proba)
        return uncertainty > uncertainty_threshold

    def get_confidence_distribution_stats(self, y_proba: np.ndarray) -> Dict[str, float]:
        """Get statistics about confidence distribution across predictions."""
        confidence = self.estimate_probability_confidence(y_proba)
        return {
            "mean_confidence": float(np.mean(confidence)),
            "median_confidence": float(np.median(confidence)),
            "std_confidence": float(np.std(confidence)),
            "min_confidence": float(np.min(confidence)),
            "max_confidence": float(np.max(confidence)),
            "pct_high_confidence": float(np.sum(confidence > 0.8) / len(confidence) * 100),
            "pct_medium_confidence": float(
                np.sum((confidence >= 0.6) & (confidence <= 0.8)) / len(confidence) * 100
            ),
            "pct_low_confidence": float(np.sum(confidence < 0.6) / len(confidence) * 100),
        }
