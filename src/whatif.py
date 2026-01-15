"""What-if analysis module for interactive predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class FeatureStats:
    min: float
    max: float
    mean: float
    std: float


class WhatIfAnalyzer:
    """Simulates predictions with different input scenarios.

    Supports:
    - numeric features (sliders)
    - categorical features:
        * label-encoded (single column per category)
        * one-hot encoded (multiple columns like feature_value)
    """

    def __init__(
        self,
        pipeline,
        feature_names: List[str],
        numeric_cols: List[str],
        categorical_cols: List[str],
        label_encoders: Dict[str, LabelEncoder],
        onehot_sep: str = "_",
    ):
        """
        Args:
            pipeline: MLPipeline instance with trained models
            feature_names: Names of features expected by the trained model
                          (after preprocessing, if applicable)
            numeric_cols: Original numeric column names
            categorical_cols: Original categorical column names
            label_encoders: Dict of fitted LabelEncoders for categoricals (if using label encoding)
            onehot_sep: Separator used in one-hot feature names (default "_")
        """
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.label_encoders = label_encoders
        self.onehot_sep = onehot_sep

        self.X_train = pipeline.X_train
        self.X_test = pipeline.X_test

        # Build fast index lookup for feature_names
        self._feat_to_idx = {f: i for i, f in enumerate(self.feature_names)}

        # Detect whether feature_names looks like one-hot for any categorical col
        self._is_onehot = self._detect_onehot()

        # Cache stats for numeric features for defaults & slider ranges
        self._numeric_stats = self._compute_numeric_stats()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _to_dense_2d(self, X) -> np.ndarray:
        """Return a dense numpy array for DataFrame / numpy / sparse matrix."""
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        # scipy sparse matrices have toarray()
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)

    def _col_values(self, X, i: int) -> np.ndarray:
        """Safely get column i as 1D numpy array."""
        Xd = self._to_dense_2d(X)
        return Xd[:, i]

    def _detect_onehot(self) -> bool:
        """Heuristic: if any categorical feature appears as multiple one-hot columns."""
        # Look for columns that start with "<cat>_" (or configured sep)
        for cat in self.categorical_cols:
            prefix = f"{cat}{self.onehot_sep}"
            if any(f.startswith(prefix) for f in self.feature_names):
                return True
        return False

    def _compute_numeric_stats(self) -> Dict[str, FeatureStats]:
        """Compute stats for numeric features (only those present in feature_names)."""
        stats: Dict[str, FeatureStats] = {}

        # If you're one-hot, numeric features likely still appear by their original names
        for feat in self.numeric_cols:
            if feat in self._feat_to_idx:
                i = self._feat_to_idx[feat]
                col = self._col_values(self.X_train, i)
                stats[feat] = FeatureStats(
                    min=float(np.min(col)),
                    max=float(np.max(col)),
                    mean=float(np.mean(col)),
                    std=float(np.std(col)),
                )

        return stats

    def _default_value_for_feature(self, feature: str, i: int) -> float:
        """Fallback default if user doesn't provide a value."""
        # Prefer cached numeric mean if available
        if feature in self._numeric_stats:
            return self._numeric_stats[feature].mean

        # Otherwise compute mean from training column i
        col = self._col_values(self.X_train, i)
        return float(np.mean(col))

    def _set_onehot(self, arr: np.ndarray, base_feature: str, value: Any) -> None:
        """Set one-hot columns for a base categorical feature."""
        prefix = f"{base_feature}{self.onehot_sep}"

        # Clear all one-hot columns for this base feature
        for fname, idx in self._feat_to_idx.items():
            if fname.startswith(prefix):
                arr[0, idx] = 0.0

        # Set the chosen category column if it exists
        chosen = f"{base_feature}{self.onehot_sep}{value}"
        if chosen in self._feat_to_idx:
            arr[0, self._feat_to_idx[chosen]] = 1.0
        else:
            # unseen category: leave all zeros (or optionally choose a default)
            # Leaving zeros is safer than guessing.
            pass

    def _set_label_encoded(self, arr: np.ndarray, feature: str, value: Any) -> None:
        """Set a label-encoded categorical feature."""
        idx = self._feat_to_idx.get(feature)
        if idx is None:
            return

        if feature in self.label_encoders:
            enc = self.label_encoders[feature]
            try:
                arr[0, idx] = float(enc.transform([value])[0])
            except ValueError:
                # unseen category: choose 0
                arr[0, idx] = 0.0
        else:
            # No encoder available, try numeric cast
            arr[0, idx] = float(value)

    def _build_feature_array(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """Build the model input array (shape 1 x n_features) from user-provided values."""
        feature_array = np.zeros((1, len(self.feature_names)), dtype=float)

        # First fill defaults
        for i, feature in enumerate(self.feature_names):
            feature_array[0, i] = self._default_value_for_feature(feature, i)

        # Then overwrite from input_dict
        for key, value in input_dict.items():
            # Numeric features: direct set if present
            if key in self._feat_to_idx and key in self.numeric_cols:
                feature_array[0, self._feat_to_idx[key]] = float(value)
                continue

            # Categorical features:
            if key in self.categorical_cols:
                if self._is_onehot:
                    self._set_onehot(feature_array, key, value)
                else:
                    self._set_label_encoded(feature_array, key, value)
                continue

            # If user passes a processed feature name directly (advanced users)
            if key in self._feat_to_idx:
                feature_array[0, self._feat_to_idx[key]] = float(value)

        return feature_array

    # -------------------------
    # Public API
    # -------------------------
    def predict_scenario(self, model_name: str, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction for a custom scenario.

        Args:
            model_name: Name of trained model to use
            input_dict: Dict with feature values.
                       - If using one-hot: provide categoricals using original base name, e.g. {"customer_region": "North"}
                       - If using label encoding: same style works.
                       - Numeric features: use original column name, e.g. {"order_value": 120.0}
        """
        try:
            feature_array = self._build_feature_array(input_dict)

            prediction = self.pipeline.get_model_predictions(model_name, feature_array)[0]

            confidence = None
            model = self.pipeline.model_manager.get_model(model_name)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_array)
                confidence = float(np.max(proba))

            pred_out: Any
            if isinstance(prediction, (np.integer, int)):
                pred_out = int(prediction)
            else:
                pred_out = float(prediction)

            return {
                "status": "success",
                "model": model_name,
                "prediction": pred_out,
                "confidence": confidence,
                "input_features": input_dict,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def compare_predictions(self, input_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all trained models for a scenario."""
        results: Dict[str, Dict[str, Any]] = {}
        for model_name in self.pipeline.model_manager.get_models().keys():
            results[model_name] = self.predict_scenario(model_name, input_dict)
        return results

    def get_sensitivity_analysis(
        self,
        model_name: str,
        base_input: Dict[str, Any],
        feature: str,
        values: List[float],
    ) -> pd.DataFrame:
        """Perform sensitivity analysis by varying one numeric feature."""
        results = []
        for value in values:
            scenario = base_input.copy()
            scenario[feature] = value
            pred_result = self.predict_scenario(model_name, scenario)

            if pred_result["status"] == "success":
                results.append(
                    {
                        "feature_value": value,
                        "prediction": pred_result["prediction"],
                        "confidence": pred_result.get("confidence", None),
                    }
                )

        return pd.DataFrame(results)

    def get_feature_ranges(self) -> Dict[str, Dict[str, float]]:
        """Get min, max, mean, std for numeric features from training data."""
        ranges: Dict[str, Dict[str, float]] = {}

        # Only numeric features that exist in feature_names can be ranged
        for feat, st in self._numeric_stats.items():
            ranges[feat] = {
                "min": st.min,
                "max": st.max,
                "mean": st.mean,
                "std": st.std,
            }

        return ranges

    def get_categorical_options(self) -> Dict[str, List[str]]:
        """Get available categorical values.

        If using one-hot: inferred from feature_names.
        If using label encoding: from LabelEncoder.classes_.
        """
        options: Dict[str, List[str]] = {}

        if self._is_onehot:
            for base in self.categorical_cols:
                prefix = f"{base}{self.onehot_sep}"
                vals = []
                for fname in self.feature_names:
                    if fname.startswith(prefix):
                        vals.append(fname[len(prefix) :])
                options[base] = sorted(set(vals))
            return options

        # label-encoded case
        for feature in self.categorical_cols:
            enc = self.label_encoders.get(feature)
            if enc is not None:
                options[feature] = enc.classes_.tolist()
            else:
                options[feature] = []

        return options

    def recommend_scenario(
        self,
        model_name: str,
        target_value: Any,
        base_input: Dict[str, Any],
        variable_features: List[str],
        n_scenarios: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend feature changes to achieve a target prediction.

        Simple heuristic:
        - vary up to 2 numeric features over their observed range
        - return scenarios that match target_value
        """
        recommendations: List[Dict[str, Any]] = []

        feature_ranges = self.get_feature_ranges()

        # Only vary numeric features for now (categorical search explodes quickly)
        candidates = [f for f in variable_features if f in feature_ranges]

        for var_feature in candidates[: min(2, len(candidates))]:
            min_val = feature_ranges[var_feature]["min"]
            max_val = feature_ranges[var_feature]["max"]

            test_values = np.linspace(min_val, max_val, n_scenarios)

            for test_val in test_values:
                scenario = base_input.copy()
                scenario[var_feature] = float(test_val)

                result = self.predict_scenario(model_name, scenario)
                if result["status"] == "success" and result["prediction"] == target_value:
                    recommendations.append(scenario)

                if len(recommendations) >= n_scenarios:
                    return recommendations[:n_scenarios]

        return recommendations[:n_scenarios]
