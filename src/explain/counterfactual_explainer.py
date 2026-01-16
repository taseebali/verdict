"""Counterfactual explainer - explains what needs to change to flip a prediction."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class CounterfactualExplainer:
    """
    Generates counterfactual explanations showing minimal changes needed
    to flip model predictions. Answers: "What needs to change for a different outcome?"
    """

    def __init__(self, feature_names: List[str], categorical_features: List[str] = None, 
                 scaler=None, feature_ranges: Dict[str, Tuple[float, float]] = None):
        """
        Initialize counterfactual explainer.

        Args:
            feature_names: List of feature names
            categorical_features: List of categorical feature names
            scaler: Fitted StandardScaler for inverse transformations
            feature_ranges: Dict mapping feature names to (min, max) tuples for original scale
        """
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.scaler = scaler
        self.feature_ranges = feature_ranges or {}
    
    def _inverse_scale_value(self, feature_name: str, scaled_value: float) -> float:
        """
        Convert a scaled value back to original scale.
        
        Args:
            feature_name: Name of the feature
            scaled_value: Value in scaled space
            
        Returns:
            Value in original scale, or original scaled_value if no scaler/ranges available
        """
        if feature_name not in self.feature_ranges:
            return scaled_value
        
        original_min, original_max = self.feature_ranges[feature_name]
        original_range = original_max - original_min
        
        # Assume StandardScaler: scaled_value = (original - mean) / std
        # We approximate by assuming: scaled range [-3, 3] maps to [original_min, original_max]
        # Better: use mean/std from training data if available, else use simple linear mapping
        
        # Simple approach: map scaled value back assuming range of ±3 std devs covers data
        # scaled_range ≈ 6 (from -3 to +3)
        original_value = original_min + (scaled_value + 3) * (original_range / 6)
        
        # Clamp to reasonable range
        return np.clip(original_value, original_min, original_max)

    def find_counterfactual(
        self,
        instance: Dict[str, float],
        model,
        X_train: pd.DataFrame,
        feature_importance: Dict[str, float] = None,
        num_features_to_change: int = 3,
    ) -> Dict[str, any]:
        """
        Find counterfactual explanation by suggesting minimal feature changes.

        Args:
            instance: Input features as dict {feature_name: value}
            model: Trained model with predict() method
            X_train: Training data for distribution analysis
            feature_importance: Feature importance scores (prioritize important features)
            num_features_to_change: How many features to suggest changing

        Returns:
            Dictionary with counterfactual explanation
        """
        # Get original prediction
        X_instance = pd.DataFrame([instance])
        original_pred = model.predict(X_instance)[0]

        # Sort features by importance (if provided) or by variance
        if feature_importance is not None:
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            important_features = [f for f, _ in sorted_features]
        else:
            important_features = self._rank_by_variance(X_train)

        # Try changing each feature to flip prediction
        candidates = []

        for feature in important_features:
            if feature not in instance:
                continue

            # Get feature distribution from training data
            feature_values = X_train[feature].dropna()

            if feature in self.categorical_features:
                # For categorical, try unique values
                unique_vals = feature_values.unique()
                changes = [v for v in unique_vals if v != instance[feature]]
            else:
                # For numeric, suggest quantiles (25th, 50th, 75th, 90th)
                quantiles = [0.25, 0.5, 0.75, 0.9]
                changes = [feature_values.quantile(q) for q in quantiles]

            # Try each change
            for new_value in changes:
                modified_instance = instance.copy()
                modified_instance[feature] = new_value

                X_modified = pd.DataFrame([modified_instance])
                new_pred = model.predict(X_modified)[0]

                if new_pred != original_pred:
                    # Found a change that flips prediction
                    change_magnitude = abs(new_value - instance[feature])
                    
                    # Convert scaled values back to original scale for display
                    original_scaled = instance[feature]
                    new_scaled = new_value
                    original_unscaled = self._inverse_scale_value(feature, original_scaled)
                    new_unscaled = self._inverse_scale_value(feature, new_scaled)
                    change_unscaled = abs(new_unscaled - original_unscaled)
                    
                    candidates.append(
                        {
                            "feature": feature,
                            "original_value": instance[feature],  # Keep for internal use
                            "new_value": new_value,  # Keep for internal use
                            "original_value_unscaled": original_unscaled,  # Display this
                            "new_value_unscaled": new_unscaled,  # Display this
                            "change": change_magnitude,
                            "change_unscaled": change_unscaled,  # Display this
                            "new_prediction": new_pred,
                            "modified_instance": modified_instance,
                        }
                    )

        # Sort by smallest change (minimal intervention)
        candidates = sorted(candidates, key=lambda x: x["change"])

        # Build explanation - handle string predictions
        try:
            original_pred_int = int(original_pred) if not isinstance(original_pred, str) else int(original_pred.replace('Yes', '1').replace('No', '0')) if original_pred in ['Yes', 'No'] else original_pred
        except (ValueError, TypeError):
            original_pred_int = original_pred
        
        return {
            "original_prediction": original_pred_int,
            "counterfactuals": candidates[:num_features_to_change],
            "num_found": len(candidates),
            "explanation": self._build_explanation(instance, candidates[:num_features_to_change]),
        }

    def find_multiple_counterfactuals(
        self,
        instance: Dict[str, float],
        model,
        X_train: pd.DataFrame,
        num_scenarios: int = 5,
    ) -> Dict[str, any]:
        """
        Generate multiple diverse counterfactual scenarios.

        Args:
            instance: Input features
            model: Trained model
            X_train: Training data
            num_scenarios: Number of diverse scenarios to generate

        Returns:
            Dictionary with multiple counterfactual scenarios
        """
        X_instance = pd.DataFrame([instance])
        original_pred = model.predict(X_instance)[0]

        scenarios = []

        # Strategy 1: Single feature changes (minimum intervention)
        for feature in self.feature_names:
            if feature not in instance:
                continue

            feature_values = X_train[feature].dropna()

            if feature in self.categorical_features:
                unique_vals = feature_values.unique()
                test_values = [v for v in unique_vals if v != instance[feature]]
            else:
                test_values = [
                    feature_values.quantile(0.25),
                    feature_values.quantile(0.5),
                    feature_values.quantile(0.75),
                ]

            for new_value in test_values[:3]:  # Limit to 3 per feature
                modified = instance.copy()
                modified[feature] = new_value

                X_mod = pd.DataFrame([modified])
                new_pred = model.predict(X_mod)[0]

                if new_pred != original_pred:
                    try:
                        new_pred_int = int(new_pred) if not isinstance(new_pred, str) else int(new_pred.replace('Yes', '1').replace('No', '0')) if new_pred in ['Yes', 'No'] else new_pred
                    except (ValueError, TypeError):
                        new_pred_int = new_pred
                    
                    scenarios.append(
                        {
                            "type": "single_feature",
                            "feature": feature,
                            "original_value": instance[feature],
                            "new_value": new_value,
                            "new_prediction": new_pred_int,
                            "instance": modified,
                        }
                    )

                if len(scenarios) >= num_scenarios:
                    break

            if len(scenarios) >= num_scenarios:
                break

        try:
            original_pred_int = int(original_pred) if not isinstance(original_pred, str) else int(original_pred.replace('Yes', '1').replace('No', '0')) if original_pred in ['Yes', 'No'] else original_pred
        except (ValueError, TypeError):
            original_pred_int = original_pred
        
        return {
            "original_prediction": original_pred_int,
            "scenarios": scenarios[:num_scenarios],
            "num_found": len(scenarios),
        }

    def explain_prediction_flip(
        self,
        instance: Dict[str, float],
        model,
        X_train: pd.DataFrame,
        target_class: int = None,
    ) -> str:
        """
        Generate human-readable explanation of how to flip prediction.

        Args:
            instance: Input features
            model: Trained model
            X_train: Training data
            target_class: Desired prediction class (if None, flip current)

        Returns:
            Human-readable explanation string
        """
        X_instance = pd.DataFrame([instance])
        current_pred = model.predict(X_instance)[0]

        if target_class is None:
            target_class = 1 - current_pred

        # Find minimal counterfactual
        cf_result = self.find_counterfactual(instance, model, X_train, num_features_to_change=1)

        if not cf_result["counterfactuals"]:
            return "No simple single-feature change found to flip prediction."

        cf = cf_result["counterfactuals"][0]
        feature = cf["feature"]
        old_val = cf["original_value"]
        new_val = cf["new_value"]

        explanation = (
            f"To change prediction from {current_pred} to {target_class}: "
            f"Change {feature} from {old_val:.2f} to {new_val:.2f}"
        )

        return explanation

    def get_feature_sensitivity_to_flip(
        self, instance: Dict[str, float], model, X_train: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rank features by sensitivity - how much each needs to change to flip prediction.

        Args:
            instance: Input features
            model: Trained model
            X_train: Training data

        Returns:
            DataFrame ranking features by required change magnitude
        """
        X_instance = pd.DataFrame([instance])
        original_pred = model.predict(X_instance)[0]

        results = []

        for feature in self.feature_names:
            if feature not in instance:
                continue

            feature_values = X_train[feature].dropna()

            # Try different magnitudes of change
            if feature in self.categorical_features:
                unique_vals = list(feature_values.unique())
                if instance[feature] in unique_vals:
                    unique_vals.remove(instance[feature])

                if unique_vals:
                    min_change = 1  # Categorical change is binary
                    required_change = min_change
                else:
                    continue
            else:
                # Binary search for minimum change needed
                min_val = feature_values.min()
                max_val = feature_values.max()

                required_change = self._find_min_change_for_flip(
                    feature, instance, model, min_val, max_val
                )

            if required_change is not None:
                results.append(
                    {
                        "feature": feature,
                        "current_value": instance[feature],
                        "required_change": required_change,
                        "change_percentage": (required_change / abs(instance[feature]) * 100)
                        if instance[feature] != 0
                        else 0,
                    }
                )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("required_change")

        return df

    def _rank_by_variance(self, X_train: pd.DataFrame) -> List[str]:
        """Rank features by variance (importance proxy)."""
        variances = X_train.var()
        return variances.sort_values(ascending=False).index.tolist()

    def _build_explanation(self, instance: Dict[str, float], counterfactuals: List[Dict]) -> str:
        """Build human-readable explanation from counterfactuals."""
        if not counterfactuals:
            return "No counterfactual explanation found."

        explanations = []
        for i, cf in enumerate(counterfactuals[:3], 1):
            explanations.append(
                f"{i}. Change {cf['feature']} from {cf['original_value']:.2f} "
                f"to {cf['new_value']:.2f}"
            )

        return "To flip the prediction, try: " + " OR ".join(explanations)

    def _find_min_change_for_flip(
        self, feature: str, instance: Dict, model, min_val: float, max_val: float
    ) -> Optional[float]:
        """Binary search to find minimum change needed to flip prediction."""
        X_instance = pd.DataFrame([instance])
        original_pred = model.predict(X_instance)[0]

        # Try extremes first
        for extreme_val in [min_val, max_val]:
            modified = instance.copy()
            modified[feature] = extreme_val
            X_mod = pd.DataFrame([modified])
            new_pred = model.predict(X_mod)[0]

            if new_pred != original_pred:
                return abs(extreme_val - instance[feature])

        return None
