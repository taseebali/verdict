"""Counterfactual explainer - explains what needs to change to flip a prediction."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class CounterfactualExplainer:
    """
    Generates counterfactual explanations showing minimal changes needed
    to flip model predictions. Answers: "What needs to change for a different outcome?"
    """

    def __init__(self, feature_names: List[str], categorical_features: List[str] = None):
        """
        Initialize counterfactual explainer.

        Args:
            feature_names: List of feature names
            categorical_features: List of categorical feature names
        """
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []

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
                    candidates.append(
                        {
                            "feature": feature,
                            "original_value": instance[feature],
                            "new_value": new_value,
                            "change": change_magnitude,
                            "new_prediction": new_pred,
                            "modified_instance": modified_instance,
                        }
                    )

        # Sort by smallest change (minimal intervention)
        candidates = sorted(candidates, key=lambda x: x["change"])

        # Build explanation
        return {
            "original_prediction": int(original_pred),
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
                    scenarios.append(
                        {
                            "type": "single_feature",
                            "feature": feature,
                            "original_value": instance[feature],
                            "new_value": new_value,
                            "new_prediction": int(new_pred),
                            "instance": modified,
                        }
                    )

                if len(scenarios) >= num_scenarios:
                    break

            if len(scenarios) >= num_scenarios:
                break

        return {
            "original_prediction": int(original_pred),
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
