"""Decision abstraction layer - maps predictions to business actions."""

from typing import Dict, List, Any, Optional
import json


class DecisionMapper:
    """Maps ML predictions to real-world business actions and decisions."""

    def __init__(self):
        """Initialize decision mapper."""
        self.action_mappings: Dict[str, Dict[str, Any]] = {}
        self.decision_context: Dict[str, Any] = {}

    def define_outcome(
        self,
        name: str,
        positive_label: str,
        negative_label: str,
        positive_action: str,
        negative_action: str,
        description: str = "",
    ) -> None:
        """
        Define what positive/negative predictions mean and their corresponding actions.

        Args:
            name: Outcome name (e.g., "customer_churn", "loan_approval")
            positive_label: What positive prediction means (e.g., "Will Churn")
            negative_label: What negative prediction means (e.g., "Will Stay")
            positive_action: Action to take if positive (e.g., "Send retention offer")
            negative_action: Action to take if negative (e.g., "Standard outreach")
            description: Business context description
        """
        self.action_mappings[name] = {
            "positive_label": positive_label,
            "negative_label": negative_label,
            "positive_action": positive_action,
            "negative_action": negative_action,
            "description": description,
        }

    def get_action(
        self, outcome_name: str, prediction: int, confidence: float = None
    ) -> Dict[str, Any]:
        """
        Get the recommended action for a prediction.

        Args:
            outcome_name: Outcome type (must be defined first)
            prediction: Predicted class (0 or 1)
            confidence: Model confidence score (optional)

        Returns:
            Dictionary with action details
        """
        if outcome_name not in self.action_mappings:
            return {
                "error": f"Outcome '{outcome_name}' not defined",
                "action": None,
            }

        mapping = self.action_mappings[outcome_name]

        if prediction == 1:
            label = mapping["positive_label"]
            action = mapping["positive_action"]
        else:
            label = mapping["negative_label"]
            action = mapping["negative_action"]

        result = {
            "prediction": prediction,
            "label": label,
            "action": action,
            "confidence": confidence,
            "description": mapping["description"],
        }

        return result

    def get_decision_matrix(self, outcome_name: str) -> Dict[str, Any]:
        """
        Get the full decision matrix for an outcome.

        Returns decision tree mapping predictions to actions.
        """
        if outcome_name not in self.action_mappings:
            return {"error": f"Outcome '{outcome_name}' not defined"}

        mapping = self.action_mappings[outcome_name]
        return {
            "outcome": outcome_name,
            "description": mapping["description"],
            "positive": {
                "prediction": 1,
                "label": mapping["positive_label"],
                "action": mapping["positive_action"],
            },
            "negative": {
                "prediction": 0,
                "label": mapping["negative_label"],
                "action": mapping["negative_action"],
            },
        }

    def get_all_outcomes(self) -> Dict[str, Dict[str, Any]]:
        """Get all defined outcomes and their action mappings."""
        return self.action_mappings.copy()

    def export_mappings(self, filepath: str) -> str:
        """Export action mappings to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.action_mappings, f, indent=2)
        return filepath

    def import_mappings(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        """Import action mappings from JSON file."""
        with open(filepath, "r") as f:
            self.action_mappings = json.load(f)
        return self.action_mappings
