"""Cost-aware model evaluation and optimal model selection."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class CostAnalyzer:
    """
    Analyzes classification models based on business costs.
    Incorporates false positive and false negative costs for realistic evaluation.
    """

    def __init__(self, fp_cost: float = 100.0, fn_cost: float = 500.0):
        """
        Initialize cost analyzer.

        Args:
            fp_cost: Cost of false positive (default: $100)
            fn_cost: Cost of false negative (default: $500)
        """
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    def calculate_expected_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fp_cost: float = None,
        fn_cost: float = None,
    ) -> Dict[str, float]:
        """
        Calculate expected cost for a model's predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            fp_cost: False positive cost (overrides default)
            fn_cost: False negative cost (overrides default)

        Returns:
            Dictionary with cost breakdown
        """
        fp_cost = fp_cost or self.fp_cost
        fn_cost = fn_cost or self.fn_cost

        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        total_fp_cost = fp * fp_cost
        total_fn_cost = fn * fn_cost
        total_cost = total_fp_cost + total_fn_cost

        num_tests = len(y_true)
        average_cost_per_decision = total_cost / num_tests if num_tests > 0 else 0

        return {
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "fp_cost": float(fp_cost),
            "fn_cost": float(fn_cost),
            "total_fp_cost": float(total_fp_cost),
            "total_fn_cost": float(total_fn_cost),
            "total_cost": float(total_cost),
            "average_cost_per_decision": float(average_cost_per_decision),
        }

    def compare_model_costs(
        self,
        model_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        fp_cost: float = None,
        fn_cost: float = None,
    ) -> pd.DataFrame:
        """
        Compare expected costs across multiple models.

        Args:
            model_predictions: Dict of {model_name: predictions}
            y_true: True labels
            fp_cost: False positive cost (overrides default)
            fn_cost: False negative cost (overrides default)

        Returns:
            DataFrame comparing costs across models
        """
        fp_cost = fp_cost or self.fp_cost
        fn_cost = fn_cost or self.fn_cost

        results = []
        for model_name, y_pred in model_predictions.items():
            cost_dict = self.calculate_expected_cost(y_true, y_pred, fp_cost, fn_cost)
            cost_dict["model"] = model_name
            results.append(cost_dict)

        df = pd.DataFrame(results)
        # Sort by total cost (ascending = best first)
        df = df.sort_values("total_cost").reset_index(drop=True)

        return df

    def find_optimal_model(
        self,
        model_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        fp_cost: float = None,
        fn_cost: float = None,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Find the cost-optimal model.

        Args:
            model_predictions: Dict of {model_name: predictions}
            y_true: True labels
            fp_cost: False positive cost (overrides default)
            fn_cost: False negative cost (overrides default)

        Returns:
            Tuple of (optimal_model_name, cost_details)
        """
        cost_comparison = self.compare_model_costs(model_predictions, y_true, fp_cost, fn_cost)

        if len(cost_comparison) == 0:
            return None, {}

        # Best model is first (lowest total cost)
        best_row = cost_comparison.iloc[0]
        optimal_model = best_row["model"]
        cost_details = best_row.to_dict()

        return optimal_model, cost_details

    def threshold_vs_cost(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        fp_cost: float = None,
        fn_cost: float = None,
        step: float = 0.01,
    ) -> pd.DataFrame:
        """
        Analyze how total cost changes across different decision thresholds.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            fp_cost: False positive cost (overrides default)
            fn_cost: False negative cost (overrides default)
            step: Threshold step size

        Returns:
            DataFrame with costs at each threshold
        """
        fp_cost = fp_cost or self.fp_cost
        fn_cost = fn_cost or self.fn_cost

        thresholds = np.arange(0, 1 + step, step)
        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost_dict = self.calculate_expected_cost(y_true, y_pred, fp_cost, fn_cost)
            cost_dict["threshold"] = round(threshold, 2)
            results.append(cost_dict)

        df = pd.DataFrame(results)
        return df

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        fp_cost: float = None,
        fn_cost: float = None,
        step: float = 0.01,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find the threshold that minimizes expected cost.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            fp_cost: False positive cost (overrides default)
            fn_cost: False negative cost (overrides default)
            step: Threshold step size

        Returns:
            Tuple of (optimal_threshold, cost_details)
        """
        cost_df = self.threshold_vs_cost(y_true, y_proba, fp_cost, fn_cost, step)

        # Find threshold with minimum total cost
        best_idx = cost_df["total_cost"].idxmin()
        best_row = cost_df.iloc[best_idx]

        optimal_threshold = best_row["threshold"]
        cost_details = best_row.to_dict()

        return float(optimal_threshold), cost_details

    def get_cost_sensitivity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fp_cost_range: Tuple[float, float] = (10, 1000),
        fn_cost_range: Tuple[float, float] = (10, 1000),
        steps: int = 5,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of model cost to FP and FN costs.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            fp_cost_range: Tuple of (min, max) false positive costs
            fn_cost_range: Tuple of (min, max) false negative costs
            steps: Number of steps for sensitivity analysis

        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []

        fp_costs = np.linspace(fp_cost_range[0], fp_cost_range[1], steps)
        fn_costs = np.linspace(fn_cost_range[0], fn_cost_range[1], steps)

        for fp_cost in fp_costs:
            for fn_cost in fn_costs:
                cost_dict = self.calculate_expected_cost(y_true, y_pred, fp_cost, fn_cost)
                cost_dict["fp_cost"] = float(fp_cost)
                cost_dict["fn_cost"] = float(fn_cost)
                results.append(cost_dict)

        return pd.DataFrame(results)

    def cost_benefit_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        benefit_per_correct_positive: float = 1000.0,
    ) -> Dict[str, float]:
        """
        Calculate net benefit (benefits - costs) of a model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            benefit_per_correct_positive: Monetary benefit for each true positive

        Returns:
            Dictionary with benefit analysis
        """
        cost_dict = self.calculate_expected_cost(y_true, y_pred)

        tp = cost_dict["true_positives"]
        total_benefit = tp * benefit_per_correct_positive
        total_cost = cost_dict["total_cost"]
        net_benefit = total_benefit - total_cost

        return {
            "total_benefit": float(total_benefit),
            "total_cost": float(total_cost),
            "net_benefit": float(net_benefit),
            "roi": float(net_benefit / total_cost * 100) if total_cost > 0 else 0,
        }
