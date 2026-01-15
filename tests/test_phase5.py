"""
Phase 5 Integration Tests - Decision Intelligence & Governance Features
Demonstrates all Phase 5.1 and Phase 5.2 features working together.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.decision.decision_mapper import DecisionMapper
from src.decision.threshold_analyzer import ThresholdAnalyzer
from src.decision.cost_analyzer import CostAnalyzer
from src.explain.counterfactual_explainer import CounterfactualExplainer
from src.decision.confidence_estimator import ConfidenceEstimator
from src.decision.data_quality_analyzer import DataQualityAnalyzer
from src.decision.decision_audit_logger import DecisionAuditLogger


def test_decision_mapper():
    """Test Decision Abstraction Layer."""
    print("\n" + "=" * 60)
    print("TEST 1: Decision Mapper - Predictions to Business Actions")
    print("=" * 60)

    mapper = DecisionMapper()

    mapper.define_outcome(
        name="customer_churn",
        positive_label="Customer will churn",
        negative_label="Customer will stay",
        positive_action="Send $50 retention offer + call",
        negative_action="Standard marketing outreach",
        description="Customer churn prediction for retention",
    )

    action1 = mapper.get_action("customer_churn", prediction=1, confidence=0.85)
    action2 = mapper.get_action("customer_churn", prediction=0, confidence=0.92)

    # ✅ Assertions (real tests)
    assert action1["label"] == "Customer will churn"
    assert "retention offer" in action1["action"].lower()
    assert action1["confidence"] == pytest.approx(0.85)

    assert action2["label"] == "Customer will stay"
    assert "outreach" in action2["action"].lower()
    assert action2["confidence"] == pytest.approx(0.92)


def test_threshold_control():
    """Test Threshold Control & Tradeoff Analysis."""
    print("\n" + "=" * 60)
    print("TEST 2: Threshold Control - Precision-Recall Tradeoffs")
    print("=" * 60)

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    analyzer = ThresholdAnalyzer()
    thresholds_df = analyzer.analyze_thresholds(y_test, y_proba)

    # ✅ Assertions
    assert thresholds_df is not None
    assert len(thresholds_df) > 0
    assert {"threshold", "precision", "recall", "f1", "accuracy"}.issubset(thresholds_df.columns)

    optimal_threshold = analyzer.get_optimal_threshold(y_test, y_proba, metric="f1")
    assert 0.0 <= float(optimal_threshold) <= 1.0


def test_cost_aware_selection():
    """Test Cost-Aware Model Selection."""
    print("\n" + "=" * 60)
    print("TEST 3: Cost-Aware Model Selection")
    print("=" * 60)

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    cost_analyzer = CostAnalyzer(fp_cost=100, fn_cost=500)

    model_predictions = {
        "logistic_regression": y_pred_lr,
        "random_forest": y_pred_rf,
    }

    cost_df = cost_analyzer.compare_model_costs(model_predictions, y_test)

    # ✅ Assertions
    assert cost_df is not None
    assert len(cost_df) == 2
    assert {"model", "false_positives", "false_negatives", "total_cost"}.issubset(cost_df.columns)

    optimal_model, cost_details = cost_analyzer.find_optimal_model(model_predictions, y_test)
    assert optimal_model in model_predictions
    assert "average_cost_per_decision" in cost_details
    assert cost_details["average_cost_per_decision"] >= 0


def test_counterfactual_explanation():
    """Test Counterfactual Explanations."""
    print("\n" + "=" * 60)
    print("TEST 4: Counterfactual Explanations - What Needs to Change?")
    print("=" * 60)

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    X_train, X_test, y_train, _ = train_test_split(X_df, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    test_sample = X_test.iloc[0].to_dict()
    explainer = CounterfactualExplainer(list(X_train.columns))
    cf_result = explainer.find_counterfactual(test_sample, model, X_train, num_features_to_change=2)

    # ✅ Assertions
    assert isinstance(cf_result, dict)
    assert "explanation" in cf_result
    assert "counterfactuals" in cf_result
    assert isinstance(cf_result["counterfactuals"], list)


def test_confidence_estimation():
    """Test Confidence & Uncertainty Estimation."""
    print("\n" + "=" * 60)
    print("TEST 5: Confidence Estimation - Probability vs Confidence")
    print("=" * 60)

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)

    estimator = ConfidenceEstimator()

    confidence = estimator.estimate_probability_confidence(y_proba)
    margin_conf = estimator.estimate_margin_confidence(y_proba)
    uncertainty = estimator.estimate_uncertainty(y_proba)
    reliability_df = estimator.get_reliability_indicators(y_proba)

    # ✅ Assertions
    assert len(confidence) == len(y_proba)
    assert len(margin_conf) == len(y_proba)
    assert len(uncertainty) == len(y_proba)
    assert not reliability_df.empty
    assert "confidence_level" in reliability_df.columns
    assert "reliability_score" in reliability_df.columns


def test_data_quality():
    """Test Data Quality Analysis."""
    print("\n" + "=" * 60)
    print("TEST 6: Data Quality Analysis - Leakage & Drift Detection")
    print("=" * 60)

    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

    # Add leaky feature
    X_df["leaky_feature"] = y + np.random.normal(0, 0.1, len(y))

    X_train, X_test, y_train, _ = train_test_split(X_df, y, test_size=0.3, random_state=42)

    analyzer = DataQualityAnalyzer()

    leakage_report = analyzer.detect_target_leakage(X_train, y_train)
    drift_report = analyzer.detect_distribution_drift(X_train, X_test)
    imbalance_report = analyzer.detect_class_imbalance(y_train)

    # ✅ Assertions
    assert isinstance(leakage_report, dict)
    assert "has_leakage" in leakage_report
    assert isinstance(drift_report, dict)
    assert "has_drift" in drift_report
    assert isinstance(imbalance_report, dict)
    assert "imbalance_ratio" in imbalance_report
    assert "severity" in imbalance_report


def test_audit_logging(tmp_path):
    """Test Decision Audit Logging."""
    print("\n" + "=" * 60)
    print("TEST 7: Decision Audit Logging - Compliance & Traceability")
    print("=" * 60)

    logger = DecisionAuditLogger()

    logger.log_prediction(
        prediction=1,
        probability=0.85,
        confidence=0.82,
        model_name="random_forest",
        threshold=0.5,
        recommended_action="Send retention offer",
    )

    logger.log_prediction(
        prediction=0,
        probability=0.32,
        confidence=0.68,
        model_name="random_forest",
        threshold=0.5,
        recommended_action="Standard outreach",
    )

    logger.log_threshold_change(
        model_name="random_forest",
        old_threshold=0.5,
        new_threshold=0.6,
        reason="Balance precision and recall",
        changed_by="admin",
    )

    logger.log_action_taken(
        prediction_id=0,
        action="Sent $50 retention offer",
        action_result="Customer engaged",
        actioned_by="retention_team",
    )

    stats = logger.get_statistics()

    # ✅ Assertions
    assert stats["total_predictions"] == 2
    assert stats["total_records"] >= 3
    assert 0.0 <= stats["avg_confidence"] <= 1.0

    # If your logger allows specifying output dir, do that.
    # Otherwise, just ensure it returns a path-like string.
    log_file = logger.save_audit_log()
    csv_file = logger.export_as_csv()

    assert isinstance(log_file, str) and len(log_file) > 0
    assert isinstance(csv_file, str) and len(csv_file) > 0


# Optional: keep a manual runner, but do NOT let pytest collect it as a test.
def run_all_tests():
    """Manual runner for humans (not used by pytest)."""
    tests = [
        test_decision_mapper,
        test_threshold_control,
        test_cost_aware_selection,
        test_counterfactual_explanation,
        test_confidence_estimation,
        test_data_quality,
        test_audit_logging,
    ]
    for t in tests:
        t()
    return True


if __name__ == "__main__":
    success = run_all_tests()
    raise SystemExit(0 if success else 1)
