"""Data quality analysis - detect leakage, drift, and suspicious patterns."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class DataQualityAnalyzer:
    """Analyzes data quality, detects target leakage, distribution drift, and other suspicious patterns."""

    def __init__(self):
        """Initialize data quality analyzer."""
        self.quality_report = {}

    def detect_target_leakage(
        self, X_train: pd.DataFrame, y_train: np.ndarray, correlation_threshold: float = 0.9
    ) -> Dict:
        """Detect potential target leakage by finding extremely high correlations."""
        leakage_risk = []
        if y_train.dtype == object:
            y_numeric = pd.factorize(y_train)[0]
        else:
            y_numeric = y_train

        for col in X_train.columns:
            try:
                if X_train[col].dtype in ["object", "category"]:
                    continue
                correlation = abs(
                    np.corrcoef(X_train[col].fillna(X_train[col].mean()), y_numeric)[0, 1]
                )
                if correlation > correlation_threshold:
                    leakage_risk.append(
                        {
                            "feature": col,
                            "correlation_with_target": correlation,
                            "risk_level": "CRITICAL",
                        }
                    )
                elif correlation > 0.8:
                    leakage_risk.append(
                        {
                            "feature": col,
                            "correlation_with_target": correlation,
                            "risk_level": "HIGH",
                        }
                    )
            except Exception:
                continue

        return {
            "has_leakage": len(leakage_risk) > 0,
            "suspicious_features": leakage_risk,
            "recommendation": (
                "⚠️ Potential target leakage detected! Review suspicious features before using model."
                if leakage_risk
                else "✅ No obvious target leakage detected."
            ),
        }

    def detect_distribution_drift(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, p_value_threshold: float = 0.05
    ) -> Dict:
        """Detect distribution drift between train and test sets using KS test."""
        drift_detected = []
        for col in X_train.columns:
            if X_train[col].dtype in ["object", "category"]:
                continue
            try:
                train_col = X_train[col].fillna(X_train[col].mean())
                test_col = X_test[col].fillna(X_test[col].mean())
                ks_stat, p_value = ks_2samp(train_col, test_col)
                if p_value < p_value_threshold:
                    drift_detected.append(
                        {
                            "feature": col,
                            "ks_statistic": float(ks_stat),
                            "p_value": float(p_value),
                            "train_mean": float(train_col.mean()),
                            "test_mean": float(test_col.mean()),
                            "train_std": float(train_col.std()),
                            "test_std": float(test_col.std()),
                        }
                    )
            except Exception:
                continue

        return {
            "has_drift": len(drift_detected) > 0,
            "drifted_features": drift_detected,
            "recommendation": (
                "⚠️ Distribution drift detected! Model performance may degrade on new data."
                if drift_detected
                else "✅ No significant distribution drift detected."
            ),
        }

    def detect_class_imbalance(self, y: np.ndarray) -> Dict:
        """Detect severe class imbalance in target variable."""
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count
        severity = "Low" if imbalance_ratio < 2 else "Medium" if imbalance_ratio < 5 else "High"

        return {
            "class_distribution": class_dist,
            "imbalance_ratio": float(imbalance_ratio),
            "severity": severity,
            "recommendation": (
                "⚠️ Severe class imbalance detected! Consider resampling or cost-weighting."
                if severity == "High"
                else "✅ Acceptable class balance."
            ),
        }

    def detect_missing_values(self, X: pd.DataFrame) -> Dict:
        """Analyze missing values in dataset."""
        missing_info = []
        total_cells = X.shape[0] * X.shape[1]

        for col in X.columns:
            missing_count = X[col].isna().sum()
            missing_pct = (missing_count / X.shape[0]) * 100
            if missing_count > 0:
                missing_info.append(
                    {
                        "feature": col,
                        "missing_count": int(missing_count),
                        "missing_percentage": float(missing_pct),
                        "severity": "High" if missing_pct > 50 else "Medium" if missing_pct > 10 else "Low",
                    }
                )

        total_missing_pct = (sum(m["missing_count"] for m in missing_info) / total_cells) * 100

        return {
            "features_with_missing": missing_info,
            "total_missing_percentage": float(total_missing_pct),
            "recommendation": (
                "⚠️ Significant missing values detected! Ensure preprocessing handles them correctly."
                if total_missing_pct > 5
                else "✅ Missing values are minimal."
            ),
        }

    def detect_outliers(self, X_train: pd.DataFrame, outlier_threshold: float = 3.0) -> Dict:
        """Detect outliers using z-score method."""
        outlier_features = []

        for col in X_train.columns:
            if X_train[col].dtype in ["object", "category"]:
                continue
            try:
                data = X_train[col].fillna(X_train[col].mean())
                z_scores = np.abs((data - data.mean()) / data.std())
                outlier_count = np.sum(z_scores > outlier_threshold)

                if outlier_count > 0:
                    outlier_pct = (outlier_count / len(data)) * 100
                    outlier_features.append(
                        {
                            "feature": col,
                            "outlier_count": int(outlier_count),
                            "outlier_percentage": float(outlier_pct),
                        }
                    )
            except Exception:
                continue

        return {
            "features_with_outliers": outlier_features,
            "total_features_with_outliers": len(outlier_features),
            "recommendation": (
                "✓ Check outliers, but they may be legitimate domain values."
                if outlier_features
                else "✅ No significant outliers detected."
            ),
        }

    def generate_quality_report(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray = None,
    ) -> Dict:
        """Generate comprehensive data quality report."""
        report = {
            "target_leakage": self.detect_target_leakage(X_train, y_train),
            "distribution_drift": self.detect_distribution_drift(X_train, X_test),
            "class_imbalance": self.detect_class_imbalance(y_train),
            "missing_values_train": self.detect_missing_values(X_train),
            "missing_values_test": self.detect_missing_values(X_test),
            "outliers": self.detect_outliers(X_train),
        }

        if y_test is not None:
            report["test_class_imbalance"] = self.detect_class_imbalance(y_test)

        issues_count = (
            len(report["target_leakage"]["suspicious_features"])
            + len(report["distribution_drift"]["drifted_features"])
            + len(report["outliers"]["features_with_outliers"])
        )

        quality_score = max(0, 100 - issues_count * 10)
        report["overall_quality_score"] = float(quality_score)
        report["warnings"] = [
            r["recommendation"] for r in report.values() if isinstance(r, dict) and "recommendation" in r
        ]

        return report

    def get_quality_summary(self, quality_report: Dict) -> str:
        """Get human-readable summary of quality report."""
        issues = []

        if quality_report["target_leakage"]["has_leakage"]:
            issues.append("⚠️ Target leakage detected")

        if quality_report["distribution_drift"]["has_drift"]:
            issues.append("⚠️ Distribution drift detected")

        if quality_report["class_imbalance"]["severity"] == "High":
            issues.append("⚠️ Severe class imbalance")

        if quality_report["missing_values_train"]["total_missing_percentage"] > 5:
            issues.append("⚠️ Significant missing values")

        if not issues:
            return f"✅ Data quality looks good (Score: {quality_report['overall_quality_score']:.0f}/100)"

        return (
            f"⚠️ Data quality issues found (Score: {quality_report['overall_quality_score']:.0f}/100):\n"
            + "\n".join(issues)
        )
