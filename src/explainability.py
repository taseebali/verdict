"""Explainability module using SHAP and permutation importance."""

from __future__ import annotations

import warnings
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class ExplainabilityAnalyzer:
    """Analyzes model predictions and feature importance."""

    def __init__(self, model, X_train, X_test, feature_names: List[str]):
        """
        Args:
            model: Trained model
            X_train: Training features (numpy / pandas / sparse)
            X_test: Test features (numpy / pandas / sparse)
            feature_names: Feature names aligned with model input space
        """
        self.model = model
        self.feature_names = feature_names

        self.X_train = self._to_dense_2d(X_train)
        self.X_test = self._to_dense_2d(X_test)

        self.explainer: Optional[Any] = None
        self.use_fallback: bool = False

        # lazily computed cache (optional)
        self._global_shap_cached: bool = False
        self._global_shap_values: Optional[Any] = None  # can be array/list/Explanation

        self._init_explainer()

    # -------------------------
    # Utilities
    # -------------------------
    def _to_dense_2d(self, X) -> np.ndarray:
        """Convert DataFrame / numpy / sparse matrix to a dense 2D numpy array."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif hasattr(X, "toarray"):  # scipy sparse
            X = X.toarray()
        else:
            X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _is_tree_model(self) -> bool:
        return hasattr(self.model, "estimators_") or hasattr(self.model, "tree_")

    def _is_linear_model(self) -> bool:
        return hasattr(self.model, "coef_")

    def _background_sample(self, X: np.ndarray, n: int = 100) -> np.ndarray:
        """Sample background rows for Kernel/Linear explainers."""
        if X.shape[0] <= n:
            return X
        idx = np.random.choice(X.shape[0], n, replace=False)
        return X[idx]

    def _predict_fn(self):
        """
        For classification: explain P(class=1) if possible.
        For regression: explain predicted value.
        """
        if hasattr(self.model, "predict_proba"):
            return lambda X: self.model.predict_proba(X)[:, 1]
        return lambda X: self.model.predict(X)

    def _init_explainer(self) -> None:
        """Initialize best available SHAP explainer for this model."""
        try:
            if self._is_tree_model():
                # Fast + stable for tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                return

            if self._is_linear_model():
                # Good for linear/logistic
                bg = self._background_sample(self.X_train, n=200)
                # For some SHAP versions, feature_names parameter here may not exist; keep safe
                try:
                    self.explainer = shap.LinearExplainer(self.model, bg, feature_names=self.feature_names)
                except TypeError:
                    self.explainer = shap.LinearExplainer(self.model, bg)
                return

            # Fallback: KernelExplainer (slow)
            bg = self._background_sample(self.X_train, n=50)
            self.explainer = shap.KernelExplainer(self._predict_fn(), bg)

        except Exception as e:
            print(
                f"Warning: Could not create SHAP explainer ({type(e).__name__}: {e}). "
                "Using permutation importance fallback."
            )
            self.explainer = None
            self.use_fallback = True

    # -------------------------
    # Permutation importance
    # -------------------------
    def get_feature_importance_permutation(
        self, y_true: np.ndarray, scoring: str = "accuracy"
    ) -> Dict[str, float]:
        """Calculate permutation importance."""
        try:
            result = permutation_importance(
                self.model,
                self.X_test,
                y_true,
                n_repeats=10,
                random_state=42,
                scoring=scoring,
                n_jobs=-1,
            )
            importance_dict = dict(zip(self.feature_names, result.importances_mean))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"Error calculating permutation importance: {e}")
            return {}

    # -------------------------
    # SHAP helpers
    # -------------------------
    def _compute_shap_for_row(self, x_row: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Returns:
          shap_values_vector: shape (n_features,)
          base_value: float or None
        """
        if self.explainer is None:
            return None, None

        # Try modern SHAP API: explainer(X) -> shap.Explanation
        try:
            exp = self.explainer(x_row)
            if hasattr(exp, "values"):
                values = exp.values
                base = getattr(exp, "base_values", None)

                # values can be (1, n_features) or (1, n_features, classes)
                values = np.asarray(values)

                # base can be scalar or array-like
                base_val = None
                if base is not None:
                    base_arr = np.asarray(base)
                    # could be shape (1,) or (1, classes)
                    base_val = float(base_arr.flatten()[0])

                # reduce to 1D
                if values.ndim == 3:
                    # (1, n_features, classes) -> take class 1 if exists else class 0
                    cls_idx = 1 if values.shape[2] > 1 else 0
                    v = values[0, :, cls_idx]
                else:
                    v = values.reshape(-1)

                return v, base_val
        except Exception:
            pass

        # Compatibility / older explainers
        try:
            sv = self.explainer.shap_values(x_row)

            # sv can be:
            # - array (1, n_features)
            # - list of arrays for multiclass: [ (1,n_features), ...]
            if isinstance(sv, list):
                # choose class 1 if exists; else class 0
                cls_idx = 1 if len(sv) > 1 else 0
                v = np.asarray(sv[cls_idx]).reshape(-1)
            else:
                v = np.asarray(sv).reshape(-1)

            base_val = None
            ev = getattr(self.explainer, "expected_value", None)
            if ev is not None:
                ev_arr = np.asarray(ev)
                base_val = float(ev_arr.flatten()[0])

            return v, base_val
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return None, None

    def get_shap_values(self) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Optional: compute global SHAP values for the full test set (can be slow).
        Returns cached values + explainer.
        """
        if self.explainer is None:
            return None, None
        try:
            # Try calling explainer(X) first
            try:
                exp = self.explainer(self.X_test)
                self._global_shap_values = exp
                self._global_shap_cached = True
                return exp, self.explainer
            except Exception:
                pass

            # Fallback to shap_values API
            sv = self.explainer.shap_values(self.X_test)
            self._global_shap_values = sv
            self._global_shap_cached = True
            return sv, self.explainer
        except Exception as e:
            print(f"Error computing global SHAP values: {e}")
            return None, None

    # -------------------------
    # Prediction + explanation API
    # -------------------------
    def predict_with_confidence(self, X) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get predictions with confidence estimates."""
        Xd = self._to_dense_2d(X)
        predictions = self.model.predict(Xd)

        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xd)
            confidence = np.max(proba, axis=1)

        return predictions, confidence

    def explain_prediction(self, sample_idx: int) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP (preferred).
        If SHAP isn't available, returns a fallback explanation (permutation importance not per-row).
        """
        x_row = self.X_test[sample_idx : sample_idx + 1]
        pred = self.model.predict(x_row)[0]

        # SHAP path
        if self.explainer is not None and not self.use_fallback:
            shap_vec, base_val = self._compute_shap_for_row(x_row)

            if shap_vec is None:
                return {"error": "Could not compute SHAP values for this sample", "prediction": pred}

            # map contributions
            contrib = dict(zip(self.feature_names, shap_vec))
            sorted_contrib = dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))

            return {
                "prediction": pred,
                "feature_contributions": sorted_contrib,
                "base_value": base_val,
                "sample_features": dict(zip(self.feature_names, x_row.flatten())),
                "method": "shap",
            }

        # Fallback (no SHAP): provide zeros + sample values (keeps UI alive)
        return {
            "prediction": pred,
            "feature_contributions": {k: 0.0 for k in self.feature_names},
            "base_value": None,
            "sample_features": dict(zip(self.feature_names, x_row.flatten())),
            "method": "fallback_no_shap",
        }

    # -------------------------
    # Plotting helpers
    # -------------------------
    def plot_shap_summary(self) -> Optional[plt.Figure]:
        """Create SHAP summary plot (global)."""
        if self.explainer is None:
            return None

        # Ensure global shap exists (compute if needed)
        if not self._global_shap_cached:
            self.get_shap_values()

        if self._global_shap_values is None:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # If we cached a shap.Explanation, use its .values
            if hasattr(self._global_shap_values, "values"):
                vals = self._global_shap_values.values
                vals = np.asarray(vals)
                # reduce multiclass if needed
                if vals.ndim == 3:
                    cls_idx = 1 if vals.shape[2] > 1 else 0
                    vals = vals[:, :, cls_idx]
                shap.summary_plot(vals, self.X_test, feature_names=self.feature_names, show=False)
            else:
                # list/array style
                sv = self._global_shap_values
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                shap.summary_plot(sv, self.X_test, feature_names=self.feature_names, show=False)

            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            return None

    def plot_shap_force(self, sample_idx: int):
        """Create SHAP force plot object for a single prediction (works best in notebooks)."""
        if self.explainer is None:
            return None

        x_row = self.X_test[sample_idx : sample_idx + 1]
        shap_vec, base_val = self._compute_shap_for_row(x_row)
        if shap_vec is None:
            return None

        try:
            # force_plot returns an object that can be rendered in some environments
            return shap.force_plot(
                base_val if base_val is not None else 0.0,
                shap_vec,
                x_row.flatten(),
                feature_names=self.feature_names,
                show=False,
            )
        except Exception as e:
            print(f"Error creating SHAP force plot: {e}")
            return None
