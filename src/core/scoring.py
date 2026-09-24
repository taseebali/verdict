"""Binary outcome modeling with honest out-of-fold scores.

One sklearn Pipeline does imputation, scaling and encoding, so there is no
leakage inside cross-validation and raw rows (new files, what-if edits,
unseen categories) can be scored directly.
"""
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from src.core.constants import LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS, RANDOM_SEED

METHODS = ("random_forest", "logistic_regression")
N_FOLDS = 5
MIN_ROWS_FOR_ID = 50
MAX_TARGET_VALUES = 20
OUTCOME_WORDS = ("churn", "target", "label", "outcome", "default", "fraud", "converted")


@dataclass
class Roles:
    numeric: list[str]
    categorical: list[str]
    identifiers: list[str]


def infer_roles(df: pd.DataFrame, target: Optional[str] = None) -> Roles:
    """Split columns into numeric features, categorical features and identifiers.

    An identifier is a text or integer column whose every value is distinct,
    on a frame big enough (>= 50 rows) for that to mean something.
    """
    numeric, categorical, identifiers = [], [], []
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        all_distinct = len(df) >= MIN_ROWS_FOR_ID and series.nunique(dropna=False) == len(df)
        if pd.api.types.is_bool_dtype(series):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series) and all_distinct:
                identifiers.append(col)
            else:
                numeric.append(col)
        elif all_distinct:
            identifiers.append(col)
        else:
            categorical.append(col)
    return Roles(numeric, categorical, identifiers)


def target_suggestions(df: pd.DataFrame) -> list[str]:
    """Columns usable as an outcome (2-20 values), most likely first."""
    ranked = []
    for col in df.columns:
        n = int(df[col].nunique())
        if 2 <= n <= MAX_TARGET_VALUES:
            outcome_like = any(word in str(col).lower() for word in OUTCOME_WORDS)
            ranked.append((0 if outcome_like else 1, 0 if n == 2 else 1, n, str(col)))
    return [name for *_, name in sorted(ranked)]


def prepare_features(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Raw frame -> model input: numerics coerced to float, categoricals to object."""
    X = pd.DataFrame(index=df.index)
    for col in numeric:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in categorical:
        values = df[col].astype(object)
        X[col] = values.where(df[col].notna(), np.nan)
    return X


def _as_str(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def build_pipeline(numeric: list[str], categorical: list[str], method: str) -> Pipeline:
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of: {', '.join(METHODS)}.")
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="(missing)", keep_empty_features=True)),
            ("text", FunctionTransformer(_as_str)),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical))
    clf = (RandomForestClassifier(**RANDOM_FOREST_PARAMS) if method == "random_forest"
           else LogisticRegression(**LOGISTIC_REGRESSION_PARAMS))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", clf)])


@dataclass
class TrainedModel:
    target: str
    positive_class: str
    method: str
    numeric: list[str]
    categorical: list[str]
    identifiers: list[str]
    pipeline: Pipeline
    row_ids: np.ndarray
    oof_proba: np.ndarray
    actual: np.ndarray
    roc_auc: float
    base_rate: float
    roc_points: list[tuple[float, float]]
    rows_skipped: int
    importance: list[tuple[str, float]] = field(default_factory=list)
    drivers: list = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


def _downsample(fpr: np.ndarray, tpr: np.ndarray, n: int = 101) -> list[tuple[float, float]]:
    idx = range(len(fpr)) if len(fpr) <= n else np.unique(np.linspace(0, len(fpr) - 1, n).round().astype(int))
    return [(float(fpr[i]), float(tpr[i])) for i in idx]


def fit_with_oof(df: pd.DataFrame, target: str, positive_class: str,
                 method: str = "random_forest", excluded: Iterable[str] = ()) -> TrainedModel:
    """Score every row with a model that never saw it, then fit a final model on all rows."""
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found.")
    values = df[target].dropna().astype(str)
    n_values = values.nunique()
    if not 2 <= n_values <= MAX_TARGET_VALUES:
        raise ValueError(
            f"'{target}' has {n_values} distinct values. "
            f"Pick a column with 2–{MAX_TARGET_VALUES} categories, like yes/no."
        )
    if positive_class not in set(values):
        raise ValueError(f"'{positive_class}' does not appear in '{target}'.")

    keep = df[target].notna()
    data = df.loc[keep]
    y = (data[target].astype(str) == positive_class).to_numpy()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if min(n_pos, n_neg) < N_FOLDS:
        raise ValueError(
            f"Need at least {N_FOLDS} rows of each outcome; found {n_pos} "
            f"'{positive_class}' and {n_neg} other."
        )

    roles = infer_roles(data, target)
    skip = set(excluded)
    numeric = [c for c in roles.numeric if c not in skip]
    categorical = [c for c in roles.categorical if c not in skip]
    if not numeric and not categorical:
        raise ValueError("No usable feature columns remain. Include at least one column.")

    X = prepare_features(data, numeric, categorical)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof = cross_val_predict(build_pipeline(numeric, categorical, method), X, y,
                            cv=cv, method="predict_proba")[:, 1]
    final = build_pipeline(numeric, categorical, method).fit(X, y)
    fpr, tpr, _ = roc_curve(y, oof)

    return TrainedModel(
        target=target,
        positive_class=positive_class,
        method=method,
        numeric=numeric,
        categorical=categorical,
        identifiers=roles.identifiers,
        pipeline=final,
        row_ids=data.index.to_numpy(),
        oof_proba=oof,
        actual=y,
        roc_auc=float(roc_auc_score(y, oof)),
        base_rate=float(y.mean()),
        roc_points=_downsample(fpr, tpr),
        rows_skipped=int((~keep).sum()),
    )
