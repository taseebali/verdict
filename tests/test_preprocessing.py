import numpy as np
import pandas as pd

from src.core.preprocessing import Preprocessor


def _df(n: int = 100) -> pd.DataFrame:
    """Deterministic frame: one numeric feature, one text feature, text target.
    n must be divisible by 4 so both target classes stay balanced."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(loc=10, scale=3, size=n),
        "color": ["red", "blue"] * (n // 2),
        "target": ["Yes", "Yes", "No", "No"] * (n // 4),
    })


def test_text_target_is_label_encoded():
    p = Preprocessor(_df(), "target")
    _, _, y_train, y_test = p.prepare_data()
    assert p.target_encoder is not None
    assert list(p.target_encoder.classes_) == ["No", "Yes"]
    assert set(pd.unique(y_train)) <= {0, 1}
    assert set(pd.unique(y_test)) <= {0, 1}


def test_numeric_target_is_left_alone():
    df = _df()
    df["target"] = [1, 1, 0, 0] * (len(df) // 4)
    p = Preprocessor(df, "target")
    p.prepare_data()
    assert p.target_encoder is None


def test_scaler_is_fit_on_training_rows_only():
    p = Preprocessor(_df(), "target")
    X_train, X_test, _, _ = p.prepare_data()
    assert p.scaler.n_samples_seen_ == len(X_train)
    assert abs(X_train["x"].mean()) < 1e-9


def test_all_unique_text_column_is_dropped_as_identifier():
    df = _df()
    df["customer_id"] = [f"C{i:04d}" for i in range(len(df))]
    p = Preprocessor(df, "target")
    p.prepare_data()
    assert "customer_id" not in p.get_feature_names()
    assert p.dropped_columns == ["customer_id"]


def test_small_frames_keep_unique_text_columns():
    df = _df(n=20)
    df["name"] = [f"n{i}" for i in range(20)]
    p = Preprocessor(df, "target")
    p.prepare_data()
    assert "name" in p.get_feature_names()
    assert p.dropped_columns == []
