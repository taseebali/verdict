"""Streamlit UI Utilities - Simplified"""

import streamlit as st
import pandas as pd
from typing import Dict

from src.core.formatters import format_value
from src.core.validators import DataValidator


@st.cache_data
def load_demo_dataset() -> pd.DataFrame:
    """Load demo dataset with caching."""
    try:
        return pd.read_csv("data/verdict_demo.csv")
    except FileNotFoundError:
        st.error("Demo dataset not found at data/verdict_demo.csv")
        st.stop()


def is_binary_feature(df: pd.DataFrame, feature_name: str) -> bool:
    """Check if feature is binary (0/1 or True/False)."""
    return df[feature_name].nunique() <= 2


def get_feature_statistics(df: pd.DataFrame, features: list) -> Dict:
    """Get min/max/mean/median for numeric features only."""
    stats = {}
    for feature in features:
        if feature in df.columns:
            # Only get statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[feature]):
                stats[feature] = {
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'mean': df[feature].mean(),
                    'median': df[feature].median()
                }
    return stats


def render_binary_input(feature_name: str, default: int = 0, key: str = None) -> int:
    """Render radio button for binary feature."""
    return st.radio(
        feature_name,
        options=[0, 1],
        format_func=lambda x: ['0 - No', '1 - Yes'][x],
        horizontal=True,
        key=key or f"binary_{feature_name}"
    )


def render_numeric_slider(feature_name: str, min_val: float, max_val: float, default_val: float = None, key: str = None) -> float:
    """Render slider with formatted caption."""
    value = st.slider(
        feature_name,
        float(min_val),
        float(max_val),
        float(default_val or (min_val + max_val) / 2),
        key=key or f"slider_{feature_name}"
    )
    st.caption(f"{feature_name}: {format_value(feature_name, value)}")
    return value


def validate_dataset(df: pd.DataFrame, target_col: str = None) -> Dict:
    """Validate dataset using centralized validator.
    
    Wrapper around DataValidator for backward compatibility.
    """
    return DataValidator.validate_quality(df, target_col)


def get_error_suggestion(error_msg: str) -> str:
    """Get helpful suggestion based on error type."""
    suggestions = {
        'stratify': "Increase test_size % or ensure 2+ samples per class",
        'feature': "Ensure all selected features have numeric values",
        'memory': "Use fewer features or reduce dataset size",
        'type': "Check data types - all features should be numeric",
        'nan': "Remove rows with NaN or infinite values",
        'inf': "Remove rows with NaN or infinite values",
        'dimension': "Ensure input features match training features"
    }
    
    for key, suggestion in suggestions.items():
        if key in error_msg.lower():
            return suggestion
    return "Check your data and settings"


def export_predictions_to_csv(predictions_df: pd.DataFrame, filename: str = "predictions.csv") -> bytes:
    """Convert predictions to CSV bytes."""
    return predictions_df.to_csv(index=False).encode('utf-8')


def export_audit_trail_to_csv(audit_trail: list) -> bytes:
    """Convert audit trail to CSV bytes."""
    return pd.DataFrame(audit_trail).to_csv(index=False).encode('utf-8')
