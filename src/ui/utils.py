"""Shared Streamlit UI Utilities - Consolidated across pages"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any


@st.cache_data
def load_demo_dataset() -> pd.DataFrame:
    """Load demo dataset with caching."""
    try:
        return pd.read_csv("data/verdict_demo.csv")
    except FileNotFoundError:
        st.error("Demo dataset not found at data/verdict_demo.csv")
        st.stop()


def format_value(feature_name: str, value: float) -> str:
    """Format raw value to human-readable format based on feature name."""
    feature_lower = feature_name.lower()
    
    if 'month' in feature_lower or 'tenure' in feature_lower:
        return f"{int(value)} months"
    if 'age' in feature_lower:
        return f"{int(value)} years"
    if 'charge' in feature_lower or 'price' in feature_lower or 'cost' in feature_lower:
        return f"${value:.2f}"
    if 'rate' in feature_lower or 'percent' in feature_lower or '%' in feature_lower:
        return f"{value:.1f}%"
    if 'hours' in feature_lower or 'usage' in feature_lower:
        return f"{value:.1f} hrs"
    if 'days' in feature_lower:
        return f"{int(value)} days"
    
    return f"{value:.2f}"


def is_binary_feature(df: pd.DataFrame, feature_name: str) -> bool:
    """Check if feature is binary (yes/no, 0/1)."""
    if feature_name not in df.columns:
        return False
    
    unique_vals = df[feature_name].unique()
    return len(unique_vals) <= 2


def get_feature_statistics(df: pd.DataFrame, features: list) -> Dict[str, Dict[str, float]]:
    """Get min/max/mean/median statistics for multiple features."""
    stats = {}
    for feature in features:
        if feature in df.columns:
            stats[feature] = {
                'min': df[feature].min(),
                'max': df[feature].max(),
                'mean': df[feature].mean(),
                'median': df[feature].median()
            }
    return stats


def render_metric_card(label: str, value: str, delta: str = None) -> None:
    """Render a formatted metric card."""
    st.metric(label, value, delta)


def render_error_message(title: str, description: str, suggestion: str = None) -> None:
    """Render user-friendly error message."""
    st.error(f"❌ {title}")
    st.caption(description)
    if suggestion:
        st.info(f"💡 Try this: {suggestion}")


def render_success_message(title: str, details: str = None) -> None:
    """Render success message."""
    st.success(f"✅ {title}")
    if details:
        st.caption(details)


def render_binary_input(feature_name: str, default: int = 0, key: str = None) -> int:
    """Render radio button for binary feature."""
    options = ['0 - No', '1 - Yes']
    selected = st.radio(
        feature_name,
        options=[0, 1],
        format_func=lambda x: options[x],
        horizontal=True,
        key=key or f"binary_{feature_name}"
    )
    return selected


def render_numeric_slider(
    feature_name: str,
    min_val: float,
    max_val: float,
    default_val: float = None,
    key: str = None
) -> float:
    """Render slider for numeric feature with formatted caption."""
    if default_val is None:
        default_val = (min_val + max_val) / 2
    
    value = st.slider(
        feature_name,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        key=key or f"slider_{feature_name}"
    )
    
    formatted = format_value(feature_name, value)
    st.caption(f"{feature_name}: {formatted}")
    
    return value


def validate_input_data(input_dict: Dict[str, Any], required_features: list) -> Tuple[bool, str]:
    """Validate input data dictionary."""
    missing = [f for f in required_features if f not in input_dict]
    if missing:
        return False, f"Missing features: {', '.join(missing)}"
    
    for feature, value in input_dict.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False, f"Invalid value for {feature}"
    
    return True, ""


def get_color_for_confidence(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"

def validate_dataset(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """Validate dataset and return quality metrics and warnings."""
    warnings = []
    metrics = {}
    
    # Check size
    if len(df) < 100:
        warnings.append("⚠️ **Small dataset:** Only {len(df)} rows. Consider collecting more data for better model performance.")
    
    metrics['rows'] = len(df)
    metrics['columns'] = len(df.columns)
    
    # Check for missing values
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    if missing_pct > 5:
        warnings.append(f"⚠️ **Missing data:** {missing_pct:.1f}% of data is missing. Consider imputation or removal.")
    metrics['missing_percent'] = missing_pct
    
    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = dup_count / len(df) * 100
        warnings.append(f"⚠️ **Duplicate rows:** {dup_count} ({dup_pct:.1f}%) exact duplicates found.")
    metrics['duplicates'] = dup_count
    
    # Check numeric columns for variance
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            warnings.append(f"⚠️ **Zero variance:** Column '{col}' has constant value. It won't help predictions.")
    
    # Check target column imbalance
    if target_col and target_col in df.columns:
        target_dist = df[target_col].value_counts(normalize=True)
        if len(target_dist) > 1:
            min_pct = target_dist.min() * 100
            max_pct = target_dist.max() * 100
            if min_pct < 10:
                warnings.append(f"⚠️ **Class imbalance:** Smallest class is {min_pct:.1f}% (largest {max_pct:.1f}%). Consider techniques like SMOTE or class weights.")
    
    # Check correlation (high multicollinearity)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            pairs_str = ', '.join([f"'{p[0]}' & '{p[1]}' ({p[2]:.2f})" for p in high_corr_pairs[:3]])
            warnings.append(f"⚠️ **High correlation:** Features are highly correlated ({pairs_str}). Consider removing one from each pair.")
    
    metrics['warnings'] = warnings
    return metrics


def get_error_suggestion(error_msg: str) -> str:
    """Get helpful suggestion based on error type."""
    error_lower = error_msg.lower()
    
    if "stratify" in error_lower:
        return "Increase test_size % or ensure you have at least 2 samples per class"
    elif "feature" in error_lower:
        return "Ensure all selected features have valid numeric values"
    elif "memory" in error_lower:
        return "Use fewer features or reduce dataset size"
    elif "type" in error_lower:
        return "Check data types - all features should be numeric"
    elif "nan" in error_lower or "inf" in error_lower:
        return "Remove rows with NaN or infinite values"
    elif "dimension" in error_lower:
        return "Ensure all input features match training features"
    else:
        return "Check your data and try again with different settings"


def export_predictions_to_csv(predictions_df: pd.DataFrame, filename: str = "predictions.csv") -> bytes:
    """Convert predictions dataframe to CSV bytes for download."""
    return predictions_df.to_csv(index=False).encode('utf-8')


def export_audit_trail_to_csv(audit_trail: list) -> bytes:
    """Convert audit trail to CSV bytes for download."""
    df = pd.DataFrame(audit_trail)
    return df.to_csv(index=False).encode('utf-8')