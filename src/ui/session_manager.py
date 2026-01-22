"""Session State Management - Centralized initialization and access"""

import streamlit as st
from typing import Any, Optional
import pandas as pd
from datetime import datetime


def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    defaults = {
        'df': None,
        'df_name': 'Demo Dataset',
        'trained_model': None,
        'model_features': [],
        'target_column': None,
        'train_acc': None,
        'test_acc': None,
        'test_precision': None,
        'test_recall': None,
        'test_f1': None,
        'model_metadata': {},
        'audit_trail': [],
        'model_registry': {},
        'trained_models_history': [],
        'current_page': 'Home',
        'last_trained_at': None,
        'last_prediction_at': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def ensure_data_loaded() -> bool:
    """Check if data is loaded. Shows error if not."""
    if st.session_state.df is None:
        st.error("❌ No data loaded")
        st.info("Go to 📊 Data Explorer and load data first")
        st.stop()
        return False
    return True


def ensure_model_trained() -> bool:
    """Check if model is trained. Shows error if not."""
    if st.session_state.trained_model is None:
        st.error("❌ No trained model found")
        st.info("Go to 🤖 Model Training and train a model first")
        st.stop()
        return False
    return True


def get_session_data(key: str, default: Any = None) -> Any:
    """Safely get session state value."""
    return st.session_state.get(key, default)


def set_session_data(key: str, value: Any) -> None:
    """Safely set session state value."""
    st.session_state[key] = value


def reset_session() -> None:
    """Reset all session state variables."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def reset_model() -> None:
    """Reset model-related session variables."""
    st.session_state.trained_model = None
    st.session_state.model_features = []
    st.session_state.target_column = None
    st.session_state.train_acc = None
    st.session_state.test_acc = None


def reset_data() -> None:
    """Reset data-related session variables."""
    st.session_state.df = None
    st.session_state.df_name = 'Demo Dataset'
    reset_model()


def add_audit_entry(feature: str, value: Any, prediction: Any, confidence: float = None) -> None:
    """Log prediction to audit trail."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'feature': feature,
        'value': value,
        'prediction': prediction,
        'confidence': confidence,
    }
    st.session_state.audit_trail.append(entry)
    st.session_state.last_prediction_at = datetime.now()


def get_audit_trail_df() -> pd.DataFrame:
    """Get audit trail as DataFrame."""
    if not st.session_state.audit_trail:
        return pd.DataFrame(columns=['timestamp', 'feature', 'value', 'prediction', 'confidence'])
    return pd.DataFrame(st.session_state.audit_trail)


def is_ready_for_predictions() -> bool:
    """Check if everything is ready for making predictions."""
    checks = [
        (st.session_state.df is not None, "Data"),
        (st.session_state.trained_model is not None, "Model"),
        (st.session_state.model_features, "Features"),
    ]
    
    for check, name in checks:
        if not check:
            st.error(f"❌ Missing: {name}")
            return False
    
    return True


def save_model_to_registry(model_name: str) -> None:
    """Save current trained model to model registry."""
    if st.session_state.trained_model is None:
        st.error("No model to save")
        return
    
    st.session_state.model_registry[model_name] = {
        'model': st.session_state.trained_model,
        'features': st.session_state.model_features,
        'target': st.session_state.target_column,
        'train_acc': st.session_state.train_acc,
        'test_acc': st.session_state.test_acc,
        'timestamp': datetime.now().isoformat()
    }
    st.success(f"✅ Model '{model_name}' saved to registry")


def load_model_from_registry(model_name: str) -> bool:
    """Load model from registry."""
    if model_name not in st.session_state.model_registry:
        st.error(f"Model '{model_name}' not found in registry")
        return False
    
    model_data = st.session_state.model_registry[model_name]
    st.session_state.trained_model = model_data['model']
    st.session_state.model_features = model_data['features']
    st.session_state.target_column = model_data['target']
    st.session_state.train_acc = model_data['train_acc']
    st.session_state.test_acc = model_data['test_acc']
    
    return True


def get_model_registry() -> dict:
    """Get all saved models in registry."""
    return st.session_state.model_registry