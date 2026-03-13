"""Session State Management - Simplified"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Any


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
        'cv_scores': None,
        'audit_trail': [],
        'model_registry': {},
        'trained_models_history': [],
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
    return True


def ensure_model_trained() -> bool:
    """Check if model is trained. Shows error if not."""
    if st.session_state.trained_model is None:
        st.error("❌ No trained model found")
        st.info("Go to 🤖 Model Training and train a model first")
        st.stop()
    return True


def save_model_to_registry(model_name: str) -> None:
    """Save current trained model to registry."""
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
    st.success(f"✅ Model '{model_name}' saved")


def load_model_from_registry(model_name: str) -> bool:
    """Load model from registry."""
    if model_name not in st.session_state.model_registry:
        st.error(f"Model '{model_name}' not found")
        return False
    
    model_data = st.session_state.model_registry[model_name]
    st.session_state.trained_model = model_data['model']
    st.session_state.model_features = model_data['features']
    st.session_state.target_column = model_data['target']
    st.session_state.train_acc = model_data['train_acc']
    st.session_state.test_acc = model_data['test_acc']
    return True


def add_audit_entry(prediction: Any, confidence: float, features: dict, target: str) -> None:
    """Log prediction to audit trail."""
    st.session_state.audit_trail.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': prediction,
        'confidence': confidence,
        'model_name': f"{target}_model",
        'status': 'success'
    })


def get_audit_trail_df() -> pd.DataFrame:
    """Get audit trail as DataFrame."""
    if not st.session_state.audit_trail:
        return pd.DataFrame(columns=['timestamp', 'prediction', 'confidence', 'model_name', 'status'])
    return pd.DataFrame(st.session_state.audit_trail)


def get_model_registry() -> dict:
    """Get all saved models in registry."""
    return st.session_state.model_registry

