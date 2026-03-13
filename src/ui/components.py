"""Reusable Streamlit Components - Extract common UI patterns"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
import pandas as pd
import numpy as np

from src.ui.utils import load_demo_dataset, format_value, get_feature_statistics
from src.core.validators import DataValidator


class StreamlitComponent(ABC):
    """Base class for reusable Streamlit components."""

    @abstractmethod
    def render(self) -> Any:
        """Render the component and return result."""
        pass


class DataLoadingComponent(StreamlitComponent):
    """Handles data loading from file upload or demo dataset."""

    def __init__(self, container=None):
        """Initialize data loader.
        
        Args:
            container: Optional Streamlit container (st or st.container())
        """
        self.container = container or st
        self.df = None

    def render(self) -> Optional[pd.DataFrame]:
        """Render data loading interface.
        
        Returns:
            Loaded DataFrame or None
        """
        st.markdown("## 📤 Load Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Use Demo Dataset", width='stretch'):
                try:
                    self.df = load_demo_dataset()
                    st.session_state.df = self.df
                    st.success(f"✅ Demo dataset loaded ({len(self.df):,} rows)")
                    return self.df
                except FileNotFoundError:
                    st.error("⚠️ Demo dataset not found")
                    return None
        
        with col2:
            uploaded_file = st.file_uploader("📥 Or upload your CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    self.df = pd.read_csv(uploaded_file)
                    st.session_state.df = self.df
                    st.success(f"✅ Loaded {uploaded_file.name} ({len(self.df):,} rows)")
                    return self.df
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    return None
        
        # Load from session or default
        if 'df' in st.session_state and st.session_state.df is not None:
            self.df = st.session_state.df
            return self.df
        
        return None

    def validate(self) -> Dict[str, Any]:
        """Validate loaded dataset.
        
        Returns:
            Validation results dictionary
        """
        if self.df is None:
            return {'valid': False, 'message': 'No data loaded'}
        
        is_valid, msg = DataValidator.validate_basic(self.df)
        return {'valid': is_valid, 'message': msg, 'df': self.df}


class TargetColumnSelector(StreamlitComponent):
    """Component for selecting target column for model training."""

    def __init__(self, df: pd.DataFrame):
        """Initialize target selector.
        
        Args:
            df: Input DataFrame
        """
        self.df = df

    def render(self) -> Tuple[str, int]:
        """Render target column selection.
        
        Returns:
            Tuple of (target_column_name, unique_value_count)
        """
        st.markdown("## 1️⃣ Select What to Predict (Target Column)")
        
        # Find columns with 2-10 unique values (good for classification)
        target_candidates = []
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            if 2 <= unique_count <= 10:
                target_candidates.append(f"{col} ({unique_count} classes)")
        
        if target_candidates:
            selected_target = st.selectbox(
                "Select target column:",
                target_candidates,
                help="Column with 2-10 unique values works best for classification"
            )
            target_col = selected_target.split(" (")[0]
            unique_count = int(selected_target.split("(")[1].split(" ")[0])
        else:
            target_col = st.selectbox("Select target column:", self.df.columns)
            unique_count = self.df[target_col].nunique()
        
        return target_col, unique_count


class FeatureSelector(StreamlitComponent):
    """Component for selecting and excluding features."""

    def __init__(self, df: pd.DataFrame, target_col: str):
        """Initialize feature selector.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
        """
        self.df = df
        self.target_col = target_col

    def render(self) -> List[str]:
        """Render feature selection interface.
        
        Returns:
            List of selected feature names
        """
        st.markdown("## 2️⃣ Select Features to Train With")
        
        # Get all columns except target
        all_features = [col for col in self.df.columns if col != self.target_col]
        
        # Separate numeric and categorical
        numeric_features = self.df[all_features].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.df[all_features].select_dtypes(include=['object']).columns.tolist()
        
        st.info(f"📊 **Numeric**: {len(numeric_features)} columns | 🏷️ **Categorical**: {len(categorical_features)} columns")
        
        # Suggest excluding ID-like columns
        id_like = [col for col in all_features if 'id' in col.lower() or 'code' in col.lower()]
        if id_like:
            st.warning(f"⚠️ Consider excluding ID columns: {', '.join(id_like)}")
        
        # Multi-select
        selected_features = st.multiselect(
            "Select features to use:",
            all_features,
            default=all_features,
            help="Remove ID/code columns if present"
        )
        
        return selected_features


class PredictionInputComponent(StreamlitComponent):
    """Component for rendering prediction input controls."""

    def __init__(self, df: pd.DataFrame, features: List[str]):
        """Initialize prediction input.
        
        Args:
            df: Training DataFrame (for ranges)
            features: List of feature names
        """
        self.df = df
        self.features = features

    def render(self) -> Dict[str, float]:
        """Render input controls for each feature.
        
        Returns:
            Dictionary mapping feature names to input values
        """
        st.markdown("## Enter Feature Values")
        
        input_data = {}
        feature_stats = get_feature_statistics(self.df, self.features)
        
        cols = st.columns(3)
        
        for idx, feature in enumerate(self.features):
            with cols[idx % 3]:
                if feature in feature_stats:
                    stats = feature_stats[feature]
                    min_val = stats['min']
                    max_val = stats['max']
                    default_val = stats['mean']
                    
                    # Check if binary
                    unique_count = self.df[feature].nunique()
                    if unique_count <= 2:
                        # Binary: radio button
                        value = st.radio(
                            feature,
                            options=[0, 1],
                            format_func=lambda x: ['No (0)', 'Yes (1)'][x],
                            horizontal=True,
                            key=f"input_{feature}"
                        )
                        input_data[feature] = value
                    else:
                        # Numeric: slider
                        value = st.slider(
                            feature,
                            float(min_val),
                            float(max_val),
                            float(default_val),
                            key=f"slider_{feature}"
                        )
                        formatted = format_value(feature, value)
                        st.caption(f"{formatted}")
                        input_data[feature] = value
                else:
                    # Fallback
                    input_data[feature] = st.number_input(
                        feature,
                        value=0.0,
                        key=f"number_{feature}"
                    )
        
        return input_data


class AuditTrailComponent(StreamlitComponent):
    """Component for displaying audit trail/prediction history."""

    def __init__(self):
        """Initialize audit trail display."""
        pass

    def render(self) -> None:
        """Render audit trail table and controls."""
        if 'audit_trail' not in st.session_state:
            st.session_state.audit_trail = []
        
        audit_trail = st.session_state.audit_trail
        
        if not audit_trail:
            st.info("📋 No predictions yet")
            return
        
        # Convert to DataFrame for display
        audit_df = pd.DataFrame(audit_trail)
        
        st.markdown(f"## 📋 Audit Trail ({len(audit_trail)} records)")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(audit_trail))
        with col2:
            avg_confidence = audit_df['confidence'].mean() if 'confidence' in audit_df.columns else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        with col3:
            st.metric("Models Used", audit_df['model'].nunique() if 'model' in audit_df.columns else 0)
        
        # Table
        st.dataframe(audit_df, width='stretch')
        
        # Export button
        csv = audit_df.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV",
            csv,
            file_name="audit_trail.csv",
            mime="text/csv"
        )


class ModelPerformanceComponent(StreamlitComponent):
    """Component for displaying model performance metrics."""

    def __init__(self, metrics: Dict[str, float]):
        """Initialize performance display.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        self.metrics = metrics

    def render(self) -> None:
        """Render model performance metrics."""
        if not self.metrics:
            st.info("ℹ️ Train a model to see performance metrics")
            return
        
        st.markdown("## 📊 Model Performance")
        
        cols = st.columns(len(self.metrics))
        
        for idx, (metric_name, metric_value) in enumerate(self.metrics.items()):
            with cols[idx]:
                st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.4f}")


class DataQualityComponent(StreamlitComponent):
    """Component for displaying data quality analysis."""

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """Initialize data quality display.
        
        Args:
            df: DataFrame to analyze
            target_col: Optional target column name
        """
        self.df = df
        self.target_col = target_col

    def render(self) -> None:
        """Render data quality checks and warnings."""
        quality = DataValidator.validate_quality(self.df, self.target_col)
        
        st.markdown("## 🔍 Data Quality Analysis")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", quality['rows'])
        with col2:
            st.metric("Columns", quality['columns'])
        with col3:
            st.metric("Missing %", f"{quality['missing_percentage']:.1f}%")
        
        # Warnings
        if quality['warnings']:
            st.warning("⚠️ **Issues Found:**")
            for warning in quality['warnings']:
                st.write(f"  • {warning}")
        else:
            st.success("✅ **No issues detected**")
        
        # Details
        with st.expander("📋 Detailed Report"):
            st.json({
                'shape': f"{quality['rows']} × {quality['columns']}",
                'missing_count': quality['missing_count'],
                'duplicate_rows': quality['duplicate_rows'],
                'numeric_columns': quality['numeric_columns'],
                'categorical_columns': quality['categorical_columns']
            })
