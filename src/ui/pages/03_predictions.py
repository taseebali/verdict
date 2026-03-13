"""Streamlit Predictions Page - Clean & Complete (FIXED)"""

import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from src.ui.utils import (
    format_value, is_binary_feature, get_feature_statistics,
    render_binary_input, render_numeric_slider
)
from src.ui.session_manager import init_session_state, ensure_model_trained, add_audit_entry
from src.ui.charts import plot_whatif_analysis
from src.ui.components import PredictionInputComponent

st.set_page_config(page_title="Make Predictions", page_icon="🎯", layout="wide")

# Initialize session state FIRST
init_session_state()
st.title("🎯 Make Predictions")

# Check if model is trained
ensure_model_trained()

model = st.session_state.trained_model
model_features = st.session_state.model_features  # Features the model expects
selected_features = st.session_state.get('selected_features', model_features)  # Original feature selection
label_encoders = st.session_state.get('label_encoders', {})  # Encoders from training

st.markdown(f"**Using model trained with {len(model_features)} features**")

# Load data to get ranges
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("data/verdict_demo.csv")

df = st.session_state.df

# Get feature statistics for sliders (numeric features only)
feature_stats = get_feature_statistics(df, selected_features)

# Filter features to only use numeric ones for input
numeric_features = [f for f in selected_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
if len(numeric_features) == 0:
    st.error("❌ No numeric features available for making predictions")
    st.stop()
input_features = numeric_features

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "❓ What-If Analysis", "📊 Feature Ranges"])

# ===== TAB 1: SINGLE PREDICTION =====
with tab1:
    st.markdown("## Make a Single Prediction")

    # Use PredictionInputComponent (only for numeric input features)
    pred_component = PredictionInputComponent(df, input_features)
    input_data = pred_component.render()

    if st.button("🚀 Make Prediction", type="primary", width='stretch'):
        with st.spinner("⏳ Generating prediction..."):
            # Prepare X with same features the model was trained on
            X_input = df[selected_features].iloc[0:1].copy()  # Get template shape
            
            # Fill in user input for numeric features
            for feature in input_features:
                if feature in input_data:
                    X_input[feature] = input_data[feature]
            
            # Encode categorical features using stored encoders
            for col in X_input.columns:
                if col in label_encoders:
                    X_input[col] = label_encoders[col].transform(X_input[col].astype(str))
            
            # Select only the features the model expects
            X_input = X_input[model_features]
            prediction = model.predict(X_input)[0]
            
            # Get probabilities if available, otherwise use default confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_input)[0]
                confidence = max(probabilities) * 100
            else:
                # Fallback for models without predict_proba (e.g., VotingClassifier with SVM)
                probabilities = np.array([0.5, 0.5])  # Default confidence
                confidence = 50.0

            # Log to audit trail
            add_audit_entry(int(prediction), confidence / 100, input_data, st.session_state.target_column)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", prediction)
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        st.success("✅ Prediction saved to audit log")

        st.markdown("### Class Probabilities")
        prob_df = pd.DataFrame({"Class": range(len(probabilities)), "Probability (%)": probabilities * 100})

        fig = go.Figure(data=[go.Bar(x=prob_df["Class"], y=prob_df["Probability (%)"])])
        fig.update_layout(
            title="Probability by Class",
            xaxis_title="Class",
            yaxis_title="Probability (%)",
            height=400,
        )
        st.plotly_chart(fig, width='stretch')

        # Export option
        from src.ui.utils import export_predictions_to_csv

        export_df = pd.DataFrame({**input_data, "Prediction": [prediction], "Confidence (%)": [f"{confidence:.1f}"]})
        csv_data = export_predictions_to_csv(export_df)
        st.download_button(
            label="📥 Download Prediction as CSV",
            data=csv_data,
            file_name="prediction.csv",
            mime="text/csv",
            key="download_prediction",
        )

# ===== TAB 2: WHAT-IF ANALYSIS =====
with tab2:
    st.markdown("## What-If Analysis")
    st.markdown("**Change one feature at a time and see how predictions change**")

    feature_to_vary = st.selectbox("Select feature to vary:", numeric_features)

    if feature_to_vary in feature_stats:
        stats = feature_stats[feature_to_vary]

        baseline_data = {}
        st.markdown("### Baseline Values (from training data)")

        cols = st.columns(3)
        for idx, feature in enumerate(selected_features):
            with cols[idx % 3]:
                # For numeric features with statistics
                if feature in feature_stats:
                    default_val = feature_stats[feature]["mean"]
                    stats_vals = feature_stats[feature]

                    # ✅ FIX: pass df + feature name
                    if is_binary_feature(df, feature):
                        unique_vals = sorted(df[feature].dropna().unique())
                        options = [
                            f"{v} - {'Yes' if v == 1 else 'No'}" if v in [0, 1] else str(v)
                            for v in unique_vals
                        ]
                        selected_idx = st.radio(
                            feature,
                            range(len(unique_vals)),
                            format_func=lambda i: options[i],
                            label_visibility="collapsed",
                            key=f"baseline_{feature}",
                            horizontal=True,
                        )
                        value = unique_vals[selected_idx]
                        st.caption(
                            f"**{feature}:** {'Yes' if value == 1 else 'No' if value == 0 else value}"
                        )
                        baseline_data[feature] = value

                    else:
                        step = (stats_vals["max"] - stats_vals["min"]) / 100
                        if step == 0:
                            step = 1.0  # avoid Streamlit slider step=0

                        baseline_value = st.slider(
                            feature,
                            min_value=float(stats_vals["min"]),
                            max_value=float(stats_vals["max"]),
                            value=float(default_val),
                            step=float(step),
                            label_visibility="collapsed",
                            key=f"baseline_{feature}",
                        )
                        st.caption(f"**{feature}:** {format_value(feature, baseline_value)}")
                        baseline_data[feature] = baseline_value
                
                # For categorical features (not in feature_stats)
                elif feature in df.columns and not pd.api.types.is_numeric_dtype(df[feature]):
                    unique_vals = sorted(df[feature].dropna().unique())
                    selected_val = st.selectbox(
                        feature,
                        unique_vals,
                        label_visibility="collapsed",
                        key=f"baseline_{feature}"
                    )
                    st.caption(f"**{feature}:** {selected_val}")
                    baseline_data[feature] = selected_val

        # Range to test
        st.markdown(f"### Vary {feature_to_vary}")
        st.markdown(
            f"Range: {format_value(feature_to_vary, stats['min'])} → {format_value(feature_to_vary, stats['max'])}"
        )

        num_points = st.slider("Number of test points:", 5, 20, 10)

        # Default: continuous sweep
        test_values = np.linspace(stats["min"], stats["max"], num_points)

        # ✅ SAFETY FIX: if the varied feature is binary, only test valid unique values
        if is_binary_feature(df, feature_to_vary):
            test_values = np.array(sorted(df[feature_to_vary].dropna().unique()))

        with st.spinner("⏳ Running sensitivity analysis..."):
            # OPTIMIZATION: Batch predictions instead of one-by-one
            batch_inputs = []
            formatted_test_values = []
            
            for test_val in test_values:
                input_row = baseline_data.copy()
                input_row[feature_to_vary] = test_val
                batch_inputs.append(input_row)
                formatted_test_values.append(format_value(feature_to_vary, test_val))
            
            # Single batch prediction call (10-20x faster)
            X_batch = pd.DataFrame(batch_inputs)
            
            # Encode categorical features using stored encoders
            for col in X_batch.columns:
                if col in label_encoders:
                    X_batch[col] = label_encoders[col].transform(X_batch[col].astype(str))
            
            # Select only the features the model expects
            X_batch = X_batch[model_features]
            
            predictions = model.predict(X_batch)
            
            # Get probabilities if available, otherwise use default confidence
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_batch)
                confidences = [float(np.max(proba) * 100) for proba in probas]
            else:
                # Fallback for models without predict_proba (e.g., VotingClassifier with SVM)
                confidences = [50.0] * len(predictions)  # Default confidence for each prediction

        # Plot results using centralized chart function
        plot_whatif_analysis(test_values, formatted_test_values, confidences, feature_to_vary)

        # Results table
        st.markdown("### Prediction Results")
        results_df = pd.DataFrame(
            {
                feature_to_vary: formatted_test_values,
                "Predicted Class": predictions,
                "Confidence (%)": confidences,
            }
        )
        st.dataframe(results_df, width='stretch')

# ===== TAB 3: FEATURE RANGES =====
with tab3:
    st.markdown("## Feature Ranges (from dataset)")

    ranges_data = []
    for feature in selected_features:
        if feature in feature_stats:
            stats = feature_stats[feature]
            ranges_data.append(
                {
                    "Feature": feature,
                    "Min": stats["min"],
                    "Mean": stats["mean"],
                    "Max": stats["max"],
                }
            )

    ranges_df = pd.DataFrame(ranges_data)
    st.dataframe(ranges_df, width='stretch')

    st.markdown("**Note:** All sliders use these real ranges from your data, with human-readable formatting")
