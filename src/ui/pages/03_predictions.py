"""Streamlit Predictions Page - Clean & Complete (FIXED)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

st.set_page_config(page_title="Make Predictions", page_icon="🎯", layout="wide")

# Initialize session state FIRST
init_session_state()
st.title("🎯 Make Predictions")

# Check if model is trained
ensure_model_trained()

model = st.session_state.trained_model
features = st.session_state.model_features

st.markdown(f"**Using model trained with {len(features)} features**")

# Load data to get ranges
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("data/verdict_demo.csv")

df = st.session_state.df

# Get feature statistics for sliders
feature_stats = get_feature_statistics(df, features)

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "❓ What-If Analysis", "📊 Feature Ranges"])

# ===== TAB 1: SINGLE PREDICTION =====
with tab1:
    st.markdown("## Make a Single Prediction")

    input_data = {}
    cols = st.columns(3)

    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            if feature in feature_stats:
                stats = feature_stats[feature]

                # Binary features: radio buttons
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
                        key=f"single_{feature}",
                        horizontal=True,
                    )
                    value = unique_vals[selected_idx]
                    st.caption(
                        f"**{feature}:** {'Yes' if value == 1 else 'No' if value == 0 else value}"
                    )
                    input_data[feature] = value

                # Continuous features: sliders
                else:
                    step = (stats["max"] - stats["min"]) / 100
                    if step == 0:
                        step = 1.0  # avoid Streamlit slider step=0

                    value = st.slider(
                        feature,
                        min_value=float(stats["min"]),
                        max_value=float(stats["max"]),
                        value=float(stats["mean"]),
                        step=float(step),
                        label_visibility="collapsed",
                        key=f"single_{feature}",
                    )
                    st.caption(f"**{feature}:** {format_value(feature, value)}")
                    input_data[feature] = value
            else:
                input_data[feature] = st.number_input(feature, value=0.0, key=f"single_{feature}")

    if st.button("🚀 Make Prediction", type="primary", use_container_width=True):
        with st.spinner("⏳ Generating prediction..."):
            X_input = pd.DataFrame([input_data])
            prediction = model.predict(X_input)[0]
            probabilities = model.predict_proba(X_input)[0]
            confidence = max(probabilities) * 100

            # Log to audit trail
            add_audit_entry(int(prediction), confidence / 100, features, st.session_state.target_column)

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
        st.plotly_chart(fig, use_container_width=True)

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

    feature_to_vary = st.selectbox("Select feature to vary:", features)

    if feature_to_vary in feature_stats:
        stats = feature_stats[feature_to_vary]

        baseline_data = {}
        st.markdown("### Baseline Values (from training data)")

        cols = st.columns(3)
        for idx, feature in enumerate(features):
            with cols[idx % 3]:
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
            predictions = model.predict(X_batch)
            probas = model.predict_proba(X_batch)
            confidences = [float(np.max(proba) * 100) for proba in probas]

        # Plot results using centralized chart function
        plot_whatif_analysis(test_values, formatted_test_values, confidences, feature_to_vary)

        # Results table
        st.markdown("### Prediction Results")
        results_df = pd.DataFrame(
            {
                feature_to_vary: formatted_test_values,
                "Predicted Class": predictions,
                "Confidence (%)": [f"{c:.1f}%" for c in confidences],
            }
        )
        st.dataframe(results_df, use_container_width=True)

# ===== TAB 3: FEATURE RANGES =====
with tab3:
    st.markdown("## Feature Ranges (from dataset)")

    ranges_data = []
    for feature in features:
        if feature in feature_stats:
            stats = feature_stats[feature]
            ranges_data.append(
                {
                    "Feature": feature,
                    "Min": format_value(feature, stats["min"]),
                    "Mean": format_value(feature, stats["mean"]),
                    "Max": format_value(feature, stats["max"]),
                }
            )

    ranges_df = pd.DataFrame(ranges_data)
    st.dataframe(ranges_df, use_container_width=True)

    st.markdown("**Note:** All sliders use these real ranges from your data, with human-readable formatting")
