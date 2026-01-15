"""Verdict - AI Decision Copilot (Streamlit Frontend Application)."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import MODEL_CONFIGS
from src.pipeline import MLPipeline

# IMPORTANT: your pasted file is "visualization.py"
from src.visualizations import Visualizer

from src.explainability import ExplainabilityAnalyzer
from src.whatif import WhatIfAnalyzer
from src.exporter import ModelExporter
from src.report_gen import ReportGenerator


# -----------------------------
# Helpers
# -----------------------------
def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit uses Arrow under the hood for st.dataframe.
    Arrow can choke on dtype objects and mixed object columns.
    """
    out = df.copy()

    # Make object columns safe
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str)

    # Convert nullable dtypes when possible
    out = out.convert_dtypes()
    return out


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Verdict",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ Verdict")
st.markdown(
    """
**Automatically build, evaluate, and explain machine learning models for decision support.**

Upload a CSV, select a prediction target, and Verdict will:
- Automatically preprocess your data
- Train multiple models
- Compare their performance
- Explain predictions with feature importance
- Simulate scenarios with what-if analysis
"""
)

# -----------------------------
# Session state
# -----------------------------
st.session_state.setdefault("pipeline", None)
st.session_state.setdefault("df", None)
st.session_state.setdefault("target_col", None)
st.session_state.setdefault("results", None)
st.session_state.setdefault("training_complete", False)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📁 Data Upload")

    upload_option = st.radio("Choose data source:", ["Upload CSV", "Use Demo Dataset"])

    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(
                f"✅ Loaded {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns"
            )
    else:
        demo_file = "data/demo_business_dataset.csv"
        if os.path.exists(demo_file):
            st.session_state.df = pd.read_csv(demo_file)
            st.info(
                f"🧪 Demo: Customer churn prediction\n\n"
                f"{st.session_state.df.shape[0]} rows · {st.session_state.df.shape[1]} columns"
            )
        else:
            st.warning("Demo dataset not found. Please upload a CSV file.")


# -----------------------------
# Main content
# -----------------------------
if st.session_state.df is None:
    st.warning("⬅️ Please upload or select a dataset to begin.")
    st.stop()

df = st.session_state.df

tab1, tab2, tab3, tab4 = st.tabs(
    ["👀 Data Preview", "🏋️ Model Training", "📈 Results", "🧠 Explanations"]
)

# -----------------------------
# Tab 1: Data Preview
# -----------------------------
with tab1:
    st.subheader("Data Overview")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.write("### Dataset Preview")
    st.dataframe(df.head(10), width="stretch")

    st.write("### Column Information")

    # ✅ FIX: convert dtype objects to strings to avoid pyarrow ArrowInvalid
    schema_df = pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Type": df.dtypes.astype(str),  # << FIXED
            "Missing": df.isnull().sum().astype(int),
            "Unique": df.nunique(dropna=True).astype(int),
        }
    )

    st.dataframe(make_arrow_safe(schema_df), width="stretch")

    st.write("### Feature Distributions")
    selected_feature = st.selectbox("Select feature to visualize:", df.columns)
    fig = Visualizer.plot_feature_distribution(df, selected_feature)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Tab 2: Model Training
# -----------------------------
with tab2:
    st.subheader("Model Training Configuration")

    left, right = st.columns(2)

    with left:
        st.session_state.target_col = st.selectbox(
            "Select prediction target:",
            df.columns,
            key="target_select",
        )

    with right:
        t = st.session_state.target_col
        st.write(f"**Target variable:** {t}")
        st.write(f"**Unique values:** {df[t].nunique()}")

    st.write("### Select Models to Train")

    c1, c2 = st.columns(2)
    with c1:
        train_lr = st.checkbox("Logistic Regression", value=True)
    with c2:
        train_rf = st.checkbox("Random Forest", value=True)

    model_list = []
    if train_lr:
        model_list.append("logistic_regression")
    if train_rf:
        model_list.append("random_forest")

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            try:
                st.session_state.pipeline = MLPipeline(df, st.session_state.target_col)
                st.session_state.results = st.session_state.pipeline.run_full_pipeline(model_list)

                if st.session_state.results.get("status") == "success":
                    st.success("✅ Models trained successfully!")
                    st.session_state.training_complete = True
                else:
                    st.session_state.training_complete = False
                    st.error(
                        f"❌ Training failed: {st.session_state.results.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                st.session_state.training_complete = False
                st.error(f"❌ Error: {str(e)}")


# -----------------------------
# Tab 3: Results
# -----------------------------
with tab3:
    if not (st.session_state.results and st.session_state.results.get("status") == "success"):
        st.info("🏋️ Train models first in the 'Model Training' tab to see results.")
    else:
        st.subheader("Model Evaluation Results")

        eval_results = st.session_state.results.get("eval_results", {})
        task_type = st.session_state.results.get("task_type", "classification")

        st.write("### Metrics Summary")
        c1, c2 = st.columns(2)

        with c1:
            metric = "f1" if task_type == "classification" else "r2"
            fig = Visualizer.plot_metrics_comparison(eval_results, metric)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = Visualizer.plot_metrics_table(eval_results)
            st.plotly_chart(fig, use_container_width=True)

        st.write("### Detailed Metrics")
        for model_name, metrics in eval_results.items():
            with st.expander(f"📌 {model_name.replace('_', ' ').title()}"):
                metric_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                st.dataframe(make_arrow_safe(metric_df), width="stretch")

        if task_type == "classification":
            st.write("### Confusion Matrices")
            X_test, y_test = st.session_state.pipeline.get_test_data()

            for model_name in st.session_state.pipeline.model_manager.get_models().keys():
                y_pred = st.session_state.pipeline.get_model_predictions(model_name, X_test)
                fig = Visualizer.plot_confusion_matrix(
                    y_test, y_pred, f"{model_name.replace('_', ' ').title()} - Confusion Matrix"
                )
                st.pyplot(fig)

        st.write("---")
        st.write("### 📦 Export & Reports")

        e1, e2, e3 = st.columns(3)

        with e1:
            if st.button("💾 Export All Models", type="secondary", key="export_models"):
                with st.spinner("Exporting models..."):
                    try:
                        os.makedirs("models", exist_ok=True)
                        exporter = ModelExporter(output_dir="models")
                        exports = exporter.export_all_models(st.session_state.pipeline, eval_results)

                        st.success("✅ Models exported successfully!")
                        with st.expander("📋 Export Summary"):
                            for model_name, export_info in exports.items():
                                st.write(f"**{model_name}**")
                                st.write(f"- File: `{export_info['model_path']}`")
                                st.write(f"- Size: {export_info['file_size_mb']:.2f} MB")
                                st.write(f"- Exported: {export_info['timestamp']}")
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")

        with e2:
            if st.button("📄 Generate HTML Report", type="secondary", key="gen_report"):
                with st.spinner("Generating report..."):
                    try:
                        os.makedirs("reports", exist_ok=True)
                        data_info = {
                            "rows": int(len(df)),
                            "columns": int(len(df.columns)),
                            "missing": int(df.isnull().sum().sum()),
                        }

                        generator = ReportGenerator("Verdict Report")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        report_file = f"reports/report_{timestamp}.html"

                        report_path = generator.generate_html_report(
                            st.session_state.pipeline, eval_results, data_info, output_file=report_file
                        )

                        st.success("✅ Report generated successfully!")
                        st.write(f"📍 Saved to: `{report_path}`")

                        with open(report_path, "r", encoding="utf-8") as f:
                            report_content = f.read()

                        st.download_button(
                            label="⬇️ Download HTML Report",
                            data=report_content,
                            file_name=f"report_{timestamp}.html",
                            mime="text/html",
                            key="download_report",
                        )
                    except Exception as e:
                        st.error(f"❌ Report generation failed: {str(e)}")

        with e3:
            if st.button("📊 View Model Metadata", type="secondary", key="view_metadata"):
                st.write("### Model Metadata")
                with st.expander("Logistic Regression"):
                    st.json(
                        {
                            "Model": "Logistic Regression",
                            "Task Type": task_type,
                            "Max Iterations": MODEL_CONFIGS["logistic_regression"]["params"]["max_iter"],
                            "Solver": MODEL_CONFIGS["logistic_regression"]["params"]["solver"],
                        }
                    )
                with st.expander("Random Forest"):
                    st.json(
                        {
                            "Model": "Random Forest",
                            "Task Type": task_type,
                            "Estimators": MODEL_CONFIGS["random_forest"]["params"]["n_estimators"],
                            "Max Depth": MODEL_CONFIGS["random_forest"]["params"]["max_depth"],
                        }
                    )


# -----------------------------
# Tab 4: Explanations
# -----------------------------
with tab4:
    st.subheader("🔍 Feature Importance & Explainability")

    if not (st.session_state.pipeline and st.session_state.results and st.session_state.results.get("status") == "success"):
        st.info("👈 Train models first in the 'Model Training' tab to see explanations.")
    else:
        exp_tab1, exp_tab2, exp_tab3 = st.tabs(
            ["📊 Feature Importance", "🔮 What-If Analysis", "💡 Prediction Explanations"]
        )

        # ---- Feature Importance ----
        with exp_tab1:
            st.write("### Feature Importance Analysis")

            selected_model = st.selectbox(
                "Select model for feature importance:",
                list(st.session_state.pipeline.model_manager.get_models().keys()),
                key="importance_model",
            )

            with st.spinner("Computing feature importance..."):
                try:
                    X_test, y_test = st.session_state.pipeline.get_test_data()
                    model = st.session_state.pipeline.model_manager.get_model(selected_model)
                    feature_names = st.session_state.results.get("feature_names", [])

                    analyzer = ExplainabilityAnalyzer(
                        model, st.session_state.pipeline.X_train, X_test, feature_names
                    )

                    task_type = st.session_state.results.get("task_type", "classification")
                    scoring = "accuracy" if task_type == "classification" else "r2"

                    importance = analyzer.get_feature_importance_permutation(y_test, scoring=scoring)

                    if importance:
                        fig = Visualizer.plot_feature_importance_permutation(importance)
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("### Top Features")
                        importance_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
                        st.dataframe(make_arrow_safe(importance_df), width="stretch")
                    else:
                        st.warning("Could not compute feature importance for this model type.")
                except Exception as e:
                    st.error(f"❌ Error computing importance: {str(e)}")

        # ---- What-If ----
        with exp_tab2:
            st.write("### What-If Analysis: Scenario Simulation")

            feature_names = st.session_state.results.get("feature_names", [])
            numeric_cols = st.session_state.pipeline.get_numeric_columns()
            categorical_cols = st.session_state.pipeline.get_categorical_columns()
            label_encoders = st.session_state.pipeline.get_label_encoders()

            whatif = WhatIfAnalyzer(
                st.session_state.pipeline,
                feature_names,
                numeric_cols,
                categorical_cols,
                label_encoders,
            )

            feature_ranges = whatif.get_feature_ranges()
            categorical_options = whatif.get_categorical_options()

            st.write("#### Adjust Feature Values")
            col1, col2 = st.columns(2)
            input_dict = {}

            ui_features = list(numeric_cols) + list(categorical_cols)

            for i, feature in enumerate(ui_features):
                col = col1 if i % 2 == 0 else col2
                with col:
                    if feature in feature_ranges:
                        r = feature_ranges[feature]
                        step = float(r["std"] / 10) if r["std"] > 0 else 0.1
                        input_dict[feature] = st.slider(
                            f"{feature}:",
                            min_value=float(r["min"]),
                            max_value=float(r["max"]),
                            value=float(r["mean"]),
                            step=step,
                        )
                    elif feature in categorical_options and categorical_options[feature]:
                        input_dict[feature] = st.selectbox(
                            f"{feature}:",
                            categorical_options[feature],
                            key=f"cat_{feature}",
                        )

            st.write("---")

            if st.button("🚀 Make Prediction with Current Scenario", type="primary"):
                with st.spinner("Making predictions..."):
                    results_dict = whatif.compare_predictions(input_dict)

                    st.write("### Predictions for Current Scenario")
                    p1, p2 = st.columns(2)

                    for i, (model_name, result) in enumerate(results_dict.items()):
                        col = p1 if i % 2 == 0 else p2
                        with col:
                            if result.get("status") == "success":
                                conf = result.get("confidence", None)
                                st.metric(
                                    model_name.replace("_", " ").title(),
                                    f"{result['prediction']}",
                                    f"Confidence: {conf:.2%}" if conf is not None else "Confidence: N/A",
                                )
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown')}")

            st.write("### Sensitivity Analysis")

            sens_feature = st.selectbox(
                "Select numeric feature to vary:",
                list(feature_ranges.keys()),
                key="sens_feature",
            )

            sens_model = st.selectbox(
                "Select model for sensitivity analysis:",
                list(st.session_state.pipeline.model_manager.get_models().keys()),
                key="sens_model",
            )

            n_points = st.slider("Number of test points:", 5, 20, 10)

            if st.button("📈 Run Sensitivity Analysis"):
                with st.spinner("Running sensitivity analysis..."):
                    r = feature_ranges[sens_feature]
                    test_values = np.linspace(r["min"], r["max"], n_points)
                    sens_df = whatif.get_sensitivity_analysis(sens_model, input_dict, sens_feature, test_values)

                    if not sens_df.empty:
                        fig = Visualizer.plot_sensitivity_analysis(sens_df, sens_feature)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(make_arrow_safe(sens_df), width="stretch")

        # ---- Prediction Explanations ----
        with exp_tab3:
            st.write("### Prediction Explanations")

            X_test, y_test = st.session_state.pipeline.get_test_data()
            feature_names = st.session_state.results.get("feature_names", [])

            selected_model = st.selectbox(
                "Select model for explanation:",
                list(st.session_state.pipeline.model_manager.get_models().keys()),
                key="explain_model",
            )

            sample_idx = st.slider("Select test sample to explain:", 0, len(X_test) - 1, 0)

            if st.button("🔬 Explain This Prediction"):
                with st.spinner("Computing explanation..."):
                    try:
                        model = st.session_state.pipeline.model_manager.get_model(selected_model)
                        analyzer = ExplainabilityAnalyzer(
                            model, st.session_state.pipeline.X_train, X_test, feature_names
                        )

                        pred, confidence = analyzer.predict_with_confidence(X_test[sample_idx : sample_idx + 1])

                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Prediction", str(pred[0]))
                        with c2:
                            st.metric("Confidence", f"{confidence[0]:.2%}" if confidence else "N/A")

                        explanation = analyzer.explain_prediction(sample_idx)

                        if "error" in explanation:
                            st.warning(f"Could not generate explanation: {explanation['error']}")
                        else:
                            st.write("### Feature Contributions (SHAP)")
                            contributions = explanation.get("feature_contributions", {})

                            if contributions:
                                contrib_df = pd.DataFrame(
                                    [{"Feature": k, "Impact": v} for k, v in contributions.items()]
                                )
                                st.dataframe(make_arrow_safe(contrib_df), width="stretch")

                            st.write("### Feature Values for This Sample")
                            sample_features = explanation.get("sample_features", {})
                            if sample_features:
                                features_df = pd.DataFrame(
                                    [{"Feature": k, "Value": v} for k, v in sample_features.items()]
                                )
                                st.dataframe(make_arrow_safe(features_df), width="stretch")
                    except Exception as e:
                        st.error(f"❌ Error generating explanation: {str(e)}")


# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; font-size: 12px;'>
Verdict v1.0 | Decision Copilot
</div>
""",
    unsafe_allow_html=True,
)
