import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config.settings import MODEL_CONFIGS
from src.core.pipeline import MLPipeline
from src.ui.visualizations import Visualizer
from src.explain.explainability import ExplainabilityAnalyzer
from src.explain.whatif import WhatIfAnalyzer
from src.artifacts.exporter import ModelExporter
from src.artifacts.report_gen import ReportGenerator
from src.decision.decision_mapper import DecisionMapper
from src.decision.threshold_analyzer import ThresholdAnalyzer
from src.decision.cost_analyzer import CostAnalyzer
from src.explain.counterfactual_explainer import CounterfactualExplainer
from src.decision.confidence_estimator import ConfidenceEstimator
from src.decision.data_quality_analyzer import DataQualityAnalyzer
from src.decision.decision_audit_logger import DecisionAuditLogger
from src.artifacts.model_card_generator import ModelCardGenerator


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
st.session_state.setdefault("dataset_name", None)
st.session_state.setdefault("decision_mapper", DecisionMapper())
st.session_state.setdefault("threshold_analyzer", ThresholdAnalyzer())
st.session_state.setdefault("cost_analyzer", CostAnalyzer())
st.session_state.setdefault("confidence_estimator", ConfidenceEstimator())
st.session_state.setdefault("data_quality_analyzer", DataQualityAnalyzer())
st.session_state.setdefault("audit_logger", DecisionAuditLogger())

# Business action variables
st.session_state.setdefault("positive_label", None)
st.session_state.setdefault("negative_label", None)
st.session_state.setdefault("action_positive", None)
st.session_state.setdefault("action_negative", None)
st.session_state.setdefault("selected_domain", "Custom")


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
            st.session_state.dataset_name = uploaded_file.name
            st.success(
                f"✅ Loaded {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns"
            )
    else:
        demo_file = "data/demo_business_dataset.csv"
        if os.path.exists(demo_file):
            st.session_state.df = pd.read_csv(demo_file)
            st.session_state.dataset_name = "Demo Dataset (Customer Churn)"
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "👀 Data Preview",
        "🏋️ Model Training",
        "📈 Results",
        "🧠 Explanations",
        "📋 Audit History",
        "📊 Model Cards",
    ]
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

    # Data Quality Check (Phase 5.2)
    with st.expander("🔍 Data Quality Analysis", expanded=False):
        st.info("Running data quality checks...")
        try:
            if st.session_state.training_complete and st.session_state.pipeline:
                X_train, X_test, y_test = (
                    st.session_state.pipeline.X_train,
                    st.session_state.pipeline.X_test,
                    st.session_state.pipeline.get_test_data()[1],
                )
                y_train = st.session_state.pipeline.y_train

                quality_report = st.session_state.data_quality_analyzer.generate_quality_report(
                    X_train, X_test, y_train, y_test
                )

                # Display warnings
                if quality_report["target_leakage"]["has_leakage"]:
                    st.warning("⚠️ **Target Leakage Detected!**")
                    for feat in quality_report["target_leakage"]["suspicious_features"][:3]:
                        st.write(
                            f"- {feat['feature']}: correlation={feat['correlation_with_target']:.3f}"
                        )

                if quality_report["distribution_drift"]["has_drift"]:
                    st.warning("⚠️ **Distribution Drift Detected**")
                    st.write("Model performance may degrade on new data")

                if (
                    quality_report["class_imbalance"]["severity"] == "High"
                ):
                    st.warning("⚠️ **Severe Class Imbalance**")
                    st.write(
                        f"Ratio: {quality_report['class_imbalance']['imbalance_ratio']:.1f}:1"
                    )

                # Quality score
                score = quality_report["overall_quality_score"]
                color = "🟢" if score > 75 else "🟡" if score > 50 else "🔴"
                st.write(f"{color} **Quality Score: {score:.0f}/100**")
            else:
                st.info("Train models first to enable data quality analysis")
        except Exception as e:
            st.warning(f"Data quality analysis unavailable: {str(e)}")

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
    st.subheader("🏋️ Model Training Configuration")
    
    st.markdown("""
    Configure and train machine learning models on your dataset.
    - Select your prediction target
    - Choose models to compare
    - Click Train to build and evaluate
    """)

    left, right = st.columns(2)

    with left:
        st.session_state.target_col = st.selectbox(
            "Select prediction target:",
            df.columns,
            key="target_select",
            help="The column you want to predict"
        )

    with right:
        t = st.session_state.target_col
        st.metric("Target Variable", t)
        st.metric("Unique Values", df[t].nunique())

    st.write("### Model Selection")
    
    st.markdown("""
    **Compare multiple models to find the best performer:**
    - **Logistic Regression**: Fast, interpretable, good baseline
    - **Random Forest**: Powerful ensemble, handles non-linearity well
    """)

    c1, c2 = st.columns(2)
    with c1:
        train_lr = st.checkbox(
            "Logistic Regression", 
            value=True,
            help="Fast, interpretable linear model - good baseline"
        )
    with c2:
        train_rf = st.checkbox(
            "Random Forest", 
            value=True,
            help="Powerful ensemble model - handles complex patterns"
        )

    if not (train_lr or train_rf):
        st.warning("⚠️ Please select at least one model to train")
    else:
        model_list = []
        if train_lr:
            model_list.append("logistic_regression")
        if train_rf:
            model_list.append("random_forest")

        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            # Create progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                with progress_placeholder.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                status_text.info("📊 Initializing pipeline...")
                progress_bar.progress(10)
                
                st.session_state.pipeline = MLPipeline(df, st.session_state.target_col)
                
                status_text.info("🔄 Training models (this may take a moment)...")
                progress_bar.progress(30)
                
                st.session_state.results = st.session_state.pipeline.run_full_pipeline(model_list)
                
                progress_bar.progress(90)
                
                if st.session_state.results.get("status") == "success":
                    progress_bar.progress(100)
                    st.session_state.training_complete = True
                    
                    # Clear progress indicators
                    progress_placeholder.empty()
                    
                    st.success("✅ Models trained successfully! Jump to 'Results' tab to see performance.")
                    st.balloons()
                else:
                    st.session_state.training_complete = False
                    progress_placeholder.empty()
                    error_msg = st.session_state.results.get('error', 'Unknown error')
                    st.error(f"""
                    ❌ **Training Failed**
                    
                    Error: {error_msg}
                    
                    **Troubleshooting Tips:**
                    - Ensure your target column is valid
                    - Check that features are numeric or categorical
                    - Verify you have enough data (minimum 10 rows)
                    """)
            except Exception as e:
                st.session_state.training_complete = False
                progress_placeholder.empty()
                st.error(f"""
                ❌ **Training Error**
                
                {str(e)}
                
                **Common Issues:**
                - Missing values in data (use Data Upload to handle)
                - Inconsistent data types
                - Memory issues with large datasets
                
                **Next Steps:** Try uploading cleaner data or checking the Data Quality tab
                """)


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

        # Recommend best model based on primary metric
        metric = "f1" if task_type == "classification" else "r2"
        best_model = max(eval_results.items(), key=lambda x: x[1].get(metric, 0))[0] if eval_results else None
        
        st.write("### 🏆 Model Recommendations")
        rec_cols = st.columns(len(eval_results))
        
        for col, (model_name, metrics) in zip(rec_cols, eval_results.items()):
            with col:
                model_metric_val = metrics.get(metric, 0)
                is_best = model_name == best_model
                
                # Create badge
                if is_best:
                    st.success(f"⭐ **{model_name.replace('_', ' ').title()}**")
                    st.caption("✨ **RECOMMENDED** - Best performance")
                else:
                    st.info(f"📊 **{model_name.replace('_', ' ').title()}**")
                
                # Show key metric
                st.metric(
                    metric.upper(),
                    f"{model_metric_val:.4f}",
                    delta=None,
                    help=f"{metric.upper()} score - Higher is better" if task_type == "classification" else "R² score - Higher is better"
                )

        st.write("### Metrics Summary")
        c1, c2 = st.columns(2)

        with c1:
            st.write(f"**Primary Metric: {metric.upper()}**")
            with st.spinner("Generating comparison chart..."):
                fig = Visualizer.plot_metrics_comparison(eval_results, metric)
                st.plotly_chart(fig, width='stretch')

        with c2:
            st.write("**All Metrics Overview**")
            with st.spinner("Preparing metrics table..."):
                fig = Visualizer.plot_metrics_table(eval_results)
                st.plotly_chart(fig, width='stretch')

        st.write("### Detailed Metrics")
        for model_name, metrics in eval_results.items():
            # Add badge for best model
            badge_icon = "⭐ " if model_name == best_model else "📌 "
            
            with st.expander(f"{badge_icon}{model_name.replace('_', ' ').title()}"):
                metric_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                
                # Format numeric values
                metric_df["Value"] = metric_df["Value"].apply(
                    lambda x: f"{float(x):.4f}" if isinstance(x, (int, float)) else x
                )
                
                st.dataframe(make_arrow_safe(metric_df), width="stretch")
                
                # Add interpretation
                if model_name == best_model:
                    st.success(
                        "✅ This is the recommended model based on the primary metric. "
                        "Consider deploying this model for production use."
                    )

        if task_type == "classification":
            st.write("### Confusion Matrices")
            X_test, y_test = st.session_state.pipeline.get_test_data()

            confusion_cols = st.columns(2)
            for idx, model_name in enumerate(st.session_state.pipeline.model_manager.get_models().keys()):
                with confusion_cols[idx % 2]:
                    try:
                        with st.spinner(f"Generating {model_name} confusion matrix..."):
                            y_pred = st.session_state.pipeline.get_model_predictions(model_name, X_test)
                            fig = Visualizer.plot_confusion_matrix(
                                y_test, y_pred, f"{model_name.replace('_', ' ').title()}"
                            )
                            st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate confusion matrix for {model_name}: {str(e)}")

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

        st.write("---")
        st.write("### 🎯 Decision Intelligence (Phase 5.1)")

        # Decision Intelligence Features
        di_col1, di_col2 = st.columns(2)

        # --- Threshold Control ---
        with di_col1:
            st.write("#### 📊 Threshold Control & Tradeoffs")
            st.info("💡 Adjust decision threshold to balance Precision vs Recall")

            if task_type == "classification":
                selected_model = st.selectbox(
                    "Select model for threshold analysis:",
                    list(st.session_state.pipeline.model_manager.get_models().keys()),
                    key="threshold_model",
                )

                X_test, y_test = st.session_state.pipeline.get_test_data()
                y_proba = st.session_state.pipeline.model_manager.predict_proba(
                    selected_model, X_test
                )

                threshold = st.slider(
                    "Decision Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="decision_threshold",
                )

                # Get metrics at threshold
                metrics_at_threshold = (
                    st.session_state.threshold_analyzer.get_metrics_at_threshold(
                        y_test, y_proba[:, 1], threshold
                    )
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precision", f"{metrics_at_threshold['precision']:.3f}")
                with col2:
                    st.metric("Recall", f"{metrics_at_threshold['recall']:.3f}")
                with col3:
                    st.metric("F1 Score", f"{metrics_at_threshold['f1']:.3f}")
                with col4:
                    st.metric("Accuracy", f"{metrics_at_threshold['accuracy']:.3f}")

                # Plot precision-recall curve
                pr_data = st.session_state.threshold_analyzer.get_precision_recall_curve_data(
                    y_test, y_proba[:, 1]
                )
                fig_pr = Visualizer.plot_precision_recall_curve(pr_data, threshold)
                st.plotly_chart(fig_pr, use_container_width=True)

        # --- Cost-Aware Model Selection ---
        with di_col2:
            st.write("#### 💰 Cost-Aware Model Selection")
            st.info("💡 Define FP/FN costs to find optimal model")

            if task_type == "classification":
                fp_cost = st.number_input(
                    "Cost of False Positive ($):",
                    min_value=0.0,
                    value=100.0,
                    step=10.0,
                    key="fp_cost_input",
                )
                fn_cost = st.number_input(
                    "Cost of False Negative ($):",
                    min_value=0.0,
                    value=500.0,
                    step=50.0,
                    key="fn_cost_input",
                )

                # Update cost analyzer
                st.session_state.cost_analyzer = CostAnalyzer(fp_cost=fp_cost, fn_cost=fn_cost)

                # Get predictions for all models
                X_test, y_test = st.session_state.pipeline.get_test_data()
                model_predictions = {}
                for model_name in st.session_state.pipeline.model_manager.get_models().keys():
                    y_pred = st.session_state.pipeline.get_model_predictions(model_name, X_test)
                    model_predictions[model_name] = y_pred

                # Compare costs
                cost_df = st.session_state.cost_analyzer.compare_model_costs(
                    model_predictions, y_test, fp_cost, fn_cost
                )

                # Highlight optimal model
                optimal_model = cost_df.iloc[0]["model"]
                st.success(f"✅ Optimal Model (Lowest Cost): **{optimal_model.replace('_', ' ').title()}**")
                st.write(
                    f"Expected Cost per Decision: ${cost_df.iloc[0]['average_cost_per_decision']:.2f}"
                )

                # Show comparison table
                st.write("**Cost Comparison Across Models:**")
                cost_display = cost_df[
                    ["model", "false_positives", "false_negatives", "total_cost", "average_cost_per_decision"]
                ].copy()
                cost_display.columns = ["Model", "FP", "FN", "Total Cost ($)", "Avg Cost/Decision ($)"]
                st.dataframe(make_arrow_safe(cost_display), use_container_width=True)

        # --- Counterfactual Explanations ---
        st.write("#### 🔄 Counterfactual Explanations")
        st.info("💡 What needs to change to get a different prediction?")

        if task_type == "classification":
            selected_model_cf = st.selectbox(
                "Select model for counterfactual analysis:",
                list(st.session_state.pipeline.model_manager.get_models().keys()),
                key="cf_model",
            )

            X_test, y_test = st.session_state.pipeline.get_test_data()

            # Select a test sample
            sample_idx = st.slider(
                "Select test sample:", 0, len(X_test) - 1, key="sample_idx_cf"
            )

            test_sample = X_test.iloc[sample_idx].to_dict()
            feature_names = list(test_sample.keys())

            try:
                explainer = CounterfactualExplainer(feature_names)
                model_obj = st.session_state.pipeline.model_manager.get_models()[selected_model_cf]

                cf_result = explainer.find_counterfactual(
                    test_sample, model_obj, X_test, num_features_to_change=3
                )

                st.write(f"**Current Prediction:** {cf_result['original_prediction']}")
                st.write(cf_result["explanation"])

                if cf_result["counterfactuals"]:
                    st.write("**Suggested Changes:**")
                    for i, cf in enumerate(cf_result["counterfactuals"], 1):
                        st.write(
                            f"{i}. {cf['feature']}: {cf['original_value']:.3f} → {cf['new_value']:.3f} "
                            f"(Change: {cf['change']:.3f})"
                        )

            except Exception as e:
                st.warning(f"Counterfactual analysis not available: {str(e)}")

        # --- Decision Mapper ---
        with st.expander("� Define Business Actions (Optional)", expanded=False):
            st.write("Map predictions to specific business decisions and actions.")
            
            col1, col2 = st.columns(2)
            
            # ---- Auto-detect target variable labels ----
            with col1:
                st.write("**Step 1: Define Outcome Labels**")
                target_col = st.session_state.target_col or "target"
                
                # Get unique values from the target column
                if target_col in st.session_state.df.columns:
                    unique_values = sorted(st.session_state.df[target_col].unique().astype(str).tolist())
                    
                    # Try to infer positive label
                    suggested_positive = None
                    if len(unique_values) == 2:
                        # For binary classification
                        if any(x.lower() in ['yes', '1', 'true', 'churn', 'fraud', 'positive'] for x in unique_values):
                            for val in unique_values:
                                if val.lower() in ['yes', '1', 'true', 'churn', 'fraud', 'positive']:
                                    suggested_positive = val
                                    break
                    
                    st.session_state.positive_label = st.selectbox(
                        "Positive Prediction Label:",
                        unique_values,
                        index=unique_values.index(suggested_positive) if suggested_positive else 0,
                        help="Which outcome is the 'positive' prediction? (e.g., 'Yes' for churn, '1' for fraud)"
                    )
                    
                    remaining_labels = [v for v in unique_values if v != st.session_state.positive_label]
                    if remaining_labels:
                        st.session_state.negative_label = st.selectbox(
                            "Negative Prediction Label:",
                            remaining_labels,
                            help="Which outcome is the 'negative' prediction?"
                        )
                else:
                    st.warning(f"Target column '{target_col}' not found")
            
            # ---- Domain-specific action presets ----
            with col2:
                st.write("**Step 2: Action Templates**")
                
                domain_templates = {
                    "Customer Churn": {
                        "positive": ["Send retention offer", "Assign account manager", "Offer discount", "Request feedback", "Do nothing"],
                        "negative": ["Monitor account", "Send upsell offer", "Standard engagement", "No action"]
                    },
                    "Fraud Detection": {
                        "positive": ["Block transaction", "Flag for review", "Request verification", "Approve with review"],
                        "negative": ["Approve automatically", "Monitor account", "No action"]
                    },
                    "Loan Approval": {
                        "positive": ["Approve", "Approve with conditions", "Request more info", "Manual review"],
                        "negative": ["Reject", "Request more info", "Suggest alternatives"]
                    },
                    "Custom": {
                        "positive": [],
                        "negative": []
                    }
                }
                
                domain = st.selectbox(
                    "Select Domain Template:",
                    list(domain_templates.keys()),
                    help="Pre-populated action suggestions based on common use cases"
                )
                
                st.session_state.selected_domain = domain
            
            # ---- Define custom actions ----
            st.write("**Step 3: Define Actions**")
            
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                st.write(f"**Action for {st.session_state.positive_label} (Positive):**")
                
                if st.session_state.selected_domain != "Custom":
                    preset_options = domain_templates[st.session_state.selected_domain]["positive"]
                    action_pos = st.selectbox(
                        f"Select action:",
                        preset_options + ["[Custom]"],
                        key="action_positive_preset",
                        help="Choose from recommended actions or enter custom"
                    )
                    
                    if action_pos == "[Custom]":
                        st.session_state.action_positive = st.text_area(
                            "Enter custom action:",
                            placeholder="e.g., Call customer within 24 hours",
                            key="action_positive_custom"
                        )
                    else:
                        st.session_state.action_positive = action_pos
                else:
                    st.session_state.action_positive = st.text_area(
                        "Enter action for positive prediction:",
                        placeholder="e.g., Send retention offer",
                        key="action_positive_free"
                    )
                
                if st.session_state.action_positive:
                    st.info(f"✓ {st.session_state.action_positive}")
            
            with col_neg:
                st.write(f"**Action for {st.session_state.negative_label} (Negative):**")
                
                if st.session_state.selected_domain != "Custom":
                    preset_options = domain_templates[st.session_state.selected_domain]["negative"]
                    action_neg = st.selectbox(
                        f"Select action:",
                        preset_options + ["[Custom]"],
                        key="action_negative_preset",
                        help="Choose from recommended actions or enter custom"
                    )
                    
                    if action_neg == "[Custom]":
                        st.session_state.action_negative = st.text_area(
                            "Enter custom action:",
                            placeholder="e.g., Monitor usage patterns",
                            key="action_negative_custom"
                        )
                    else:
                        st.session_state.action_negative = action_neg
                else:
                    st.session_state.action_negative = st.text_area(
                        "Enter action for negative prediction:",
                        placeholder="e.g., Monitor account",
                        key="action_negative_free"
                    )
                
                if st.session_state.action_negative:
                    st.info(f"✓ {st.session_state.action_negative}")
            
            # ---- Save actions ----
            st.divider()
            
            button_cols = st.columns([1, 3])
            with button_cols[0]:
                if st.button("💾 Save Actions", type="primary"):
                    if st.session_state.action_positive and st.session_state.action_negative:
                        # Save to decision mapper
                        outcome_name = st.session_state.target_col or "prediction"
                        st.session_state.decision_mapper.define_outcome(
                            outcome_name,
                            st.session_state.positive_label,
                            st.session_state.negative_label,
                            st.session_state.action_positive,
                            st.session_state.action_negative
                        )
                        st.success("✅ Business actions saved!")
                        st.balloons()
                    else:
                        st.error("❌ Please fill in both positive and negative actions")
            
            with button_cols[1]:
                if st.session_state.action_positive and st.session_state.action_negative:
                    st.write(f"**Summary:** {st.session_state.positive_label} → {st.session_state.action_positive} | {st.session_state.negative_label} → {st.session_state.action_negative}")


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


# ------
# Tab 5: Audit History (Phase 5.2)
# ------
with tab5:
    st.subheader("📋 Decision Audit Log")
    st.info(
        "View complete audit trail of all predictions, thresholds, confidence scores, and actions taken."
    )

    if st.session_state.training_complete:
        # Get audit statistics
        stats = st.session_state.audit_logger.get_statistics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", stats.get("total_predictions", 0))
        with col2:
            st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.3f}")
        with col3:
            st.metric("High Confidence", stats.get("high_confidence_count", 0))
        with col4:
            st.metric("Positive Preds", stats.get("positive_predictions", 0))

        # Display audit records
        if stats.get("total_predictions", 0) > 0:
            st.write("### Recent Audit Records")

            # Convert audit records to DataFrame for display
            audit_records = st.session_state.audit_logger.get_audit_trail()
            if audit_records:
                display_records = []
                for record in audit_records[-20:]:  # Last 20 records
                    if "prediction" in record:
                        display_records.append(
                            {
                                "Timestamp": record.get("timestamp", "")[:19],
                                "Model": record.get("model", ""),
                                "Prediction": record.get("prediction", ""),
                                "Confidence": f"{record.get('confidence', 0):.3f}",
                                "Threshold": record.get("threshold", ""),
                                "Action": record.get("recommended_action", "")[:50],
                            }
                        )

                if display_records:
                    st.dataframe(
                        pd.DataFrame(display_records), use_container_width=True
                    )

            # Export audit logs
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export as CSV", key="export_audit_csv"):
                    csv_file = st.session_state.audit_logger.export_as_csv()
                    st.success(f"✅ Audit log exported to: {csv_file}")

            with col2:
                if st.button("📥 Export as JSON", key="export_audit_json"):
                    json_file = st.session_state.audit_logger.save_audit_log()
                    st.success(f"✅ Audit log exported to: {json_file}")

            # Audit report
            if st.button("📊 Generate Audit Report", key="gen_audit_report"):
                report = st.session_state.audit_logger.get_audit_report()
                st.text(report)

        else:
            st.info("No predictions logged yet. Make predictions to start tracking.")

    else:
        st.info("Train models first to enable audit logging.")


# ------
# Tab 6: Model Cards (Phase 5.3)
# ------
with tab6:
    st.subheader("📊 Standardized Model Cards")
    st.info(
        "Automatically generated model documentation for governance and responsible AI."
    )

    if st.session_state.training_complete:
        # Select model to document
        selected_model = st.selectbox(
            "Select model for documentation:",
            list(st.session_state.pipeline.model_manager.get_models().keys()),
            key="model_card_select",
        )

        st.write(f"### {selected_model.replace('_', ' ').title()} Model Card")

        # Generate model card
        card = ModelCardGenerator(selected_model, version="1.0")

        # Add model details
        card.add_model_details(
            model_type=selected_model.replace("_", " ").title(),
            framework="scikit-learn",
            task_type="classification" if st.session_state.pipeline.model_manager.task_type == "classification" else "regression",
            created_date=datetime.now().isoformat(),
        )

        # Add intended use
        card.add_intended_use(
            primary_use=f"Automated {st.session_state.pipeline.model_manager.task_type} for decision support",
            primary_users=["Data Scientists", "Business Analysts", "Decision Makers"],
            out_of_scope_uses=[
                "Mission-critical decisions without human review",
                "Automated individual decisions with legal consequences",
            ],
        )

        # Add training data
        X_train, X_test = (
            st.session_state.pipeline.X_train,
            st.session_state.pipeline.X_test,
        )
        card.add_training_data(
            dataset_name=getattr(st.session_state, 'dataset_name', 'Uploaded Dataset'),
            dataset_size=len(X_train),
            features=list(X_train.columns),
            target_variable=st.session_state.target_col or "target",
            data_preprocessing=[
                "Categorical encoding (LabelEncoder)",
                "Numeric scaling (StandardScaler)",
                "Missing value handling (median/mode imputation)",
            ],
            data_splits={"train": 0.7, "test": 0.3},
        )

        # Add performance metrics
        if st.session_state.results:
            model_metrics = st.session_state.results.get(selected_model, {})
            card.add_performance_metrics(metrics=model_metrics)

        # Add limitations
        card.add_limitations(
            [
                "Model trained on historical data - may not capture future trends",
                "Performance depends on data quality and feature engineering",
                "Requires regular retraining with new data",
                "Should not be used as sole basis for high-stakes decisions",
                "May have different performance across demographic groups",
            ]
        )

        # Add ethical considerations
        card.add_ethical_considerations(
            fairness_considerations=[
                "Test for performance disparities across demographic groups",
                "Consider fairness metrics beyond overall accuracy",
            ],
            bias_mitigation=[
                "Use stratified train-test splits",
                "Monitor model performance over time",
                "Include diverse stakeholders in model development",
            ],
            privacy_measures=[
                "Data anonymization where applicable",
                "Secure model storage and access controls",
                "Regular audit logging of predictions",
            ],
        )

        # Add recommendations
        card.add_recommendations(
            recommended_actions=[
                "Start with lower-risk pilot deployments",
                "Implement human-in-the-loop review process",
                "Establish clear escalation procedures",
            ],
            monitoring_recommendations=[
                "Track prediction distributions over time",
                "Monitor for data drift",
                "Log all production predictions",
                "Monthly performance review against baseline",
            ],
        )

        # Display card summary
        summary = card.get_card_summary()
        st.write(f"**Sections Completed:** {len(summary['sections_completed'])}/7")

        # Save options
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🌐 Export as HTML", key="save_card_html", help="Generate beautiful HTML card for sharing"):
                try:
                    os.makedirs("model_cards", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    card_file = f"model_cards/card_{selected_model}_{timestamp}.html"
                    card.export_html(card_file)
                    st.success(f"✅ Model card exported: {card_file}")
                    st.info(f"📂 Open in browser: `{card_file}`")
                except Exception as e:
                    st.error(f"❌ Error exporting HTML: {str(e)}")

        with col2:
            if st.button("📋 Export as JSON", key="save_card_json", help="Export structured data for programmatic use"):
                try:
                    os.makedirs("model_cards", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    card_file = f"model_cards/card_{selected_model}_{timestamp}.json"
                    card.export_json(card_file)
                    st.success(f"✅ Card data exported: {card_file}")
                except Exception as e:
                    st.error(f"❌ Error exporting JSON: {str(e)}")

        with col3:
            if st.button("👁️ Preview Card", key="preview_card", help="View model card in application"):
                try:
                    with st.spinner("Generating preview..."):
                        html_content = card._generate_html()
                        st.markdown("##### Model Card Preview")
                        st.write("⬇️ Scroll down to see complete card")
                        components.html(html_content, height=1200, scrolling=True)
                except Exception as e:
                    st.error(f"❌ Error previewing card: {str(e)}")

        # Phase 5.3: Model Recommendations & Comparison
        st.write("---")
        st.write("### ⭐ Model Recommendations")

        # Get evaluation results
        eval_results = st.session_state.results.get("eval_results", {})
        task_type = st.session_state.results.get("task_type", "classification")
        metric = "f1" if task_type == "classification" else "r2"

        if eval_results:
            # Find best model for card
            best_model_for_card = max(eval_results.items(), key=lambda x: x[1].get(metric, 0))[0]
            
            st.success(
                f"⭐ **Recommended Model:** {best_model_for_card.replace('_', ' ').title()}\n\n"
                f"**Reason:** Highest {metric.upper()} score of {eval_results[best_model_for_card].get(metric, 0):.4f}"
            )
            
            # Show comparison
            st.write("**Performance Comparison:**")
            comparison_data = []
            for model_name, metrics in eval_results.items():
                is_recommended = model_name == best_model_for_card
                comparison_data.append({
                    "Model": ("⭐ " if is_recommended else "  ") + model_name.replace('_', ' ').title(),
                    metric.upper(): f"{metrics.get(metric, 0):.4f}",
                    "Status": "✅ RECOMMENDED" if is_recommended else "✓ Alternative"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run model training to get recommendations")

        # Phase 5.3: Card Completion Status
        st.write("---")
        st.write("### 📋 Model Card Completeness")
        
        summary = card.get_card_summary()
        completion_pct = (len(summary['sections_completed']) / 7) * 100
        
        st.progress(completion_pct / 100, text=f"{completion_pct:.0f}% Complete")
        
        col_status1, col_status2 = st.columns(2)
        
        with col_status1:
            st.write("**✅ Completed Sections:**")
            for section in summary['sections_completed']:
                st.write(f"  ✓ {section.replace('_', ' ').title()}")
        
        with col_status2:
            if summary['sections_missing']:
                st.write("**⚠️ Missing Sections:**")
                for section in summary['sections_missing']:
                    st.write(f"  ○ {section.replace('_', ' ').title()}")
            else:
                st.write("**🎉 All Sections Complete!**")
                st.success("Your model card is fully documented.")

    else:
        st.info("👈 Train models first (in 'Model Training' tab) to generate comprehensive model cards.")


# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; font-size: 12px;'>
Verdict v1.0 | Decision Copilot | Phase 5 Complete: Decision Intelligence & Governance
</div>
""",
    unsafe_allow_html=True,
)
