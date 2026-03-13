"""Streamlit Audit Logs Page"""

import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from src.ui.session_manager import (
    init_session_state, get_audit_trail_df, get_model_registry,
    load_model_from_registry, save_model_to_registry
)
from src.ui.charts import plot_feature_importance
from src.ui.components import AuditTrailComponent, ModelPerformanceComponent
from src.core.ml_operations import DriftDetector

# Configure page FIRST
st.set_page_config(
    page_title="Audit & Registry",
    page_icon="📋",
    layout="wide"
)

# Initialize session state ONCE
init_session_state()

st.title("📋 Model & Prediction Audit")

# ===== MODEL REGISTRY =====
st.markdown("## 🏪 Model Registry & Comparison")

model_registry = get_model_registry()

if len(model_registry) > 0:
    col_reg1, col_reg2, col_reg3 = st.columns([2, 1, 1])
    with col_reg1:
        selected_model = st.selectbox("Select model to load:", list(model_registry.keys()), key="model_selector")
    with col_reg2:
        if st.button("📂 Load Model", key="load_model_btn"):
            if load_model_from_registry(selected_model):
                st.success(f"Loaded model: {selected_model}")
                st.rerun()
    with col_reg3:
        if st.button("🗑️ Delete", key="delete_model_btn"):
            del st.session_state.model_registry[selected_model]
            st.success(f"Deleted model: {selected_model}")
            st.rerun()
    
    # Show all models comparison table
    st.markdown("### 📊 All Saved Models")
    model_comparison = pd.DataFrame([
        {
            "Model Name": name,
            "Train Accuracy": f"{data['train_acc']:.2%}",
            "Test Accuracy": f"{data['test_acc']:.2%}",
            "Features": len(data['features']),
            "Saved At": data['timestamp']
        }
        for name, data in model_registry.items()
    ])
    st.dataframe(model_comparison, width='stretch')
    st.markdown("---")

# Save current model button
if st.session_state.trained_model is not None:
    st.markdown("### 💾 Save Current Model")
    col_save1, col_save2 = st.columns([3, 1])
    with col_save1:
        model_save_name = st.text_input("Model name:", value=f"Model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}", key="save_model_name")
    with col_save2:
        if st.button("Save to Registry", key="save_model_btn", type="primary"):
            save_model_to_registry(model_save_name)
    st.markdown("---")

# ===== MODEL INFO =====
if st.session_state.trained_model is not None:
    st.markdown("## ✅ Current Trained Model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Features Used", len(st.session_state.model_features))
    with col2:
        st.metric("Target", st.session_state.target_column)
    with col3:
        st.metric("Train Accuracy", f"{st.session_state.train_acc:.2%}")
    with col4:
        st.metric("Test Accuracy", f"{st.session_state.test_acc:.2%}")
    
    # Use ModelPerformanceComponent for detailed metrics
    metrics_dict = {
        "Test Accuracy": st.session_state.test_acc,
        "Precision": st.session_state.test_precision,
        "Recall": st.session_state.test_recall,
        "F1-Score": st.session_state.test_f1
    }
    st.markdown("---")
    st.markdown("### 📊 Detailed Metrics")
    perf_component = ModelPerformanceComponent(metrics_dict)
    perf_component.render()
    
    st.markdown("### Feature Importance")
    if hasattr(st.session_state.trained_model, 'feature_importances_'):
        importance_dict = dict(zip(
            st.session_state.model_features,
            st.session_state.trained_model.feature_importances_
        ))
        plot_feature_importance(importance_dict)
    
else:
    st.warning("⏳ No trained model yet. Train a model in the Model Training page.")

st.markdown("---")

# ===== PREDICTION HISTORY =====
st.markdown("## 📊 Prediction History")

records = st.session_state.audit_trail

if len(records) > 0:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", len(records))
    
    with col2:
        # Safely calculate average confidence, handling non-numeric values
        valid_confidences = []
        for r in records:
            try:
                conf = r.get("confidence", 0)
                if isinstance(conf, (int, float)):
                    valid_confidences.append(float(conf))
                elif isinstance(conf, str):
                    valid_confidences.append(float(conf))
            except (ValueError, TypeError):
                # Skip invalid confidence values
                continue
        
        if valid_confidences:
            avg_confidence = sum(valid_confidences) / len(valid_confidences)
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col3:
        success_count = sum(1 for r in records if r.get("status") == "success")
        st.metric("✅ Successful", success_count)
    
    st.markdown("### 📋 Recent Predictions")
    
    # Initialize page state
    if 'audit_page' not in st.session_state:
        st.session_state.audit_page = 0
    
    # Pagination settings
    items_per_page = 50
    total_records = len(records)
    total_pages = (total_records + items_per_page - 1) // items_per_page  # Ceiling division
    
    # Calculate slice indices
    start_idx = st.session_state.audit_page * items_per_page
    end_idx = min(start_idx + items_per_page, total_records)
    
    # Display current page info
    col_pg1, col_pg2, col_pg3 = st.columns([2, 1, 2])
    with col_pg1:
        if st.button("← Previous", disabled=st.session_state.audit_page == 0):
            st.session_state.audit_page -= 1
            st.rerun()
    with col_pg2:
        st.write(f"**Page {st.session_state.audit_page + 1} of {total_pages}**")
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_records}")
    with col_pg3:
        if st.button("Next →", disabled=st.session_state.audit_page >= total_pages - 1):
            st.session_state.audit_page += 1
            st.rerun()
    
    # Display paginated data
    audit_df = pd.DataFrame([
        {
            "Timestamp": r.get("timestamp", "-"),
            "Model": r.get("model_name", "-"),
            "Prediction": r.get("prediction", "-"),
            "Confidence": f"{float(r.get('confidence', 0)):.1%}",
            "Status": r.get("status", "-")
        }
        for r in records[start_idx:end_idx]
    ])
    
    st.dataframe(audit_df, width='stretch')
    
    # Export audit trail
    from src.ui.utils import export_audit_trail_to_csv
    csv_data = export_audit_trail_to_csv(records)
    st.download_button(
        label="📥 Download Audit Trail as CSV",
        data=csv_data,
        file_name=f"audit_trail_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="download_audit"
    )
    
    # ===== STATISTICS & TRENDS =====
    st.markdown("## 📊 Analysis")
    
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        st.markdown("### 🤖 Models Used")
        models = {}
        for r in records:
            model = r.get("model_name", "unknown")
            models[model] = models.get(model, 0) + 1
        
        for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
            st.write(f"  • **{model}**: {count} predictions")
    
    with col_analysis2:
        st.markdown("### 📊 Status Distribution")
        statuses = {}
        for r in records:
            status = r.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        
        for status, count in statuses.items():
            emoji = "✅" if status == "success" else "❌"
            st.write(f"  {emoji} **{status}**: {count}")

else:
    st.info("⏳ No predictions made yet. Go to Predictions page and make some predictions!")

# ===== DRIFT ANALYSIS =====
st.markdown("---")
st.markdown("## 📊 Drift Detection Analysis")

if st.session_state.trained_model is not None and st.session_state.get('X_train') is not None:
    detector = DriftDetector()
    
    col_drift1, col_drift2 = st.columns([2, 1])
    
    with col_drift1:
        st.markdown("### Feature Drift Detection")
        st.caption("Compares training data distribution with test/current data distribution")
        
        # Check if we have test data
        if st.session_state.X_test is not None and len(st.session_state.X_test) > 0:
            
            # Detect drift for all features
            try:
                drift_results = detector.detect_feature_drift(
                    train_df=st.session_state.X_train,
                    test_df=st.session_state.X_test,
                    feature_columns=st.session_state.model_features,
                    categorical_features=[]  # Customize based on your data
                )
                
                # Overall drift assessment
                overall_drift = detector.assess_overall_drift(drift_results)
                
                # Display overall drift status
                col_overall1, col_overall2, col_overall3 = st.columns(3)
                
                with col_overall1:
                    if overall_drift.overall_drift_detected:
                        st.error(f"⚠️ Drift Detected: {overall_drift.drift_percentage*100:.1f}%")
                    else:
                        st.success(f"✅ No Drift: {overall_drift.drift_percentage*100:.1f}%")
                
                with col_overall2:
                    severity_emoji = {"none": "✅", "low": "⚠️", "medium": "🔴", "high": "🛑"}
                    emoji = severity_emoji.get(overall_drift.severity, "❓")
                    st.metric(f"{emoji} Overall Severity", overall_drift.severity.upper())
                
                with col_overall3:
                    st.metric("Features Drifted", f"{overall_drift.num_features_drifted}/{overall_drift.total_features_checked}")
                
                # Create drift summary table
                drift_summary_df = detector.get_drift_summary_dataframe(drift_results)
                
                # Format the dataframe for display
                display_df = drift_summary_df.copy()
                display_df['Drift Detected'] = display_df['Drift Detected'].map({True: '🔴 Yes', False: '✅ No'})
                display_df['Severity'] = display_df['Severity'].map({
                    'none': '✅ None',
                    'low': '⚠️ Low',
                    'medium': '🔴 Medium',
                    'high': '🛑 High'
                })
                
                st.dataframe(display_df, width='stretch')
                
                # Detailed feature analysis
                with st.expander("📋 Detailed Feature Analysis"):
                    for i, result in enumerate(drift_results):
                        st.markdown(f"**{result.feature}** ({result.feature_type})")
                        col_d1, col_d2, col_d3 = st.columns(3)
                        
                        with col_d1:
                            st.metric("Statistic", f"{result.statistic:.4f}")
                        with col_d2:
                            st.metric("P-Value", f"{result.p_value:.4f}")
                        with col_d3:
                            st.metric("Threshold", f"{result.threshold:.4f}")
                        
                        st.caption(result.description)
                        
                        # Show distributions if available
                        if result.train_dist and result.test_dist:
                            col_dist1, col_dist2 = st.columns(2)
                            with col_dist1:
                                st.markdown("**Training Distribution**")
                                if isinstance(result.train_dist, dict) and 'mean' in result.train_dist:
                                    st.write(f"Mean: {result.train_dist.get('mean', 'N/A'):.4f}")
                                    st.write(f"Std: {result.train_dist.get('std', 'N/A'):.4f}")
                                else:
                                    st.write(result.train_dist)
                            with col_dist2:
                                st.markdown("**Test Distribution**")
                                if isinstance(result.test_dist, dict) and 'mean' in result.test_dist:
                                    st.write(f"Mean: {result.test_dist.get('mean', 'N/A'):.4f}")
                                    st.write(f"Std: {result.test_dist.get('std', 'N/A'):.4f}")
                                else:
                                    st.write(result.test_dist)
                        
                        st.divider()
                
            except Exception as e:
                st.error(f"Error in drift detection: {str(e)}")
        else:
            st.warning("⏳ No test data available. Train a model with test data to enable drift analysis.")
    
    with col_drift2:
        # Model performance drift
        st.markdown("### Model Performance")
        st.caption("Monitor for model degradation")
        
        if hasattr(st.session_state, 'train_scores') and hasattr(st.session_state, 'test_scores'):
            try:
                perf_drift_detected, degradation_ratio, perf_description = detector.detect_model_performance_drift(
                    train_scores=st.session_state.train_scores,
                    test_scores=st.session_state.test_scores,
                    metric_name="accuracy",
                    performance_threshold=0.05
                )
                
                if perf_drift_detected:
                    st.error(f"⚠️ Performance Degradation")
                    st.metric("Degradation", f"{degradation_ratio*100:.1f}%")
                else:
                    st.success(f"✅ Performance Stable")
                    st.metric("Degradation", f"{degradation_ratio*100:.1f}%")
                
                st.caption(perf_description)
            except Exception as e:
                st.info("Train/test scores not available for performance drift analysis")
        else:
            st.info("Train and test scores needed for performance drift analysis")

else:
    st.warning("⏳ Train a model with test data to enable drift analysis.")

# ===== REFRESH =====
st.markdown("---")
if st.button("🔄 Refresh Data", width='stretch'):
    st.rerun()

st.markdown("""
### ℹ️ About Audit Logs
- All predictions are automatically logged
- Logs help track model performance over time
- Use this to verify model behavior
- Check for anomalies or unexpected patterns
""")