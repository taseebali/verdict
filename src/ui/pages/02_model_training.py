"""Streamlit Model Training Page - WORKING VERSION"""

import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import plotly.express as px

from src.ui.utils import load_demo_dataset, get_feature_statistics
from src.ui.session_manager import init_session_state
from src.ui.charts import plot_feature_importance
from src.ui.components import DataLoadingComponent, TargetColumnSelector, FeatureSelector, ModelPerformanceComponent
from src.core.ml_operations import EnsembleManager
from src.explain.explainability import ExplainabilityAnalyzer

st.set_page_config(page_title="Train Model", page_icon="🤖", layout="wide")

# Initialize session state FIRST
init_session_state()
st.title("🤖 Train Model")

# Ensure data is loaded
if st.session_state.df is None:
    st.session_state.df = load_demo_dataset()

df = st.session_state.df

# Show data info
st.markdown(f"**Dataset:** {len(df):,} rows × {len(df.columns)} columns")

# ===== STEP 1: SELECT TARGET =====
target_selector = TargetColumnSelector(df)
target_col, unique_count = target_selector.render()

if target_col is None:
    st.error("❌ Please select a target column")
    st.stop()

# ===== STEP 2: SELECT FEATURES =====
feature_selector = FeatureSelector(df, target_col)
numeric_feature_cols = feature_selector.render()

if len(numeric_feature_cols) == 0:
    st.error("❌ No numeric features available for training")
    st.stop()

# Check for class imbalance
target_counts = df[target_col].value_counts()
min_class = target_counts.min()
max_class = target_counts.max()
imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

if imbalance_ratio > 3:
    st.warning(f"⚠️ **Class Imbalance Detected:** Ratio {imbalance_ratio:.1f}:1. Majority class has {imbalance_ratio:.1f}x more samples. Consider: using class weights, oversampling minority, or collecting more balanced data.")
    with st.expander("📊 Class Distribution"):
        for val, count in target_counts.items():
            pct = count / len(df) * 100
            st.write(f"  • {val}: {count:,} ({pct:.1f}%)")

# ===== STEP 3: TRAINING PARAMETERS =====
st.markdown("## 3️⃣ Training Settings")

col1, col2, col3, col4 = st.columns(4)

with col1:
    test_size = st.slider("Test size %:", 10, 40, 20) / 100
with col2:
    n_estimators = st.slider("Number of trees:", 10, 200, 100)
with col3:
    max_depth = st.slider("Tree depth:", 2, 20, 10)
with col4:
    use_cv = st.checkbox("Use K-Fold CV", value=False, help="Cross-validation for more robust evaluation")

if use_cv:
    n_folds = st.slider("Number of folds:", 3, 10, 5)

# ===== STEP 3B: SELECT ENSEMBLE METHOD =====
st.markdown("## 3️⃣B Select Training Method")

col_method1, col_method2 = st.columns([3, 1])

with col_method1:
    ensemble_manager = EnsembleManager()
    available_methods = ensemble_manager.available_methods
    
    method_descriptions = {
        'random_forest': '🌲 Random Forest (baseline, fast, interpretable)',
        'xgboost': '⚡ XGBoost (fast, accurate, gradient boosting)',
        'lightgbm': '💡 LightGBM (very fast, memory efficient)',
        'voting': '🗳️ Voting Ensemble (combines multiple models)',
        'stacking': '📚 Stacking (meta-learner approach, more complex)'
    }
    
    method_display = [method_descriptions.get(m, m) for m in available_methods if m in method_descriptions]
    selected_method_display = st.selectbox(
        "Select training method:",
        method_display,
        help="Different methods have different speed/accuracy tradeoffs"
    )
    # Extract method name from display string
    training_method = 'random_forest'
    for method in available_methods:
        if method_descriptions.get(method, '') == selected_method_display:
            training_method = method
            break

with col_method2:
    use_ensemble_tuning = st.checkbox("Tune params", value=False, help="Hyperparameter tuning for gradient boosting")

# ===== STEP 4: TRAIN =====
st.markdown("## 4️⃣ Train Model")

if st.button("🚀 TRAIN MODEL", type="primary", width='stretch'):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Prepare data
        status_text.write("📊 **Step 1/4:** Preparing data...")
        progress_bar.progress(25)
        
        # Prepare data - start with selected features
        X = df[numeric_feature_cols].copy()
        y = df[target_col].copy()
        
        # Encode categorical features in X
        from sklearn.preprocessing import LabelEncoder as LE
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':  # If categorical
                le = LE()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                st.info(f"✅ Encoded '{col}': {len(le.classes_)} unique values")
        
        # Encode target if needed
        if y.dtype == 'object':
            le = LE()
            y = le.fit_transform(y)
            st.session_state.label_encoder = le
            st.session_state.label_encoders = label_encoders
        
        # Step 2: Split data
        status_text.write("✂️ **Step 2/4:** Splitting train/test...")
        progress_bar.progress(50)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        # Step 3: Train model
        status_text.write("🤖 **Step 3/4:** Training model...")
        progress_bar.progress(75)
        
        # Train using selected method
        if training_method == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        else:
            # Use EnsembleManager for other methods
            try:
                ensemble_result = None
                if use_ensemble_tuning and training_method in ['xgboost', 'lightgbm']:
                    ensemble_result = ensemble_manager.train_with_tuning(
                        training_method,
                        X_train, X_test, y_train, y_test,
                        cv=3,
                        param_grid_preset='balanced'
                    )
                elif training_method == 'xgboost':
                    ensemble_result = ensemble_manager.train_xgboost(X_train, X_test, y_train, y_test)
                elif training_method == 'lightgbm':
                    ensemble_result = ensemble_manager.train_lightgbm(X_train, X_test, y_train, y_test)
                elif training_method == 'voting':
                    ensemble_result = ensemble_manager.train_voting(X_train, X_test, y_train, y_test)
                elif training_method == 'stacking':
                    ensemble_result = ensemble_manager.train_stacking(X_train, X_test, y_train, y_test)
                
                if ensemble_result:
                    model = ensemble_result.model
                    train_acc = ensemble_result.train_score
                    test_acc = ensemble_result.test_score
                    test_precision = ensemble_result.precision
                    test_recall = ensemble_result.recall
                    test_f1 = ensemble_result.f1
                    # Skip to evaluation step
                    progress_bar.progress(95)
                    status_text.write("📈 **Step 4/4:** Evaluating performance...")
                    
                    cv_scores = ensemble_result.cv_scores
                    y_pred = model.predict(X_test)
                    
                    # Store ensemble info
                    st.session_state.ensemble_method = training_method
                    st.session_state.ensemble_params = ensemble_result.params
                    
                    # Skip standard evaluation since ensemble already did it
                    progress_bar.progress(100)
                    status_text.write("✅ **Complete:** Model trained successfully!")
            except Exception as e:
                st.error(f"Ensemble training failed: {str(e)}")
                st.info("💡 Falling back to Random Forest...")
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                ensemble_result = None  # Reset ensemble result for standard evaluation
        
        # Step 4: Evaluate (skip if ensemble already computed)
        if 'train_acc' not in locals():  # Only evaluate if not already done by ensemble
            status_text.write("📈 **Step 4/4:** Evaluating performance...")
            progress_bar.progress(95)
            
            # Evaluate
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            y_pred = model.predict(X_test)
            test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation if enabled
            cv_scores = None
            if use_cv:
                from sklearn.model_selection import cross_validate
                status_text.write(f"🔄 **Bonus:** Running {n_folds}-fold cross-validation...")
                cv_results = cross_validate(
                    model, X, y,
                    cv=n_folds,
                    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                    n_jobs=-1
                )
                cv_scores = {
                    'accuracy': (cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std()),
                    'precision': (cv_results['test_precision_weighted'].mean(), cv_results['test_precision_weighted'].std()),
                    'recall': (cv_results['test_recall_weighted'].mean(), cv_results['test_recall_weighted'].std()),
                    'f1': (cv_results['test_f1_weighted'].mean(), cv_results['test_f1_weighted'].std())
                }
            
            progress_bar.progress(100)
            status_text.write("✅ **Complete:** Model trained successfully!")
        
        # Save to session
        st.session_state.trained_model = model
        st.session_state.model_features = list(X.columns)  # Store actual trained columns (after encoding)
        st.session_state.selected_features = numeric_feature_cols  # Store original feature selection
        st.session_state.label_encoders = label_encoders  # Store encoders for prediction
        st.session_state.target_column = target_col
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.train_acc = train_acc
        st.session_state.test_acc = test_acc
        st.session_state.test_precision = test_precision
        st.session_state.test_recall = test_recall
        st.session_state.test_f1 = test_f1
        st.session_state.cv_scores = cv_scores  # Store CV results if available
        
        # Detect overfitting
        overfit_gap = train_acc - test_acc
        if overfit_gap > 0.15:
            st.warning(f"⚠️ **Possible Overfitting Detected!** Train accuracy ({train_acc:.1%}) is {overfit_gap:.1%} higher than test accuracy ({test_acc:.1%}). Consider: reducing max_depth, adding more data, or using regularization.")
        elif overfit_gap > 0.10:
            st.info(f"ℹ️ **Moderate gap** between train ({train_acc:.1%}) and test ({test_acc:.1%}) accuracy. Model may benefit from tuning.")
        
        # Store lightweight model metadata (NOT the full model object to save memory)
        if 'trained_models_history' not in st.session_state:
            st.session_state.trained_models_history = []
        
        st.session_state.trained_models_history.append({
            'timestamp': pd.Timestamp.now(),
            'features': numeric_feature_cols,
            'target': target_col,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'test_size': test_size,
            'overfit_gap': overfit_gap,
            'cv_scores': cv_scores,  # Include CV results in history
        })
        
        st.success("✅ Model trained successfully!")
        
        
    except ValueError as e:
        if "stratify" in str(e).lower():
            st.error("❌ Not enough samples for stratified split")
            st.info("💡 Try: Increase test_size % or ensure you have at least 2 samples per class")
        elif "no numeric features" in str(e).lower():
            st.error("❌ No numeric features available for training")
            st.info("💡 Try: Exclude fewer columns or include numeric columns")
        else:
            st.error(f"❌ Data issue: {str(e)}")
    except Exception as e:
        error_msg = str(e).lower()
        if "feature" in error_msg:
            st.error("❌ Feature mismatch in training")
            st.info("💡 Try: Ensure all selected features have valid numeric values")
        elif "memory" in error_msg:
            st.error("❌ Out of memory during training")
            st.info("💡 Try: Use fewer features or reduce dataset size")
        else:
            st.error(f"❌ Training failed: {str(e)}")

# ===== DISPLAY RESULTS =====
if st.session_state.trained_model is not None:
    st.markdown("## ✨ Comprehensive Training Results")
    
    # Use ModelPerformanceComponent for results display
    metrics_dict = {
        "Test Accuracy": st.session_state.test_acc,
        "Precision": st.session_state.test_precision,
        "Recall": st.session_state.test_recall,
        "F1-Score": st.session_state.test_f1
    }
    perf_component = ModelPerformanceComponent(metrics_dict)
    perf_component.render()
    
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Model", "🏆 Best Model", "📈 Model Comparison", "📋 Model History"])
    
    # ===== TAB 1: CURRENT MODEL =====
    with tab1:
        st.markdown("### Current Model Performance")
        
        # Detailed metrics in a nice layout
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**Classification Metrics**")
            metrics_data = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Train (%)": [st.session_state.train_acc * 100, None, None, None],
                "Test (%)": [
                    st.session_state.test_acc * 100,
                    st.session_state.test_precision * 100,
                    st.session_state.test_recall * 100,
                    st.session_state.test_f1 * 100
                ]
            })
            st.dataframe(metrics_data, hide_index=True, width='stretch')
        
        with col_m2:
            st.markdown("**Model Configuration**")
            config_data = pd.DataFrame({
                "Parameter": ["Number of Trees", "Max Depth", "Test Size (%)", "Features"],
                "Value": [
                    n_estimators,
                    max_depth,
                    int(test_size*100),
                    len(st.session_state.model_features)
                ]
            })
            st.dataframe(config_data, hide_index=True, width='stretch')
        
        # Display CV results if available
        if st.session_state.get('cv_scores') is not None:
            st.markdown("---")
            st.markdown("**Cross-Validation Results**")
            cv_scores = st.session_state.cv_scores
            cv_data = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Mean": [
                    f"{cv_scores['accuracy'][0]:.2%}",
                    f"{cv_scores['precision'][0]:.2%}",
                    f"{cv_scores['recall'][0]:.2%}",
                    f"{cv_scores['f1'][0]:.2%}"
                ],
                "Std Dev": [
                    f"± {cv_scores['accuracy'][1]:.2%}",
                    f"± {cv_scores['precision'][1]:.2%}",
                    f"± {cv_scores['recall'][1]:.2%}",
                    f"± {cv_scores['f1'][1]:.2%}"
                ]
            })
            st.dataframe(cv_data, hide_index=True, width='stretch')
            st.caption("Cross-validation provides a more robust estimate of model performance across different data splits.")
        
        st.markdown("### Feature Importance")
        # Permutation importance works for any model (tree, linear, ensemble),
        # unlike .feature_importances_ which only tree-based models expose.
        try:
            analyzer = ExplainabilityAnalyzer(
                st.session_state.trained_model,
                st.session_state.get('X_train', st.session_state.X_test),
                st.session_state.X_test,
                st.session_state.model_features,
            )
            importance_dict = analyzer.get_feature_importance(
                use_cache=False, y_test=st.session_state.y_test
            )
        except Exception:
            importance_dict = None

        if importance_dict:
            plot_feature_importance(importance_dict)
            st.caption("Permutation importance: drop in accuracy when a feature's values are shuffled, measured against true test labels.")
        elif hasattr(st.session_state.trained_model, 'feature_importances_'):
            importance_dict = dict(zip(
                st.session_state.model_features,
                st.session_state.trained_model.feature_importances_
            ))
            plot_feature_importance(importance_dict)
        else:
            st.info("ℹ️ Feature importance unavailable for this model.")
    
    # ===== TAB 2: BEST MODEL =====
    with tab2:
        st.markdown("### 🏆 Best Performing Model")
        
        try:
            history = st.session_state.get('trained_models_history', [])
            
            if history and len(history) > 0:
                # Find best model by F1 score
                best_idx = 0
                best_f1 = history[0].get('f1', 0)
                
                for i, m in enumerate(history):
                    if m.get('f1', 0) > best_f1:
                        best_f1 = m.get('f1', 0)
                        best_idx = i
                
                best_model = history[best_idx]
                
                col_best1, col_best2, col_best3, col_best4 = st.columns(4)
                with col_best1:
                    st.metric("🥇 Best F1-Score", f"{best_model.get('f1', 0):.2%}")
                with col_best2:
                    st.metric("Accuracy", f"{best_model.get('test_acc', 0):.2%}")
                with col_best3:
                    st.metric("Precision", f"{best_model.get('precision', 0):.2%}")
                with col_best4:
                    st.metric("Recall", f"{best_model.get('recall', 0):.2%}")
                
                st.success(f"""
                ✅ **Recommendation**: Model #{best_idx + 1} is the best performer
                
                **Why?** This model achieves the highest F1-Score ({best_model.get('f1', 0):.2%}), 
                providing the best balance between precision and recall.
                
                **Config**: {best_model.get('n_estimators', 'N/A')} trees, max depth {best_model.get('max_depth', 'N/A')}, 
                test size {int(best_model.get('test_size', 0.2)*100)}%
                """)
            else:
                st.info("ℹ️ Train at least one model to see recommendations")
        except Exception as e:
            st.error(f"Error displaying best model: {str(e)}")
    
    # ===== TAB 3: MODEL COMPARISON =====
    with tab3:
        st.markdown("### 📈 All Models Comparison")
        
        try:
            history = st.session_state.get('trained_models_history', [])
            
            if history and len(history) > 1:
                # Create comparison dataframe
                comparison_data = []
                for i, m in enumerate(history):
                    comparison_data.append({
                        "Model #": i+1,
                        "Test Acc": f"{m.get('test_acc', 0):.2%}",
                        "Precision": f"{m.get('precision', 0):.2%}",
                        "Recall": f"{m.get('recall', 0):.2%}",
                        "F1-Score": f"{m.get('f1', 0):.2%}",
                        "Trees": m.get('n_estimators', 'N/A'),
                        "Depth": m.get('max_depth', 'N/A'),
                        "Time": m.get('timestamp', 'N/A')
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, hide_index=True, width='stretch')
                
                # Visualization - Performance metrics over time
                st.markdown("**Performance Trend**")
                
                trend_data = {
                    "Model #": [i+1 for i in range(len(history))],
                    "F1-Score": [m.get('f1', 0) for m in history],
                    "Accuracy": [m.get('test_acc', 0) for m in history],
                    "Precision": [m.get('precision', 0) for m in history],
                    "Recall": [m.get('recall', 0) for m in history]
                }
                
                trend_df = pd.DataFrame(trend_data)
                
                fig = px.line(trend_df, x="Model #", y=["F1-Score", "Accuracy", "Precision", "Recall"], 
                             markers=True, title="Model Performance Trend",
                             labels={"value": "Score", "variable": "Metric"})
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("💡 Train another model with different parameters to compare")
        except Exception as e:
            st.error(f"Error displaying comparison: {str(e)}")
    
    # ===== TAB 4: MODEL HISTORY =====
    with tab4:
        st.markdown("### 📋 Training History")
        
        try:
            history = st.session_state.get('trained_models_history', [])
            
            if history and len(history) > 0:
                history_data = []
                for i, m in enumerate(history):
                    history_data.append({
                        "Model": f"#{i+1}",
                        "Test Accuracy": f"{m.get('test_acc', 0):.2%}",
                        "Precision": f"{m.get('precision', 0):.2%}",
                        "Recall": f"{m.get('recall', 0):.2%}",
                        "F1-Score": f"{m.get('f1', 0):.2%}",
                        "Trees": m.get('n_estimators', 'N/A'),
                        "Depth": m.get('max_depth', 'N/A'),
                        "Test Split": f"{int(m.get('test_size', 0.2)*100)}%",
                        "Time": str(m.get('timestamp', 'N/A'))[:19]
                    })
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, hide_index=True, width='stretch')
                
                col_h1, col_h2, col_h3 = st.columns(3)
                with col_h1:
                    st.metric("Total Models", len(history))
                with col_h2:
                    f1_scores = [m.get('f1', 0) for m in history]
                    avg_f1 = np.mean(f1_scores) if f1_scores else 0
                    st.metric("Average F1-Score", f"{avg_f1:.2%}")
                with col_h3:
                    best_f1 = max([m.get('f1', 0) for m in history]) if history else 0
                    st.metric("Best F1-Score", f"{best_f1:.2%}")
            else:
                st.info("No training history yet")
        except Exception as e:
            st.error(f"Error displaying history: {str(e)}")
    
    st.markdown("---")
    st.markdown("✅ **Model is ready!** Go to 🎯 Predictions page to make predictions or 📋 Audit & Registry to save models.")
else:
    st.info("⏳ Train a model to see results here")
