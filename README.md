# ⚖️ Verdict

**Business-oriented AutoML with decision intelligence, explainability, and governance built-in.**

Verdict is an end-to-end machine learning platform that transforms CSV data into production-ready decision systems. It automates preprocessing, trains multiple models, explains predictions with SHAP, maps outputs to business actions, and generates audit trails for governance.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.46-green.svg)](https://github.com/slundberg/shap)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Or with Docker:
```bash
docker-compose up --build
```

Visit `http://localhost:8501` to access the UI.

---

## Architecture Overview 

Verdict is structured as a modular pipeline with four core subsystems:

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  MLPipeline (src/core/pipeline.py)                          │
│  ├─ DataHandler: Validation, missing values                │
│  ├─ Preprocessor: Encoding, scaling, train/test split      │
│  ├─ ModelManager: Training, prediction                     │
│  └─ MetricsCalculator: Performance evaluation              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Explainability Layer (src/explain/)                        │
│  ├─ ExplainabilityAnalyzer: SHAP, permutation importance   │
│  ├─ WhatIfAnalyzer: Scenario simulation, sensitivity       │
│  └─ CounterfactualExplainer: Minimal changes to flip pred  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Decision Intelligence (src/decision/)                      │
│  ├─ DecisionMapper: Predictions → Business actions         │
│  ├─ ThresholdAnalyzer: Precision/recall tradeoffs          │
│  ├─ CostAnalyzer: FP/FN cost-aware model selection         │
│  ├─ ConfidenceEstimator: Uncertainty quantification        │
│  ├─ DataQualityAnalyzer: Leakage, drift detection          │
│  └─ DecisionAuditLogger: Complete prediction audit trail   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Artifacts & Export (src/artifacts/)                        │
│  ├─ ModelExporter: Joblib serialization with metadata      │
│  ├─ ReportGenerator: HTML reports with all metrics         │
│  └─ ModelCardGenerator: Governance documentation           │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. The ML Pipeline

The `MLPipeline` class in [src/core/pipeline.py](src/core/pipeline.py) orchestrates the entire workflow:

```python
from src.core.pipeline import MLPipeline

# Initialize with dataframe and target column
pipeline = MLPipeline(df, target_col="churn")

# Validate data (checks for missing target, empty data, etc.)
is_valid, message = pipeline.validate()

# Preprocess: handles missing values, encodes categoricals, scales numerics
pipeline.preprocess()
# Creates: X_train, X_test, y_train, y_test (80/20 split by default)

# Train models (defaults to both logistic regression and random forest)
train_results = pipeline.train(model_names=["logistic_regression", "random_forest"])

# Evaluate on test set
eval_results = pipeline.evaluate()
# Returns: {"logistic_regression": {"accuracy": 0.85, ...}, ...}

# Or run everything at once
results = pipeline.run_full_pipeline()
```

**Under the hood:**
- **DataHandler** ([src/core/data_handler.py](src/core/data_handler.py)): Validates schema, handles missing values with mean/mode imputation, provides column type detection
- **Preprocessor** ([src/core/preprocessing.py](src/core/preprocessing.py)): LabelEncoder for categoricals, StandardScaler for numerics, automatic train/test split
- **ModelManager** ([src/core/models.py](src/core/models.py)): Wraps scikit-learn models with unified interface, maintains model registry
- **MetricsCalculator** ([src/core/metrics.py](src/core/metrics.py)): Computes accuracy, precision, recall, F1, ROC-AUC (classification) or R², MAE, RMSE (regression)

### 2. Explainability System

The `ExplainabilityAnalyzer` uses SHAP to explain predictions:

```python
from src.explain.explainability import ExplainabilityAnalyzer

# Initialize with trained model and data
analyzer = ExplainabilityAnalyzer(
    model=pipeline.get_model("random_forest"),
    X_train=pipeline.X_train,
    X_test=pipeline.X_test,
    feature_names=pipeline.get_feature_names()
)

# Global feature importance (permutation-based)
importance = analyzer.get_feature_importance_permutation(
    y_test=pipeline.y_test,
    n_repeats=10
)
# Returns: {"tenure": 0.23, "monthly_charges": 0.18, ...}

# SHAP values for all test samples
shap_values = analyzer.get_global_shap_values()
# Returns: numpy array of shape (n_samples, n_features)

# Explain individual prediction
explanation = analyzer.explain_prediction(
    sample_idx=0,
    method="shap"  # or "lime" (not implemented yet)
)
# Returns: feature contributions for that prediction
```

**SHAP Explainer Selection Logic** ([src/explain/explainability.py](src/explain/explainability.py#L86)):
```python
def _init_explainer(self):
    if self._is_tree_model():
        # Fast TreeExplainer for RandomForest, XGBoost
        self.explainer = shap.TreeExplainer(self.model)
    elif self._is_linear_model():
        # LinearExplainer for LogisticRegression
        bg = self._background_sample(self.X_train, n=200)
        self.explainer = shap.LinearExplainer(self.model, bg)
    else:
        # KernelExplainer fallback (slow, model-agnostic)
        bg = self._background_sample(self.X_train, n=50)
        self.explainer = shap.KernelExplainer(self._predict_fn(), bg)
```

### 3. What-If Analysis

The `WhatIfAnalyzer` enables scenario simulation:

```python
from src.explain.whatif import WhatIfAnalyzer

whatif = WhatIfAnalyzer(
    pipeline=pipeline,
    feature_names=pipeline.get_feature_names(),
    numeric_cols=pipeline.get_numeric_columns(),
    categorical_cols=pipeline.get_categorical_columns(),
    label_encoders=pipeline.preprocessor.get_label_encoders()
)

# Simulate a custom scenario
scenario = {
    "age": 35,
    "tenure": 24,
    "monthly_charges": 75.0,
    "contract_type": "Month-to-month"
}
result = whatif.predict_scenario("random_forest", scenario)
# Returns: {"prediction": 1, "probability": 0.73, "confidence": "High"}

# Sensitivity analysis: vary one feature, hold others constant
sensitivity_df = whatif.get_sensitivity_analysis(
    model_name="random_forest",
    base_input=scenario,
    feature_to_vary="monthly_charges",
    values_to_test=np.linspace(20, 150, 20)
)
# Returns: DataFrame with columns [monthly_charges, prediction, probability]
```

**Key Implementation Detail**: The analyzer handles both scaled (model input) and original (user-facing) feature spaces by maintaining mappings via label encoders and scalers.

### 4. Decision Intelligence Layer

This is where predictions become **business decisions**:

#### Decision Mapping

Map ML predictions to real-world actions:

```python
from src.decision.decision_mapper import DecisionMapper

mapper = DecisionMapper()

# Define what predictions mean in business terms
mapper.define_outcome(
    name="customer_churn",
    positive_label="Will Churn",
    negative_label="Will Stay",
    positive_action="Send retention offer + discount",
    negative_action="Standard service",
    description="Customer churn prediction for telecom"
)

# Get action recommendation
action = mapper.get_action(
    outcome_name="customer_churn",
    prediction=1,  # model predicted churn
    confidence=0.85
)
# Returns: {
#   "prediction": 1,
#   "label": "Will Churn",
#   "action": "Send retention offer + discount",
#   "confidence": 0.85
# }
```

#### Threshold Analysis

Find optimal decision thresholds:

```python
from src.decision.threshold_analyzer import ThresholdAnalyzer

threshold_analyzer = ThresholdAnalyzer()

# Analyze precision/recall tradeoffs across thresholds
analysis = threshold_analyzer.analyze_thresholds(
    y_true=pipeline.y_test,
    y_proba=model.predict_proba(pipeline.X_test)[:, 1],
    thresholds=np.linspace(0.1, 0.9, 50)
)
# Returns: DataFrame with columns [threshold, precision, recall, f1, ...]

# Find optimal threshold for specific goal
optimal = threshold_analyzer.find_optimal_threshold(
    y_true=pipeline.y_test,
    y_proba=y_proba,
    optimization_metric="f1"  # or "precision", "recall", "balanced"
)
# Returns: {"threshold": 0.42, "precision": 0.78, "recall": 0.82, "f1": 0.80}
```

#### Cost-Aware Model Selection

Choose models based on business costs:

```python
from src.decision.cost_analyzer import CostAnalyzer

cost_analyzer = CostAnalyzer()

# Define costs of false positives and false negatives
cost_analyzer.set_costs(fp_cost=100, fn_cost=500)

# Compare models on cost basis
cost_comparison = cost_analyzer.compare_models_by_cost(
    y_true=pipeline.y_test,
    predictions_dict={
        "logistic_regression": lr_predictions,
        "random_forest": rf_predictions
    }
)
# Returns: {"logistic_regression": {"total_cost": 12500, ...}, ...}

# Get the cheapest model
best = cost_analyzer.get_cheapest_model(cost_comparison)
# Returns: "random_forest"
```

#### Confidence Estimation

Distinguish between **probability** and **confidence**:

```python
from src.decision.confidence_estimator import ConfidenceEstimator

conf_estimator = ConfidenceEstimator()

# Probability confidence (highest class probability)
prob_conf = conf_estimator.estimate_probability_confidence(y_proba)

# Margin confidence (difference between top 2 classes)
margin_conf = conf_estimator.estimate_margin_confidence(y_proba)

# Uncertainty (entropy of probability distribution)
uncertainty = conf_estimator.estimate_uncertainty(y_proba)

# Comprehensive reliability report
reliability = conf_estimator.get_reliability_indicators(
    y_proba=y_proba,
    y_pred=predictions
)
# Returns: DataFrame with columns [prediction_id, probability_confidence,
#          margin_confidence, uncertainty, confidence_level, reliability_score]

# Flag uncertain predictions for manual review
uncertain_mask = conf_estimator.flag_uncertain_predictions(
    y_proba=y_proba,
    uncertainty_threshold=0.5
)
# Returns: boolean array marking high-uncertainty predictions
```

#### Counterfactual Explanations

Answer "What needs to change to flip this prediction?"

```python
from src.explain.counterfactual_explainer import CounterfactualExplainer

cf_explainer = CounterfactualExplainer(
    feature_names=pipeline.get_feature_names(),
    categorical_features=pipeline.get_categorical_columns(),
    scaler=pipeline.preprocessor.scaler,
    feature_ranges=pipeline.get_feature_ranges()
)

# Find minimal changes to flip prediction
instance = {"age": 25, "tenure": 3, "monthly_charges": 85}
counterfactual = cf_explainer.find_counterfactual(
    instance=instance,
    model=model,
    X_train=pipeline.X_train,
    feature_importance=importance_dict,
    num_features_to_change=3
)
# Returns: {
#   "original_prediction": 1,
#   "target_prediction": 0,
#   "changes": {
#       "tenure": {"from": 3, "to": 12},
#       "contract_type": {"from": "Month-to-month", "to": "Two year"}
#   },
#   "explanation": "Increase tenure by 9 months and switch to 2-year contract"
# }
```

#### Data Quality Analysis

Detect data issues before they become model issues:

```python
from src.decision.data_quality_analyzer import DataQualityAnalyzer

quality_analyzer = DataQualityAnalyzer()

# Detect target leakage (features too correlated with target)
leakage_report = quality_analyzer.detect_target_leakage(
    X_train=pipeline.X_train,
    y_train=pipeline.y_train,
    correlation_threshold=0.9
)
# Returns: {
#   "has_leakage": True,
#   "suspicious_features": [
#       {"feature": "customer_id", "correlation": 0.95, "risk_level": "CRITICAL"}
#   ],
#   "recommendation": "⚠️ Remove customer_id before training"
# }

# Detect distribution drift between train/test
drift_report = quality_analyzer.detect_distribution_drift(
    X_train=pipeline.X_train,
    X_test=pipeline.X_test,
    p_value_threshold=0.05
)
# Returns: {
#   "has_drift": True,
#   "drifted_features": [
#       {"feature": "tenure", "ks_statistic": 0.23, "p_value": 0.001}
#   ]
# }

# Detect class imbalance
imbalance_report = quality_analyzer.detect_class_imbalance(
    y=pipeline.y_train,
    imbalance_threshold=0.3
)
# Returns: {
#   "is_imbalanced": True,
#   "class_distribution": {0: 0.75, 1: 0.25},
#   "recommendation": "Consider SMOTE or class_weight='balanced'"
# }
```

### 5. Audit Trail System

Complete logging for governance and compliance:

```python
from src.decision.decision_audit_logger import DecisionAuditLogger

audit_logger = DecisionAuditLogger(log_dir="audit_logs")

# Log every prediction with full context
audit_logger.log_decision(
    model_name="random_forest",
    input_features=scenario,
    prediction=1,
    probability=0.73,
    confidence=0.85,
    threshold=0.5,
    action="Send retention offer",
    metadata={
        "user_id": "analyst_123",
        "session_id": "abc-def-ghi"
    }
)
# Creates timestamped JSON log entry

# Retrieve decision history
history = audit_logger.get_decision_history(
    model_name="random_forest",
    start_date="2026-01-01",
    end_date="2026-01-29"
)
# Returns: List of all logged decisions in date range

# Export audit trail for compliance
audit_logger.export_audit_trail(
    output_file="audit_report_2026.csv",
    format="csv"  # or "json"
)
```

### 6. Model Cards for Governance

Auto-generate standardized documentation:

```python
from src.artifacts.model_card_generator import ModelCardGenerator

card_gen = ModelCardGenerator(
    model_name="customer_churn_rf",
    version="1.0"
)

# Add model details
card_gen.add_model_details(
    model_type="Random Forest Classifier",
    framework="scikit-learn 1.5.2",
    task_type="binary_classification"
)

# Add intended use
card_gen.add_intended_use(
    primary_use="Predict customer churn for retention campaigns",
    primary_users=["Marketing team", "Customer success"],
    out_of_scope_uses=["Credit scoring", "Hiring decisions"]
)

# Add performance metrics
card_gen.add_performance_metrics(
    metrics={"accuracy": 0.85, "precision": 0.78, "recall": 0.82},
    test_data_info={"size": 200, "date_range": "2025-12-01 to 2026-01-15"}
)

# Add limitations and biases
card_gen.add_limitations(
    known_limitations=[
        "Performance degrades for customers with <3 months tenure",
        "Lower accuracy for international customers"
    ],
    bias_assessment="No significant bias detected across age/gender groups"
)

# Export as JSON or HTML
card_gen.export_card(output_file="model_card.json", format="json")
card_gen.export_card(output_file="model_card.html", format="html")
```

---

## Configuration

All model hyperparameters live in [config/settings.py](config/settings.py):

```python
# config/settings.py
MODEL_CONFIGS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "params": {
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
        },
    },
    "random_forest": {
        "name": "Random Forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
}

# Data preprocessing settings
MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with >50% missing
TEST_SIZE = 0.2  # 80/20 train/test split
RANDOM_SEED = 42
```

To add a new model:
1. Add config to `MODEL_CONFIGS`
2. Update `ModelManager._get_model_instance()` in [src/core/models.py](src/core/models.py)
3. Import the model class from scikit-learn (or other library)

---

## Project Structure

```
verdict1.0(bk)/
├── app.py                              # Streamlit UI entry point
├── requirements.txt                    # Python dependencies
├── docker-compose.yml                  # Docker orchestration
├── Dockerfile                          # Container definition
│
├── config/
│   ├── __init__.py
│   └── settings.py                     # Model configs, hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── core/                           # Core ML pipeline
│   │   ├── __init__.py
│   │   ├── data_handler.py            # Data validation, missing values
│   │   ├── preprocessing.py           # Encoding, scaling, splitting
│   │   ├── models.py                  # Model training, prediction
│   │   ├── metrics.py                 # Performance calculation
│   │   └── pipeline.py                # Orchestration layer
│   │
│   ├── explain/                        # Explainability subsystem
│   │   ├── __init__.py
│   │   ├── explainability.py          # SHAP, permutation importance
│   │   ├── whatif.py                  # Scenario simulation
│   │   └── counterfactual_explainer.py # Minimal change suggestions
│   │
│   ├── decision/                       # Decision intelligence
│   │   ├── __init__.py
│   │   ├── decision_mapper.py         # Predictions → Actions
│   │   ├── threshold_analyzer.py      # Precision/recall tradeoffs
│   │   ├── cost_analyzer.py           # FP/FN cost analysis
│   │   ├── confidence_estimator.py    # Uncertainty quantification
│   │   ├── data_quality_analyzer.py   # Leakage, drift detection
│   │   └── decision_audit_logger.py   # Audit trail logging
│   │
│   ├── artifacts/                      # Export and documentation
│   │   ├── __init__.py
│   │   ├── exporter.py                # Model serialization
│   │   ├── report_gen.py              # HTML report generation
│   │   └── model_card_generator.py    # Governance docs
│   │
│   └── ui/
│       ├── __init__.py
│       └── visualizations.py          # Plotly charts, confusion matrices
│
├── data/
│   ├── demo_business_dataset.csv      # Sample dataset (500 rows)
│   ├── generate_demo.py               # Demo data generator
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── tests/
    ├── quick.py                        # Quick sanity check
    ├── test_core.py                    # Unit tests
    └── test_phase5.py                  # Integration tests
```

---

## Data Flow

Here's how data flows through the system:

```
1. CSV Upload
   ↓
2. DataHandler.validate_data()
   → Checks for: empty data, missing target, dtype issues
   ↓
3. DataHandler.handle_missing_values()
   → Numeric: mean imputation
   → Categorical: mode imputation
   ↓
4. Preprocessor.prepare_data()
   → LabelEncoder for categoricals (stored for inverse transform)
   → StandardScaler for numerics (stored for inverse transform)
   → train_test_split (80/20 default)
   ↓
5. ModelManager.train()
   → model.fit(X_train_scaled, y_train)
   → Stores trained model in self.models dict
   ↓
6. MetricsCalculator.calculate()
   → predictions = model.predict(X_test_scaled)
   → accuracy, precision, recall, F1, ROC-AUC
   ↓
7. ExplainabilityAnalyzer.get_global_shap_values()
   → SHAP TreeExplainer/LinearExplainer/KernelExplainer
   → Returns feature contributions for all test samples
   ↓
8. DecisionMapper.get_action()
   → prediction (0/1) → business action ("Send offer" / "Standard service")
   ↓
9. DecisionAuditLogger.log_decision()
   → Writes to audit_logs/decisions_YYYYMMDD.json
   ↓
10. ModelExporter.export_all_models()
    → joblib.dump(model, "models/model_name_timestamp.joblib")
    → Saves metadata (metrics, feature names, label encoders)
```

---

## Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.41.1 | Web UI framework |
| `pandas` | 2.2.3 | Data manipulation |
| `numpy` | 2.1.3 | Numerical operations |
| `scikit-learn` | 1.5.2 | ML algorithms, preprocessing |
| `shap` | 0.46.0 | SHAP explainability |
| `plotly` | 5.24.1 | Interactive visualizations |
| `matplotlib` | 3.9.2 | Static plots |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `joblib` | 1.4.2 | Model serialization |
| `scipy` | 1.14.1 | KS test for drift detection |

Install all with:
```bash
pip install -r requirements.txt
```

---

## Extending Verdict

### Add a New Model

1. **Update config** ([config/settings.py](config/settings.py)):
```python
MODEL_CONFIGS["xgboost"] = {
    "name": "XGBoost Classifier",
    "params": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    }
}
```

2. **Update ModelManager** ([src/core/models.py](src/core/models.py)):
```python
from xgboost import XGBClassifier, XGBRegressor

def _get_model_instance(self, model_name: str):
    # ... existing code ...
    elif model_name == "xgboost":
        if self.task_type == "classification":
            return XGBClassifier(**MODEL_CONFIGS["xgboost"]["params"])
        else:
            return XGBRegressor(**MODEL_CONFIGS["xgboost"]["params"])
```

3. **Add to UI** ([app.py](app.py)):
```python
model_options = st.multiselect(
    "Select models to train:",
    ["logistic_regression", "random_forest", "xgboost"],  # Add here
    default=["random_forest", "xgboost"]
)
```

### Add a New Metric

Update [src/core/metrics.py](src/core/metrics.py):

```python
from sklearn.metrics import matthews_corrcoef

def calculate_classification_metrics(self, y_true, y_pred, y_proba=None):
    # ... existing metrics ...
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    return metrics
```

### Add Custom Business Actions

```python
# In your Streamlit app
from src.decision.decision_mapper import DecisionMapper

mapper = DecisionMapper()
mapper.define_outcome(
    name="loan_approval",
    positive_label="Approve Loan",
    negative_label="Reject Loan",
    positive_action="Send approval email + offer letter",
    negative_action="Send rejection email with reapplication timeline",
    description="Loan approval decision system"
)

# Use it
action = mapper.get_action("loan_approval", prediction=1, confidence=0.92)
st.write(action["action"])
```

---

## Deployment

### Local Development
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Docker Deployment
```bash
# Build image
docker build -t verdict:latest .

# Run container
docker run -p 8501:8501 verdict:latest

# Or use docker-compose
docker-compose up -d
```

### Streamlit Cloud
1. Push to GitHub
2. Visit https://share.streamlit.io
3. Connect repo, select `app.py`
4. Deploy (free tier available)

### Production Considerations
- **Scaling**: Use Streamlit's `@st.cache_data` for expensive computations
- **Security**: Store API keys in `.streamlit/secrets.toml` (never commit)
- **Monitoring**: Audit logs in `audit_logs/` directory track all predictions
- **Model Versioning**: ModelExporter adds timestamps to all saved models
- **Data Privacy**: No data is stored permanently; all processing in-memory

---

## Testing

```bash
# Run unit tests
pytest tests/test_core.py -v

# Quick sanity check
python tests/quick.py

# Integration tests
pytest tests/test_phase5.py -v
```

---

## Common Use Cases

### Customer Churn Prediction
```python
# Load data
df = pd.read_csv("customer_data.csv")

# Train pipeline
pipeline = MLPipeline(df, target_col="churned")
results = pipeline.run_full_pipeline()

# Map to business actions
mapper.define_outcome(
    name="churn",
    positive_label="Will Churn",
    negative_label="Will Stay",
    positive_action="Send retention discount",
    negative_action="Standard engagement"
)

# Make prediction
pred = pipeline.predict("random_forest", new_customer_data)
action = mapper.get_action("churn", pred["prediction"], pred["probability"])
```

### Fraud Detection
```python
# Cost-aware model selection (FN = missed fraud is expensive)
cost_analyzer.set_costs(fp_cost=10, fn_cost=1000)
best_model = cost_analyzer.get_cheapest_model(cost_comparison)

# Adjust threshold to maximize fraud detection
optimal_threshold = threshold_analyzer.find_optimal_threshold(
    y_true, y_proba, optimization_metric="recall"
)
```

### Credit Scoring
```python
# Generate model card for compliance
card_gen = ModelCardGenerator("credit_score_model", version="2.1")
card_gen.add_ethical_considerations(
    fairness_assessment="Model audited for disparate impact",
    protected_attributes=["age", "gender", "race"]
)
card_gen.export_card("credit_model_card.html", format="html")
```

---

## FAQ

**Q: How does SHAP explainer selection work?**  
A: Verdict automatically selects the best SHAP explainer:
- Tree models (RandomForest) → `TreeExplainer` (fast, exact)
- Linear models (LogisticRegression) → `LinearExplainer` (fast, exact)
- Other models → `KernelExplainer` (slow, model-agnostic)

**Q: Can I use this for regression?**  
A: Yes! The pipeline auto-detects regression tasks (continuous target) and switches to:
- LinearRegression instead of LogisticRegression
- RandomForestRegressor instead of RandomForestClassifier
- R², MAE, RMSE metrics instead of accuracy/precision/recall

**Q: How are categorical features handled?**  
A: `LabelEncoder` for ordinal/binary categoricals, with encoders stored in `preprocessor.label_encoders` dict for inverse transforms during what-if analysis.

**Q: What's the difference between probability and confidence?**  
A:
- **Probability**: `model.predict_proba()` output (e.g., 0.73 for class 1)
- **Confidence**: How certain the model is, accounting for:
  - Margin between top 2 classes
  - Entropy of probability distribution
  - Ensemble agreement (if using multiple models)

**Q: How do I debug data quality issues?**  
A: Use `DataQualityAnalyzer`:
```python
quality_analyzer = DataQualityAnalyzer()
leakage = quality_analyzer.detect_target_leakage(X_train, y_train)
drift = quality_analyzer.detect_distribution_drift(X_train, X_test)
imbalance = quality_analyzer.detect_class_imbalance(y_train)
```

**Q: Where are trained models saved?**  
A: `ModelExporter` saves to `models/` directory with format:
```
models/
  random_forest_20260129_143022.joblib       # Model
  random_forest_20260129_143022_metadata.json  # Metrics, feature names, etc.
```

---

## License

MIT License - See LICENSE file for details

---

## Contributing

Pull requests welcome! Key areas:
- XGBoost/LightGBM model support
- LIME explainer integration
- Time-series forecasting
- Custom metric functions
- API endpoint (FastAPI) for batch predictions

---

**Built with scikit-learn, SHAP, and Streamlit**

*Last Updated: January 29, 2026*
