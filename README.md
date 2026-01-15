#  Verdict
<!-- 
**An Intelligent Decision-Support System Powered by AutoML & Explainable AI**

> Upload your data. Predict your future. Understand your models.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-green.svg)](https://github.com/slundberg/shap)

##  Features

###  Phase 1-2: AutoML Pipeline
- **Automatic Data Processing**: Handles missing values, categorical encoding, feature scaling
- **Multi-Model Training**: Logistic Regression, Random Forest with automated hyperparameters
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC (classification) + R, MAE, RMSE (regression)
- **Interactive Dashboard**: Data preview, model comparison, visualizations with Plotly

###  Phase 3: Explainability & What-If
- **SHAP Explanations**: TreeExplainer for fast insights, KernelExplainer for flexibility
- **Feature Importance**: Permutation-based importance ranking
- **What-If Analysis**: Adjust features with sliders, see predictions change in real-time
- **Sensitivity Analysis**: Vary single features, track prediction curves
- **Confidence Scores**: Probability-based confidence for all predictions

###  Phase 4: Export & Deployment  
- **Model Export**: Save trained models as joblib artifacts with metadata
- **HTML Reports**: Beautiful, self-contained reports with all metrics
- **Docker Support**: Containerized deployment with docker-compose

###  Phase 5: Decision Intelligence & Governance ⭐ NEW
- **Decision Mapping**: Map predictions to business actions and outcomes
- **Threshold Control**: Dynamic threshold adjustment with precision-recall tradeoffs
- **Cost-Aware Selection**: FP/FN costs for business-aligned model selection
- **Counterfactual Explanations**: "What needs to change to flip a prediction?"
- **Confidence Estimation**: Distinguish probability from actual confidence, uncertainty quantification
- **Data Quality Analysis**: Detect target leakage, distribution drift, class imbalance
- **Decision Audit Trail**: Complete logging of predictions, thresholds, confidence, and actions
- **Model Cards**: Standardized documentation for governance and responsible AI
- **Streamlit Cloud Ready**: One-click deployment to Streamlit Cloud

---

##  Quick Start

### Local Development

**1. Clone & Setup**
```bash
cd c:\Development\verdict
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**2. Generate Demo Data**
```bash
python data/generate_demo.py
```

**3. Run the App**
```bash
streamlit run app.py
```

Visit `http://localhost:8501`

### Docker Deployment

**1. Build & Run**
```bash
docker-compose up --build
```

**2. Access**
Visit `http://localhost:8501`

---

##  Project Structure

```
verdict/
 app.py                          # Main Streamlit application
 requirements.txt                # Python dependencies
 Dockerfile                      # Docker image configuration
 docker-compose.yml              # Docker compose setup

 src/                            # Core application modules
    __init__.py
    data_handler.py            # Data validation & exploration
    preprocessing.py           # Encoding, scaling, train-test split
    models.py                  # Logistic Regression, Random Forest
    metrics.py                 # Classification & regression metrics
    pipeline.py                # ML orchestration
    explainability.py          # SHAP & permutation importance
    whatif.py                  # What-if scenario simulation
    exporter.py                # Model export functionality
    report_gen.py              # HTML report generation
    visualizations.py          # Plotly & Matplotlib charts

 config/                        # Configuration
    settings.py               # Model hyperparameters & settings

 tests/                         # Unit & integration tests
    test_core.py              # Core pipeline tests
    quick.py                  # Quick demo script

 data/                          # Data directory
    demo_business_dataset.csv # Demo dataset (500 rows  11 features)
    generate_demo.py           # Dataset generator

 models/                        # Exported models (created on export)

 reports/                       # Generated reports (created on export)

 .streamlit/                    # Streamlit configuration
     config.toml               # UI theme & settings
     secrets.toml              # API keys & sensitive data
```

---

##  Use Cases

### 1. Business Operations
- **Customer Churn Prediction**: Identify at-risk customers before they leave
- **Sales Forecasting**: Predict revenue based on historical patterns
- **Inventory Optimization**: Forecast stockouts and optimize reorders

### 2. Finance & Risk
- **Default Prediction**: Assess loan default probability
- **Fraud Detection**: Identify suspicious transactions
- **Credit Scoring**: Automated lending decisions

### 3. HR Analytics
- **Employee Attrition**: Predict who might leave
- **Offer Acceptance**: Optimize job offers
- **Performance Prediction**: Early identification of top performers

### 4. Healthcare
- **Patient Risk Stratification**: Identify high-risk patients
- **Treatment Response**: Predict which treatment works best
- **Readmission Prediction**: Prevent unnecessary hospitalizations

---

##  API Reference

### MLPipeline
```python
from src.pipeline import MLPipeline

# Initialize and train
pipeline = MLPipeline(df, target_col="churn")
results = pipeline.run_full_pipeline()

# Access trained models
predictions = pipeline.get_model_predictions("random_forest", X_test)

# Get feature information
numeric_cols = pipeline.get_numeric_columns()
categorical_cols = pipeline.get_categorical_columns()
```

### ExplainabilityAnalyzer
```python
from src.explainability import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(model, X_train, X_test, feature_names)

# Get feature importance
importance = analyzer.get_feature_importance_permutation(y_test)

# Explain individual prediction
explanation = analyzer.explain_prediction(sample_idx=0)
```

### WhatIfAnalyzer
```python
from src.whatif import WhatIfAnalyzer

whatif = WhatIfAnalyzer(pipeline, feature_names, numeric_cols, categorical_cols, label_encoders)

# Make prediction for scenario
result = whatif.predict_scenario("random_forest", {"age": 35, "income": 75000})

# Sensitivity analysis
sens_df = whatif.get_sensitivity_analysis("random_forest", base_input, "age", [20, 30, 40, 50])
```

### ModelExporter
```python
from src.exporter import ModelExporter

exporter = ModelExporter(output_dir="models")

# Export all models
exports = exporter.export_all_models(pipeline, eval_results)

# Load model later
model = exporter.load_model("models/random_forest_20260114_120000.joblib")
```

### ReportGenerator
```python
from src.report_gen import ReportGenerator

generator = ReportGenerator("My ML Report")
report_path = generator.generate_html_report(
    pipeline, eval_results, 
    {"rows": 500, "columns": 11, "missing": 0},
    output_file="my_report.html"
)
```

---

##  Workflow

### Step 1: Data Upload
- Upload CSV or use demo dataset (demo_business_dataset.csv)
- View data preview, missing values, distributions
- Automatic data type detection

### Step 2: Model Training
- Select prediction target
- Choose models (Logistic Regression, Random Forest)
- Click "Train Models" - automatic preprocessing & training

### Step 3: Results & Comparison
- View metrics dashboard with model comparison
- See confusion matrices for classification
- Feature distributions by target class

### Step 4: Explainability
- **Feature Importance**: See which features matter most
- **What-If Analysis**: Adjust inputs, see predictions change
- **Sensitivity Analysis**: Test impact of varying single features
- **Prediction Explanations**: Understand why model made specific prediction

### Step 5: Export & Share
- Export trained models (joblib format)
- Generate beautiful HTML reports
- Deploy with Docker/Streamlit Cloud

---

##  Deployment

### Streamlit Cloud (Free)
```bash
# 1. Push to GitHub
git push origin main

# 2. Visit https://share.streamlit.io
# 3. Connect GitHub repository
# 4. Select main/app.py
# 5. Deploy!
```

### Docker (Local/Cloud)
```bash
# Build
docker build -t ai-copilot .

# Run
docker run -p 8501:8501 ai-copilot

# Or with docker-compose
docker-compose up
```

### AWS/Azure/GCP
- Use Docker image in your preferred cloud platform
- Estimated cost: ~$50-100/month for small always-on instance
- Supports CPU-only deployment (no GPU needed)

---

##  Testing

```bash
# Run unit tests
pytest tests/test_core.py -v

# Run quick demo
python tests/quick.py
```

---

##  Key Technologies

| Component | Technology | Version |
|-----------|-----------|---------|
| Frontend | Streamlit | 1.28+ |
| ML Framework | scikit-learn | 1.3+ |
| Data Processing | Pandas, NumPy | 2.1+, 1.26+ |
| Explainability | SHAP | 0.43+ |
| Visualization | Plotly, Matplotlib | 5.17+, 3.8+ |
| Serialization | joblib | 1.3+ |
| Containerization | Docker | 20.10+ |

---

##  Learning Resources

- **SHAP Documentation**: https://github.com/slundberg/shap
- **scikit-learn Guide**: https://scikit-learn.org/stable/user_guide.html
- **Streamlit Docs**: https://docs.streamlit.io/
- **ML Interpretability**: https://christophm.github.io/interpretable-ml-book/

---

##  Configuration

### Model Hyperparameters
Edit `config/settings.py`:
```python
MODEL_CONFIGS = {
    "logistic_regression": {
        "params": {
            "max_iter": 1000,
            "solver": "lbfgs",
        }
    },
    "random_forest": {
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
        }
    }
}
```

### Streamlit Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#3498db"
backgroundColor = "#f8f9fa"
```

---

##  Contributing

Pull requests welcome! Areas for contribution:
- XGBoost model support
- Advanced feature engineering
- Time-series support
- Model serving API (FastAPI)
- Batch prediction
- Custom metric functions

---

##  License

MIT License - See LICENSE file for details

---

##  Acknowledgments

- SHAP library for explainability
- scikit-learn for ML algorithms
- Streamlit for amazing UI framework
- The open-source ML community

---

##  FAQ

**Q: Can I use my own dataset?**  
A: Yes! Upload any CSV file with your data. The system automatically handles preprocessing.

**Q: What file formats are supported?**  
A: Currently CSV. Parquet, Excel support coming soon.

**Q: Can I export trained models?**  
A: Yes! Models are exported as joblib files with full metadata.

**Q: Is GPU required?**  
A: No! Runs on CPU-only hardware. GPU support available in future releases.

**Q: How can I deploy this?**  
A: Streamlit Cloud (free), Docker, AWS/Azure/GCP, or on-premises.

**Q: What's the data size limit?**  
A: Currently ~100MB per file. Larger datasets coming in next release.

---

##  📧 Contact & Support

- **Documentation**: See README.md (this file)
- **Issues**: Create GitHub issue
- **Discussions**: GitHub Discussions tab
- **Email**: support@example.com (future)

---

**Made with ❤️ by the AI Decision Copilot Team**

*Last Updated: January 14, 2026* -->
