# VERDICT ML Platform - Source Code Organization

## 📁 Directory Structure

```
src/
├── api/                          # FastAPI REST Application (v2.0)
│   ├── app.py                   # Consolidated API with all endpoints
│   ├── schemas.py               # Pydantic request/response models
│   └── __init__.py
│
├── core/                         # Core ML Pipeline & Data Processing
│   ├── pipeline.py              # Main ML pipeline orchestration
│   ├── data_handler.py          # Data loading and basic operations
│   ├── preprocessing.py         # Data preprocessing & feature engineering
│   ├── models.py                # Model definitions and wrappers
│   ├── metrics.py               # Evaluation metrics
│   ├── cross_validation.py      # CV and model selection
│   └── __init__.py
│
├── decision/                     # Decision Support & Analysis
│   ├── decision_audit_logger.py # Prediction audit trail
│   ├── threshold_analyzer.py    # Threshold optimization
│   ├── cost_analyzer.py         # Cost/benefit analysis
│   ├── confidence_estimator.py  # Confidence calibration
│   ├── multiclass_handler.py    # Multi-class classification support
│   ├── decision_mapper.py       # Decision mapping utilities
│   ├── data_quality_analyzer.py # Data quality metrics
│   └── __init__.py
│
├── explain/                      # Explainability & What-If
│   ├── explainability.py        # Feature importance & SHAP
│   ├── counterfactual_explorer.py # Counterfactual explanations
│   ├── whatif.py                # What-if scenario analysis
│   └── __init__.py
│
├── artifacts/                    # Model Artifacts & Persistence
│   ├── model_serializer.py      # Model versioning & serialization
│   ├── exporter.py              # Model export utilities
│   ├── report_gen.py            # Report generation
│   ├── model_card_generator.py  # Model card creation
│   └── __init__.py
│
├── ui/                          # Streamlit Dashboard Application
│   ├── dashboard.py             # Main dashboard entry point
│   ├── visualizations.py        # Reusable visualization components
│   ├── __init__.py
│   └── pages/                   # Multi-page dashboard
│       ├── 01_data_explorer.py  # Data exploration
│       ├── 02_model_training.py # Model training interface
│       ├── 03_predictions.py    # What-If & Predictions 🆕
│       └── 04_audit_logs.py     # Audit trail viewer
│
└── __init__.py
```

## 🎯 Key Improvements (Phase 2 - Current Session)

### 1. **Consolidated API (✅ DONE)**
- **Before**: Two separate API files (`app.py` and `app_v2.py`) with overlapping code
- **After**: Single consolidated `src/api/app.py` (v2.0) with all functionality
- **Benefits**: No code duplication, clearer maintenance, unified endpoints

### 2. **Comprehensive Request/Response Schemas (✅ DONE)**
- **File**: `src/api/schemas.py` 
- **New Models**:
  - `PredictRequest/Response` - Predictions with confidence
  - `WhatIfRequest/Response` - Scenario analysis
  - `RecommendationResponse` - Feature recommendations
  - `FeatureInfo/FeaturesResponse` - Feature metadata
  - `AuditRecord/AuditResponse` - Audit logging
  - `HealthResponse` - API health

### 3. **Human-Readable Value Formatting (✅ DONE)**
- **Function**: `format_value(feature_name, value)`
- **Examples**:
  - `monthlyCharges=89.5` → `"$89.50"`
  - `churnRate=12.5` → `"12.5%"`
  - `tenure=24` → `"24 months"`
  - `age=35` → `"35 years"`

### 4. **What-If Scenario Analysis (✅ DONE)**
- **Endpoint**: `POST /whatif`
- **Features**:
  - Compare current vs. hypothetical scenarios
  - Show impact of feature changes
  - Formatted before/after/change display
  - Prediction change detection

### 5. **Feature Recommendations Engine (✅ DONE)**
- **Endpoint**: `GET /recommendations`
- **Features**:
  - Mean, median, std, min, max for each feature
  - UI-friendly ranges for sliders
  - Training data statistics

### 6. **Enhanced Predictions Page (✅ DONE)**
- **File**: `src/ui/pages/03_predictions.py` (NEW)
- **Features**:
  - Feature sliders with recommendations
  - What-if scenario builder
  - Model comparison
  - Audit trail viewer
  - Real-time predictions with formatting

### 7. **Fixed Dashboard Import Issues (✅ DONE)**
- **Problem**: `ModuleNotFoundError: No module named 'src'`
- **Solution**: Added `sys.path.insert(0, ...)` for absolute imports
- **Location**: `src/ui/dashboard.py` line 22

### 8. **Fixed Training Data Issues (✅ DONE)**
- **Problem**: "could not convert string to float" with mixed data types
- **Solution**: Auto-filter numeric columns, encode categorical targets
- **Location**: `src/ui/dashboard.py` lines 280-300

### 9. **Cleaned Up Redundant Files (✅ DONE)**
- Deleted: `src/api/app_v2.py` (duplicate)
- Deleted: Old `src/api/schemas.py` (limited models)
- Deleted: `tests/test_api_v2.py` (referenced deleted app_v2.py)
- Recreated: `tests/test_api_v2.py` with new consolidated tests

## 📊 API Endpoints (v2.0)

### Health & Info
- `GET /` - API overview and available endpoints
- `GET /health` - Health check with available models

### Predictions
- `POST /predict` - Real predictions with confidence and formatting
  ```json
  {
    "features": {"age": 35, "monthlyCharges": 89.50},
    "model_name": "random_forest"
  }
  ```

### What-If Analysis
- `POST /whatif` - Scenario analysis comparing current vs. changes
  ```json
  {
    "current_features": {"age": 35, "tenure": 24},
    "scenario_changes": {"tenure": 36},
    "model_name": "random_forest"
  }
  ```

### Information & Recommendations
- `GET /recommendations` - Feature recommendations and ranges
- `GET /features` - Available features with metadata
- `GET /audit?limit=20` - Prediction audit trail

## 🧪 Test Status

| Component | Tests | Status |
|-----------|-------|--------|
| Phase 1 Core | 155 | ✅ PASS |
| P2.1 Model Persistence | 19 | ✅ PASS |
| P2.2 Enhanced API (Consolidated) | 24 | ✅ PASS |
| P2.3 Streamlit Dashboard | 53 | ✅ PASS |
| **TOTAL** | **244** | **✅ PASS** |

## 🚀 Running the Platform

### Start API Server
```bash
cd c:\Development\verdict
uvicorn src.api.app:app --reload --port 8000
```

### Start Streamlit Dashboard
```bash
cd c:\Development\verdict
streamlit run src/ui/dashboard.py
```

### Run Tests
```bash
cd c:\Development\verdict
pytest tests/ -q --tb=short
```

## 📋 Module Responsibilities

### `src/api/`
- RESTful API endpoints
- Request/response validation
- Model predictions
- What-if analysis
- Audit logging

### `src/core/`
- ML pipeline orchestration
- Data preprocessing
- Model training/evaluation
- Cross-validation

### `src/decision/`
- Decision support analysis
- Threshold optimization
- Cost/benefit analysis
- Data quality assessment
- Audit trail management

### `src/explain/`
- Feature importance
- Explainability analysis
- What-if explanations
- Counterfactual examples

### `src/artifacts/`
- Model versioning
- Model serialization
- Report generation
- Model card creation

### `src/ui/`
- Streamlit dashboard
- Data exploration
- Model training UI
- Predictions & what-if
- Audit trail viewer

## 🔧 Configuration

- **API Settings**: `config/settings.py`
- **Model Configs**: `config/settings.py` → `MODEL_CONFIGS`
- **Log Settings**: Individual modules with logging config

## ✨ Key Features

- ✅ **Real-time Predictions** - Single and batch
- ✅ **What-If Analysis** - Scenario testing
- ✅ **Feature Recommendations** - From training data statistics
- ✅ **Human-Readable Formatting** - Currency, percentages, time units
- ✅ **Model Versioning** - Track and manage versions
- ✅ **Audit Logging** - Complete prediction history
- ✅ **Explainability** - Feature importance and counterfactuals
- ✅ **Interactive Dashboard** - Streamlit multi-page UI
- ✅ **REST API** - FastAPI with comprehensive documentation

## 📈 Last Update

**Session**: Phase 2 - Consolidation & Enhancement
**Date**: Current
**Changes**: API consolidation, What-If analysis, recommendations, predictions page
**Tests**: 244/244 passing ✅
