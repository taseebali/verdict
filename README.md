# Verdict - ML Pipeline Foundation

**An intelligent ML pipeline for binary & multiclass classification with comprehensive data validation.**

> Phase 1: Foundation Complete ✅ | Phase 2: In Development 🚀

---

## ✨ What's Included

### Core Features (Phase 1 ✅)
- ✅ Binary & multiclass classification (OvR strategy)
- ✅ Stratified K-fold cross-validation
- ✅ Comprehensive data quality validation
- ✅ Multiclass metrics (macro/weighted/per-class)
- ✅ FastAPI framework with 4 production endpoints
- ✅ 155 passing tests (100% success rate)

### Phase 2 (In Development 🚀)
- Model persistence & serialization
- Regression models (Prophet, statsmodels)
- Feature importance (SHAP, permutation)
- Ensemble methods
- Streamlit dashboard

---

## 🚀 Quick Start

### Installation
```bash
cd verdict
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run Tests
```bash
pytest tests/ -q
# Expected: 155 passed in ~15s
```

### Use the Pipeline
```python
import pandas as pd
from src.core.data_handler import DataHandler
from src.core.models import ModelManager

# Load & validate
df = pd.read_csv('data.csv')
handler = DataHandler(df)
is_valid, msg = handler.validate_data()

# Train & predict
manager = ModelManager(task_type="classification", strategy="ovr")
X = df.drop('target', axis=1)
y = df['target']
manager.train("random_forest", X, y)
predictions = manager.predict("random_forest", X)
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [PHASE_1_HANDOFF.md](PHASE_1_HANDOFF.md) | Complete technical docs, API reference, examples |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick start guide & common tasks |
| [PHASE_2_PLAN.md](PHASE_2_PLAN.md) | Phase 2 roadmap and implementation plan |

---

## 📁 Project Structure

```
verdict/
├── src/
│   ├── core/                    # Production modules
│   │   ├── data_handler.py      # Data validation
│   │   ├── metrics.py           # Multiclass metrics
│   │   ├── models.py            # OvR classification
│   │   ├── cross_validation.py  # Stratified CV
│   │   └── preprocessing.py     # Feature scaling
│   ├── decision/
│   │   └── multiclass_handler.py # Problem detection
│   └── api/
│       └── app.py               # FastAPI framework
│
├── tests/                        # 155 verified tests
│   ├── test_data_handler_edge_cases.py
│   ├── test_multiclass_handler_enhanced.py
│   ├── test_ovr_multiclass.py
│   ├── test_multiclass_metrics.py
│   ├── test_binary_regression.py
│   ├── test_cv_integration.py
│   ├── test_api_endpoints.py
│   └── test_multiclass_integration_v2.py
│
├── requirements.txt
├── pyproject.toml
├── PHASE_1_HANDOFF.md
├── QUICK_REFERENCE.md
└── PHASE_2_PLAN.md
```

---

## 🎯 Phase 1 Status

| Component | Status | Tests |
|-----------|--------|-------|
| Data Quality | ✅ Complete | 24 |
| Problem Detection | ✅ Complete | 36 |
| Classification (OvR) | ✅ Complete | 20 |
| Metrics | ✅ Complete | 14 |
| Cross-Validation | ✅ Complete | 13 |
| API Framework | ✅ Complete | 24 |
| End-to-End Pipeline | ✅ Complete | 12 |
| Multiclass Integration | ✅ Complete | 12 |
| **TOTAL** | **✅ 100%** | **155** |

---

## 🔧 Key Modules

### DataHandler
```python
handler = DataHandler(df)
is_valid, msg = handler.validate_data()
nulls = handler.detect_null_rows()
dups = handler.detect_duplicates()
```

### MultiClassDetector
```python
problem_type = MultiClassDetector.detect_problem_type(target)
valid, warnings = MultiClassDetector.validate_target(target)
classes = MultiClassDetector.get_unique_classes(target)
```

### ModelManager (OvR)
```python
manager = ModelManager(task_type="classification", strategy="ovr")
manager.train("random_forest", X, y)
predictions = manager.predict("random_forest", X)
probabilities = manager.predict_proba("random_forest", X)
```

### CrossValidationEngine
```python
cv = CrossValidationEngine(n_splits=5)
result = cv.run_cv(X, y, model, "random_forest", "multiclass")
```

### MetricsCalculator
```python
metrics = MetricsCalculator.calculate_multiclass_metrics(y, pred, proba)
```

---

## ⚙️ Configuration

Supported models in `src/core/models.py`:
- `"random_forest"` - RandomForestClassifier (default)
- `"logistic_regression"` - LogisticRegression

All models use `random_state=42` for reproducibility.

---

## 🐛 Common Issues & Solutions

| Error | Solution |
|-------|----------|
| `ValueError: Unknown model: rf` | Use `"random_forest"` not `"rf"` |
| `AttributeError: no attribute 'iloc'` | Convert arrays to DataFrame |
| `TypeError: missing 2 arguments` | Pass DataFrame + target: `Preprocessor(df, 'target')` |

See [PHASE_1_HANDOFF.md](PHASE_1_HANDOFF.md) for complete troubleshooting guide.

---

## 📈 Next Steps

**Phase 2 Implementation Plan:** See [PHASE_2_PLAN.md](PHASE_2_PLAN.md)

Key priorities:
1. Model persistence (joblib serialization)
2. Real model integration in API
3. Streamlit dashboard
4. Regression models

---

**Questions?** See [PHASE_1_HANDOFF.md](PHASE_1_HANDOFF.md) or [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
