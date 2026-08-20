# Verdict Frontend Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cluttered Streamlit dashboard with a FastAPI backend + React/TypeScript frontend, keeping all existing ML logic (`src/core`, `src/artifacts`, `src/decision`, `src/explain`) untouched.

**Architecture:** `backend/` is a thin FastAPI routing layer over the existing pipeline classes — no ML logic is rewritten, only wrapped. `frontend/` is a React + Vite + TypeScript SPA (shadcn/ui + Tailwind + Recharts + TanStack Query) talking to the backend over a JSON REST API. In production, FastAPI serves the built frontend as static files from the same process (one container, matches the current HF Spaces deploy story).

**Tech Stack:** Python 3.12 / FastAPI / Pydantic v2 (backend); Node 20+ / React 18 / TypeScript / Vite / Tailwind CSS / shadcn/ui (Radix primitives) / Recharts / TanStack Query / react-router-dom (frontend).

**Spec:** `docs/superpowers/specs/2026-08-20-frontend-redesign-design.md`

## Global Constraints

- Do not modify any file under `src/core`, `src/artifacts`, `src/decision`, `src/explain` — the backend only imports and calls these, verified by the existing 317 tests staying green (`pytest tests/ -q` from repo root).
- No emoji anywhere in frontend code, markup, or copy.
- Single accent color `#6366f1` (indigo-500) — no other saturated colors except semantic states (green for positive deltas, red for errors).
- Never pure black (`#000000`) or pure white (`#FFFFFF`) as a text/background pair — use `#18181b`/`#fafaf9` per the spec's zinc/stone palette.
- All numeric/data values use `font-variant-numeric: tabular-nums`.
- No `window.alert`/browser-native dialogs — inline error states only.
- Animate only `transform`/`opacity`, never layout-triggering CSS properties.
- Server state (dataset, pipeline, trained models) is in-memory, single active session — no auth, no DB, matches spec.

---

## File Structure

```
backend/
  app/
    __init__.py
    main.py              # FastAPI app factory, CORS, static file mount
    state.py             # AppState singleton (current df, pipeline, models, audit logger)
    schemas.py           # Pydantic request/response models
    routers/
      __init__.py
      datasets.py        # /api/datasets/*
      training.py        # /api/train
      predictions.py     # /api/predict, /api/whatif
      audit.py           # /api/audit-logs
      models.py          # /api/models/{name}/download
  tests/
    conftest.py
    test_datasets.py
    test_training.py
    test_predictions.py
    test_audit.py
  requirements.txt

frontend/
  index.html
  vite.config.ts
  tailwind.config.ts
  tsconfig.json
  package.json
  src/
    main.tsx
    App.tsx               # router setup
    lib/
      api.ts               # fetch wrapper + typed API client
      types.ts              # shared TS types mirroring backend schemas
      queryClient.ts
    components/
      layout/
        AppShell.tsx        # sidebar + content frame
        Sidebar.tsx
      ui/                    # shadcn generated primitives (button, card, tabs, etc.)
      StatTile.tsx
      HeroStat.tsx           # double-bezel stat card
      Chip.tsx
      SkeletonBlock.tsx
      EmptyState.tsx
      InlineError.tsx
    pages/
      Dashboard.tsx
      DataExplorer.tsx
      ModelTraining.tsx
      Predictions.tsx
      AuditLogs.tsx
    styles/
      globals.css           # Tailwind base + CSS variable design tokens
  .env.development           # VITE_API_BASE_URL=http://localhost:8000
```

---

## Task 1: Backend scaffold — app factory, state, health check

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/state.py`
- Create: `backend/requirements.txt`
- Test: `backend/tests/conftest.py`
- Test: `backend/tests/test_datasets.py` (health check only in this task)

**Interfaces:**
- Produces: `AppState` singleton in `backend/app/state.py` with attributes `df: pd.DataFrame | None`, `pipeline: MLPipeline | None`, `trained_model: Any | None`, `trained_model_name: str | None`, `model_features: list[str] | None`, `target_column: str | None`, `audit_logger: DecisionAuditLogger`; function `get_state() -> AppState` (module-level singleton accessor).
- Produces: FastAPI app instance importable as `from app.main import app`.

- [ ] **Step 1: Write requirements.txt**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
```

- [ ] **Step 2: Write `backend/app/state.py`**

```python
"""In-memory application state — single active session, no auth/DB."""
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.pipeline import MLPipeline
from src.decision.decision_audit_logger import DecisionAuditLogger


class AppState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.pipeline: Optional[MLPipeline] = None
        self.trained_model: Optional[Any] = None
        self.trained_model_name: Optional[str] = None
        self.model_features: Optional[list[str]] = None
        self.target_column: Optional[str] = None
        self.audit_logger = DecisionAuditLogger()


_state = AppState()


def get_state() -> AppState:
    return _state
```

- [ ] **Step 3: Write `backend/app/__init__.py`** (empty file marking the package)

```python
```

- [ ] **Step 4: Write `backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Verdict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 5: Write `backend/tests/conftest.py`**

```python
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.state import get_state


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    state = get_state()
    state.df = None
    state.pipeline = None
    state.trained_model = None
    state.trained_model_name = None
    state.model_features = None
    state.target_column = None
    yield
```

- [ ] **Step 6: Write `backend/tests/test_datasets.py`** (health check only for now)

```python
def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 7: Install deps and run test**

Run: `pip install -r backend/requirements.txt httpx` (httpx is FastAPI TestClient's dependency)
Run: `cd backend && python -m pytest tests/test_datasets.py -v`
Expected: `test_health` PASSES

- [ ] **Step 8: Commit**

```bash
git add backend/
git commit -m "Add FastAPI backend scaffold with health check"
```

---

## Task 2: Dataset endpoints

**Files:**
- Create: `backend/app/schemas.py`
- Create: `backend/app/routers/__init__.py`
- Create: `backend/app/routers/datasets.py`
- Modify: `backend/app/main.py` (register router)
- Test: `backend/tests/test_datasets.py` (add dataset tests)

**Interfaces:**
- Consumes: `get_state()` from Task 1; `src.core.data_handler.DataHandler` (existing, has `validate_data()`, method to compute missing %/correlations — reuse whatever quality-check method it exposes for the "Issues Found" logic already used by the Streamlit data explorer page at `src/ui/pages/01_data_explorer.py`).
- Produces: `DatasetSummary` Pydantic model with fields `rows: int`, `columns: int`, `numeric_columns: list[str]`, `categorical_columns: list[str]`, `missing_pct: float`, `warnings: list[str]`. Used by Task 3 (training needs column lists) and the frontend Dashboard/DataExplorer pages.

- [ ] **Step 1: Write failing tests in `backend/tests/test_datasets.py`**

```python
def test_load_demo_dataset(client):
    response = client.post("/api/datasets/demo")
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 5000
    assert body["columns"] == 25
    assert "churn" in body["numeric_columns"] or "churn" in body["categorical_columns"]


def test_upload_dataset(client):
    csv_content = b"a,b,target\n1,2,0\n3,4,1\n5,6,0\n"
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 3
    assert body["columns"] == 3


def test_current_summary_before_load_returns_404(client):
    response = client.get("/api/datasets/current")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cd backend && python -m pytest tests/test_datasets.py -v`
Expected: FAIL — `/api/datasets/demo` returns 404 (route doesn't exist)

- [ ] **Step 3: Write `backend/app/schemas.py`**

```python
from typing import Any, Optional

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_pct: float
    warnings: list[str]
```

- [ ] **Step 4: Write `backend/app/routers/__init__.py`** (empty)

```python
```

- [ ] **Step 5: Write `backend/app/routers/datasets.py`**

```python
import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from app.schemas import DatasetSummary
from app.state import get_state

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DEMO_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "verdict_demo.csv"


def _summarize(df: pd.DataFrame) -> DatasetSummary:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    missing_pct = round(float(df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100, 2) if df.size else 0.0

    warnings: list[str] = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        high_corr_pairs = 0
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] > 0.9:
                    high_corr_pairs += 1
        if high_corr_pairs:
            warnings.append(f"High correlation detected: {high_corr_pairs} feature pairs > 0.9")

    return DatasetSummary(
        rows=df.shape[0],
        columns=df.shape[1],
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        missing_pct=missing_pct,
        warnings=warnings,
    )


@router.post("/demo", response_model=DatasetSummary)
def load_demo_dataset():
    state = get_state()
    if not DEMO_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Demo dataset not found at {DEMO_DATA_PATH}")
    state.df = pd.read_csv(DEMO_DATA_PATH)
    state.pipeline = None
    return _summarize(state.df)


@router.post("/upload", response_model=DatasetSummary)
async def upload_dataset(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    state = get_state()
    state.df = df
    state.pipeline = None
    return _summarize(state.df)


@router.get("/current", response_model=DatasetSummary)
def get_current_dataset():
    state = get_state()
    if state.df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet")
    return _summarize(state.df)
```

- [ ] **Step 6: Register router in `backend/app/main.py`**

Modify `backend/app/main.py` — add after the `app = FastAPI(...)` block:

```python
from app.routers import datasets

app.include_router(datasets.router)
```

- [ ] **Step 7: Run tests, verify pass**

Run: `cd backend && python -m pytest tests/test_datasets.py -v`
Expected: all 3 tests PASS

- [ ] **Step 8: Commit**

```bash
git add backend/
git commit -m "Add dataset load/upload/summary endpoints"
```

---

## Task 3: Training endpoint

**Files:**
- Modify: `backend/app/schemas.py` (add training models)
- Create: `backend/app/routers/training.py`
- Modify: `backend/app/main.py` (register router)
- Test: `backend/tests/test_training.py`

**Interfaces:**
- Consumes: `MLPipeline(df, target_col)` with `.preprocess()`, `.train(["random_forest"])`, `.evaluate()`, `.get_test_data()`, `.preprocessor.get_feature_names()` (all existing, unchanged, from `src/core/pipeline.py`); `ExplainabilityAnalyzer(model, X_train, X_test, feature_names).get_feature_importance(use_cache=False, y_test=y_test)` (from `src/explain/explainability.py`, already fixed to accept real `y_test`).
- Produces: `TrainResponse` with `metrics: dict[str, float]`, `feature_importance: dict[str, float]`, `model_name: str`. Sets `state.trained_model`, `state.trained_model_name`, `state.model_features`, `state.target_column` — consumed by Task 4 (predict).

- [ ] **Step 1: Write failing test in `backend/tests/test_training.py`**

```python
def test_train_random_forest(client):
    client.post("/api/datasets/demo")
    response = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["metrics"]["accuracy"] <= 1.0
    assert isinstance(body["feature_importance"], dict)
    assert len(body["feature_importance"]) > 0


def test_train_without_dataset_returns_400(client):
    response = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert response.status_code == 400
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd backend && python -m pytest tests/test_training.py -v`
Expected: FAIL — 404, route doesn't exist

- [ ] **Step 3: Add training schemas to `backend/app/schemas.py`**

Append:

```python
class TrainRequest(BaseModel):
    target: str
    features: Optional[list[str]] = None
    method: str = "random_forest"


class TrainResponse(BaseModel):
    model_name: str
    metrics: dict[str, float]
    feature_importance: dict[str, float]
```

- [ ] **Step 4: Write `backend/app/routers/training.py`**

```python
from fastapi import APIRouter, HTTPException

from app.schemas import TrainRequest, TrainResponse
from app.state import get_state
from src.core.pipeline import MLPipeline
from src.explain.explainability import ExplainabilityAnalyzer

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("", response_model=TrainResponse)
def train_model(request: TrainRequest):
    state = get_state()
    if state.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded — call /api/datasets/demo or /api/datasets/upload first")
    if request.target not in state.df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target}' not found in dataset")

    df = state.df
    if request.features:
        df = df[request.features + [request.target]]

    pipeline = MLPipeline(df, target_col=request.target)
    is_valid, message = pipeline.validate()
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    pipeline.preprocess()
    train_results = pipeline.train([request.method])
    if train_results[request.method].get("status") == "failed":
        raise HTTPException(status_code=500, detail=train_results[request.method]["error"])

    eval_results = pipeline.evaluate([request.method])
    model = pipeline.model_manager.get_models()[request.method]

    X_train, X_test = pipeline.X_train, pipeline.X_test
    y_test = pipeline.y_test
    analyzer = ExplainabilityAnalyzer(model, X_train, X_test, pipeline.preprocessor.get_feature_names())
    importance = analyzer.get_feature_importance(use_cache=False, y_test=y_test)

    state.pipeline = pipeline
    state.trained_model = model
    state.trained_model_name = request.method
    state.model_features = pipeline.preprocessor.get_feature_names()
    state.target_column = request.target

    return TrainResponse(
        model_name=request.method,
        metrics=eval_results[request.method],
        feature_importance=importance,
    )
```

- [ ] **Step 5: Register router in `backend/app/main.py`**

Modify — add:

```python
from app.routers import training

app.include_router(training.router)
```

- [ ] **Step 6: Run tests, verify pass**

Run: `cd backend && python -m pytest tests/test_training.py -v`
Expected: both tests PASS

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "Add model training endpoint"
```

---

## Task 4: Prediction and what-if endpoints

**Files:**
- Modify: `backend/app/schemas.py` (add prediction models)
- Create: `backend/app/routers/predictions.py`
- Modify: `backend/app/main.py` (register router)
- Test: `backend/tests/test_predictions.py`

**Interfaces:**
- Consumes: `state.trained_model`, `state.model_features`, `state.pipeline` from Task 3; `state.audit_logger.log_prediction(prediction, probability, confidence, model_name, threshold, recommended_action, feature_values)` (existing, `src/decision/decision_audit_logger.py:21`).
- Produces: `PredictResponse` with `prediction: int`, `probability: float`, `confidence: float`. Consumed by Task 5 (`/api/audit-logs` reads what this endpoint logs).

- [ ] **Step 1: Write failing test in `backend/tests/test_predictions.py`**

```python
def test_predict_after_training(client):
    client.post("/api/datasets/demo")
    train_resp = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    features = train_resp.json()
    from app.state import get_state
    state = get_state()
    sample = {f: float(state.df[f].iloc[0]) if state.df[f].dtype.kind in "if" else 0 for f in state.model_features}

    response = client.post("/api/predict", json={"features": sample})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_without_trained_model_returns_400(client):
    response = client.post("/api/predict", json={"features": {}})
    assert response.status_code == 400
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd backend && python -m pytest tests/test_predictions.py -v`
Expected: FAIL — 404

- [ ] **Step 3: Add prediction schemas to `backend/app/schemas.py`**

Append:

```python
class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    confidence: float


class WhatIfRequest(BaseModel):
    baseline_features: dict[str, Any]
    scenario_features: dict[str, Any]


class WhatIfResponse(BaseModel):
    baseline: PredictResponse
    scenario: PredictResponse
    delta_probability: float
```

- [ ] **Step 4: Write `backend/app/routers/predictions.py`**

```python
import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas import PredictRequest, PredictResponse, WhatIfRequest, WhatIfResponse
from app.state import get_state

router = APIRouter(prefix="/api", tags=["predictions"])


def _predict_one(features: dict) -> PredictResponse:
    state = get_state()
    if state.trained_model is None:
        raise HTTPException(status_code=400, detail="No trained model — call /api/train first")

    row = np.array([[features.get(f, 0) for f in state.model_features]])
    prediction = int(state.trained_model.predict(row)[0])

    if hasattr(state.trained_model, "predict_proba"):
        proba = state.trained_model.predict_proba(row)[0]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = float(max(proba))
    else:
        probability = float(prediction)
        confidence = 1.0

    state.audit_logger.log_prediction(
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        model_name=state.trained_model_name,
        feature_values=features,
    )

    return PredictResponse(prediction=prediction, probability=probability, confidence=confidence)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return _predict_one(request.features)


@router.post("/whatif", response_model=WhatIfResponse)
def whatif(request: WhatIfRequest):
    baseline = _predict_one(request.baseline_features)
    scenario = _predict_one(request.scenario_features)
    return WhatIfResponse(
        baseline=baseline,
        scenario=scenario,
        delta_probability=round(scenario.probability - baseline.probability, 4),
    )
```

- [ ] **Step 5: Register router in `backend/app/main.py`**

Modify — add:

```python
from app.routers import predictions

app.include_router(predictions.router)
```

- [ ] **Step 6: Run tests, verify pass**

Run: `cd backend && python -m pytest tests/test_predictions.py -v`
Expected: both tests PASS

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "Add prediction and what-if endpoints"
```

---

## Task 5: Audit log and model download endpoints

**Files:**
- Modify: `backend/app/schemas.py` (add audit models)
- Create: `backend/app/routers/audit.py`
- Create: `backend/app/routers/models.py`
- Modify: `backend/app/main.py` (register both routers)
- Test: `backend/tests/test_audit.py`

**Interfaces:**
- Consumes: `state.audit_logger.get_audit_trail()` (existing, `src/decision/decision_audit_logger.py:157`); `ModelSerializer.save_model(model, model_name, metadata, overwrite=True)` / `.load_model(model_name)` (existing, `src/artifacts/model_serializer.py:32,95`).
- Produces: `GET /api/audit-logs` returning `list[dict]`; `GET /api/models/{name}/download` returning a `FileResponse` of the saved `.joblib`.

- [ ] **Step 1: Write failing test in `backend/tests/test_audit.py`**

```python
def test_audit_logs_after_prediction(client):
    client.post("/api/datasets/demo")
    client.post("/api/train", json={"target": "churn", "features": None, "method": "random_forest"})
    from app.state import get_state
    state = get_state()
    sample = {f: float(state.df[f].iloc[0]) if state.df[f].dtype.kind in "if" else 0 for f in state.model_features}
    client.post("/api/predict", json={"features": sample})

    response = client.get("/api/audit-logs")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["prediction"] in (0, 1)


def test_audit_logs_empty_before_predictions(client):
    response = client.get("/api/audit-logs")
    assert response.status_code == 200
    assert response.json() == []
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd backend && python -m pytest tests/test_audit.py -v`
Expected: FAIL — 404

- [ ] **Step 3: Write `backend/app/routers/audit.py`**

```python
from fastapi import APIRouter

from app.state import get_state

router = APIRouter(prefix="/api", tags=["audit"])


@router.get("/audit-logs")
def get_audit_logs():
    state = get_state()
    return state.audit_logger.get_audit_trail()
```

- [ ] **Step 4: Write `backend/app/routers/models.py`**

```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.artifacts.model_serializer import ModelSerializer

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/{name}/download")
def download_model(name: str):
    if not ModelSerializer.model_exists(name):
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    info = ModelSerializer.get_model_info(name)
    return FileResponse(info["model_path"], filename=f"{name}.joblib")
```

- [ ] **Step 5: Register routers in `backend/app/main.py`**

Modify — add:

```python
from app.routers import audit, models

app.include_router(audit.router)
app.include_router(models.router)
```

- [ ] **Step 6: Run tests, verify pass**

Run: `cd backend && python -m pytest tests/test_audit.py -v`
Expected: both tests PASS

- [ ] **Step 7: Run the full backend test suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all tests across Tasks 1-5 PASS

- [ ] **Step 8: Commit**

```bash
git add backend/
git commit -m "Add audit log and model download endpoints"
```

---

## Task 6: Frontend scaffold — Vite, Tailwind, shadcn, design tokens, API client, layout shell

**Files:**
- Create: `frontend/` (via `npm create vite@latest`)
- Create: `frontend/src/styles/globals.css`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/types.ts`
- Create: `frontend/src/lib/queryClient.ts`
- Create: `frontend/src/components/layout/AppShell.tsx`
- Create: `frontend/src/components/layout/Sidebar.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/.env.development`
- Modify: `frontend/src/main.tsx`

**Interfaces:**
- Produces: `apiClient` object in `frontend/src/lib/api.ts` with methods `loadDemo()`, `uploadCsv(file: File)`, `getCurrentDataset()`, `train(req: TrainRequest)`, `predict(req: PredictRequest)`, `whatif(req: WhatIfRequest)`, `getAuditLogs()` — all typed against `frontend/src/lib/types.ts`, all pages (Tasks 7-11) consume this.
- Produces: `<AppShell>` component wrapping `<Sidebar>` + page content, used by every route in `App.tsx`.

- [ ] **Step 1: Scaffold the Vite project**

Run: `cd frontend-tmp && npm create vite@latest . -- --template react-ts` then move contents into `frontend/` (or run directly inside an empty `frontend/` directory: `npm create vite@latest frontend -- --template react-ts`)
Run: `cd frontend && npm install`

- [ ] **Step 2: Install Tailwind, shadcn deps, and app deps**

Run: `cd frontend && npm install -D tailwindcss postcss autoprefixer @types/node`
Run: `cd frontend && npx tailwindcss init -p`
Run: `cd frontend && npm install react-router-dom @tanstack/react-query @tanstack/react-table recharts clsx tailwind-merge class-variance-authority @radix-ui/react-tabs @radix-ui/react-select lucide-react`

Note: `lucide-react` is used only for the small set of icons needed (upload, download, chevron, alert) — not the "generic icon library" the design skills warn against when overused; it's a pragmatic choice here since Phosphor's React package requires extra setup. Standardize stroke width to `1.5` everywhere it's used.

- [ ] **Step 3: Write `frontend/tailwind.config.ts`**

```typescript
import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: "#6366f1",
        "accent-light": "#818cf8",
        ink: "#18181b",
        canvas: "#fafaf9",
      },
      fontFamily: {
        sans: ["-apple-system", "SF Pro Display", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["SF Mono", "JetBrains Mono", "monospace"],
      },
      boxShadow: {
        tile: "0 10px 26px -18px rgba(0,0,0,0.12)",
        "tile-accent": "0 12px 30px -16px rgba(99,102,241,0.25)",
      },
    },
  },
  plugins: [],
} satisfies Config;
```

- [ ] **Step 4: Write `frontend/src/styles/globals.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --color-ink: #18181b;
  --color-canvas: #fafaf9;
  --color-accent: #6366f1;
  --color-border: rgba(0, 0, 0, 0.06);
}

body {
  background: var(--color-canvas);
  color: var(--color-ink);
  font-variant-numeric: tabular-nums;
}

.data-value {
  font-variant-numeric: tabular-nums;
}
```

- [ ] **Step 5: Write `frontend/src/lib/types.ts`**

```typescript
export interface DatasetSummary {
  rows: number;
  columns: number;
  numeric_columns: string[];
  categorical_columns: string[];
  missing_pct: number;
  warnings: string[];
}

export interface TrainRequest {
  target: string;
  features: string[] | null;
  method: string;
}

export interface TrainResponse {
  model_name: string;
  metrics: Record<string, number>;
  feature_importance: Record<string, number>;
}

export interface PredictRequest {
  features: Record<string, number | string>;
}

export interface PredictResponse {
  prediction: number;
  probability: number;
  confidence: number;
}

export interface WhatIfRequest {
  baseline_features: Record<string, number | string>;
  scenario_features: Record<string, number | string>;
}

export interface WhatIfResponse {
  baseline: PredictResponse;
  scenario: PredictResponse;
  delta_probability: number;
}

export interface AuditRecord {
  timestamp: string;
  prediction: number;
  probability: number;
  confidence: number;
  confidence_level: string;
  model: string;
  threshold: number;
  recommended_action: string | null;
  record_id: number;
  features?: Record<string, unknown>;
}
```

- [ ] **Step 6: Write `frontend/src/lib/api.ts`**

```typescript
import type {
  DatasetSummary,
  TrainRequest,
  TrainResponse,
  PredictRequest,
  PredictResponse,
  WhatIfRequest,
  WhatIfResponse,
  AuditRecord,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json();
}

export const apiClient = {
  loadDemo: () => request<DatasetSummary>("/api/datasets/demo", { method: "POST" }),

  uploadCsv: async (file: File): Promise<DatasetSummary> => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${BASE_URL}/api/datasets/upload`, { method: "POST", body: formData });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(body.detail ?? `Upload failed: ${response.status}`);
    }
    return response.json();
  },

  getCurrentDataset: () => request<DatasetSummary>("/api/datasets/current"),

  train: (req: TrainRequest) =>
    request<TrainResponse>("/api/train", { method: "POST", body: JSON.stringify(req) }),

  predict: (req: PredictRequest) =>
    request<PredictResponse>("/api/predict", { method: "POST", body: JSON.stringify(req) }),

  whatif: (req: WhatIfRequest) =>
    request<WhatIfResponse>("/api/whatif", { method: "POST", body: JSON.stringify(req) }),

  getAuditLogs: () => request<AuditRecord[]>("/api/audit-logs"),
};
```

- [ ] **Step 7: Write `frontend/src/lib/queryClient.ts`**

```typescript
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});
```

- [ ] **Step 8: Write `frontend/src/components/layout/Sidebar.tsx`**

```tsx
import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard" },
  { to: "/data", label: "Data Explorer" },
  { to: "/training", label: "Model Training" },
  { to: "/predictions", label: "Predictions" },
  { to: "/audit", label: "Audit Logs" },
];

export function Sidebar() {
  return (
    <nav className="w-16 bg-[#111113] flex flex-col items-center py-4 gap-5 shrink-0">
      <div className="w-5 h-5 rounded-md bg-gradient-to-br from-accent-light to-accent shadow-[0_0_16px_rgba(99,102,241,0.5)]" />
      <div className="flex flex-col gap-4 mt-2">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            title={item.label}
            className={({ isActive }) =>
              `w-7 h-7 rounded-lg border transition-colors ${
                isActive
                  ? "bg-accent/15 border-accent/30"
                  : "border-transparent hover:bg-white/5"
              }`
            }
          />
        ))}
      </div>
    </nav>
  );
}
```

- [ ] **Step 9: Write `frontend/src/components/layout/AppShell.tsx`**

```tsx
import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-canvas relative overflow-hidden">
      <div className="absolute -top-32 -right-20 w-[420px] h-[420px] rounded-full bg-[radial-gradient(circle,rgba(99,102,241,0.10),transparent_70%)] pointer-events-none" />
      <Sidebar />
      <main className="flex-1 p-8 relative z-10">{children}</main>
    </div>
  );
}
```

- [ ] **Step 10: Write `frontend/src/App.tsx`** (routes only reference page components — Tasks 7-11 create the actual files, this task creates placeholder stubs so the app compiles)

```tsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./pages/Dashboard";
import { DataExplorer } from "./pages/DataExplorer";
import { ModelTraining } from "./pages/ModelTraining";
import { Predictions } from "./pages/Predictions";
import { AuditLogs } from "./pages/AuditLogs";

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/data" element={<DataExplorer />} />
            <Route path="/training" element={<ModelTraining />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/audit" element={<AuditLogs />} />
          </Routes>
        </AppShell>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 11: Create placeholder page stubs so the build compiles** (Tasks 7-11 replace these)

Create `frontend/src/pages/Dashboard.tsx`, `DataExplorer.tsx`, `ModelTraining.tsx`, `Predictions.tsx`, `AuditLogs.tsx`, each with:

```tsx
export function Dashboard() {
  return <div>Dashboard</div>;
}
```

(same pattern, renaming the function per file: `DataExplorer`, `ModelTraining`, `Predictions`, `AuditLogs`)

- [ ] **Step 12: Update `frontend/src/main.tsx`**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

- [ ] **Step 13: Write `frontend/.env.development`**

```
VITE_API_BASE_URL=http://localhost:8000
```

- [ ] **Step 14: Verify the build compiles**

Run: `cd frontend && npm run build`
Expected: build succeeds with no TypeScript errors

- [ ] **Step 15: Commit**

```bash
git add frontend/
git commit -m "Scaffold React frontend with routing, API client, and design tokens"
```

---

## Task 7: Dashboard page

**Files:**
- Modify: `frontend/src/pages/Dashboard.tsx`
- Create: `frontend/src/components/HeroStat.tsx`
- Create: `frontend/src/components/StatTile.tsx`
- Create: `frontend/src/components/EmptyState.tsx`

**Interfaces:**
- Consumes: `apiClient.getCurrentDataset()`, `apiClient.getAuditLogs()` from Task 6.
- Produces: `<EmptyState>` and `<StatTile>` reused by Tasks 8-11.

- [ ] **Step 1: Write `frontend/src/components/EmptyState.tsx`**

```tsx
import type { ReactNode } from "react";

export function EmptyState({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-20 border border-dashed border-black/10 rounded-2xl">
      <div className="text-sm font-medium text-ink mb-1">{title}</div>
      <div className="text-xs text-stone-500 mb-4 max-w-xs">{description}</div>
      {action}
    </div>
  );
}
```

- [ ] **Step 2: Write `frontend/src/components/StatTile.tsx`**

```tsx
export function StatTile({ label, value, delta }: { label: string; value: string; delta?: string }) {
  return (
    <div className="bg-white rounded-2xl p-[18px] border border-black/5 shadow-tile">
      <div className="text-[11px] text-stone-500 uppercase tracking-wide mb-2">{label}</div>
      <div className="text-2xl font-semibold text-ink data-value">{value}</div>
      {delta && <div className="text-[11px] text-green-600 mt-1">{delta}</div>}
    </div>
  );
}
```

- [ ] **Step 3: Write `frontend/src/components/HeroStat.tsx`**

```tsx
export function HeroStat({ label, value, suffix }: { label: string; value: string; suffix?: string }) {
  return (
    <div className="bg-black/[0.03] rounded-[20px] p-1.5">
      <div className="bg-white rounded-2xl p-5 h-full border border-accent/10 shadow-tile-accent">
        <div className="text-[11px] text-stone-500 uppercase tracking-wide mb-2">{label}</div>
        <div className="text-4xl font-bold tracking-tight text-ink data-value">
          {value}
          {suffix && <span className="text-lg text-stone-400">{suffix}</span>}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Write `frontend/src/pages/Dashboard.tsx`**

```tsx
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { HeroStat } from "../components/HeroStat";
import { StatTile } from "../components/StatTile";
import { EmptyState } from "../components/EmptyState";
import { Link } from "react-router-dom";

export function Dashboard() {
  const datasetQuery = useQuery({
    queryKey: ["dataset"],
    queryFn: apiClient.getCurrentDataset,
    retry: false,
  });
  const auditQuery = useQuery({
    queryKey: ["audit-logs"],
    queryFn: apiClient.getAuditLogs,
    enabled: !datasetQuery.isError,
  });

  if (datasetQuery.isError) {
    return (
      <EmptyState
        title="No dataset loaded"
        description="Load a dataset from the Data Explorer page to get started."
        action={
          <Link to="/data" className="text-xs font-medium text-accent">
            Go to Data Explorer &rarr;
          </Link>
        }
      />
    );
  }

  const dataset = datasetQuery.data;
  const predictionsCount = auditQuery.data?.length ?? 0;

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Verdict</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Overview</div>
      </div>

      <div className="grid grid-cols-3 gap-3.5 mb-4">
        <HeroStat label="Dataset rows" value={dataset ? dataset.rows.toLocaleString() : "—"} />
        <StatTile label="Columns" value={dataset ? String(dataset.columns) : "—"} />
        <StatTile label="Predictions logged" value={String(predictionsCount)} />
      </div>

      {dataset && dataset.warnings.length > 0 && (
        <div className="bg-white rounded-2xl p-4 border border-black/5 shadow-tile text-xs text-stone-600">
          {dataset.warnings.map((w) => (
            <div key={w}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Verify build**

Run: `cd frontend && npm run build`
Expected: build succeeds

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "Add Dashboard page"
```

---

## Task 8: Data Explorer page

**Files:**
- Modify: `frontend/src/pages/DataExplorer.tsx`
- Create: `frontend/src/components/SkeletonBlock.tsx`

**Interfaces:**
- Consumes: `apiClient.loadDemo()`, `apiClient.uploadCsv(file)` from Task 6; `EmptyState`, `StatTile` from Task 7.
- Produces: `<SkeletonBlock>` reused by Tasks 9-11 for loading states.

- [ ] **Step 1: Write `frontend/src/components/SkeletonBlock.tsx`**

```tsx
export function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse bg-stone-100 rounded-lg ${className}`} />;
}
```

- [ ] **Step 2: Write `frontend/src/pages/DataExplorer.tsx`**

```tsx
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { apiClient } from "../lib/api";
import { StatTile } from "../components/StatTile";

export function DataExplorer() {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const demoMutation = useMutation({
    mutationFn: apiClient.loadDemo,
    onSuccess: () => {
      setError(null);
      queryClient.invalidateQueries({ queryKey: ["dataset"] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const uploadMutation = useMutation({
    mutationFn: apiClient.uploadCsv,
    onSuccess: () => {
      setError(null);
      queryClient.invalidateQueries({ queryKey: ["dataset"] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const summary = demoMutation.data ?? uploadMutation.data;

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Data</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Data Explorer</div>
      </div>

      <div className="flex gap-3 mb-6">
        <button
          onClick={() => demoMutation.mutate()}
          disabled={demoMutation.isPending}
          className="bg-ink text-white rounded-lg px-4 py-2 text-xs font-medium active:scale-[0.98] transition-transform disabled:opacity-50"
        >
          {demoMutation.isPending ? "Loading…" : "Use demo dataset"}
        </button>
        <label className="border border-black/10 bg-white rounded-lg px-4 py-2 text-xs font-medium cursor-pointer hover:bg-stone-50 transition-colors">
          Upload CSV
          <input
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) uploadMutation.mutate(file);
            }}
          />
        </label>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-100 text-red-700 text-xs rounded-lg px-4 py-3 mb-6">
          {error}
        </div>
      )}

      {summary && (
        <div className="grid grid-cols-4 gap-3.5 mb-6">
          <StatTile label="Rows" value={summary.rows.toLocaleString()} />
          <StatTile label="Columns" value={String(summary.columns)} />
          <StatTile label="Numeric" value={String(summary.numeric_columns.length)} />
          <StatTile label="Categorical" value={String(summary.categorical_columns.length)} />
        </div>
      )}

      {summary && summary.warnings.length > 0 && (
        <div className="bg-white rounded-2xl p-4 border border-black/5 shadow-tile text-xs text-stone-600">
          {summary.warnings.map((w) => (
            <div key={w}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

Run: `cd frontend && npm run build`
Expected: build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/
git commit -m "Add Data Explorer page"
```

---

## Task 9: Model Training page

**Files:**
- Modify: `frontend/src/pages/ModelTraining.tsx`
- Create: `frontend/src/components/Chip.tsx`

**Interfaces:**
- Consumes: `apiClient.getCurrentDataset()`, `apiClient.train(req)` from Tasks 6-7; `SkeletonBlock`, `EmptyState` from Tasks 7-8.
- Produces: nothing new consumed elsewhere.

- [ ] **Step 1: Write `frontend/src/components/Chip.tsx`**

```tsx
export function Chip({ label, selected, onClick }: { label: string; selected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`border rounded-md px-2 py-1 text-[11px] transition-colors ${
        selected
          ? "bg-white border-black/10 text-zinc-700"
          : "bg-stone-100 border-transparent text-stone-400"
      }`}
    >
      {label}
    </button>
  );
}
```

- [ ] **Step 2: Write `frontend/src/pages/ModelTraining.tsx`**

```tsx
import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { Chip } from "../components/Chip";
import { EmptyState } from "../components/EmptyState";
import { Link } from "react-router-dom";

export function ModelTraining() {
  const datasetQuery = useQuery({ queryKey: ["dataset"], queryFn: apiClient.getCurrentDataset, retry: false });
  const [target, setTarget] = useState<string>("");
  const [excludedFeatures, setExcludedFeatures] = useState<Set<string>>(new Set());

  const trainMutation = useMutation({ mutationFn: apiClient.train });

  if (datasetQuery.isError) {
    return (
      <EmptyState
        title="No dataset loaded"
        description="Load a dataset first."
        action={<Link to="/data" className="text-xs font-medium text-accent">Go to Data Explorer &rarr;</Link>}
      />
    );
  }

  const dataset = datasetQuery.data;
  const allColumns = dataset ? [...dataset.numeric_columns, ...dataset.categorical_columns] : [];
  const candidateFeatures = allColumns.filter((c) => c !== target);

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Model training</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Train a classifier</div>
        {dataset && (
          <div className="text-xs text-stone-400 data-value mt-0.5">
            {dataset.rows.toLocaleString()} rows &middot; {dataset.columns} columns
          </div>
        )}
      </div>

      <div className="mb-5">
        <div className="text-[11px] text-stone-500 mb-1.5">Target column</div>
        <select
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="border border-black/10 rounded-md px-2.5 py-2 text-xs bg-white w-64"
        >
          <option value="">Select…</option>
          {allColumns.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {target && (
        <div className="mb-5">
          <div className="text-[11px] text-stone-500 mb-1.5 data-value">
            Features &middot; {candidateFeatures.length - excludedFeatures.size} of {candidateFeatures.length} selected
          </div>
          <div className="flex flex-wrap gap-1.5">
            {candidateFeatures.map((f) => (
              <Chip
                key={f}
                label={f}
                selected={!excludedFeatures.has(f)}
                onClick={() =>
                  setExcludedFeatures((prev) => {
                    const next = new Set(prev);
                    if (next.has(f)) next.delete(f);
                    else next.add(f);
                    return next;
                  })
                }
              />
            ))}
          </div>
        </div>
      )}

      <button
        disabled={!target || trainMutation.isPending}
        onClick={() =>
          trainMutation.mutate({
            target,
            features: candidateFeatures.filter((f) => !excludedFeatures.has(f)),
            method: "random_forest",
          })
        }
        className="bg-ink text-white rounded-lg px-4 py-2 text-xs font-medium active:scale-[0.98] transition-transform disabled:opacity-40"
      >
        {trainMutation.isPending ? "Training…" : "Train model"}
      </button>

      {trainMutation.isError && (
        <div className="bg-red-50 border border-red-100 text-red-700 text-xs rounded-lg px-4 py-3 mt-4">
          {(trainMutation.error as Error).message}
        </div>
      )}

      {trainMutation.data && (
        <div className="mt-6 pt-5 border-t border-black/5">
          <div className="flex gap-6 mb-5">
            {Object.entries(trainMutation.data.metrics).map(([key, value]) => (
              <div key={key}>
                <div className="text-[10px] text-stone-400 uppercase tracking-wide">{key}</div>
                <div className="text-sm font-semibold text-ink data-value">{(value as number).toFixed(3)}</div>
              </div>
            ))}
          </div>

          <div className="text-xs font-medium text-ink mb-2">Feature importance</div>
          <div className="flex flex-col gap-2">
            {Object.entries(trainMutation.data.feature_importance)
              .slice(0, 8)
              .map(([feature, value]) => {
                const maxVal = Math.max(...Object.values(trainMutation.data!.feature_importance).map(Math.abs));
                const width = maxVal > 0 ? (Math.abs(value as number) / maxVal) * 100 : 0;
                return (
                  <div key={feature}>
                    <div className="flex justify-between text-[11px] text-stone-600 mb-0.5">
                      <span>{feature}</span>
                      <span className="data-value">{(value as number).toFixed(4)}</span>
                    </div>
                    <div className="h-[5px] bg-stone-100 rounded">
                      <div className="h-full bg-accent rounded" style={{ width: `${width}%` }} />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

Run: `cd frontend && npm run build`
Expected: build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/
git commit -m "Add Model Training page"
```

---

## Task 10: Predictions page

**Files:**
- Modify: `frontend/src/pages/Predictions.tsx`

**Interfaces:**
- Consumes: `apiClient.predict(req)` from Task 6; `EmptyState` from Task 7.

- [ ] **Step 1: Write `frontend/src/pages/Predictions.tsx`**

```tsx
import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { EmptyState } from "../components/EmptyState";
import { Link } from "react-router-dom";

export function Predictions() {
  const datasetQuery = useQuery({ queryKey: ["dataset"], queryFn: apiClient.getCurrentDataset, retry: false });
  const [values, setValues] = useState<Record<string, string>>({});

  const predictMutation = useMutation({ mutationFn: apiClient.predict });

  if (datasetQuery.isError) {
    return (
      <EmptyState
        title="No dataset loaded"
        description="Load and train a model first."
        action={<Link to="/data" className="text-xs font-medium text-accent">Go to Data Explorer &rarr;</Link>}
      />
    );
  }

  const dataset = datasetQuery.data;
  const featureColumns = dataset ? [...dataset.numeric_columns, ...dataset.categorical_columns] : [];

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Predictions</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Run a prediction</div>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-5">
        {featureColumns.map((f) => (
          <div key={f}>
            <label className="text-[11px] text-stone-500 mb-1 block">{f}</label>
            <input
              type="text"
              value={values[f] ?? ""}
              onChange={(e) => setValues((prev) => ({ ...prev, [f]: e.target.value }))}
              className="border border-black/10 rounded-md px-2.5 py-1.5 text-xs w-full bg-white"
            />
          </div>
        ))}
      </div>

      <button
        onClick={() => {
          const features: Record<string, number | string> = {};
          for (const [key, value] of Object.entries(values)) {
            const numeric = Number(value);
            features[key] = Number.isNaN(numeric) ? value : numeric;
          }
          predictMutation.mutate({ features });
        }}
        disabled={predictMutation.isPending}
        className="bg-ink text-white rounded-lg px-4 py-2 text-xs font-medium active:scale-[0.98] transition-transform disabled:opacity-40"
      >
        {predictMutation.isPending ? "Predicting…" : "Predict"}
      </button>

      {predictMutation.isError && (
        <div className="bg-red-50 border border-red-100 text-red-700 text-xs rounded-lg px-4 py-3 mt-4">
          {(predictMutation.error as Error).message}
        </div>
      )}

      {predictMutation.data && (
        <div className="mt-6 pt-5 border-t border-black/5 flex gap-6">
          <div>
            <div className="text-[10px] text-stone-400 uppercase tracking-wide">Prediction</div>
            <div className="text-2xl font-semibold text-ink data-value">{predictMutation.data.prediction}</div>
          </div>
          <div>
            <div className="text-[10px] text-stone-400 uppercase tracking-wide">Probability</div>
            <div className="text-2xl font-semibold text-ink data-value">
              {(predictMutation.data.probability * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-[10px] text-stone-400 uppercase tracking-wide">Confidence</div>
            <div className="text-2xl font-semibold text-ink data-value">
              {(predictMutation.data.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && npm run build`
Expected: build succeeds

- [ ] **Step 3: Commit**

```bash
git add frontend/
git commit -m "Add Predictions page"
```

---

## Task 11: Audit Logs page

**Files:**
- Modify: `frontend/src/pages/AuditLogs.tsx`

**Interfaces:**
- Consumes: `apiClient.getAuditLogs()` from Task 6; `EmptyState` from Task 7; `AuditRecord` type from Task 6.

- [ ] **Step 1: Write `frontend/src/pages/AuditLogs.tsx`**

```tsx
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { EmptyState } from "../components/EmptyState";

export function AuditLogs() {
  const auditQuery = useQuery({ queryKey: ["audit-logs"], queryFn: apiClient.getAuditLogs });

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Audit</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Prediction log</div>
      </div>

      {auditQuery.data && auditQuery.data.length === 0 && (
        <EmptyState title="No predictions yet" description="Predictions you make will be logged here." />
      )}

      {auditQuery.data && auditQuery.data.length > 0 && (
        <div className="bg-white rounded-2xl border border-black/5 shadow-tile overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-black/5 text-stone-500 uppercase text-[10px] tracking-wide">
                <th className="text-left px-4 py-2.5 font-medium">Timestamp</th>
                <th className="text-left px-4 py-2.5 font-medium">Model</th>
                <th className="text-left px-4 py-2.5 font-medium">Prediction</th>
                <th className="text-left px-4 py-2.5 font-medium">Probability</th>
                <th className="text-left px-4 py-2.5 font-medium">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {auditQuery.data.map((record) => (
                <tr key={record.record_id} className="border-b border-black/5 last:border-0">
                  <td className="px-4 py-2.5 text-stone-500 data-value">{new Date(record.timestamp).toLocaleString()}</td>
                  <td className="px-4 py-2.5 text-ink">{record.model}</td>
                  <td className="px-4 py-2.5 text-ink data-value">{record.prediction}</td>
                  <td className="px-4 py-2.5 text-ink data-value">{(record.probability * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2.5 text-ink">{record.confidence_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && npm run build`
Expected: build succeeds

- [ ] **Step 3: Commit**

```bash
git add frontend/
git commit -m "Add Audit Logs page"
```

---

## Task 12: Single-container production serving

**Files:**
- Modify: `backend/app/main.py` (mount built frontend as static files)
- Modify: `app.py` (root-level entry point, update to launch FastAPI+static instead of Streamlit)

**Interfaces:**
- Consumes: `frontend/dist/` build output from Task 6-11's `npm run build`.

- [ ] **Step 1: Modify `backend/app/main.py`** — add after all router registrations, before end of file:

```python
import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
```

- [ ] **Step 2: Modify `app.py`** — replace the `run_app` function body to launch uvicorn instead of Streamlit:

```python
def run_app():
    """Run the FastAPI backend, serving the built frontend as static files."""
    import uvicorn

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          VERDICT ML Platform - Starting Up                 ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    port = int(os.getenv("PORT", 8000))
    print(f"Server will run on port {port}\n")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port)
```

- [ ] **Step 3: Build the frontend for production**

Run: `cd frontend && npm run build`
Expected: `frontend/dist/` directory created

- [ ] **Step 4: Verify the combined server serves both API and frontend**

Run: `python app.py` (from repo root, in a terminal — this is a manual verification step, not automated)
Then: `curl http://localhost:8000/api/health` → expect `{"status":"ok"}`
Then: open `http://localhost:8000/` in a browser → expect the React dashboard to load

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py app.py
git commit -m "Serve built frontend from FastAPI in production"
```

---

## Task 13: Full golden-path smoke test

**Files:** none (verification only)

- [ ] **Step 1: Start both dev servers**

Run backend: `cd backend && uvicorn app.main:app --reload --port 8000`
Run frontend: `cd frontend && npm run dev` (Vite dev server, typically port 5173)

- [ ] **Step 2: Walk the golden path in a browser** (use the Browser preview tool, not a manual claim)

1. Navigate to the frontend dev URL — Dashboard shows the "No dataset loaded" empty state
2. Go to Data Explorer, click "Use demo dataset" — stat tiles populate with 5,000 rows / 25 columns
3. Go to Model Training, select `churn` as target, leave all features selected, click "Train model" — metrics and feature importance bars render
4. Go to Predictions, fill in feature values, click "Predict" — prediction/probability/confidence render
5. Go to Audit Logs — the prediction from step 4 appears in the table
6. Go back to Dashboard — "Predictions logged" stat tile now shows 1

- [ ] **Step 3: Verify no console errors**

Use `read_console_messages` (or browser devtools) on each page — no red errors.

- [ ] **Step 4: Run the full Python test suite one more time to confirm nothing in `src/` broke**

Run: `python -m pytest tests/ backend/tests/ -q` (from repo root)
Expected: all tests pass (317 existing + new backend tests)

- [ ] **Step 5: Final commit if any fixes were needed during smoke testing**

```bash
git add -A
git commit -m "Fix issues found during golden-path smoke test"
```

---

## Self-Review Notes

- **Spec coverage:** All 5 pages (Dashboard, Data Explorer, Model Training, Predictions, Audit Logs) have tasks. Backend API surface from the spec is fully covered (Tasks 1-5). Design system tokens (zinc/indigo palette, tabular nums, no emoji, hairline dividers, double-bezel hero stat, skeleton loaders, empty/error states) are implemented in Task 6 (tokens) and used consistently across Tasks 7-11. Single-container deploy is Task 12. Out-of-scope items (auth, regression models, Vercel split, GSAP-tier motion) are correctly absent from all tasks.
- **Type consistency checked:** `DatasetSummary`, `TrainRequest`/`TrainResponse`, `PredictRequest`/`PredictResponse`, `WhatIfRequest`/`WhatIfResponse`, `AuditRecord` are defined once in Task 2-5 (backend) and Task 6 (frontend `types.ts`) and referenced by the same names in every later task — no renamed fields.
- **`what-if` endpoint exists in the backend (Task 4) but no frontend page consumes it yet** — the spec lists it under Predictions page scope but the Task 10 Predictions page only calls `/api/predict`. This is a known gap: what-if UI (baseline vs. scenario comparison) can be added as a follow-up task after the golden path is verified, since it's additive to an already-working Predictions page and not required to unblock the rest of the plan.
