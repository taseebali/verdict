# Verdict

Upload a customer CSV, pick what you want to predict (churn, default, upgrade…), and Verdict trains a classifier, tells you how good it is, and scores new records — with every prediction logged for audit.

![CI](https://github.com/taseebali/verdict/actions/workflows/ci.yml/badge.svg)

## Try it

```bash
docker compose up --build
```

Open http://localhost:8000, click **Use demo dataset**, train on `churn`, then run a prediction.

## Run locally (dev)

Backend (Python 3.12):

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash; macOS/Linux: source .venv/bin/activate; PowerShell: .venv\Scripts\activate
pip install -r requirements-dev.txt
python app.py                 # API on http://localhost:8000
```

Frontend (Node 22), in a second terminal:

```bash
cd frontend
npm install
npm run dev                   # UI on http://localhost:5173, talks to :8000
```

Tests:

```bash
python -m pytest -q
```

## How it works

```
React (Vite, TanStack Query)  ──HTTP──▶  FastAPI  ──▶  src/core      preprocessing, training, metrics
                                                  ├─▶  src/explain   permutation feature importance
                                                  ├─▶  src/decision  audit logging
                                                  └─▶  src/artifacts model persistence (joblib)
```

- **Preprocessing:** text categories are label-encoded, identifier columns (all-unique text) are dropped, numeric text like `"29.85"` is parsed, and the scaler is fit on the training split only.
- **Models:** Random Forest or Logistic Regression, stratified 80/20 split.
- **Targets:** classification only — binary or multiclass. Continuous columns are rejected with a clear message.
- **Predictions** return the original label (`"Yes"`, not `1`) plus probability and confidence.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/datasets/demo` | Load bundled demo data |
| POST | `/api/datasets/upload` | Upload a CSV (≤ 50 MB) |
| GET | `/api/datasets/current` | Current dataset summary |
| POST | `/api/train` | Train `{target, features?, method}` |
| POST | `/api/predict` | Score one record |
| POST | `/api/whatif` | Compare baseline vs. changed record |
| GET | `/api/audit-logs` | Prediction history |
| GET | `/api/models/{name}/download` | Download trained model |

Interactive docs at `/docs` when the server is running.

## Limitations

- Single-user: all visitors share one in-memory dataset and model.
- The audit log resets when the server restarts.

## License

MIT — see [LICENSE](LICENSE).
