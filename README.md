# Verdict

**Upload customer records. Get a verdict on who to act on.**

Verdict learns from your history, scores every row honestly with out-of-fold predictions, explains each score, and turns your costs into a recommendation: *"Act on 2,429 of 5,000 rows. Net: +$142,670."* (the demo dataset at default costs).

![CI](https://github.com/taseebali/verdict/actions/workflows/ci.yml/badge.svg)

**Live demo:** see [Deploy on Render](#deploy-on-render-free). Once it's live, put its URL here.

## What it does

1. **Data:** load the demo churn dataset or upload a CSV (≤ 20 MB; row limit set by `VERDICT_MAX_ROWS`, 100,000 by default). ID-like columns are detected and left out.
2. **Outcome:** pick the column that records what happened and the outcome you want to catch (e.g. `churn = 1`).
3. **Verdict:**
   - **Decision:** enter the cost of acting, the value of a save and the success rate. Verdict recommends the risk cutoff with the highest expected net value, and you can drag it to see the trade-off.
   - **At-risk list:** every row ranked by risk, with the top reasons behind each score (SHAP for Random Forest, coefficient contributions for Logistic Regression). Click any row to test what-if changes, export the list as CSV, or score a new file with the trained model.
   - **What drives it:** plain-language segments (e.g. "support_tickets 5 – 17: 78% have churn = 1, vs 28% overall") plus permutation importance.
   - **Under the hood:** ROC AUC, base rate, precision, recall and a confusion matrix at your cutoff, plus the ROC curve.

## Screenshots

| The verdict | What drives it |
|---|---|
| ![Verdict headline, cost controls and ranked list](docs/screenshots/verdict.png) | ![Plain-language drivers](docs/screenshots/drivers.png) |
| **Data step** | **Under the hood** |
| ![Data step with preview](docs/screenshots/data.png) | ![Metrics, confusion matrix and ROC](docs/screenshots/under-the-hood.png) |

## Why the numbers are honest

- Every row is scored by a model that **never saw it**: 5-fold stratified out-of-fold predictions.
- Imputation, scaling and encoding live **inside** the cross-validated sklearn pipeline, so nothing leaks from the test folds.
- The recommendation uses those out-of-fold scores and the true outcomes, so the net value isn't flattered by training fit.

## Privacy

Your data stays in memory for this session only (1 hour) and is never saved. Each visitor gets an anonymous session cookie. There are no accounts and nothing is written to disk.

## Try it locally

```bash
docker compose up --build
```

Open http://localhost:7860 and click **Try the demo dataset**.

## Develop

Backend (Python 3.12):

```bash
python -m venv .venv
source .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python app.py                   # API on http://localhost:8000
```

Frontend (Node 22), in a second terminal:

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173 (proxies /api to :8000)
```

Tests:

```bash
python -m pytest -q
```

## How it works

```
React (Vite, TanStack Query, Tailwind, Recharts)
        │  same-origin /api, session cookie
FastAPI ├─ sessions.py        in-memory per-visitor store (30 max, 1 h idle expiry)
        ├─ routers/datasets   upload, profile, ID detection
        ├─ routers/training   5-fold out-of-fold scoring + final model
        └─ routers/results    decision curve, ranked rows + reasons, export, what-if, new-file scoring
src/core/scoring.py           sklearn ColumnTransformer pipeline, permutation importance, drivers, SHAP reasons
src/decision/decision_curve.py  net value per cutoff, recommendation
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/datasets/demo` | Load the demo dataset |
| POST | `/api/datasets/upload` | Upload a CSV |
| GET | `/api/datasets/current` | Current dataset profile |
| POST | `/api/train` | `{target, positive_class, method?, excluded?}` → summary |
| GET | `/api/results/summary` | Metrics, drivers, importance |
| POST | `/api/results/decision` | `{action_cost, saved_value, success_rate}` → curve + recommendation |
| GET | `/api/results/rows` | Ranked rows with reasons (`source=training\|new`, `offset`, `limit`) |
| GET | `/api/results/export.csv` | Ranked CSV with probability, flag and reasons |
| POST | `/api/results/whatif` | `{row_id, changes}` → baseline vs scenario |
| POST | `/api/results/score` | Score a new CSV with the trained model |
| POST | `/api/results/new/decision` | Expected net for the new file at a cutoff |

Interactive docs are at `/docs` while the server is running.

## Deploy on Render (free)

The repo includes a `render.yaml` Blueprint that builds the Dockerfile on Render's free plan.

1. Sign in at https://render.com with GitHub.
2. Choose **New → Blueprint**, select this repository, and click **Apply**.
3. Render builds the image and deploys every push to `main`. The URL is `https://verdict-<suffix>.onrender.com`.

The free plan has 512 MB RAM and 0.1 vCPU and sleeps after 15 minutes idle, so the first visit takes about a minute to wake. To fit that, `render.yaml` sets smaller limits through environment variables:

| Variable | Default | On Render |
|---|---|---|
| `VERDICT_RF_TREES` | 100 | 40 (faster training) |
| `VERDICT_MAX_SESSIONS` | 30 | 5 (memory) |
| `VERDICT_MAX_ROWS` | 100,000 | 20,000 |

## Limitations

- Classification outcomes only (2–20 distinct values). Continuous targets are rejected with a clear message.
- Sessions live in memory: a restart or 1 hour idle clears your data.
- Upload limits: 20 MB, and 100,000 rows by default (`VERDICT_MAX_ROWS`).
- Reasons in the CSV export are filled for the first 1,000 flagged rows.

## License

MIT — see [LICENSE](LICENSE).
