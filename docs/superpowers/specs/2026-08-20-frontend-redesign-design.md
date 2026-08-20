# Verdict frontend redesign

## Why

The Streamlit dashboard works but looks cluttered: default styling, emoji-numbered step headers, giant wrapping multiselect pills, no real visual hierarchy. User wants a genuinely better frontend, open to replacing the whole UI layer.

## Decisions made (via brainstorming + mockups)

- **Full rewrite of the frontend**, not a Streamlit theme patch. Streamlit's per-page reload model and widget defaults are the actual source of the clutter, not just missing CSS.
- **Stack:** React + Vite + TypeScript, shadcn/ui (Tailwind + Radix) + Recharts + TanStack Table + TanStack Query.
- **Backend:** FastAPI wrapping the existing `MLPipeline`, `ModelSerializer`, `EnsembleManager`, `ExplainabilityAnalyzer`, `DecisionAuditLogger` classes — thin routing/serialization layer, core ML logic untouched (317 existing tests keep validating it).
- **State:** single active dataset/model per server process (in-memory), mirrors current Streamlit `session_state` behavior. No auth, no multi-tenancy, no DB — matches this being a single-user tool.
- **Deployment:** primary target is HF Spaces — FastAPI serves the built React static files from one container, same one-container story as today. Frontend calls a configurable `VITE_API_BASE_URL` so splitting it onto Vercel later (frontend) + a separate host (backend) is a config change, not a rewrite.
- **Scope:** full golden path in the first build — Dashboard/overview, Data Explorer, Model Training, Predictions, Audit Logs.

## Visual design system

Synthesized from the design-taste skills installed at `~/.agents/skills/` (design-taste-frontend-v1, high-end-visual-design, minimalist-ui, redesign-existing-projects, stitch-design-taste), filtered down to what applies to a data-dense internal tool rather than a marketing site. Validated against the user via two mockup rounds in the brainstorming visual companion.

**Direction:** dark slate sidebar (`#111113`), light content area (`#fafaf9`/`#fff`), single indigo accent (`#6366f1`), used sparingly (active nav, primary actions, chart lines) — not spread across every element.

- **Typography:** system sans stack approximating Geist (`-apple-system, 'SF Pro Display', 'Segoe UI', system-ui`) for UI text; tabular/monospace numerals (`font-variant-numeric: tabular-nums`) for all metrics and data. No Inter as the "premium" choice, no serif anywhere (dashboard context). Headlines controlled via weight/tracking, not raw size.
- **Color:** neutral zinc/stone family throughout — never pure black (`#1c1917`/`#18181b` for text) or pure white background (`#fafaf9`). One accent only, capped saturation. No purple/neon "AI gradient" cliché beyond the single considered indigo.
- **Cards/surfaces:** hairline borders (`rgba(0,0,0,0.05-0.06)`) with tinted, diffused shadows (`0 10px 26px -18px rgba(0,0,0,0.12)`) instead of flat `box-shadow` or heavy borders. Cards used only where elevation communicates real hierarchy (stat tiles, chart panels) — dense areas (feature chip lists, data tables) use hairline dividers instead of boxing everything.
- **Hero metric card:** double-bezel treatment (outer tinted shell + inner white core) for the single most important number on a page.
- **No emoji anywhere** — replaced by a small consistent icon set (Phosphor or Radix icons, one stroke weight) and plain text labels. This is the single biggest fix for the "clutter" complaint — the current step badges (`1️⃣2️⃣3️⃣`) and heading icons go away entirely.
- **Components:** thin tab-style step indicator (not numbered emoji badges) for the training wizard; compact bordered chips (not oversized colored pills) for feature selection; skeleton loaders during training/prediction instead of spinners; designed empty states before data is loaded; inline error states, no `window.alert`/generic toasts only.
- **Motion:** hover/active feedback on every interactive element (subtle lift + tinted shadow growth, `scale(0.98)` on press), staggered fade-in on initial load, `transform`/`opacity`-only animations (no layout-triggering properties). Restrained — this is a working tool, not a landing page, so no scroll-triggered choreography, no perpetual looping micro-animations.

## API surface (backend/)

| Endpoint | Wraps |
|---|---|
| `POST /api/datasets/demo` | loads `data/verdict_demo.csv` |
| `POST /api/datasets/upload` | CSV upload, returns quality report |
| `GET /api/datasets/current/columns` | numeric/categorical column info |
| `POST /api/train` | `{target, features, method, params}` → `MLPipeline`/`EnsembleManager`, returns metrics + feature importance |
| `POST /api/predict` | feature values → prediction + confidence, logs via `DecisionAuditLogger` |
| `POST /api/whatif` | wraps `src/explain/whatif.py` |
| `GET /api/audit-logs` | paginated audit trail |
| `GET /api/models/{name}/download` | reuses `ModelSerializer` |

All JSON in/out. Errors return `{detail: str}` with appropriate status codes, surfaced as inline error states in the UI (per the redesign-existing-projects rule against generic alerts).

## Pages (frontend/src/)

Same 5 areas as today, sharing a persistent sidebar layout (fixes the full-page-reload-per-nav-click feel of Streamlit):
1. **Dashboard** — overview stat row + accuracy trend chart + top features (this was mocked up)
2. **Data Explorer** — upload/demo load, quality report, column stats, distributions
3. **Model Training** — target/feature selection, training method, results (this was mocked up)
4. **Predictions** — feature input form, prediction + confidence, what-if comparison
5. **Audit Logs** — paginated table of past predictions

## Out of scope for this pass

- Auth/multi-user sessions
- Regression models (Prophet/statsmodels) — still judged out of scope per earlier analysis
- Deploying to Vercel — architecture supports it later, not building it now
- The full "agency-grade" motion arsenal from the taste skills (GSAP scroll choreography, magnetic buttons, particle effects) — deliberately excluded, this is a working tool not a marketing site

## Testing

- Existing 317 Python tests keep validating the ML core unchanged.
- New: FastAPI endpoint tests (`backend/tests/`) covering the API surface above.
- Manual smoke test of the golden path (upload/demo → explore → train → predict → audit log) via the browser preview tool before calling this done.
