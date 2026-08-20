from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import datasets, training, predictions, audit, models

app = FastAPI(title="Verdict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(predictions.router)
app.include_router(audit.router)
app.include_router(models.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve built frontend files, falling back to index.html for client-side routes."""
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
