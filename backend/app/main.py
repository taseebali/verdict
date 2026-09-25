from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import uploads
from app.routers import datasets, results, training

app = FastAPI(title="Verdict API")
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(results.router)

UPLOAD_PATHS = {"/api/datasets/upload", "/api/results/score"}


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    """Refuse big uploads from Content-Length alone, before Starlette buffers the
    multipart body. The in-handler check still covers requests without the header."""
    if request.method == "POST" and request.url.path in UPLOAD_PATHS:
        limit = uploads.MAX_UPLOAD_BYTES  # read per request so tests can monkeypatch it
        try:
            length = int(request.headers.get("content-length", ""))
        except ValueError:
            length = 0
        if length > limit + 64 * 1024:
            return JSONResponse({"detail": f"File too large — the limit is {limit // (1024 * 1024)} MB."},
                                status_code=413)
    return await call_next(request)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve built frontend files, falling back to index.html for client-side routes."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)

        root = FRONTEND_DIST.resolve()
        candidate = (root / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        return FileResponse(root / "index.html")
