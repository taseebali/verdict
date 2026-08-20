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


import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
