"""
VERDICT ML Platform - Main Application
Entry point for local runs and the Docker image.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
# backend/app/main.py uses absolute imports (e.g. `from app.routers import ...`)
# that assume backend/ itself is on sys.path, same as backend/tests/conftest.py does.
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def run_app():
    """Run the FastAPI backend, serving the built frontend as static files."""
    import uvicorn

    # Hugging Face Spaces (Gradio SDK) run `python app.py`, set SPACE_ID, and expect port 7860.
    port = int(os.getenv("PORT") or (7860 if os.getenv("SPACE_ID") else 8000))
    print(f"Verdict running on http://localhost:{port}")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_app()