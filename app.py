"""
VERDICT ML Platform - Main Application
Root-level entry point for HuggingFace Spaces & Docker deployment
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

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          VERDICT ML Platform - Starting Up                 ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    port = int(os.getenv("PORT", 8000))
    print(f"Server will run on port {port}\n")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_app()