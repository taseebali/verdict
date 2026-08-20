"""
VERDICT ML Platform - Main Application
Root-level entry point for HuggingFace Spaces & Docker deployment
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def run_app():
    """Run the Streamlit UI."""

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          🎯 VERDICT ML Platform - Starting Up             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Get port from environment (HF Spaces uses PORT env var)
    ui_port = int(os.getenv("PORT", 8501))
    
    print(f"🎨 UI will run on port {ui_port}\n")

    # Start Streamlit frontend
    print("🚀 Starting Streamlit frontend...\n")
    ui_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            "src/ui/dashboard.py",
            "--server.port", str(ui_port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
    )
    
    print("\n✅ VERDICT Platform is running!")
    print(f"📊 Dashboard: http://localhost:{ui_port}")
    print("\nPress Ctrl+C to stop...\n")
    
    # Keep process alive
    try:
        ui_process.wait()
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        ui_process.terminate()
        ui_process.wait()
        print("✅ Platform stopped")
        sys.exit(0)


if __name__ == "__main__":
    run_app()