"""
Launch the Spacetime Arena API server.
Run from repo root: python run_api.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "spacetime"))
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "spacetime.api.main:app",
        host="127.0.0.1",
        port=8765,
        reload=True,
        reload_dirs=[str(ROOT / "spacetime")],
    )
