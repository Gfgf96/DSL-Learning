"""Utilities for building and serving the exported frontend."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_OUT_DIR = FRONTEND_DIR / "out"
FRONTEND_INDEX_FILE = FRONTEND_OUT_DIR / "index.html"


def frontend_export_exists() -> bool:
    """Return True when the static frontend export is available."""
    return FRONTEND_INDEX_FILE.exists()


def ensure_frontend_export() -> Path:
    """Build the frontend export if it is missing and return the export directory."""
    if frontend_export_exists():
        return FRONTEND_OUT_DIR

    npm_executable = shutil.which("npm")
    if not npm_executable:
        raise RuntimeError("npm is required to build the frontend export, but it was not found on PATH.")

    if not (FRONTEND_DIR / "package.json").exists():
        raise RuntimeError(f"Frontend package.json was not found at {FRONTEND_DIR}")

    print(" Building frontend export...")
    result = subprocess.run(
        [npm_executable, "run", "build"],
        cwd=FRONTEND_DIR,
        check=False,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Frontend build failed. Run `cd frontend && npm run build` to inspect the error."
        )

    if not frontend_export_exists():
        raise RuntimeError(
            "Frontend build completed but `frontend/out/index.html` was not produced."
        )

    return FRONTEND_OUT_DIR
