"""Shared, absolute paths for AskMyDocs runtime data."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR = BASE_DIR / "index"
FRONTEND_FILE = BASE_DIR / "index.html"


def ensure_runtime_directories() -> None:
    """Create the local storage folders used by the demo application."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
