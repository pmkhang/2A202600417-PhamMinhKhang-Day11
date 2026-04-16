"""
Environment bootstrap helpers.
"""
import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = REPO_ROOT / ".env"


def auto_load_env() -> bool:
    """Load variables from the repository .env when present."""
    return load_dotenv(dotenv_path=DOTENV_PATH)


auto_load_env()
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")
