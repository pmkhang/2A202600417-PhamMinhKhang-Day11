"""
Lab 11 — Configuration & API Key Setup
"""
import os
import sys

from core.env import auto_load_env


def setup_api_key():
    """Load Google API key from .env/env and only prompt in interactive shells."""
    auto_load_env()

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        if sys.stdin.isatty():
            api_key = input("Enter Google API Key: ").strip()
        else:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY. Add it to .env or export it before running the project."
            )

    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")
    print("API key loaded.")


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
