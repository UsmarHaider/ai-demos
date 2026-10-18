"""One place for settings. Both projects import from here.

Keys live in demo_project/.env - never hard-coded in the lesson code, because
that is the habit we want students to copy.

Everything runs on Google's Gemini API: one key for chat and one for
embeddings, and they can be the same key. Embeddings are configured separately
because they have their own model and their own offline fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KEY_URL = "https://aistudio.google.com/apikey"


# ------------------------------------------------------------------ chat
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")

# The active chat model. Kept as its own name so the lesson code reads well.
CHAT_MODEL = GEMINI_CHAT_MODEL

# ------------------------------------------------------------ embeddings
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "gemini").strip().lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")


def _die(message: str) -> None:
    print(f"\n  ERROR: {message}\n", file=sys.stderr)
    raise SystemExit(1)


def require_chat() -> None:
    """Fail loudly and helpfully instead of mid-demo with a stack trace."""
    if not GOOGLE_API_KEY:
        _die(
            f"GOOGLE_API_KEY missing from {ROOT / '.env'}\n"
            f"  Copy .env.example to .env, then get a free key from {KEY_URL}"
        )
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def require_embeddings() -> None:
    if EMBED_BACKEND == "gemini":
        if not GOOGLE_API_KEY:
            _die(
                f"EMBED_BACKEND=gemini but GOOGLE_API_KEY is missing from {ROOT / '.env'}\n"
                "  Either add a key, or set EMBED_BACKEND=local to run with no network."
            )
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    elif EMBED_BACKEND != "local":
        _die(f"EMBED_BACKEND must be 'gemini' or 'local', not '{EMBED_BACKEND}'")


def embedding_label() -> str:
    return EMBED_MODEL if EMBED_BACKEND == "gemini" else "local numpy keyword embedding"


def chat_label() -> str:
    return f"{CHAT_MODEL} (Google AI)"


def _mask(key: str) -> str:
    return f"{key[:6]}…{key[-4:]}" if key else "(not set)"


def masked_key() -> str:
    """Safe to show on a projector."""
    return _mask(GOOGLE_API_KEY)


# Kept so older copies of the lesson code keep working.
require_api_key = require_chat
