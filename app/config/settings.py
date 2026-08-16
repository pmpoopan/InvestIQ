"""Environment and application configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_settings() -> dict:
    """Return application settings from environment variables."""
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "groq_fallback_model": os.getenv("GROQ_FALLBACK_MODEL", "qwen/qwen3.6-27b"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "chroma_path": os.getenv("CHROMA_PATH", str(REPO_ROOT / "data" / "chroma")),
        "chroma_collection": os.getenv("CHROMA_COLLECTION", "investiq_chunks"),
        "top_k": int(os.getenv("TOP_K", "5")),
        "min_relevance": float(os.getenv("MIN_RELEVANCE", "0.28")),
        "processed_dir": os.getenv("PROCESSED_DIR", str(REPO_ROOT / "data" / "processed")),
    }
