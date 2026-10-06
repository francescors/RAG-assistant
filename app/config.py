"""Configuration lue dans les variables d'environnement (jamais de secret dans le code)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def docs_dir() -> Path:
    return Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs"))


def chunk_size() -> int:
    return int(os.getenv("CHUNK_SIZE", "300"))


def chunk_overlap() -> int:
    return int(os.getenv("CHUNK_OVERLAP", "60"))


def min_score() -> float:
    """Score de similarité minimal : en dessous, on considère qu'il n'y a pas de contexte."""
    return float(os.getenv("MIN_SCORE", "0.1"))


def llm_model() -> str:
    return os.getenv("LLM_MODEL", "claude-sonnet-5-5")


def anthropic_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY") or None
