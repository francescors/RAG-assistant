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


def retriever_kind() -> str:
    """'tfidf' (mots-clés) ou 'embeddings' (sens des phrases)."""
    return os.getenv("RETRIEVER", "tfidf").lower()


def embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# Les scores TF-IDF et embeddings n'ont pas la même échelle : seuils par défaut distincts.
DEFAULT_MIN_SCORE = {"tfidf": 0.1, "embeddings": 0.15}


def min_score(kind: str | None = None) -> float:
    """Score minimal : en dessous, on considère qu'il n'y a pas de contexte pertinent."""
    if os.getenv("MIN_SCORE"):
        return float(os.environ["MIN_SCORE"])
    return DEFAULT_MIN_SCORE.get(kind or retriever_kind(), 0.1)


def llm_model() -> str:
    return os.getenv("LLM_MODEL", "claude-sonnet-5-5")


def anthropic_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY") or None
