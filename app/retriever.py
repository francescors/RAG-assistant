"""Recherche des passages les plus pertinents (TF-IDF + similarité cosinus)."""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ingest import Chunk

# Mots vides français (sans lettres isolées ni accents, pour rester cohérent
# avec le tokenizer de scikit-learn).
FRENCH_STOP_WORDS = [
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "au", "aux",
    "est", "sont", "que", "qui", "quoi", "quel", "quelle", "quels", "quelles",
    "ce", "cet", "cette", "ces", "pour", "par", "sur", "dans", "il", "elle",
    "on", "je", "tu", "nous", "vous", "ils", "elles", "ne", "pas", "se", "sa",
    "son", "ses", "puis", "ou", "comment", "combien", "ai", "avez", "avoir",
    "faire", "peut", "peux", "mon", "ma", "mes", "votre", "vos",
]  # fmt: skip


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def top_hits(chunks: list[Chunk], scores: np.ndarray, k: int, min_score: float) -> list[Hit]:
    """Garde les k meilleurs scores supérieurs ou égaux au seuil."""
    best = np.argsort(scores)[::-1][:k]
    return [Hit(chunks[i], float(scores[i])) for i in best if scores[i] >= min_score]


class Retriever:
    def __init__(self, chunks: list[Chunk], min_score: float = 0.1):
        if not chunks:
            raise ValueError("Impossible de construire un index vide")
        self.chunks = chunks
        self.min_score = min_score
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            stop_words=FRENCH_STOP_WORDS,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3, min_score: float | None = None) -> list[Hit]:
        """Retourne au plus k passages, triés par score décroissant."""
        query_vec = self.vectorizer.transform([query])
        # Les vecteurs TF-IDF sont normalisés : le produit scalaire = cosinus.
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        threshold = self.min_score if min_score is None else min_score
        return top_hits(self.chunks, scores, k, threshold)


def build_retriever(chunks: list[Chunk], kind: str = "tfidf", min_score: float | None = None):
    """Construit le retriever demandé. L'import des embeddings est paresseux (PyTorch est lourd)."""
    if kind == "tfidf":
        return Retriever(chunks, min_score=0.1 if min_score is None else min_score)
    if kind == "embeddings":
        from app.embedding_retriever import EmbeddingRetriever

        return EmbeddingRetriever(chunks, min_score=0.15 if min_score is None else min_score)
    raise ValueError(f"Retriever inconnu : {kind!r} (attendu : 'tfidf' ou 'embeddings')")
