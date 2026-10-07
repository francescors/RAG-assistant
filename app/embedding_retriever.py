"""Recherche sémantique avec fastembed (embeddings multilingues, ONNX, sans PyTorch)."""

import numpy as np

from app import config
from app.ingest import Chunk
from app.retriever import Hit, top_hits


class EmbeddingRetriever:
    """Compare le sens de la question à celui de chaque chunk (similarité cosinus).

    Contrairement à TF-IDF, « annuler mon abonnement » peut retrouver un passage
    qui parle de « résilier », car les deux phrases ont un sens proche.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        min_score: float = 0.15,
        model_name: str | None = None,
        encoder=None,
    ):
        if not chunks:
            raise ValueError("Impossible de construire un index vide")
        if encoder is None:
            # Import ici : inutile (et plus lent au démarrage) pour le mode TF-IDF.
            from fastembed import TextEmbedding

            # Le modèle est téléchargé au premier usage, puis gardé en cache disque
            # (dossier défini par FASTEMBED_CACHE_PATH).
            encoder = TextEmbedding(model_name=model_name or config.embedding_model())
        self.encoder = encoder
        self.chunks = chunks
        self.min_score = min_score
        self.matrix = self._encode([c.text for c in chunks])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.encoder.embed(texts)), dtype=np.float32)
        # On normalise nous-mêmes : le produit scalaire devient la similarité cosinus.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def search(self, query: str, k: int = 3, min_score: float | None = None) -> list[Hit]:
        scores = self.matrix @ self._encode([query])[0]
        threshold = self.min_score if min_score is None else min_score
        return top_hits(self.chunks, scores, k, threshold)
