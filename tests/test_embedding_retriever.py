import os
import re
import zlib

import numpy as np
import pytest

from app import config
from app.embedding_retriever import EmbeddingRetriever
from app.ingest import Chunk, load_chunks
from app.retriever import Retriever, build_retriever


class FakeEncoder:
    """Faux modèle (même interface que fastembed.TextEmbedding) : sac de mots hashé.

    Permet de tester la logique du retriever sans télécharger de modèle.
    """

    def embed(self, texts):
        for text in texts:
            vec = np.zeros(128, dtype=np.float32)
            for word in re.findall(r"\w+", text.lower()):
                vec[zlib.crc32(word.encode()) % 128] += 1
            yield vec


CHUNKS = [
    Chunk("a.md", "le chat dort sur le canapé du salon"),
    Chunk("b.md", "la facture est payable par virement bancaire"),
]


def make(min_score=0.0):
    return EmbeddingRetriever(CHUNKS, min_score=min_score, encoder=FakeEncoder())


def test_best_chunk_first():
    hits = make().search("virement bancaire facture", k=2)
    assert hits[0].chunk.source == "b.md"


def test_scores_sorted_and_normalized():
    hits = make().search("chat canapé", k=2)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(-1.001 <= h.score <= 1.001 for h in hits)


def test_min_score_filters_everything():
    assert make(min_score=0.99).search("zzz inconnu") == []


def test_search_min_score_override():
    r = make(min_score=0.0)
    assert r.search("chat canapé", k=2, min_score=0.99) == []


def test_empty_index_raises():
    with pytest.raises(ValueError):
        EmbeddingRetriever([], encoder=FakeEncoder())


def test_factory_builds_tfidf():
    assert isinstance(build_retriever(CHUNKS, "tfidf"), Retriever)


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_retriever(CHUNKS, "magie")


@pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="télécharge le vrai modèle (RUN_SLOW=1)")
def test_real_model_understands_synonyms():
    retriever = build_retriever(load_chunks(config.docs_dir()), "embeddings")
    hits = retriever.search("Comment annuler mon abonnement ?", k=3)
    assert hits and hits[0].chunk.source == "support.md"
