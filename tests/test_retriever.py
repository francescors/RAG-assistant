import pytest

from app import config
from app.ingest import load_chunks
from app.retriever import Retriever


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_chunks(config.docs_dir()), min_score=config.min_score())


def test_finds_right_document(retriever):
    hits = retriever.search("Combien coûte l'offre Pro ?", k=3)
    assert hits
    assert hits[0].chunk.source == "offres.md"


def test_scores_are_sorted(retriever):
    scores = [h.score for h in retriever.search("chiffrement des données", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_out_of_scope_returns_nothing(retriever):
    assert retriever.search("Quelle est la capitale de l'Australie ?") == []


def test_empty_index_raises():
    with pytest.raises(ValueError):
        Retriever([])
