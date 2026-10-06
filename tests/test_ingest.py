import pytest

from app.ingest import chunk_text, load_chunks


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"mot{i}" for i in range(200))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0].startswith("mot0")
    assert chunks[-1].endswith("mot199")


def test_chunks_overlap():
    text = " ".join(f"mot{i}" for i in range(100))
    a, b = chunk_text(text, size=100, overlap=30)[:2]
    assert set(a.split()) & set(b.split())


def test_short_text_is_one_chunk():
    assert chunk_text("bonjour le monde", size=100, overlap=10) == ["bonjour le monde"]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_load_chunks_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path)
