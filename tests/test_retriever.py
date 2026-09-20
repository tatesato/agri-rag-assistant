"""Offline tests exercising real FAISS search and persistence."""
import json
import numpy as np
import pytest

from src.ingest import Chunk, chunk_text, load_documents
from src.retriever import Retriever


class FakeEmbedder:
    """Map irrigation and soil terms to deterministic two-dimensional vectors."""
    model = "offline-test"

    def embed(self, texts):
        return [[1.0, 0.0] if "water" in text else [0.0, 1.0] for text in texts]


@pytest.fixture
def retriever():
    return Retriever.build([Chunk("water log", "water.md", 0),
                            Chunk("soil log", "soil.txt", 0)], FakeEmbedder())


def test_ranking_and_top_k(retriever):
    results = retriever.search("water", 1)
    assert [r.chunk.source for r in results] == ["water.md"]
    assert results[0].score == pytest.approx(1.0)
    assert len(retriever.search("soil", 20)) == 2


def test_persistence_roundtrip(retriever, tmp_path):
    retriever.save(tmp_path)
    restored = Retriever.load(tmp_path, FakeEmbedder())
    assert restored.search("soil")[0].chunk.source == "soil.txt"
    assert restored.chunks == retriever.chunks


def test_reject_model_mismatch(retriever, tmp_path):
    retriever.save(tmp_path)
    embedder = FakeEmbedder()
    embedder.model = "different-model"
    with pytest.raises(ValueError, match="rebuild"):
        Retriever.load(tmp_path, embedder)


def test_reject_corrupt_metadata(retriever, tmp_path):
    retriever.save(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["chunks"].pop()
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        Retriever.load(tmp_path, FakeEmbedder())


@pytest.mark.parametrize("question,k", [("", 1), ("  ", 1), ("water", 0), ("water", -1)])
def test_invalid_query(retriever, question, k):
    with pytest.raises(ValueError):
        retriever.search(question, k)


def test_empty_index():
    with pytest.raises(ValueError):
        Retriever.build([], FakeEmbedder())


@pytest.mark.parametrize("vectors", [[[0, 0]], [[np.nan, 1]], [[np.inf, 1]], [], [[1], [2]]])
def test_bad_embeddings(vectors):
    class BadEmbedder(FakeEmbedder):
        def embed(self, texts):
            return vectors
    with pytest.raises(ValueError):
        Retriever.build([Chunk("water", "a.txt", 0)], BadEmbedder())


def test_dimension_mismatch(retriever):
    retriever.embedder.embed = lambda texts: [[1, 0, 0]]
    with pytest.raises(ValueError, match="dimension"):
        retriever.search("water")


def test_chunk_overlap_and_unicode():
    text = "水分観測と土壌の記録を保存する"
    parts = chunk_text(text, 6, 2)
    assert all(len(p) <= 6 for p in parts)
    assert parts[0][-2:] == parts[1][:2]
    assert parts[0] + "".join(p[2:] for p in parts[1:]) == text
    assert chunk_text("  \n ") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (4, 4), (4, -1)])
def test_invalid_chunk_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", size, overlap)


def test_load_documents(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "crop.MD").write_text("水分観測", encoding="utf-8")
    (tmp_path / "soil.txt").write_text("soil log", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("ignored")
    chunks = load_documents(tmp_path)
    assert {c.source for c in chunks} == {"nested/crop.MD", "soil.txt"}


def test_missing_empty_and_invalid_documents(tmp_path):
    with pytest.raises(ValueError):
        load_documents(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_documents(tmp_path)
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        load_documents(tmp_path)


def test_missing_index(tmp_path):
    with pytest.raises(ValueError, match="rebuild"):
        Retriever.load(tmp_path, FakeEmbedder())
