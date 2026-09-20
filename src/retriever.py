"""Cosine similarity retrieval using a persisted FAISS index and JSON metadata."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol, Sequence

import faiss
import numpy as np

from src.ingest import Chunk


class Embedder(Protocol):
    """Dependency boundary for real or offline embedding providers."""

    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input in input order."""
        ...


@dataclass(frozen=True)
class SearchResult:
    """A matching chunk and its cosine similarity score."""

    chunk: Chunk
    score: float


def _vectors(values: list[list[float]], count: int) -> np.ndarray:
    """Validate and normalize vectors before passing them to FAISS."""
    matrix = np.array(values, dtype=np.float32, order="C", copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] == 0:
        raise ValueError("Embedding shape does not match the input.")
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("Embeddings must contain finite, non-zero vectors.")
    faiss.normalize_L2(matrix)
    return matrix


class Retriever:
    """Own an exact cosine index with aligned chunk metadata."""

    def __init__(self, embedder: Embedder, chunks: list[Chunk], index: faiss.Index) -> None:
        """Validate the index and metadata alignment."""
        if not chunks or index.ntotal != len(chunks):
            raise ValueError("Index and chunk metadata are empty or inconsistent.")
        self.embedder = embedder
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "Retriever":
        """Embed chunks and build an inner-product index of normalized vectors."""
        if not chunks:
            raise ValueError("Cannot build an empty index.")
        matrix = _vectors(embedder.embed([chunk.text for chunk in chunks]), len(chunks))
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(embedder, chunks, index)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Retrieve up to top_k chunks ranked by descending cosine similarity."""
        if not question.strip() or top_k <= 0:
            raise ValueError("A non-empty question and positive top_k are required.")
        query = _vectors(self.embedder.embed([question]), 1)
        if query.shape[1] != self.index.d:
            raise ValueError("Embedding dimension changed; rebuild the index.")
        scores, ids = self.index.search(query, min(top_k, len(self.chunks)))
        return [SearchResult(self.chunks[int(i)], float(score))
                for i, score in zip(ids[0], scores[0]) if i >= 0]

    def save(self, directory: Path) -> None:
        """Persist FAISS vectors and JSON metadata; never serialize with pickle."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        metadata = {"version": 1, "model": self.embedder.model,
                    "chunks": [asdict(chunk) for chunk in self.chunks]}
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "Retriever":
        """Load a locally generated index, rejecting incompatible metadata."""
        directory = Path(directory)
        try:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if metadata["version"] != 1 or metadata["model"] != embedder.model:
                raise ValueError("Index model or version changed; rebuild the index.")
            chunks = [Chunk(**item) for item in metadata["chunks"]]
            if any(not isinstance(c.text, str) or not c.text.strip()
                   or not isinstance(c.source, str) or not c.source
                   or not isinstance(c.chunk_id, int) for c in chunks):
                raise ValueError("Invalid chunk metadata.")
            index = faiss.read_index(str(directory / "index.faiss"))
            if not isinstance(index, faiss.IndexFlatIP):
                raise ValueError("Unsupported index type.")
            return cls(embedder, chunks, index)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Cannot load index; rebuild it from your documents.") from exc
