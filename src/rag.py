"""Compose retrieval and generation with traceable excerpt citations."""
from dataclasses import dataclass
import json
import re
from typing import Protocol

from src.retriever import SearchResult


class Searcher(Protocol):
    """Search dependency that can be replaced by a test double."""

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Return ranked excerpts."""
        ...


class Answerer(Protocol):
    """Generation dependency that can be replaced by a test double."""

    def answer(self, question: str, context: str) -> str:
        """Answer using the provided evidence."""
        ...


@dataclass(frozen=True)
class RAGResponse:
    """An answer, its cited sources, and the complete retrieval evidence."""

    answer: str
    sources: list[str]
    results: list[SearchResult]


class RAGAssistant:
    """Retrieve before generating and validate numeric citation references."""

    def __init__(self, retriever: Searcher, answerer: Answerer) -> None:
        """Inject the search and generation dependencies."""
        self.retriever = retriever
        self.answerer = answerer

    def ask(self, question: str, top_k: int = 3) -> RAGResponse:
        """Pass only retrieved chunks as context and map citations to filenames."""
        if not question.strip() or top_k <= 0:
            raise ValueError("A non-empty question and positive top_k are required.")
        results = self.retriever.search(question, top_k)
        if not results:
            return RAGResponse("No matching documents were found.", [], [])
        context = json.dumps([
            {"label": f"[{i}]", "source": result.chunk.source,
             "chunk_id": result.chunk.chunk_id, "text": result.chunk.text}
            for i, result in enumerate(results, 1)
        ], ensure_ascii=False)
        answer = self.answerer.answer(question, context)
        labels = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
        if any(label < 1 or label > len(results) for label in labels):
            raise ValueError("The answer contains an unknown citation. Please retry.")
        sources = list(dict.fromkeys(results[label - 1].chunk.source for label in labels))
        return RAGResponse(answer, sources, results)
