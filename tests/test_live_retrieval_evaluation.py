import pytest

from scripts.evaluate_live_retrieval import evaluate_live, read_questions
from src.ingest import Chunk
from src.retriever import SearchResult


class FakeRetriever:
    def search(self, question, top_k=3):
        if "気温" in question:
            sources = ["heading.md", "heading.md", "heat.md"]
        else:
            sources = ["heat.md", "heading.md"]
        return [SearchResult(Chunk("text", source, i), 0.9 - i * 0.1)
                for i, source in enumerate(sources[:top_k])]


def test_live_evaluation_deduplicates_sources():
    report = evaluate_live(
        FakeRetriever(),
        [{"question": "気温と出穂", "relevant_ids": ["heading.md"]},
         {"question": "高温障害", "relevant_ids": ["heat.md"]}],
        3,
    )
    assert report["samples"] == 2
    assert report["recall_at_k"] == 1.0
    assert report["mrr_at_k"] == 1.0
    assert report["questions"][0]["retrieved_ids"] == ["heading.md", "heat.md"]


def test_reject_missing_gold():
    with pytest.raises(ValueError):
        evaluate_live(FakeRetriever(), [{"question": "出穂"}], 3)


def test_read_questions(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question":"test","relevant_ids":["x.md"]}\n', encoding="utf-8")
    assert read_questions(path)[0]["relevant_ids"] == ["x.md"]
