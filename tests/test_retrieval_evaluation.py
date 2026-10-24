import pytest

from scripts.evaluate_retrieval import evaluate, score_case


def test_metrics():
    scores = score_case(["a", "c"], ["b", "a", "d"], 2)
    assert scores == {"recall_at_k": 0.5, "mrr_at_k": 0.5, "hit_at_k": 1.0}


def test_no_hit():
    scores = score_case(["a"], ["b"], 1)
    assert scores == {"recall_at_k": 0.0, "mrr_at_k": 0.0, "hit_at_k": 0.0}


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        score_case(["a"], ["a", "a"], 2)


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        score_case(["a"], ["a"], 0)


def test_fixture_aggregation(tmp_path):
    fixture = tmp_path / "qa.jsonl"
    fixture.write_text(
        '{"relevant_ids":["a"],"retrieved_ids":["a","b"]}\n'
        '{"relevant_ids":["b"],"retrieved_ids":["a","b"]}\n',
        encoding="utf-8",
    )
    assert evaluate(fixture, 2) == {
        "samples": 2, "k": 2,
        "recall_at_k": 1.0, "mrr_at_k": 0.75, "hit_at_k": 1.0
    }
