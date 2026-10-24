"""Offline retrieval evaluation from ranked JSONL fixtures.

Each line contains relevant_ids and retrieved_ids using the same ID granularity.
This evaluates supplied rankings, not live FAISS/OpenAI retrieval.
"""
import argparse
import json
from pathlib import Path


def score_case(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")
    if len(relevant_ids) != len(set(relevant_ids)):
        raise ValueError("relevant_ids must not contain duplicates")
    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("retrieved_ids must not contain duplicates")
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = len(relevant.intersection(top_k))
    rr = next((1.0 / rank for rank, item in enumerate(top_k, 1) if item in relevant), 0.0)
    return {"recall_at_k": hits / len(relevant), "mrr_at_k": rr, "hit_at_k": float(hits > 0)}


def evaluate(path: Path, k: int) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    cases = []
    with path.open(encoding="utf-8") as f:
        for line_number, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                case = json.loads(raw)
                relevant = case["relevant_ids"]
                retrieved = case["retrieved_ids"]
                if not isinstance(relevant, list) or not isinstance(retrieved, list):
                    raise ValueError("ID fields must be lists")
                if any(not isinstance(x, str) or not x for x in relevant + retrieved):
                    raise ValueError("IDs must be non-empty strings")
                scores = score_case(relevant, retrieved, k)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
                raise ValueError(f"Invalid JSONL line {line_number}: {e}") from e
            cases.append(scores)
    if not cases:
        raise ValueError("No evaluation cases found")
    return {"samples": len(cases), "k": k, **{
        metric: round(sum(row[metric] for row in cases) / len(cases), 4)
        for metric in ("recall_at_k", "mrr_at_k", "hit_at_k")
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked retrieval results from JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input, args.k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
