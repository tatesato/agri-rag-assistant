"""Run the actual FAISS Retriever over a labelled QA set.

The answer-generating LLM is not used. OpenAI embeddings require API access.
Gold IDs use source filenames (document-level relevance).
"""
import argparse
import json
from pathlib import Path

from scripts.evaluate_retrieval import score_case
from src.ingest import load_documents
from src.llm import OpenAIEmbedder, create_client
from src.retriever import Retriever


def evaluate_live(retriever, cases: list[dict], k: int) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    if not cases:
        raise ValueError("At least one labelled question is required")
    scores = []
    details = []
    for idx, case in enumerate(cases, 1):
        question = case.get("question")
        relevant = case.get("relevant_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {idx} is empty")
        if not isinstance(relevant, list) or not relevant or any(
            not isinstance(item, str) or not item for item in relevant
        ):
            raise ValueError(f"Question {idx} needs non-empty relevant_ids")
        results = retriever.search(question, top_k=k)
        # Document-level evaluation: overlapping chunks from the same file
        # count as one retrieved document, preserving first occurrence order.
        documents = list(dict.fromkeys(result.chunk.source for result in results))
        score = score_case(relevant, documents, k)
        scores.append(score)
        details.append({"question": question, "relevant_ids": relevant,
                        "retrieved_ids": documents, **score})
    metrics = {
        metric: round(sum(s[metric] for s in scores) / len(scores), 4)
        for metric in ("recall_at_k", "mrr_at_k", "hit_at_k")
    }
    return {"samples": len(scores), "k": k, "id_level": "source_document",
            **metrics, "questions": details}


def read_questions(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for number, raw in enumerate(f, 1):
            if raw.strip():
                try:
                    cases.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {number}") from exc
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live FAISS retrieval using OpenAI embeddings")
    parser.add_argument("--documents", type=Path, default=Path("data/sample_documents"))
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    cases = read_questions(args.questions)
    client = create_client()
    embedder = OpenAIEmbedder(client)
    retriever = Retriever.build(load_documents(args.documents), embedder)
    report = evaluate_live(retriever, cases, args.k)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
