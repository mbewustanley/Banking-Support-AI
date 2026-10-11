"""
RETRIEVAL EVALUATION
"""


import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.retrieval import PolicyRetriever


def ranked_unique_sources(
    results: list[tuple[Any, float]],
    k: int,
) -> list[str]:
    """Return the first k unique source filenames in rank order."""
    sources = []

    for document, _distance in results:
        source = Path(document.metadata["source"]).name

        if source not in sources:
            sources.append(source)

        if len(sources) == k:
            break

    return sources


def recall_at_k(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> float:
    """Fraction of expected relevant sources retrieved."""
    if not relevant_sources:
        raise ValueError("relevant_sources cannot be empty")

    retrieved = set(retrieved_sources[:])
    relevant = set(relevant_sources)

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> float:
    """Fraction of retrieved unique sources that are relevant."""
    if not retrieved_sources:
        return 0.0

    retrieved = retrieved_sources
    relevant = set(relevant_sources)

    return sum(source in relevant for source in retrieved) / len(retrieved)


def reciprocal_rank(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> float:
    """Reciprocal rank of the first relevant source."""
    relevant = set(relevant_sources)

    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant:
            return 1.0 / rank

    return 0.0


def load_evaluation_cases(
    path: str = "evals/retrieval_eval.json",
) -> list[dict[str, Any]]:
    """Load and validate the retrieval evaluation dataset."""
    with Path(path).open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a non-empty list")

    required = {"id", "question", "relevant_sources"}
    seen_ids = set()

    for case in cases:
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError("Each case must contain id, question, and relevant_sources")

        if not isinstance(case["id"], str) or not case["id"].strip():
            raise ValueError("Each evaluation case needs a non-empty ID")

        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate evaluation ID: {case['id']}")

        seen_ids.add(case["id"])

        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"Invalid question in case {case['id']}")

        if (
            not isinstance(case["relevant_sources"], list)
            or not case["relevant_sources"]
            or not all(
                isinstance(source, str) and source.strip()
                for source in case["relevant_sources"]
            )
        ):
            raise ValueError(
                f"Invalid relevant_sources in case {case['id']}"
            )

    return cases


def evaluate_retrieval(
    retriever: PolicyRetriever,
    cases: list[dict[str, Any]],
    k: int = 2,
) -> dict[str, Any]:
    """Evaluate retrieval at the unique source-document level."""
    if k < 1:
        raise ValueError("k must be at least 1")

    if not cases:
        raise ValueError("Evaluation cases cannot be empty")

    rows = []

    for case in cases:
        # Fetch several chunks so repeated chunks from one source
        # are less likely to hide another relevant source.
        results = retriever.search(case["question"], k=max(10, k * 4))

        sources = ranked_unique_sources(results, k)

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "retrieved_sources": sources,
                "relevant_sources": case["relevant_sources"],
                f"recall@{k}": recall_at_k(
                    sources, case["relevant_sources"]
                ),
                f"precision@{k}": precision_at_k(
                    sources, case["relevant_sources"]
                ),
                "reciprocal_rank": reciprocal_rank(
                    sources, case["relevant_sources"]
                ),
            }
        )

    return {
        "k": k,
        "evaluation_level": "unique source documents",
        "num_cases": len(rows),
        f"mean_recall@{k}": mean(row[f"recall@{k}"] for row in rows),
        f"mean_precision@{k}": mean(row[f"precision@{k}"] for row in rows),
        "mrr": mean(row["reciprocal_rank"] for row in rows),
        "results": rows,
    }


def main() -> None:
    cases = load_evaluation_cases()

    retriever = PolicyRetriever()
    indexed_count = retriever.build_index()
    print(f"Indexed {indexed_count} policy chunks.")

    report = evaluate_retrieval(retriever, cases, k=2)

    print(json.dumps(report, indent=2))

    output_path = Path("reports/retrieval_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nEvaluation report saved to {output_path}")


if __name__ == "__main__":
    main()
