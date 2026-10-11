
import pytest

from src.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_finds_all_relevant_sources() -> None:
    retrieved = ["refunds.md", "card_fees.md"]
    relevant = ["card_fees.md", "refunds.md"]

    assert recall_at_k(retrieved, relevant) == 1.0


def test_recall_at_k_penalizes_missing_sources() -> None:
    retrieved = ["card_fees.md"]
    relevant = ["card_fees.md", "refunds.md"]

    assert recall_at_k(retrieved, relevant) == 0.5


def test_precision_at_k_measures_relevant_sources() -> None:
    retrieved = ["card_fees.md", "unrelated.md"]
    relevant = ["card_fees.md"]

    assert precision_at_k(retrieved, relevant) == 0.5


def test_precision_at_k_empty_results() -> None:
    assert precision_at_k([], ["refunds.md"]) == 0.0


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    retrieved = ["unrelated.md", "refunds.md", "card_fees.md"]
    relevant = ["refunds.md", "card_fees.md"]

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_is_relevant() -> None:
    assert reciprocal_rank(
        ["unrelated.md"],
        ["refunds.md"],
    ) == 0.0


def test_recall_rejects_empty_ground_truth() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["refunds.md"], [])


def test_load_evaluation_dataset() -> None:
    from src.evaluation import load_evaluation_cases

    cases = load_evaluation_cases()

    assert len(cases) == 5
    assert len({case["id"] for case in cases}) == 5
