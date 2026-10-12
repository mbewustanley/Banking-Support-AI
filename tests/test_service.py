
from unittest.mock import Mock

import pytest

from src.service import BankingSupportService


def test_service_returns_clean_response():
    graph = Mock()
    graph.invoke.return_value = {
        "question": "What happens with duplicate charges?",
        "answer": "A duplicate charge must be investigated. [S1]",
        "sources": [
            {
                "citation_id": "S1",
                "source": "card_fees.md",
                "document_id": "card_fees",
                "distance": 0.25,
            }
        ],
        "retrieval_results": [],
        "retrieval_documents": [],
        "abstained": False,
        "error": None,
    }

    service = BankingSupportService(graph=graph)
    result = service.ask("  What happens with duplicate charges?  ")

    assert result["answer"] == (
        "A duplicate charge must be investigated. [S1]"
    )
    assert result["abstained"] is False
    assert result["sources"][0]["citation_id"] == "S1"
    assert result["error"] is None

    graph.invoke.assert_called_once()
    state = graph.invoke.call_args.args[0]
    assert state["question"] == "What happens with duplicate charges?"


def test_service_preserves_abstention():
    graph = Mock()
    graph.invoke.return_value = {
        "answer": "The available policies do not specify that information.",
        "sources": [],
        "abstained": True,
        "error": None,
    }

    service = BankingSupportService(graph=graph)
    result = service.ask("What is my account balance?")

    assert result["abstained"] is True
    assert result["sources"] == []


@pytest.mark.parametrize("question", ["", "   ", None, 123])
def test_service_rejects_invalid_question(question):
    graph = Mock()
    service = BankingSupportService(graph=graph)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.ask(question)

    graph.invoke.assert_not_called()
