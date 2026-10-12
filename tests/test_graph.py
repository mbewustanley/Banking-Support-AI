
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.graph.nodes import (
    retrieve_policy_evidence,
    generate_grounded_answer,
)


def test_retrieval_node_returns_policy_evidence():
    document = Document(
        page_content="Duplicate charges must be investigated.",
        metadata={
            "source": "knowledge_base/policies/card_fees.md",
            "document_id": "card_fees",
        },
    )

    retriever = Mock()
    retriever.search.return_value = [(document, 0.25)]

    state = {
        "question": "  What happens with duplicate charges?  ",
        "answer": "",
        "sources": [],
        "retrieval_results": [],
        "retrieval_documents": [],
        "abstained": False,
        "error": None,
    }

    result = retrieve_policy_evidence(state, retriever)

    retriever.search.assert_called_once_with(
        "What happens with duplicate charges?",
        k=3,
    )
    assert result["question"] == "What happens with duplicate charges?"
    assert result["retrieval_results"][0]["content"] == (
        "Duplicate charges must be investigated."
    )
    assert result["retrieval_results"][0]["rank"] == 1
    assert result["retrieval_results"][0]["distance"] == 0.25
    assert result["sources"][0]["document_id"] == "card_fees"

    # The original documents must be preserved for generation.
    assert result["retrieval_documents"] == [(document, 0.25)]


def test_retrieval_node_rejects_empty_question():
    retriever = Mock()

    state = {
        "question": "   ",
        "answer": "",
        "sources": [],
        "retrieval_results": [],
        "retrieval_documents": [],
        "abstained": False,
        "error": None,
    }

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieve_policy_evidence(state, retriever)

    retriever.search.assert_not_called()


def test_retrieval_node_handles_no_results():
    retriever = Mock()
    retriever.search.return_value = []

    state = {
        "question": "What is my account balance?",
        "answer": "",
        "sources": [],
        "retrieval_results": [],
        "retrieval_documents": [],
        "abstained": False,
        "error": None,
    }

    result = retrieve_policy_evidence(state, retriever)

    assert result["retrieval_results"] == []
    assert result["retrieval_documents"] == []
    assert result["sources"] == []


def test_generation_node_returns_generator_result():
    document = Document(
        page_content="Duplicate charges must be investigated.",
        metadata={
            "source": "knowledge_base/policies/card_fees.md",
            "document_id": "card_fees",
        },
    )
    retrieved_documents = [(document, 0.25)]

    generator = Mock()
    generator.answer_with_evidence.return_value = SimpleNamespace(
        answer="A duplicate charge must be investigated. [S1]",
        sources=[
            {
                "citation_id": "S1",
                "source": "knowledge_base/policies/card_fees.md",
                "document_id": "card_fees",
                "distance": 0.25,
            }
        ],
        retrieval_results=[
            {
                "citation_id": "S1",
                "source": "knowledge_base/policies/card_fees.md",
                "document_id": "card_fees",
                "distance": 0.25,
            }
        ],
        abstained=False,
    )

    state = {
        "question": "  What happens with duplicate charges?  ",
        "answer": "",
        "sources": [],
        "retrieval_results": [],
        "retrieval_documents": retrieved_documents,
        "abstained": False,
        "error": None,
    }

    result = generate_grounded_answer(state, generator)

    generator.answer_with_evidence.assert_called_once_with(
        "What happens with duplicate charges?",
        retrieved_documents,
    )
    assert result["answer"] == (
        "A duplicate charge must be investigated. [S1]"
    )
    assert result["abstained"] is False
    assert result["sources"][0]["document_id"] == "card_fees"
    assert result["error"] is None


def test_generation_node_preserves_abstention():
    generator = Mock()
    generator.answer_with_evidence.return_value = SimpleNamespace(
        answer="The available policies do not specify that information.",
        sources=[],
        retrieval_results=[],
        abstained=True,
    )

    state = {
        "question": "What is my account balance?",
        "answer": "",
        "sources": [],
        "retrieval_results": [],
        "retrieval_documents": [],
        "abstained": False,
        "error": None,
    }

    result = generate_grounded_answer(state, generator)

    generator.answer_with_evidence.assert_called_once_with(
        "What is my account balance?",
        [],
    )
    assert result["abstained"] is True
    assert result["sources"] == []
    assert result["error"] is None
