
from unittest.mock import Mock

from langchain_core.documents import Document

from src.graph.graph import build_support_graph


def test_workflow_retrieves_and_generates_answer():
    document = Document(
        page_content="Duplicate charges must be investigated.",
        metadata={
            "source": "knowledge_base/policies/card_fees.md",
            "document_id": "card_fees",
        },
    )

    retriever = Mock()
    retriever.search.return_value = [(document, 0.25)]

    generator = Mock()
    generator.answer_with_evidence.return_value = Mock(
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

    graph = build_support_graph(
        retriever=retriever,
        generator=generator,
    )

    result = graph.invoke(
        {
            "question": "What happens with duplicate charges?",
            "answer": "",
            "sources": [],
            "retrieval_results": [],
            "retrieval_documents": [],
            "abstained": False,
            "error": None,
        }
    )

    retriever.search.assert_called_once_with(
        "What happens with duplicate charges?",
        k=3,
    )
    generator.answer_with_evidence.assert_called_once()

    assert result["answer"] == (
        "A duplicate charge must be investigated. [S1]"
    )
    assert result["abstained"] is False
    assert result["sources"][0]["document_id"] == "card_fees"
    assert result["error"] is None


def test_workflow_handles_no_retrieved_evidence():
    retriever = Mock()
    retriever.search.return_value = []

    generator = Mock()
    generator.answer_with_evidence.return_value = Mock(
        answer=(
            "I couldn't find relevant policy information to answer "
            "your question. Please contact customer support "
            "for clarification."
        ),
        sources=[],
        retrieval_results=[],
        abstained=True,
    )

    graph = build_support_graph(
        retriever=retriever,
        generator=generator,
    )

    result = graph.invoke(
        {
            "question": "What is my account balance?",
            "answer": "",
            "sources": [],
            "retrieval_results": [],
            "retrieval_documents": [],
            "abstained": False,
            "error": None,
        }
    )

    retriever.search.assert_called_once()
    generator.answer_with_evidence.assert_called_once_with(
        "What is my account balance?",
        [],
    )

    assert result["abstained"] is True
    assert result["sources"] == []
    assert result["retrieval_results"] == []
