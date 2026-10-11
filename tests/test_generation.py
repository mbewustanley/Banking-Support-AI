
"""Unit tests for grounded banking-policy answer generation."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.generation import GroundedAnswerGenerator


# ------------------------------------------------------------------
# Test doubles: no real vector database or LLM calls.
# ------------------------------------------------------------------

class FakeRetriever:
    """Return controlled retrieval results for testing."""

    def __init__(self, results=None):
        self.results = results if results is not None else []
        self.last_query = None
        self.last_k = None

    def search(self, query: str, k: int = 3):
        self.last_query = query
        self.last_k = k
        return self.results


class FakeLLM:
    """Return a predefined response instead of calling an API."""

    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return AIMessage(content=self.content)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_documents():
    """Representative retrieved banking-policy passages."""

    return [
        (
            Document(
                page_content=(
                    "Card fees may be refunded after the bank "
                    "reviews the customer's case."
                ),
                metadata={
                    "source": "card_fees.md",
                    "document_id": "card-fees",
                    "start_index": 0,
                },
            ),
            0.15,
        ),
        (
            Document(
                page_content=(
                    "Customers should contact support to dispute "
                    "an incorrect card charge."
                ),
                metadata={
                    "source": "card_disputes.md",
                    "document_id": "card-disputes",
                    "start_index": 0,
                },
            ),
            0.28,
        ),
    ]


@pytest.fixture
def generator_factory(sample_documents):
    """Build a generator with injectable test dependencies."""

    def factory(
        answer="A fee may be refunded after a review. [S1]",
        results=None,
        k=3,
    ):
        retriever = FakeRetriever(
            results=(
                sample_documents if results is None else results
            )
        )
        llm = FakeLLM(answer)

        generator = GroundedAnswerGenerator(
            retriever=retriever,
            llm=llm,
            k=k,
        )

        return generator, retriever, llm

    return factory


# ------------------------------------------------------------------
# Initialization tests
# ------------------------------------------------------------------

def test_generator_accepts_injected_dependencies(generator_factory):
    generator, retriever, llm = generator_factory()

    assert generator.retriever is retriever
    assert generator.llm is llm


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError, match="k must be at least 1"):
        GroundedAnswerGenerator(
            retriever=FakeRetriever(),
            llm=FakeLLM("Unused"),
            k=0,
        )


# ------------------------------------------------------------------
# Successful generation tests
# ------------------------------------------------------------------

def test_answer_returns_grounded_response(generator_factory):
    generator, _, _ = generator_factory(
        answer="A fee may be refunded after a review. [S1]"
    )

    result = generator.answer("Can my card fee be refunded?")

    assert result.answer == (
        "A fee may be refunded after a review. [S1]"
    )
    assert result.abstained is False


def test_answer_uses_retriever_with_question_and_k(
    generator_factory,
):
    generator, retriever, _ = generator_factory(k=2)

    generator.answer("Can my card fee be refunded?")

    assert retriever.last_query == (
        "Can my card fee be refunded?"
    )
    assert retriever.last_k == 2


def test_answer_preserves_cited_source_metadata(
    generator_factory,
):
    generator, _, _ = generator_factory(
        answer="A fee may be refunded after review. [S1]"
    )

    result = generator.answer("Can my card fee be refunded?")

    assert len(result.sources) == 1
    assert result.sources[0]["citation_id"] == "S1"
    assert result.sources[0]["source"] == "card_fees.md"
    assert result.sources[0]["document_id"] == "card-fees"
    assert result.sources[0]["distance"] == pytest.approx(0.15)


def test_only_cited_sources_are_returned(generator_factory):
    generator, _, _ = generator_factory(
        answer="Contact support about an incorrect charge. [S2]"
    )

    result = generator.answer("What if I have an incorrect charge?")

    assert result.abstained is False
    assert [source["citation_id"] for source in result.sources] == [
        "S2"
    ]
    assert result.sources[0]["source"] == "card_disputes.md"


def test_multiple_valid_citations_are_accepted(generator_factory):
    generator, _, _ = generator_factory(
        answer=(
            "A fee may be refunded after review [S1]. "
            "Contact support about incorrect charges [S2]."
        )
    )

    result = generator.answer("Tell me about card fees and disputes.")

    assert result.abstained is False
    assert {
        source["citation_id"] for source in result.sources
    } == {"S1", "S2"}


def test_retrieval_results_include_diagnostics(generator_factory):
    generator, _, _ = generator_factory()

    result = generator.answer("Can my card fee be refunded?")

    assert len(result.retrieval_results) == 2
    assert result.retrieval_results[0]["citation_id"] == "S1"
    assert result.retrieval_results[0]["distance"] == pytest.approx(
        0.15
    )


def test_result_can_be_converted_to_dictionary(generator_factory):
    generator, _, _ = generator_factory()

    result = generator.answer("Can my card fee be refunded?")
    output = result.to_dict()

    assert output["answer"] == result.answer
    assert output["sources"] == result.sources
    assert output["abstained"] is False
    assert output["retrieval_results"] == result.retrieval_results


# ------------------------------------------------------------------
# Abstention tests
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "response",
    [
        "ABSTAIN: The available policies do not specify this.",
        "ABSTAIN: Please contact support for clarification.",
    ],
)
def test_model_abstention_is_preserved(generator_factory, response):
    generator, _, _ = generator_factory(answer=response)

    result = generator.answer("Is this transaction guaranteed to clear?")

    assert result.abstained is True
    assert "available policies" in result.answer.lower() or (
        "contact support" in result.answer.lower()
    )
    assert result.sources == []


def test_empty_retrieval_abstains_without_calling_llm(
    generator_factory,
):
    generator, _, llm = generator_factory(results=[])

    result = generator.answer("What is the international transfer fee?")

    assert result.abstained is True
    assert result.sources == []
    assert result.retrieval_results == []
    assert "couldn't find relevant policy information" in (
        result.answer.lower()
    )
    assert llm.calls == []


# ------------------------------------------------------------------
# Citation validation tests
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "response",
    [
        "Your fee will be refunded automatically. [S9]",
        "Your fee will be refunded automatically. [S1] [S9]",
    ],
)
def test_fabricated_citation_causes_abstention(
    generator_factory, response
):
    generator, _, _ = generator_factory(answer=response)

    result = generator.answer("Will my fee be refunded automatically?")

    assert result.abstained is True
    assert result.sources == []
    assert "valid source citations" in result.answer.lower()


@pytest.mark.parametrize(
    "response",
    [
        "Your fee will be refunded automatically.",
        "The bank will refund you [S1, S2].",
        "The bank will refund you [s1].",
    ],
)
def test_missing_or_malformed_citations_cause_abstention(
    generator_factory, response
):
    generator, _, _ = generator_factory(answer=response)

    result = generator.answer("Will my fee be refunded automatically?")

    assert result.abstained is True
    assert result.sources == []
    assert "valid source citations" in result.answer.lower()


def test_citation_validation_is_based_on_current_retrieval(
    generator_factory,
):
    generator, _, _ = generator_factory(
        answer="The bank reviews disputed charges. [S2]",
        results=[
            (
                Document(
                    page_content="Disputed charges are reviewed.",
                    metadata={
                        "source": "disputes.md",
                        "document_id": "disputes",
                    },
                ),
                0.12,
            )
        ],
    )

    result = generator.answer("How are disputed charges handled?")

    # This request retrieved only one passage, which receives [S1].
    assert result.abstained is True
    assert result.sources == []


# ------------------------------------------------------------------
# Input and LLM response validation tests
# ------------------------------------------------------------------

@pytest.mark.parametrize("question", ["", " ", "\n\t"])
def test_empty_question_raises_value_error(
    generator_factory, question
):
    generator, retriever, llm = generator_factory()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.answer(question)

    assert retriever.last_query is None
    assert llm.calls == []


def test_question_is_stripped_before_retrieval(generator_factory):
    generator, retriever, _ = generator_factory()

    generator.answer("  Can my card fee be refunded?  ")

    assert retriever.last_query == "Can my card fee be refunded?"


def test_empty_llm_response_raises_value_error(generator_factory):
    generator, _, _ = generator_factory(answer="   ")

    with pytest.raises(ValueError, match="empty response"):
        generator.answer("Can my card fee be refunded?")


def test_non_text_llm_response_raises_value_error(generator_factory):
    generator, _, _ = generator_factory()
    generator.llm.content = [
        {"type": "text", "text": "A fee may be refunded. [S1]"}
    ]

    with pytest.raises(ValueError, match="non-text response"):
        generator.answer("Can my card fee be refunded?")
