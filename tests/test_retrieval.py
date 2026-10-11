
from pathlib import Path

import pytest

from langchain_core.embeddings import Embeddings

from src.retrieval import PolicyRetriever


class FakeEmbeddings(Embeddings):
    """Simple deterministic embeddings for unit tests."""

    @staticmethod
    def encode(text: str) -> list[float]:
        text = text.lower()

        return [
            float(any(word in text for word in ["refund", "refunded"])),
            float(any(word in text for word in ["card", "fee", "transaction"])),
            float("duplicate" in text),
            float("unrelated" in text),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.encode(text)


@pytest.fixture
def policy_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "policies"
    directory.mkdir()

    (directory / "card_fees.md").write_text(
        "# Card Fees\n\n"
        "A foreign transaction fee may apply to card purchases. "
        "Customers should review their account terms.\n\n"
        "Duplicate card charges must be investigated.",
        encoding="utf-8",
    )

    (directory / "refunds.md").write_text(
        "# Refunds\n\n"
        "Customers may request a refund review. "
        "A refund is not guaranteed and requires investigation.",
        encoding="utf-8",
    )

    return directory


@pytest.fixture
def retriever(
    policy_dir: Path,
    tmp_path: Path,
) -> PolicyRetriever:
    return PolicyRetriever(
        policy_dir=str(policy_dir),
        persist_directory=str(tmp_path / "vector_store"),
        embeddings=FakeEmbeddings(),
    )


def test_build_index_indexes_policy_chunks(
    retriever: PolicyRetriever,
) -> None:
    count = retriever.build_index()

    assert count >= 2
    assert retriever.vector_store.get()["ids"]


def test_search_returns_documents_with_metadata(
    retriever: PolicyRetriever,
) -> None:
    retriever.build_index()

    results = retriever.search("Can I get a refund?", k=2)

    assert len(results) == 2

    for document, distance in results:
        assert document.page_content
        assert document.metadata["source"]
        assert isinstance(distance, float)


def test_rebuilding_index_does_not_duplicate_chunks(
    retriever: PolicyRetriever,
) -> None:
    retriever.build_index()
    first_count = len(retriever.vector_store.get()["ids"])

    retriever.build_index()
    second_count = len(retriever.vector_store.get()["ids"])

    assert first_count == second_count


@pytest.mark.parametrize(
    ("query", "k"),
    [
        ("", 2),
        ("   ", 2),
        ("refund", 0),
        ("refund", -1),
    ],
)
def test_invalid_search_inputs_are_rejected(
    retriever: PolicyRetriever,
    query: str,
    k: int,
) -> None:
    with pytest.raises(ValueError):
        retriever.search(query, k=k)
