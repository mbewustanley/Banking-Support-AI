
from pathlib import Path

import pytest

from src.ingestion import PolicyIngestion


@pytest.fixture
def policy_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "policies"
    directory.mkdir()

    (directory / "card_fees.md").write_text(
        "# Card Fees\n\n"
        "A foreign transaction fee may apply to purchases "
        "made in a foreign currency.\n\n"
        "Customers should review their account terms.",
        encoding="utf-8",
    )

    (directory / "refunds.md").write_text(
        "# Refunds\n\n"
        "A refund request does not guarantee approval.\n\n"
        "Support must investigate before confirming a refund.",
        encoding="utf-8",
    )

    return directory


def test_load_documents(policy_dir: Path) -> None:
    ingestion = PolicyIngestion(policy_dir=str(policy_dir))

    documents = ingestion.load_documents()

    assert len(documents) == 2
    assert all(document.page_content for document in documents)
    assert {doc.metadata["document_id"] for doc in documents} == {
        "card_fees",
        "refunds",
    }


def test_split_documents_preserves_source(
    policy_dir: Path,
) -> None:
    ingestion = PolicyIngestion(
        policy_dir=str(policy_dir),
        chunk_size=100,
        chunk_overlap=20,
    )

    documents = ingestion.load_documents()
    chunks = ingestion.split_documents(documents)

    assert chunks
    assert all("source" in chunk.metadata for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_missing_directory_raises_error(
    tmp_path: Path,
) -> None:
    ingestion = PolicyIngestion(
        policy_dir=str(tmp_path / "missing")
    )

    with pytest.raises(FileNotFoundError):
        ingestion.load_documents()


def test_empty_directory_raises_error(
    tmp_path: Path,
) -> None:
    ingestion = PolicyIngestion(policy_dir=str(tmp_path))

    with pytest.raises(ValueError, match="No Markdown policies"):
        ingestion.load_documents()


def test_invalid_chunk_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        PolicyIngestion(chunk_size=100, chunk_overlap=100)