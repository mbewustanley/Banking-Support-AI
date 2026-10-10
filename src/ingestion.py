
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PolicyIngestion:
    """Load and split Markdown policy documents."""

    def __init__(
        self,
        policy_dir: str = "knowledge_base/policies",
        chunk_size: int = 700,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller "
                "than chunk_size"
            )

        self.policy_dir = Path(policy_dir)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def load_documents(self) -> list[Document]:
        """Load Markdown files with source metadata."""
        if not self.policy_dir.is_dir():
            raise FileNotFoundError(
                f"Policy directory not found: {self.policy_dir}"
            )

        files = sorted(self.policy_dir.glob("*.md"))

        if not files:
            raise ValueError(
                f"No Markdown policies found in {self.policy_dir}"
            )

        documents = []

        for file_path in files:
            content = file_path.read_text(encoding="utf-8").strip()

            if not content:
                raise ValueError(
                    f"Policy document is empty: {file_path}"
                )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path.as_posix(),
                        "document_id": file_path.stem,
                    },
                )
            )

        return documents

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Split loaded documents while retaining metadata."""
        if not documents:
            raise ValueError("Cannot split an empty document list")

        chunks = self.splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index

        return chunks

    def run(self) -> list[Document]:
        """Load policies and return their chunks."""
        documents = self.load_documents()
        return self.split_documents(documents)


if __name__ == "__main__":
    ingestion = PolicyIngestion()
    chunks = ingestion.run()

    print(f"Loaded and split policies into {len(chunks)} chunks.")

    for chunk in chunks:
        print("\n" + "=" * 60)
        print(f"Source: {chunk.metadata['source']}")
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"Characters: {len(chunk.page_content)}")
        print(chunk.page_content)