"""
SEMANTIC RETRIEVAL

- with vector database local persistence
"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion import PolicyIngestion


class PolicyRetriever:
    """Build and query a persistent policy vector index."""

    def __init__(
        self,
        policy_dir: str = "knowledge_base/policies",
        #persist_directory: str = "data/vector_store/chroma",
        persist_directory: str | None =  None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embeddings: Embeddings | None = None,
    ) -> None:
        self.persist_directory = str(
            Path(
                persist_directory
                or os.getenv(
                    "CHROMA_PERSIST_DIRECTORY",
                    "data/vector_store/chroma",
                )
            )
        )
        Path(self.persist_directory).parent.mkdir(
            parents=True, exist_ok=True
        )

        # Allow tests to inject a lightweight embedding implementation.
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.ingestion = PolicyIngestion(policy_dir=policy_dir)

        self.vector_store = Chroma(
            collection_name="banking_policies",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def build_index(self) -> int:
        """Rebuild the index from the current policy documents."""
        chunks = self.ingestion.run()

        # Remove previous entries so deleted or shortened policies
        # don't leave stale chunks in the index.
        existing = self.vector_store.get()
        existing_ids = existing.get("ids", [])

        if existing_ids:
            self.vector_store.delete(ids=existing_ids)

        ids = [
            f"{chunk.metadata['document_id']}:"
            f"{chunk.metadata.get('start_index', index)}"
            for index, chunk in enumerate(chunks)
        ]

        self.vector_store.add_documents(chunks, ids=ids)
        return len(chunks)

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[tuple[Document, float]]:
        """Return relevant chunks with their vector distances."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k < 1:
            raise ValueError("k must be at least 1")

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )


if __name__ == "__main__":
    retriever = PolicyRetriever()
    count = retriever.build_index()
    print(f"Indexed {count} policy chunks.")

    #question = "Can I get a refund for a card fee?"
    question = "can a fee be refunded automatically?"
    #question = "what should support do if my card was charged twice?"
    #question = "what happens when i request a refund?"

    results = retriever.search(question, k=3)

    print(f"\nQuestion: {question}")

    for rank, (document, distance) in enumerate(results, start=1):
        print("=" * 60)
        print(f"\nResult {rank}")
        print(f"Source: {document.metadata['source']}")
        print(f"Distance: {distance:.4f}")
        print(document.page_content)
