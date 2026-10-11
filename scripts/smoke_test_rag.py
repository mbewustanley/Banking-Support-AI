
"""Live smoke test for the Banking-Support-AI RAG pipeline."""

import sys

from src.generation import GroundedAnswerGenerator
from src.retrieval import PolicyRetriever


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "Can a fee be refunded?"

    print("=" * 70)
    print("BANKING SUPPORT AI - RAG SMOKE TEST")
    print("=" * 70)
    print(f"\nQuestion: {question}")

    # Reuse the existing persistent Chroma implementation.
    # Rebuild the index to ensure it reflects current policy files.
    retriever = PolicyRetriever()
    indexed_count = retriever.build_index()

    print(f"\nIndexed policy chunks: {indexed_count}")

    if indexed_count == 0:
        print("No policy chunks were indexed. Check the policy directory.")
        return

    generator = GroundedAnswerGenerator(
        retriever=retriever,
        k=3,
    )

    result = generator.answer(question)

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(result.answer)

    print(f"\nAbstained: {result.abstained}")

    print("\nCITED SOURCES")
    print("-" * 70)

    if result.sources:
        for source in result.sources:
            print(
                f"[{source['citation_id']}] "
                f"{source['source']} "
                f"(document_id={source['document_id']})"
            )
    else:
        print("No sources were returned.")

    print("\nRETRIEVAL DIAGNOSTICS")
    print("-" * 70)

    for item in result.retrieval_results:
        print(
            f"[{item['citation_id']}] "
            f"distance={item['distance']:.4f} "
            f"source={item['source']}"
        )


if __name__ == "__main__":
    main()
