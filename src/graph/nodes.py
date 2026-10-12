
from src.graph.state import ConversationState
from src.retrieval import PolicyRetriever
from src.generation import GroundedAnswerGenerator




""" 
FIRST LANGGRAPH NODE: PolicyRetriever

"""


def retrieve_policy_evidence(
    state: ConversationState,
    retriever: PolicyRetriever,
) -> dict:
    """Retrieve relevant policy evidence for the customer's question."""

    # -----------------------------------------------------------Define the state
    question = state["question"].strip()

    if not question:
        raise ValueError("Question cannot be empty")
    
    
    # -----------------------------------------------------------Run retrieval chain
    results = retriever.search(question, k=3)

    retrieval_results = []
    sources = []

    # -----------------------------------------------------------create dict return for node
    for rank, (document, distance) in enumerate(results, start=1):
        source = document.metadata.get("source", "unknown")
        document_id = document.metadata.get("document_id", "unknown")

        retrieval_results.append(
            {
                "rank": rank,
                "content": document.page_content,
                "source": source,
                "document_id": document_id,
                "distance": float(distance),
            }
        )

        sources.append(
            {
                "source": source,
                "document_id": document_id,
            }
        )

    return {
        "question": question,
        "retrieval_results": retrieval_results,
        "retrieval_documents": results,
        "sources": sources,
        "error": None,
    }



"""
SECOND NODE: GroundedAnswerGeneration

"""


def generate_grounded_answer(
    state: ConversationState,
    generator: GroundedAnswerGenerator,
) -> dict:
    """Generate a policy-grounded answer from the current question."""
    question = state["question"].strip()

    if not question:
        raise ValueError("Question cannot be empty")

    results = state["retrieval_documents"]
    result = generator.answer_with_evidence(question, results)

    return {
        "answer": result.answer,
        "sources": result.sources,
        "retrieval_results": result.retrieval_results,
        "abstained": result.abstained,
        "error": None,
    }
