
from langgraph.graph import END, START, StateGraph

from src.generation import GroundedAnswerGenerator
from src.graph.nodes import (
    generate_grounded_answer,
    retrieve_policy_evidence,
)
from src.graph.state import ConversationState
from src.retrieval import PolicyRetriever


def build_support_graph(
    retriever: PolicyRetriever | None = None,
    generator: GroundedAnswerGenerator | None = None,
):
    """Build and compile the banking support workflow."""

    if retriever is None:
        retriever = PolicyRetriever()

        # Initialize the policy index on first startup only.
        existing = retriever.vector_store.get()
        if not existing.get("ids"):
            retriever.build_index()

    if generator is None:
        generator = GroundedAnswerGenerator(retriever=retriever)

    workflow = StateGraph(ConversationState)

    workflow.add_node(
        "retrieve",
        lambda state: retrieve_policy_evidence(state, retriever),
    )
    workflow.add_node(
        "generate",
        lambda state: generate_grounded_answer(state, generator),
    )

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
