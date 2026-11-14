
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class MCPConversationState(TypedDict, total=False):
    """
    State shared between nodes in the MCP-enabled LangGraph workflow.

    This remains separate from the original direct RAG state.
    """

    # Stores conversation messages across multiple interactions.
    # The reducer appends new messages instead of replacing history.
    messages: Annotated[list[AnyMessage], add_messages]

    # The original customer question.
    question: str

    # The LLM response that determines whether a tool should be called.
    selection_response: Any

    # Tool calls requested by the LLM.
    tool_calls: list[dict]

    # Whether the workflow actually used MCP retrieval.
    used_retrieval: bool

    # Retrieved policy documents and their vector distances.
    retrieval_documents: list[tuple[Document, float]]

    # Final customer-facing response.
    answer: str

    # Citation information returned by the grounded generator.
    sources: list[dict]

    # Whether the generator abstained from answering.
    abstained: bool

    # Errors encountered during execution.
    error: str | None
