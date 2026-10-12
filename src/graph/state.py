from typing import TypedDict
from langchain_core.documents import Document

# typedDict because LangGraph passes a shared state between nodes
# each node reads the fields it needs and returns a dict containing the field it updates
"""
Our state captures:
- question — the customer's current question.
- answer — the generated response.
- sources — the evidence sources supporting the answer.
- retrieval_results — retrieved passages and their metadata.
- abstained — whether the assistant declined to answer based on insufficient evidence.
- error — a place to record handled workflow errors.
"""


class ConversationState(TypedDict):
    """State passed between nodes in the banking support graph"""

    question: str
    answer: str
    sources: list[dict]
    retrieval_results: list[dict]
    retrieval_documents: list[tuple[Document, float]]
    abstained: bool
    error: str | None