"""## The class created in this file will act as the boundary between the langgraph
and future interfaces such as FastApi. 

It will accept customer question, execute the graphand return a predictable response"""


"""
WHY?
- separation of concerns (fastApi does not need to know about langgraph nodes or states)
- Dependency injection
- Consistent responses
- input validation (reject blank question before invoking graph)"""


from typing import Any
from src.graph.graph import build_support_graph


class BankingSupportService:
    """Application service for policy-grounded banking support"""

    def __init__(self, graph: Any | None = None) -> None:
        # Allow a compiled graph to be injected during testing
        self.graph = graph if graph is not None else build_support_graph()

    def ask(self,question: str) -> dict[str, Any]:
        """Answer a customer's question using the support graph."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty")

        initial_state = {
            "question": question.strip(),
            "answer": "",
            "sources": [],
            "retrieval_results": [],
            "retrieval_documents": [],
            "abstained": False,
            "error": None,
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "abstained": result["abstained"],
            "error": result["error"],
        }