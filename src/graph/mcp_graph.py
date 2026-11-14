
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from src.generation import GroundedAnswerGenerator
from src.graph.mcp_nodes import (
    decide_tool_usage,
    generate_grounded_answer,
    retrieve_policy_evidence,
)
from src.graph.mcp_state import MCPConversationState


def build_mcp_support_graph(
    selection_model,
    tools,
    generator: GroundedAnswerGenerator,
):
    """Build the asynchronous MCP-enabled banking support graph."""

    # Bind the MCP tools to the model once.
    model_with_tools = selection_model.bind_tools(tools)

    # Create a lookup for executing the requested MCP tools.
    tool_lookup = {tool.name: tool for tool in tools}

    # create one checkpointer for this graph instance
    checkpointer = InMemorySaver()

    # Define the graph state.
    workflow = StateGraph(MCPConversationState)

    # Define asynchronous node wrappers.
    async def decide_node(state: MCPConversationState):
        return await decide_tool_usage(state, model_with_tools)

    async def retrieve_node(state: MCPConversationState):
        return await retrieve_policy_evidence(state, tool_lookup)

    async def generate_node(state: MCPConversationState):
        return await generate_grounded_answer(state, generator)

    # Register the nodes.
    workflow.add_node("decide", decide_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Start with the LLM's tool-selection decision.
    workflow.add_edge(START, "decide")

    # Route banking questions to retrieval.
    def route_after_decision(state: MCPConversationState):
        if state.get("tool_calls"):
            return "retrieve"

        return END

    workflow.add_conditional_edges(
        "decide",
        route_after_decision,
        {
            "retrieve": "retrieve",
            END: END,
        },
    )

    # Decide what happens after MCP retrieval.
    def route_after_retrieval(state: MCPConversationState):
        # If retrieval failed, skip generation and terminate safely.
        if state.get("error"):
            return END

        # Otherwise, continue to grounded answer generation.
        return "generate"

    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            END: END,
        },
    )

    # Finish after generating the grounded answer.
    workflow.add_edge("generate", END)

    # compile the workflow with checkpointing enabled
    return workflow.compile(checkpointer=checkpointer)
