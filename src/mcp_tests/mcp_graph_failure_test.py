
import asyncio

from langchain_core.messages import AIMessage

from src.graph.mcp_graph import build_mcp_support_graph


# Simulate an LLM that selects the banking search tool.
class MockSelectionModel:

    def bind_tools(self, tools):
        # Return this mock model instead of connecting to a real LLM.
        return self

    async def ainvoke(self, messages):
        # Return a simulated tool call in LangChain's expected format.
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_banking_policies",
                    "args": {
                        "query": "duplicate account charge",
                        "k": 3,
                    },
                    "id": "test_call_001",
                    "type": "tool_call",
                }
            ],
        )


# Simulate an MCP tool that cannot connect to its server.
class FailingMCPTool:

    name = "search_banking_policies"

    async def ainvoke(self, arguments):
        raise RuntimeError("Simulated MCP server connection failure.")


# This generator should never execute when retrieval fails.
class MockGenerator:

    def answer_with_evidence(self, question, results):
        raise AssertionError(
            "Generator should not execute after retrieval failure."
        )


async def main():

    # Supply our simulated model, tool and generator.
    graph = build_mcp_support_graph(
        selection_model=MockSelectionModel(),
        tools=[FailingMCPTool()],
        generator=MockGenerator(),
    )

    # Execute the complete LangGraph workflow.
    result = await graph.ainvoke({
        "question": "What happens if my account is charged twice?"
    })

    # Inspect the final graph state.
    print("ERROR:")
    print(result.get("error"))

    print("\nANSWER:")
    print(result.get("answer"))

    print("\nUSED MCP RETRIEVAL:")
    print(result.get("used_retrieval"))

    print("\nABSTAINED:")
    print(result.get("abstained"))


if __name__ == "__main__":
    asyncio.run(main())
