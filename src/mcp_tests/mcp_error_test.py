
import asyncio

from src.graph.mcp_nodes import retrieve_policy_evidence


# Simulate an MCP tool that fails when called.
class FailingMCPTool:

    async def ainvoke(self, arguments):
        raise RuntimeError("Simulated MCP server connection failure.")


async def main():

    # Provide a banking question and a simulated tool call.
    state = {
        "question": "What happens if my account is charged twice?",
        "tool_calls": [
            {
                "name": "search_banking_policies",
                "args": {
                    "query": "duplicate account charge",
                    "k": 3,
                },
            }
        ],
    }

    # Replace the real MCP tool with our simulated failing tool.
    tool_lookup = {
        "search_banking_policies": FailingMCPTool()
    }

    # Execute the retrieval node directly.
    result = await retrieve_policy_evidence(state, tool_lookup)

    # Display the captured error and fallback response.
    print("ERROR:")
    print(result.get("error"))

    print("\nFALLBACK ANSWER:")
    print(result.get("answer"))

    print("\nABSTAINED:")
    print(result.get("abstained"))


if __name__ == "__main__":
    asyncio.run(main())
