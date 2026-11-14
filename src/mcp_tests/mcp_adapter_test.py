import asyncio
import sys
from pathlib import Path
from langchain.mcp import MCPAdapter


# resolve the project root so the server starts from the correct directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def main():
    # configure the local MCP server as a subprocess
    # sys.executable ensures we use the python interpreter from the currently activated conda environment

    server_config ={
        "mcpServers": {
            "banking_knowledge": {
                "command": sys.executable,
                "args": ["-m", "src.banking_knowledge_server"],
                "cwd": str(PROJECT_ROOT)
            }
        }
    }


    # MCPAdapter connects to the server and translates its tools into tools langchain models can use
    async with MCPAdapter(server_config) as adapter:

        # Discover the tools advertised by the MCP server
        tools = await adapter.list_tools()

        print("\nDISCOVERED LANGCHAIN TOOLS:")
        for tool in tools:
            print(f"\nName: {tool.name}")
            print(f"Description: {tool.description}")
            print(f"Arguments: {tool.args}")

        # Verify that the expected banking search tool is available.
        expected_tool = "search_banking_policies"
        discovered_names = [tool.name for tool in tools]

        if expected_tool not in discovered_names:
            raise RuntimeError(
                f"Expected tool '{expected_tool}' was not discovered."
            )

        print("\nMCP ADAPTER CONNECTION SUCCESSFUL")


if __name__ == "__main__":
    asyncio.run(main())