
"""
BANKING KNOWLEDGE MCP CLIENT TEST

Connects to the custom Banking Knowledge MCP server,
discovers its tools, and tests policy retrieval.
"""

import time
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    # Resolve the project root so the client can launch the server
    # regardless of the current terminal directory.
    project_root = Path(__file__).resolve().parents[1]

    # Configure the MCP server process.
    # Using the current Python executable ensures the server runs
    # inside the same Conda environment as this client.
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "src.banking_knowledge_server",
        ],
        cwd=str(project_root),
    )

    # Start the server and establish an MCP connection over stdio.
    async with stdio_client(server_parameters) as (read_stream, write_stream):

        # Create a client session using the communication streams.
        async with ClientSession(read_stream, write_stream) as session:

            # Complete the MCP initialization handshake.
            await session.initialize()

            print("\nBANKING KNOWLEDGE MCP SERVER CONNECTED")

            # Discover the tools exposed by the server.
            tool_response = await session.list_tools()

            print("\nAVAILABLE MCP TOOLS:")

            for tool in tool_response.tools:
                print(f"\nTool: {tool.name}")
                print(f"Description: {tool.description}")

                # Show the input schema so we know what arguments
                # the server expects.
                print(
                    "Input schema:",
                    json.dumps(tool.input_schema, indent=2),
                )

            # Execute the banking policy search tool.
            # This is a real search against your existing ChromaDB index.
            print("\nEXECUTING BANKING POLICY SEARCH...")


            queries = [
                "can a fee be refunded automatically?",
                "What happens when a customer disputes a duplicate card charge?",
                "How are refund requests investigated?"
            ]

            for query in queries:
                print(f"\nSEARCHING: {query}")

                # start the timer immediately before calling the MCP tool.
                start_time = time.perf_counter()

                result = await session.call_tool(
                    "search_banking_policies",
                    arguments={
                        "query": query,
                        "k": 3,
                    },
                )

                # calculate the duration of individual search.
                elapsed_time = time.perf_counter() - start_time

                # Check whether the tool reported an execution error.
                if result.is_error:
                    raise RuntimeError(f"The MCP tool reported an execution error: {query}")

                print(f"search completed in {elapsed_time:.2f} seconds")

                # Display the result returned by the MCP server.
                print("\nRETRIEVAL RESULTS:")

                for content in result.content:
                    if content.type == "text":
                        print(content.text)

            # Check whether the tool reported an execution error.
            if result.is_error:
                raise RuntimeError("The MCP tool reported an execution error.")


if __name__ == "__main__":
    asyncio.run(main())
