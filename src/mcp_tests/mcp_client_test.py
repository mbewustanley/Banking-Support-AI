
import asyncio
from pathlib import Path

# MCPAdapter connects to MCP servers and converts their tools
# into tools that can be used by LangChain agents.
from langchain.mcp import MCPAdapter


async def main():

    # Resolve the sandbox directory relative to this Python file.
    # parents[1] takes us from src/ back to the project root.
    sandbox = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "mcp_sandbox"
    )

    # Configure the external filesystem MCP server.
    config = {
        "mcpServers": {
            "filesystem": {
                # cmd allows Windows to execute the npx command.
                "command": "cmd",

                # IMPORTANT: args must be a list, not a set.
                # Each item represents one command-line argument.
                "args": [
                    "/c",
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(sandbox.resolve()),
                ],
            }
        }
    }

    # Open the MCP connection.
    # The adapter starts the server and manages communication.
    async with MCPAdapter(config) as adapter:

        # Ask the server which tools it makes available.
        tools = await adapter.list_tools()

        print("\nCONNECTED TO FILESYSTEM MCP SERVER")
        print(f"Discovered {len(tools)} tools:\n")

        # Display each discovered tool and its input schema.
        for tool in tools:
            print(f"Tool: {tool.name}")
            #print(f"Description: {tool.description}")
            #print(f"Arguments: {tool.args}")
            print("-" * 60)


        # Find the text-reading tool
        read_tool = next(
            (tool for tool in tools if tool.name == "read_text_file"), None
        )
        # stop if the server does not expose the expected tool
        if read_tool is None:
            raise RuntimeError("read_text_file tool was not found.")

        # Define thefile we want the MCP server to read
        file_path = sandbox / "banking_note.txt"

        # Invoke the remote MCP tool through its langchain wrapper
        result = await read_tool.ainvoke({
            "path": str(file_path.resolve())
        })

        # Display the result returned by the MCP server
        print("\nMCP TOOL EXECUTION SUCCESSFUL")
        print(f"File: {file_path.name}")
        file_content = result[0]["text"]
        print(f"Content returned:\n{file_content.strip()}")  # leave as result for full structured content reponse

# Run the asynchronous main function when this file
# is executed directly, rather than imported as a module.
if __name__ == "__main__":
    asyncio.run(main())
