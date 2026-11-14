
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI


# Load the project's existing environment variables from .env.
load_dotenv()


async def main():
    # Locate the project root and the restricted MCP sandbox.
    # This keeps the script independent of the terminal's working directory.
    project_root = Path(__file__).resolve().parents[1]
    sandbox = project_root / "data" / "mcp_sandbox"

    if not sandbox.is_dir():
        raise FileNotFoundError(f"MCP sandbox not found: {sandbox}")

    # Read the same OpenRouter settings used by generation.py.
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")

    if not api_key or not model_name:
        raise ValueError(
            "OPENROUTER_API_KEY and OPENROUTER_MODEL must be configured."
        )

    # Reuse the existing project's LLM provider and settings.
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "12000")),
        timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60")),
        max_retries=2,
    )

    # Tell MCPAdapter how to start the existing filesystem MCP server.
    # The server is restricted to the sandbox directory.
    mcp_config = {
        "mcpServers": {
            "filesystem": {
                # Windows uses cmd to execute the npx command.
                "command": "cmd",
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

    # Connect to the MCP server and discover its available tools.
    async with MCPAdapter(mcp_config) as adapter:
        discovered_tools = await adapter.list_tools()

        # Give the agent only the read_text_file tool.
        # This is an example of limiting an agent's permissions.
        allowed_tools = [
            tool
            for tool in discovered_tools
            if tool.name == "read_text_file"
        ]

        if not allowed_tools:
            raise RuntimeError(
                "The filesystem MCP server did not expose read_text_file."
            )

        print("MCP CONNECTION SUCCESSFUL")
        print("Tools available to the agent:")
        for tool in allowed_tools:
            print(f"- {tool.name}")

        # Create an agent that can decide when to use its available tools.
        agent = create_agent(
            model=llm,
            tools=allowed_tools,
            system_prompt=(
                "You are a helpful assistant. "
                "Use the filesystem tool when you need to inspect a file. "
                "Only access files inside the permitted sandbox. "
                "After reading a file, explain its contents clearly. "
                "Never claim to have read a file unless the tool succeeded."
            ),
        )

        # Ask a question that requires the agent to use the MCP tool.
        question = (
            "Read banking_note.txt from the filesystem and tell me "
            "exactly what information it contains."
        )

        print("\nUSER QUESTION:")
        print(question)

        # Run the agent. It may call the MCP tool before answering.
        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            }
        )

        # Display the tool calls so we can see whether the agent
        # actually chose to use the MCP server.
        print("\nAGENT ACTIVITY:")
        for message in result["messages"]:
            tool_calls = getattr(message, "tool_calls", [])

            for tool_call in tool_calls:
                print(f"Tool called: {tool_call['name']}")
                print(f"Tool arguments: {tool_call['args']}")

        # The final message should contain the agent's natural-language answer.
        final_message = result["messages"][-1]

        print("\nFINAL AGENT RESPONSE:")
        print(final_message.content)


if __name__ == "__main__":
    asyncio.run(main())
