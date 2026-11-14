
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load API credentials and model configuration from the project .env file.
load_dotenv()

# Resolve the project root for launching the MCP server.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def main():
    # Configure the local banking knowledge MCP server.
    server_config = {
        "mcpServers": {
            "banking_knowledge": {
                "command": sys.executable,
                "args": ["-m", "src.banking_knowledge_server"],
                "cwd": str(PROJECT_ROOT),
            }
        }
    }

    # Initialize the LLM using the same OpenRouter configuration
    # as the existing Banking Support AI project.
    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=12000,
    )

    async with MCPAdapter(server_config) as adapter:

        # Discover the MCP tools and make them available to the LLM.
        tools = await adapter.list_tools()

        # Bind the discovered tools to the model.
        # The model can now request tool execution when appropriate.
        model_with_tools = model.bind_tools(tools)

        # This instruction guides the model's tool-selection behaviour.
        messages = [
            SystemMessage(
                content=(
                    "You are an assistant for a banking support platform. "
                    "For questions that require knowledge of banking "
                    "policies, fees, refunds, or account terms, use the "
                    "search_banking_policies tool before answering. "
                    "For greetings or general conversation that does "
                    "not require policy information, do not call a tool. "
                    "Never invent banking policy information."
                )
            ),
            HumanMessage(
                content="Can a fee be refunded automatically?"
            ),
        ]

        # Ask the LLM to decide whether it needs the MCP tool.
        response = await model_with_tools.ainvoke(messages)

        print("\nLLM RESPONSE:")
        print(response.content)

        # Inspect the response for tool calls requested by the LLM.
        if not response.tool_calls:
            print("\nThe LLM decided not to call an MCP tool.")
            return

        print("\nLLM REQUESTED TOOL CALLS:")

        # Create a lookup so we can execute the requested tool by name.
        tool_lookup = {tool.name: tool for tool in tools}

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_arguments = tool_call["args"]

            print(f"\nTool: {tool_name}")
            print(f"Arguments: {tool_arguments}")

            # Execute the tool requested by the LLM.
            selected_tool = tool_lookup[tool_name]
            result = await selected_tool.ainvoke(tool_arguments)

            print("\nMCP TOOL RESULT:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
