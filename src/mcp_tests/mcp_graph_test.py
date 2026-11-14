
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI

from src.generation import GroundedAnswerGenerator
from src.graph.mcp_graph import build_mcp_support_graph

# Represents a customer's message in the LangChain conversation history.
from langchain_core.messages import HumanMessage


# Load OpenRouter credentials and model settings.
load_dotenv()

# Resolve the project root for launching the MCP server.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def main():

    # Configure the banking knowledge server as a local MCP subprocess.
    server_config = {
        "mcpServers": {
            "banking_knowledge": {
                "command": sys.executable,
                "args": ["-m", "src.banking_knowledge_server"],
                "cwd": str(PROJECT_ROOT),
            }
        }
    }

    # Initialize the LLM that decides whether MCP retrieval is required.
    selection_model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=12000,
    )

    # Initialize the existing grounded generator.
    # Its retriever remains lazy because MCP supplies the evidence.
    generator = GroundedAnswerGenerator()

    # Keep the MCP adapter open while the graph is running.
    async with MCPAdapter(server_config) as adapter:

        # Discover the tools exposed by the banking MCP server.
        tools = await adapter.list_tools()

        # Build the MCP-enabled LangGraph workflow.
        graph = build_mcp_support_graph(
            selection_model=selection_model,
            tools=tools,
            generator=generator,
        )

        
        # Build a unique conversation ID.
        # LangGraph uses this ID to retrieve the saved state between turns.
        config = {
            "configurable": {
                "thread_id": "banking_customer_001"
            }
        }

        # Define two consecutive customer messages.
        # The second question deliberately refers to the first interaction.
        questions = [
            "What happens if my account is charged twice?",
            "Can I get a refund for that?"
        ]

        # Process each message using the same graph and thread ID.
        for question in questions:

            print("\n" + "=" * 60)
            print("CUSTOMER QUESTION:")
            print(question)

            # Each invocation adds a new customer message to the conversation.
            # The checkpointer restores the previous conversation state.
            result = await graph.ainvoke(
                {
                    "question": question,
                    "messages": [
                        HumanMessage(content=question)
                    ],
                },
                config=config,
            )

            print("\nUSED MCP RETRIEVAL:")
            print(result.get("used_retrieval", False))

            print("\nGROUNDED ANSWER:")
            print(result.get("answer", "No answer returned."))

            print("\nABSTAINED:")
            print(result.get("abstained", False))

            print("\nCONVERSATION HISTORY:")
            for message in result.get("messages", []):
                print(f"{message.type.upper()}: {message.content}")



if __name__ == "__main__":
    asyncio.run(main())
