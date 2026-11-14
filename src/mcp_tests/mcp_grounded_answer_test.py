
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from src.generation import GroundedAnswerGenerator

# Load the project's OpenRouter credentials and model configuration.
load_dotenv()

# Resolve the project root for launching the MCP server.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def convert_mcp_result_to_documents(tool_result):
    """
    Convert the MCP JSON response into the evidence format
    expected by GroundedAnswerGenerator.answer_with_evidence().

    Expected output:
        list[tuple[Document, float]]
    """

    # LangChain MCP tools return content blocks.
    # Extract the text block containing the JSON response.
    if isinstance(tool_result, str):
        json_text = tool_result

    elif isinstance(tool_result, list):
        text_blocks = [
            block["text"]
            for block in tool_result
            if block.get("type") == "text" and "text" in block
        ]

        if not text_blocks:
            raise ValueError("MCP response contained no text content.")

        # Our banking search tool currently returns one JSON text block.
        json_text = text_blocks[0]

    else:
        raise TypeError(
            f"Unexpected MCP response type: {type(tool_result)}"
        )

    # Parse the JSON string into a Python dictionary.
    payload = json.loads(json_text)

    # Extract the retrieved passages from the response.
    results = payload.get("results", [])

    if not results:
        raise ValueError("MCP search returned no policy evidence.")

    evidence = []

    for item in results:
        # Store the policy passage as a LangChain Document.
        document = Document(
            page_content=item["content"],
            metadata={
                "source": item.get("source"),
                "document_id": item.get("document_id"),
                "rank": item.get("rank"),
            },
        )

        # Preserve the vector distance alongside the Document.
        distance = float(item["distance"])

        evidence.append((document, distance))

    return evidence


async def main():
    # Configure the independent Banking Knowledge MCP server.
    server_config = {
        "mcpServers": {
            "banking_knowledge": {
                "command": sys.executable,
                "args": ["-m", "src.banking_knowledge_server"],
                "cwd": str(PROJECT_ROOT),
            }
        }
    }

    # Initialize the model responsible for deciding whether
    # the banking policy search tool should be called.
    selection_model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=12000,
    )

    # The existing generator handles the final grounded answer.
    # Its retriever should remain lazy so it doesn't load another
    # embedding model in this MCP client process.
    generator = GroundedAnswerGenerator()

    question = "Can a fee be refunded automatically?"

    async with MCPAdapter(server_config) as adapter:

        # Discover MCP tools and bind them to the selection model.
        tools = await adapter.list_tools()
        model_with_tools = selection_model.bind_tools(tools)

        # Ask the LLM to decide whether policy retrieval is necessary.
        messages = [
            SystemMessage(
                content=(
                    "You are an assistant for a banking support platform. "
                    "For questions requiring banking policy information, "
                    "use search_banking_policies. "
                    "Do not invent banking policy information."
                )
            ),
            HumanMessage(content=question),
        ]

        selection_response = await model_with_tools.ainvoke(messages)

        # This test expects a banking policy question to trigger retrieval.
        if not selection_response.tool_calls:
            raise RuntimeError(
                "The LLM did not request the banking policy search tool."
            )

        # Find the LangChain tool objects by their names.
        tool_lookup = {tool.name: tool for tool in tools}

        # Collect evidence from the requested MCP tool calls.
        evidence = []

        for tool_call in selection_response.tool_calls:
            tool_name = tool_call["name"]

            if tool_name not in tool_lookup:
                raise RuntimeError(f"Unknown tool requested: {tool_name}")

            # Execute the tool selected by the LLM.
            tool_result = await tool_lookup[tool_name].ainvoke(
                tool_call["args"]
            )

            # Convert the MCP response into LangChain Documents.
            evidence.extend(
                convert_mcp_result_to_documents(tool_result)
            )

        print(f"\nRetrieved {len(evidence)} policy passages.")

        # Pass the retrieved evidence directly to the existing generator.
        # This avoids performing another retrieval operation.
        final_result = generator.answer_with_evidence(
            question=question,
            results=evidence,
        )

        print("\nCUSTOMER QUESTION:")
        print(question)

        print("\nGROUNDED ANSWER:")
        print(final_result.answer)

        print("\nABSTAINED:")
        print(final_result.abstained)

        print("\nSOURCES:")
        for source in final_result.sources:
            print(source)


if __name__ == "__main__":
    asyncio.run(main())
