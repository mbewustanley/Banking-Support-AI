import asyncio
import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage


# Instructions controlling when the LLM should use the banking knowledge tool.
SYSTEM_PROMPT = """
You are an assistant for a banking support platform.

For questions about banking policies, fees, refunds, account terms,
transactions, and other banking-specific rules, use the
search_banking_policies tool before answering.

For greetings and general conversation that do not require banking
policy information, you may respond without using a tool.

Never invent banking policy information.
"""


def _get_response_text(content) -> str:
    """
    Extract readable text from an LLM response.

    LangChain responses may contain either a plain string or a list
    of structured content blocks.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        )

    return str(content)



def _requires_banking_retrieval(question: str) -> bool:
    """
    Identify questions that clearly require banking policy evidence.

    This is a lightweight safety fallback, not a replacement
    for the LLM's more flexible tool-selection capability.
    """

    # Convert the question to lowercase for consistent matching.
    question = question.lower()

    # Banking-related terms that commonly require policy evidence.
    banking_keywords = {
        "bank", "banking", "account", "card", "fee", "fees",
        "refund", "transaction", "transfer", "payment",
        "withdrawal", "deposit", "balance", "interest",
        "overdraft", "loan", "atm", "charge", "charges",
        "dispute", "limit", "policy", "eligibility"
    }

    # Check whether any banking-related term appears in the question.
    return any(keyword in question for keyword in banking_keywords)




async def decide_tool_usage(state, model_with_tools):
    """
    Ask the LLM whether the customer question requires policy retrieval.

    The model receives the complete conversation history so it can
    understand follow-up questions and references to earlier messages.
    """

    question = state["question"].strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # Retrieve previous customer and assistant messages from graph state.
    conversation_history = state.get("messages", [])

    # Supply system instructions followed by the conversation history.
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *conversation_history,
    ]

    # Invoke the model with the MCP tools bound to it.
    response = await model_with_tools.ainvoke(messages)

    tool_calls = response.tool_calls or []

    # Safety fallback: obvious banking questions must not bypass retrieval.
    if not tool_calls and _requires_banking_retrieval(question):
        tool_calls = [{
            "name": "search_banking_policies",
            "args": {
                "query": question,
                "k": 3,
            },
        }]

    direct_answer = ""

    # Only use a direct LLM response when retrieval isn't required.
    if not tool_calls:
        direct_answer = _get_response_text(response.content)

    return {
        "selection_response": response,
        "tool_calls": tool_calls,
        "used_retrieval": bool(tool_calls),
        "answer": direct_answer,

        # Clear temporary data from the previous interaction.
        "retrieval_documents": [],
        "sources": [],
        "abstained": False,
        "error": None,

        # Save the assistant's direct response to conversation history.
        # The customer message is already added by the graph invocation.
        "messages": (
            [AIMessage(content=direct_answer)]
            if not tool_calls and direct_answer
            else []
        ),
    }


def _parse_mcp_response(tool_result) -> dict:
    """
    Extract and parse the JSON payload returned by the MCP tool.

    The MCPAdapter currently returns a list of content blocks,
    with the JSON payload inside a text block.
    """

    if isinstance(tool_result, str):
        json_text = tool_result

    elif isinstance(tool_result, list):
        text_blocks = [
            block["text"]
            for block in tool_result
            if isinstance(block, dict)
            and block.get("type") == "text"
            and "text" in block
        ]

        if not text_blocks:
            raise ValueError("MCP response contained no text content.")

        json_text = text_blocks[0]

    else:
        raise TypeError(
            f"Unexpected MCP response type: {type(tool_result)}"
        )

    # Convert the JSON string into a Python dictionary.
    return json.loads(json_text)



async def retrieve_policy_evidence(state, tool_lookup):
    """
    Execute the MCP tools selected by the LLM.

    Convert retrieved passages into LangChain Documents.
    Capture failures in the graph state instead of crashing the workflow.
    """

    evidence = []

    try:
        # Process each MCP tool call selected by the LLM.
        for tool_call in state.get("tool_calls", []):
            tool_name = tool_call["name"]
            tool_arguments = tool_call["args"]

            # Confirm that the requested tool is available.
            if tool_name not in tool_lookup:
                raise ValueError(f"Unknown MCP tool: {tool_name}")

            # Execute the MCP tool asynchronously.
            tool_result = await tool_lookup[tool_name].ainvoke(tool_arguments)

            # Parse the returned JSON response.
            payload = _parse_mcp_response(tool_result)

            # Convert retrieved passages into LangChain Documents.
            for item in payload.get("results", []):
                document = Document(
                    page_content=item["content"],
                    metadata={
                        "source": item.get("source"),
                        "document_id": item.get("document_id"),
                        "rank": item.get("rank"),
                    },
                )

                # Preserve the vector distance for the grounded generator.
                distance = float(item["distance"])
                evidence.append((document, distance))

        # Reject empty retrieval results.
        if not evidence:
            raise ValueError(
                "MCP retrieval completed but returned no policy evidence."
            )

        # Return successfully retrieved evidence.
        return {
            "retrieval_documents": evidence,
            "used_retrieval": True,
            "error": None,
        }

    
    except Exception as exc:
        # Prepare a safe response when MCP retrieval fails.
        fallback_answer = (
            "I'm unable to retrieve the relevant banking policy "
            "information at the moment. Please try again later or "
            "contact customer support."
        )

        return {
            "retrieval_documents": [],
            "used_retrieval": True,
            "error": f"Policy retrieval failed: {str(exc)}",
            "answer": fallback_answer,
            "abstained": True,
            "sources": [],

            # Save the fallback response to conversation history.
            "messages": [
                AIMessage(content=fallback_answer)
            ],
        }



async def generate_grounded_answer(state, generator):
    """
    Generate the final answer using the existing grounded generator.

    This node does not perform another retrieval operation.
    It uses only the evidence collected by the MCP retrieval node.
    """

    # Run the synchronous generator outside the async event loop.
    # This avoids blocking other asynchronous graph operations.
    result = await asyncio.to_thread(
        generator.answer_with_evidence,
        question=state["question"],
        results=state.get("retrieval_documents", []),
    )

    return {
    "answer": result.answer,
    "sources": result.sources,
    "abstained": result.abstained,

    # Save the grounded answer as an assistant message.
    "messages": [
        AIMessage(content=result.answer)
    ],
}
