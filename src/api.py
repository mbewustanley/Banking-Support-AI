


####### TO RUN
# uvicorn src.api:app --reload

import os
import sys

import time
import uuid

import logging
from src.logging_config import configure_logging

from dotenv import load_dotenv

from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field, field_validator

from contextlib import asynccontextmanager

from src.service import BankingSupportService
from src.generation import GroundedAnswerGenerator
from src.graph.mcp_graph import build_mcp_support_graph




configure_logging()
logger = logging.getLogger(__name__)

# creates a service when the application starts and releases the reference when it shuts down.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared RAG and MCP resources when the API starts."""

    # Load API credentials and model configuration.
    load_dotenv()

    # Preserve the existing direct RAG service.
    app.state.support_service = BankingSupportService()

    # Resolve the project root for launching the MCP server.
    project_root = Path(__file__).resolve().parent.parent

    # Configure the banking knowledge server as a subprocess.
    server_config = {
        "mcpServers": {
            "banking_knowledge": {
                "command": sys.executable,
                "args": ["-m", "src.mcp.banking_knowledge_server"],
                "cwd": str(project_root),
            }
        }
    }

    # Initialise the model responsible for selecting MCP tools.
    selection_model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=12000,
    )

    # Initialise the existing grounded answer generator.
    generator = GroundedAnswerGenerator()

    # Keep the MCP adapter and its subprocess alive throughout
    # the FastAPI application's lifetime.
    async with MCPAdapter(server_config) as adapter:

        # Discover the tools exposed by the MCP server.
        tools = await adapter.list_tools()

        # Build the graph once, preserving its checkpointer
        # across requests handled by this application instance.
        app.state.mcp_graph = build_mcp_support_graph(
            selection_model=selection_model,
            tools=tools,
            generator=generator,
        )

        logger.info("MCP banking support runtime initialised")

        try:
            # FastAPI can now start accepting requests.
            yield

        finally:
            # Release application references during shutdown.
            app.state.mcp_graph = None
            app.state.support_service = None

            logger.info("Banking support application shutting down")



app = FastAPI(
    title="Banking Support AI Platform",
    description=(
        "Policy-grounded banking customer support using RAG and LangGraph."
    ),
    version="0.1.0",
    lifespan=lifespan
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request IDs, status codes, and request duration."""
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "HTTP request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response

    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "HTTP request failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration_ms, 2),
            },
        )
        raise



BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html")


class ChatRequest(BaseModel):
    """Incoming customer question."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The customer's banking support question.",
    )
    
    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value



class ChatResponse(BaseModel):
    """Structured answer returned by the support service."""

    answer: str
    sources: list[dict[str, Any]]
    abstained: bool
    error: str | None = None




class MCPChatRequest(BaseModel):
    """Incoming request for the MCP-enabled conversational agent."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The customer's banking support question.",
    )

    # Reuse this ID to continue an existing conversation.
    # If omitted, the API creates a new conversation.
    conversation_id: str | None = Field(
        default=None,
        max_length=128,
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_must_not_be_blank(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Conversation ID cannot be blank")
        return value


class MCPChatResponse(BaseModel):
    """Structured response returned by the MCP-enabled agent."""

    conversation_id: str
    answer: str
    used_retrieval: bool
    sources: list[dict[str, Any]]
    abstained: bool
    error: str | None = None




@app.get("/health")
def health_check() -> dict[str, str]:
    """Report API process health."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    chat_request: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Answer a customer question using banking policy evidence."""
    try:
        service = request.app.state.support_service

        logger.info("Starting support request")

        result = service.ask(chat_request.question)

        logger.info("Support request completed")
        
        return ChatResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Failed to process banking support question")

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred while processing the question."
            ),
        ) from exc




@app.post("/mcp/chat", response_model=MCPChatResponse)
async def mcp_chat(
    chat_request: MCPChatRequest,
    request: Request,
) -> MCPChatResponse:
    """Process a customer question using the MCP-enabled LangGraph agent."""

    try:
        # Retrieve the graph initialised during application startup.
        graph = request.app.state.mcp_graph

        # Continue an existing conversation or create a new one.
        conversation_id = (
            chat_request.conversation_id or str(uuid.uuid4())
        )

        # LangGraph uses thread_id to retrieve the correct checkpoint.
        config = {
            "configurable": {
                "thread_id": conversation_id,
            }
        }

        logger.info(
            "Starting MCP support request",
            extra={"conversation_id": conversation_id},
        )

        # Execute the graph asynchronously.
        result = await graph.ainvoke(
            {
                "question": chat_request.question,
                "messages": [
                    HumanMessage(content=chat_request.question)
                ],
            },
            config=config,
        )

        logger.info(
            "MCP support request completed",
            extra={
                "conversation_id": conversation_id,
                "used_retrieval": result.get("used_retrieval", False),
                "abstained": result.get("abstained", False),
            },
        )

        # Convert the graph state into the API response schema.
        return MCPChatResponse(
            conversation_id=conversation_id,
            answer=result.get("answer", "No answer returned."),
            used_retrieval=result.get("used_retrieval", False),
            sources=result.get("sources", []),
            abstained=result.get("abstained", False),
            error=result.get("error"),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Failed to process MCP support request")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the request.",
        ) from exc
