


####### TO RUN
# uvicorn src.api:app --reload
import time
import uuid

import logging
from src.logging_config import configure_logging

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field, field_validator

from contextlib import asynccontextmanager

from src.service import BankingSupportService

configure_logging()
logger = logging.getLogger(__name__)

# creates a service when the application starts and releases the reference when it shuts down.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared application resources."""
    app.state.support_service = BankingSupportService()
    yield
    app.state.support_service = None


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