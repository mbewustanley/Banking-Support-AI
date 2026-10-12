


####### TO RUN
# uvicorn src.api:api --reload


from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from contextlib import asynccontextmanager

from src.service import BankingSupportService


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
        result = service.ask(chat_request.question)
        return ChatResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred while processing the question."
            ),
        ) from exc