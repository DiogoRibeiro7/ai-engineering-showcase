"""FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.factory import build_agent, build_conversation_store, build_index
from ai_engineering_showcase.memory import ConversationMemory
from ai_engineering_showcase.schemas import (
    ChatRequest,
    ChatResponse,
    IndexRequest,
    QueryRequest,
    QueryResponse,
)
from ai_engineering_showcase.telemetry import configure_logging, log_event


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    settings = Settings()
    agent = build_agent(settings)
    conversation_store = build_conversation_store(settings)

    app = FastAPI(
        title="AI Engineering Showcase API",
        version="0.1.0",
        description="Evidence-grounded customer feedback intelligence agent.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return service health."""
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        """Answer a question using the feedback insight agent."""
        try:
            result = agent.answer(request.question, top_k=request.top_k)
        except Exception as exc:  # noqa: BLE001 - convert to API-safe response.
            log_event("query_failed", {"error": str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return QueryResponse(result=result)

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """Answer a message within a stored conversation.

        Omit ``conversation_id`` to start a new conversation; pass it back to
        continue the same conversation with previous turns as context.
        """
        try:
            result, conversation_id = agent.chat(
                request.message,
                store=conversation_store,
                conversation_id=request.conversation_id,
                top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - convert to API-safe response.
            log_event("chat_failed", {"error": str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatResponse(conversation_id=conversation_id, result=result)

    @app.get("/conversations/{conversation_id}", response_model=ConversationMemory)
    def get_conversation(conversation_id: str) -> ConversationMemory:
        """Return the stored turns of one conversation."""
        try:
            memory = conversation_store.get(conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if memory is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return memory

    @app.post("/index")
    def index(request: IndexRequest) -> dict[str, str | int]:
        """Rebuild the local vector index from a CSV path."""
        index_path = request.index_path or str(settings.index_path)
        try:
            vector_store = build_index(
                request.input_path,
                index_path,
                embedding_dim=settings.embedding_dim,
            )
        except Exception as exc:  # noqa: BLE001 - convert to API-safe response.
            log_event("index_failed", {"error": str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "indexed", "chunks": vector_store.size, "index_path": index_path}

    return app
