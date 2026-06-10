"""FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.factory import build_agent, build_index
from ai_engineering_showcase.schemas import IndexRequest, QueryRequest, QueryResponse
from ai_engineering_showcase.telemetry import configure_logging, log_event


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    settings = Settings()
    agent = build_agent(settings)

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
