"""Typed domain schemas used across the project."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeedbackChannel(str, Enum):
    """Supported feedback channels."""

    support_ticket = "support_ticket"
    app_review = "app_review"
    sales_call = "sales_call"
    nps_survey = "nps_survey"
    community = "community"


class FeedbackRecord(BaseModel):
    """Single raw customer feedback record."""

    model_config = ConfigDict(str_strip_whitespace=True)

    feedback_id: str = Field(min_length=1)
    customer_segment: str = Field(min_length=1)
    channel: FeedbackChannel
    rating: int = Field(ge=1, le=5)
    text: str = Field(min_length=3)
    created_at: datetime

    @field_validator("text")
    @classmethod
    def normalise_text(cls, value: str) -> str:
        """Collapse repeated whitespace in feedback text."""
        return " ".join(value.split())


class DocumentChunk(BaseModel):
    """Searchable chunk derived from feedback."""

    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any]


class SearchResult(BaseModel):
    """A retrieved chunk with similarity score."""

    chunk: DocumentChunk
    score: float


class Citation(BaseModel):
    """Evidence citation exposed to API consumers."""

    source_id: str
    quote: str
    score: float


class AgentAnswer(BaseModel):
    """Final answer returned by the insight agent."""

    question: str
    answer: str
    recommended_actions: list[str]
    citations: list[Citation]
    route: str
    confidence: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """API request for the `/query` endpoint."""

    question: str = Field(min_length=3)
    top_k: int = Field(default=4, ge=1, le=12)


class QueryResponse(BaseModel):
    """API response for the `/query` endpoint."""

    result: AgentAnswer


class IndexRequest(BaseModel):
    """API request for rebuilding the index from a CSV path."""

    input_path: str = Field(min_length=1)
    index_path: str | None = None


class EvaluationCase(BaseModel):
    """Single retrieval evaluation case."""

    question: str
    relevant_source_ids: list[str]
