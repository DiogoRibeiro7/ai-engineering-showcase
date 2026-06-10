"""Retrieval helpers."""

from __future__ import annotations

from ai_engineering_showcase.embeddings import EmbeddingModel
from ai_engineering_showcase.schemas import SearchResult
from ai_engineering_showcase.vector_store import InMemoryVectorStore


class QueryEngine:
    """Embed user questions and retrieve relevant document chunks."""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: InMemoryVectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def search(self, question: str, *, top_k: int = 4) -> list[SearchResult]:
        """Search for chunks relevant to a natural-language question."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        query_vector = self.embedding_model.embed([question])[0]
        return self.vector_store.search(query_vector, top_k=top_k)
