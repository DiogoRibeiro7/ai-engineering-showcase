"""Factories for constructing the application components."""

from __future__ import annotations

from pathlib import Path

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.chunking import feedback_to_chunks
from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.ingestion import load_feedback_csv
from ai_engineering_showcase.lexical_search import BM25Retriever
from ai_engineering_showcase.llm import DeterministicLLM, LLMProvider, OpenAIChatLLM
from ai_engineering_showcase.retrieval import HybridRetriever, QueryEngine, Retriever
from ai_engineering_showcase.schemas import DocumentChunk
from ai_engineering_showcase.vector_store import InMemoryVectorStore


def chunk_to_embedding_text(chunk: DocumentChunk) -> str:
    """Create the text representation used for embedding and retrieval.

    The visible citation keeps the original feedback text, but retrieval benefits
    from structured metadata such as segment, channel, rating, and date.
    """
    metadata = chunk.metadata
    metadata_text = " ".join(
        [
            f"segment {metadata.get('customer_segment', '')}",
            f"channel {metadata.get('channel', '')}",
            f"rating {metadata.get('rating', '')}",
            f"created {metadata.get('created_at', '')}",
        ]
    )
    return f"{metadata_text} {chunk.text}"


def build_index(
    input_path: str | Path, index_path: str | Path, *, embedding_dim: int
) -> InMemoryVectorStore:
    """Build and persist a vector index from feedback CSV data."""
    records = load_feedback_csv(input_path)
    chunks = feedback_to_chunks(records)
    embedding_model = HashingEmbeddingModel(dim=embedding_dim)
    vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
    vector_store = InMemoryVectorStore(dim=embedding_dim)
    vector_store.add(chunks, vectors)
    vector_store.save(index_path)
    return vector_store


def load_or_build_index(settings: Settings) -> InMemoryVectorStore:
    """Load an index from disk or build it from configured data."""
    settings.ensure_artifact_dir()
    if settings.index_path.exists():
        return InMemoryVectorStore.load(settings.index_path)
    return build_index(
        settings.data_path,
        settings.index_path,
        embedding_dim=settings.embedding_dim,
    )


def build_retriever(settings: Settings, vector_store: InMemoryVectorStore) -> Retriever:
    """Construct the configured retriever over an existing vector store.

    ``dense`` keeps the original vector-similarity behaviour, ``lexical`` uses
    the local BM25 index, and ``hybrid`` combines both with the configured
    ``dense_weight`` and ``lexical_weight``.
    """
    embedding_model = HashingEmbeddingModel(dim=vector_store.dim)
    dense = QueryEngine(embedding_model=embedding_model, vector_store=vector_store)
    if settings.retriever_type == "dense":
        return dense
    chunks = vector_store.chunks
    lexical = BM25Retriever(chunks, texts=[chunk_to_embedding_text(chunk) for chunk in chunks])
    if settings.retriever_type == "lexical":
        return lexical
    return HybridRetriever(
        dense=dense,
        lexical=lexical,
        dense_weight=settings.dense_weight,
        lexical_weight=settings.lexical_weight,
    )


def build_llm(settings: Settings) -> LLMProvider:
    """Construct the configured LLM provider."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_SHOWCASE_LLM_PROVIDER=openai")
        return OpenAIChatLLM(api_key=settings.openai_api_key, model=settings.openai_model)
    return DeterministicLLM()


def build_agent(settings: Settings) -> FeedbackInsightAgent:
    """Construct a fully wired feedback insight agent."""
    vector_store = load_or_build_index(settings)
    retriever = build_retriever(settings, vector_store)
    return FeedbackInsightAgent(query_engine=retriever, llm=build_llm(settings))
