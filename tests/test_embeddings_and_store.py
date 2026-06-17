from __future__ import annotations

from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.schemas import DocumentChunk
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


def test_hashing_embeddings_are_deterministic() -> None:
    model = HashingEmbeddingModel(dim=128)
    first = model.embed(["onboarding checklist"])
    second = model.embed(["onboarding checklist"])
    assert first.shape == (1, 128)
    assert (first == second).all()


def test_vector_store_returns_most_similar_chunk() -> None:
    model = HashingEmbeddingModel(dim=128)
    chunks = [
        DocumentChunk(
            chunk_id="1", source_id="fb-1", text="onboarding checklist setup", metadata={}
        ),
        DocumentChunk(chunk_id="2", source_id="fb-2", text="pricing renewal finance", metadata={}),
    ]
    vectors = model.embed([chunk.text for chunk in chunks])
    store = InMemoryVectorStore(dim=128)
    store.add(chunks, vectors)

    query_vector = model.embed(["setup checklist"])[0]
    results = store.search(query_vector, top_k=1)

    assert results[0].chunk.source_id == "fb-1"
