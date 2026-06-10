from __future__ import annotations

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.llm import DeterministicLLM
from ai_engineering_showcase.retrieval import QueryEngine
from ai_engineering_showcase.schemas import DocumentChunk
from ai_engineering_showcase.vector_store import InMemoryVectorStore


def build_test_agent() -> FeedbackInsightAgent:
    model = HashingEmbeddingModel(dim=128)
    chunks = [
        DocumentChunk(
            chunk_id="1",
            source_id="fb-1",
            text="Onboarding checklist was unclear and setup took too long.",
            metadata={"rating": 2},
        ),
        DocumentChunk(
            chunk_id="2",
            source_id="fb-2",
            text="Pricing renewal was hard to explain to finance.",
            metadata={"rating": 2},
        ),
    ]
    store = InMemoryVectorStore(dim=128)
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    query_engine = QueryEngine(embedding_model=model, vector_store=store)
    return FeedbackInsightAgent(query_engine=query_engine, llm=DeterministicLLM())


def test_agent_routes_onboarding_question() -> None:
    agent = build_test_agent()
    assert agent.route("What is wrong with onboarding?") == "onboarding"


def test_agent_answer_contains_citations() -> None:
    agent = build_test_agent()
    answer = agent.answer("Why is onboarding slow?", top_k=1)
    assert answer.citations
    assert answer.confidence > 0
    assert answer.recommended_actions
