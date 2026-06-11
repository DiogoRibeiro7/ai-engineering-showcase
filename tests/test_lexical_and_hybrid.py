from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_engineering_showcase.cli import app
from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.factory import build_retriever, load_or_build_index
from ai_engineering_showcase.lexical_search import BM25Retriever, tokenize
from ai_engineering_showcase.retrieval import HybridRetriever, QueryEngine, min_max_normalize
from ai_engineering_showcase.schemas import DocumentChunk
from ai_engineering_showcase.vector_store import InMemoryVectorStore

runner = CliRunner()


def make_chunk(suffix: str, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=f"c-{suffix}", source_id=f"fb-{suffix}", text=text, metadata={})


def domain_corpus() -> list[DocumentChunk]:
    """Corpus where one chunk carries a rare exact domain term (Salesforce)."""
    return [
        make_chunk("a", "Salesforce keeps disconnecting."),
        make_chunk("b", "The integration sync failure happens when the export integration runs."),
        make_chunk("c", "Integration sync failure reports arrive daily."),
        make_chunk("d", "Another integration sync failure was logged."),
        make_chunk("e", "Our integration sync failure dashboard is busy."),
        make_chunk("f", "Weekly integration sync failure summary email."),
    ]


def build_dense(chunks: list[DocumentChunk], dim: int = 256) -> QueryEngine:
    model = HashingEmbeddingModel(dim=dim)
    store = InMemoryVectorStore(dim=dim)
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    return QueryEngine(embedding_model=model, vector_store=store)


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Salesforce ERR_5021 broke!") == ["salesforce", "err_5021", "broke"]


def test_bm25_ranks_exact_domain_term_first() -> None:
    lexical = BM25Retriever(domain_corpus())
    results = lexical.search("Salesforce integration sync failure", top_k=3)
    assert results
    assert results[0].chunk.source_id == "fb-a"
    assert results[0].score > results[-1].score


def test_lexical_beats_dense_on_exact_domain_terms() -> None:
    """The rare exact term dominates BM25 but is diluted in dense similarity."""
    chunks = domain_corpus()
    lexical = BM25Retriever(chunks)
    dense = build_dense(chunks)
    question = "Salesforce integration sync failure"
    assert lexical.search(question, top_k=1)[0].chunk.source_id == "fb-a"
    assert dense.search(question, top_k=1)[0].chunk.source_id != "fb-a"


def test_bm25_returns_empty_for_no_overlap() -> None:
    lexical = BM25Retriever(domain_corpus())
    assert lexical.search("pricing renewal finance", top_k=4) == []


def test_bm25_rejects_invalid_input() -> None:
    lexical = BM25Retriever(domain_corpus())
    with pytest.raises(ValueError):
        lexical.search("   ", top_k=4)
    with pytest.raises(ValueError):
        lexical.search("salesforce", top_k=0)
    with pytest.raises(ValueError):
        BM25Retriever(domain_corpus(), texts=["only one text"])


def test_dense_search_still_handles_semantic_queries() -> None:
    """Dense retrieval keeps working for paraphrased questions without rare terms."""
    chunks = [
        make_chunk("on", "Onboarding checklist was unclear and setup took too long."),
        make_chunk("pr", "Pricing renewal was hard to explain to finance."),
    ]
    dense = build_dense(chunks)
    results = dense.search("slow setup during onboarding", top_k=1)
    assert results[0].chunk.source_id == "fb-on"


def test_hybrid_combines_both_retrievers_without_duplicates() -> None:
    chunks = domain_corpus()
    hybrid = HybridRetriever(
        dense=build_dense(chunks),
        lexical=BM25Retriever(chunks),
        dense_weight=0.5,
        lexical_weight=0.5,
    )
    results = hybrid.search("Salesforce integration sync failure", top_k=4)
    chunk_ids = [result.chunk.chunk_id for result in results]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert len(results) <= 4
    assert all(0.0 <= result.score <= 1.0 for result in results)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    # The exact-term document surfaces even though dense search alone misses it.
    assert "c-a" in chunk_ids


def test_hybrid_weighting_controls_ranking() -> None:
    chunks = domain_corpus()
    dense = build_dense(chunks)
    lexical = BM25Retriever(chunks)
    question = "Salesforce integration sync failure"

    lexical_only = HybridRetriever(dense=dense, lexical=lexical, dense_weight=0, lexical_weight=1)
    assert (
        lexical_only.search(question, top_k=1)[0].chunk.chunk_id
        == lexical.search(question, top_k=1)[0].chunk.chunk_id
    )

    dense_only = HybridRetriever(dense=dense, lexical=lexical, dense_weight=1, lexical_weight=0)
    assert (
        dense_only.search(question, top_k=1)[0].chunk.chunk_id
        == dense.search(question, top_k=1)[0].chunk.chunk_id
    )


def test_hybrid_rejects_invalid_weights() -> None:
    chunks = domain_corpus()
    dense = build_dense(chunks)
    lexical = BM25Retriever(chunks)
    with pytest.raises(ValueError):
        HybridRetriever(dense=dense, lexical=lexical, dense_weight=-0.1, lexical_weight=0.5)
    with pytest.raises(ValueError):
        HybridRetriever(dense=dense, lexical=lexical, dense_weight=0.0, lexical_weight=0.0)


def test_min_max_normalize_handles_edge_cases() -> None:
    assert min_max_normalize([]) == []
    assert min_max_normalize([0.7, 0.7]) == [1.0, 1.0]
    assert min_max_normalize([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]


def test_factory_builds_each_retriever_type(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"
    expected = {"dense": QueryEngine, "lexical": BM25Retriever, "hybrid": HybridRetriever}
    for retriever_type, expected_class in expected.items():
        settings = Settings.model_validate(
            {"index_path": index_path, "retriever_type": retriever_type}
        )
        vector_store = load_or_build_index(settings)
        retriever = build_retriever(settings, vector_store)
        assert isinstance(retriever, expected_class)
        results = retriever.search("Why are enterprise customers unhappy with onboarding?")
        assert results
        assert len({result.chunk.chunk_id for result in results}) == len(results)


def test_query_command_supports_hybrid_retriever(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"
    result = runner.invoke(
        app,
        [
            "query",
            "Which Salesforce integration problems were reported?",
            "--index-path",
            str(index_path),
            "--retriever",
            "hybrid",
            "--dense-weight",
            "0.5",
            "--lexical-weight",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"answer"' in result.output
    assert '"citations"' in result.output


def test_query_command_rejects_unknown_retriever(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "query",
            "Anything",
            "--index-path",
            str(tmp_path / "vector_store.json"),
            "--retriever",
            "sparse",
        ],
    )
    assert result.exit_code != 0
