"""Prompt regression tests.

These tests pin the exact bytes of the main RAG answer prompt. If a prompt
template changes — intentionally or by accident — the snapshots below fail.
Intentional changes must register a new prompt version and refresh the
snapshots deliberately (see docs/prompts.md).
"""

from __future__ import annotations

from pathlib import Path

from ai_engineering_showcase.prompts import PROMPT_REGISTRY, build_grounded_prompt
from ai_engineering_showcase.schemas import DocumentChunk, SearchResult

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def read_snapshot(filename: str) -> str:
    """Read a snapshot with newline normalisation so git EOL settings cannot break it."""
    raw = (SNAPSHOT_DIR / filename).read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n")


def sample_results() -> list[SearchResult]:
    return [
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="fb-001::chunk-0",
                source_id="fb-001",
                text=(
                    "Implementation took three weeks longer than expected. "
                    "We also had no clear onboarding checklist and did not know "
                    "who owned each setup step."
                ),
                metadata={
                    "customer_segment": "enterprise",
                    "channel": "support_ticket",
                    "rating": 2,
                },
            ),
            score=0.532497,
        ),
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="fb-007::chunk-0",
                source_id="fb-007",
                text="Onboarding felt fragmented and ownership of setup steps was unclear.",
                metadata={
                    "customer_segment": "mid_market",
                    "channel": "nps_survey",
                    "rating": 3,
                },
            ),
            score=0.501882,
        ),
    ]


def test_rag_answer_template_matches_golden_snapshot() -> None:
    template = PROMPT_REGISTRY.get("rag_answer", "v1").template
    assert template == read_snapshot("rag_answer_v1_template.txt")


def test_grounded_prompt_matches_golden_snapshot() -> None:
    prompt = build_grounded_prompt(
        "Why are enterprise customers unhappy with onboarding?",
        sample_results(),
        route="onboarding",
    )
    assert prompt == read_snapshot("rag_answer_v1_rendered.txt")


def test_grounded_prompt_preserves_citation_labels_and_sections() -> None:
    prompt = build_grounded_prompt(
        "Why are enterprise customers unhappy with onboarding?",
        sample_results(),
        route="onboarding",
    )
    assert "citation: [1]" in prompt
    assert "citation: [2]" in prompt
    assert prompt.endswith("Answer:\nRecommended actions:\nCitations:\n")
