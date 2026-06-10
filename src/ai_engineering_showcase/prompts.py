"""Prompt construction utilities."""

from __future__ import annotations

from ai_engineering_showcase.schemas import SearchResult

SYSTEM_PROMPT = """You are a careful AI product analyst.
Use only the evidence provided in the context.
Return a concise answer, recommended actions, and cite the source IDs.
Do not invent customer facts that are not present in the context.
"""


def build_grounded_prompt(question: str, results: list[SearchResult], *, route: str) -> str:
    """Build a grounded prompt for an LLM provider."""
    context_blocks = []
    for result in results:
        metadata = result.chunk.metadata
        context_blocks.append(
            "\n".join(
                [
                    f"source_id: {result.chunk.source_id}",
                    f"score: {result.score:.3f}",
                    f"segment: {metadata.get('customer_segment', 'unknown')}",
                    f"channel: {metadata.get('channel', 'unknown')}",
                    f"rating: {metadata.get('rating', 'unknown')}",
                    f"text: {result.chunk.text}",
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"""{SYSTEM_PROMPT}

Route: {route}

Question:
{question}

Context:
{context}

Return the response with these sections:
Answer:
Recommended actions:
Citations:
"""
