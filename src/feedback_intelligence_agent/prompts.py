"""Prompt construction utilities and the default prompt registry.

All production prompts are registered in :data:`PROMPT_REGISTRY` with a name,
version, declared variables, and a changelog note. The agent always renders
through the registry, so prompt changes are explicit, versioned, and covered
by snapshot tests (see ``tests/test_prompt_snapshot.py`` and ``docs/prompts.md``).
"""

from __future__ import annotations

from feedback_intelligence_agent.citations import build_citations, citation_marker
from feedback_intelligence_agent.prompt_registry import PromptRegistry, PromptTemplate
from feedback_intelligence_agent.schemas import SearchResult

SYSTEM_PROMPT = """You are a careful AI product analyst.
Use only the evidence provided in the context.
Return a concise answer, recommended actions, and cite the source IDs.
Cite evidence inline with bracketed markers such as [1] or [2] that refer to
the citation numbers of the context blocks below.
Do not cite any source that is not present in the context.
Do not invent customer facts that are not present in the context.
"""

_RAG_ANSWER_BODY = """
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

PROMPT_REGISTRY = PromptRegistry()
"""Default registry holding every production prompt used by the system."""

RAG_SYSTEM_V1 = PROMPT_REGISTRY.register(
    PromptTemplate(
        name="rag_system",
        version="v1",
        template=SYSTEM_PROMPT,
        required_variables=(),
        changelog="Initial citation-aware system prompt: grounded, cites [n] markers only.",
    )
)

RAG_ANSWER_V1 = PROMPT_REGISTRY.register(
    PromptTemplate(
        name="rag_answer",
        version="v1",
        template=SYSTEM_PROMPT + "\n" + _RAG_ANSWER_BODY,
        required_variables=("question",),
        optional_variables={"route": "general_insight", "context": ""},
        changelog=(
            "Initial grounded RAG answer prompt with citation-aware context blocks "
            "(citation: [n] labels) and sectioned response format."
        ),
    )
)


def build_grounded_prompt(question: str, results: list[SearchResult], *, route: str) -> str:
    """Build a grounded prompt for an LLM provider."""
    marker_by_document = {
        citation.document_id: citation_marker(citation.citation_id)
        for citation in build_citations(results)
    }
    context_blocks = []
    for result in results:
        metadata = result.chunk.metadata
        context_blocks.append(
            "\n".join(
                [
                    f"citation: {marker_by_document[result.chunk.source_id]}",
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
    return PROMPT_REGISTRY.render("rag_answer", question=question, route=route, context=context)
