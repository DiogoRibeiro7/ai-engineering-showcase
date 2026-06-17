"""Deterministic safety guardrails for the feedback intelligence agent.

The module implements two transparent, regex/keyword-based gates:

1. :func:`check_input` runs **before retrieval** and blocks empty queries,
   prompt-injection attempts, requests for hidden system instructions,
   requests to ignore the retrieved context, and unsupported data access
   requests (other customers' PII, raw database access).
2. :func:`check_context` runs **before answer generation** and flags
   retrieved chunks that contain instruction-override content (indirect
   prompt injection planted in feedback text).

Every rule is a documented regular expression, so decisions are fully
deterministic, reproducible in CI, and easy to audit. No model call is
involved in any guardrail decision.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

Severity = Literal["low", "medium", "high"]
"""Severity of a guardrail finding, from informational to clear abuse."""

SAFE_REFUSAL = (
    "I can only answer questions grounded in the indexed customer feedback. "
    "Please ask a question about customer feedback themes, segments, or channels."
)


class GuardrailDecision(BaseModel):
    """Outcome of one guardrail evaluation.

    Attributes:
        allowed: Whether the request may proceed.
        reason: Human-readable explanation, including the rule that fired.
        severity: ``low`` (benign or malformed), ``medium`` (policy bypass
            attempts), or ``high`` (injection or data exfiltration attempts).
        suggested_response: Safe text the caller can return verbatim when the
            request is blocked.
    """

    allowed: bool
    reason: str
    severity: Severity = "low"
    suggested_response: str | None = None


@dataclass(frozen=True)
class GuardrailRule:
    """One deterministic guardrail rule built from compiled regex patterns."""

    name: str
    severity: Severity
    patterns: tuple[re.Pattern[str], ...]
    suggested_response: str

    def first_match(self, text: str) -> re.Pattern[str] | None:
        """Return the first pattern that matches ``text``, if any."""
        for pattern in self.patterns:
            if pattern.search(text):
                return pattern
        return None


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    """Compile case-insensitive guardrail patterns."""
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


_PROMPT_INJECTION = GuardrailRule(
    name="prompt_injection",
    severity="high",
    patterns=_compile(
        r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|earlier|above)\s+"
        r"(?:instructions?|prompts?|rules|directions)\b",
        r"\bdisregard\s+(?:all\s+|any\s+)?(?:previous|prior|earlier|above|your)\s+"
        r"(?:instructions?|prompts?|rules|guidelines)\b",
        r"\bforget\s+(?:everything|all\s+your|your)\s+(?:instructions?|rules|training)\b",
        r"\boverride\s+(?:your|the|all)\s+(?:instructions?|rules|safety|guardrails|guidelines)\b",
        r"\bnew\s+instructions?\s*:",
        r"\byou\s+are\s+now\s+(?:a|an|in)\b",
        r"\bpretend\s+(?:to\s+be|you\s+are)\b",
        r"\bact\s+as\s+(?:if\s+you\s+(?:have\s+no|are\s+not)|an?\s+unrestricted)\b",
        r"\bjailbreak\b",
        r"\b(?:dan|developer)\s+mode\b",
    ),
    suggested_response=(
        "I can't follow instructions that try to override how I operate. "
        "I can answer questions about the indexed customer feedback instead."
    ),
)

_SYSTEM_PROMPT_DISCLOSURE = GuardrailRule(
    name="system_prompt_disclosure",
    severity="high",
    patterns=_compile(
        r"\b(?:reveal|show|print|repeat|display|output|expose|leak|share)\b[^.?!]{0,80}?"
        r"\b(?:system|hidden|initial|internal|developer)\s+(?:prompt|instructions?|message)\b",
        r"\bwhat\s+(?:is|are)\s+your\s+"
        r"(?:system|hidden|initial|internal)\s+(?:prompt|instructions?)\b",
        r"\btell\s+me\s+your\s+(?:system|hidden|initial|internal)\s+(?:prompt|instructions?)\b",
    ),
    suggested_response=(
        "I can't share hidden system instructions or internal prompts. "
        "I can answer questions about the indexed customer feedback instead."
    ),
)

_CONTEXT_OVERRIDE = GuardrailRule(
    name="context_override",
    severity="medium",
    patterns=_compile(
        r"\bignore\s+(?:the\s+|all\s+)?(?:retrieved\s+)?(?:context|evidence|sources|documents)\b",
        r"\banswer\s+without\s+(?:using\s+)?(?:the\s+)?"
        r"(?:context|evidence|sources|retrieved|citations?)\b",
        r"\bdo\s*(?:n.?t|\s+not)\s+use\s+(?:the\s+)?"
        r"(?:context|evidence|sources|retrieved|citations?)\b",
        r"\bmake\s+(?:up\s+an?\s+answer|something\s+up)\b",
    ),
    suggested_response=(
        "Answers are always grounded in retrieved feedback evidence, so I can't skip "
        "or ignore the context. Please ask a question about the customer feedback."
    ),
)

_DATA_ACCESS = GuardrailRule(
    name="data_access",
    severity="high",
    patterns=_compile(
        r"\b(?:other|another|all|every)\s+customers?['’]?s?\s+"
        r"(?:pii|personal|private|contact|email|phone|payment|data|records|details)\b",
        r"\b(?:email\s+address(?:es)?|phone\s+numbers?|home\s+address(?:es)?)\s+"
        r"(?:of|for|belonging\s+to)\b",
        r"\b(?:credit\s+card|social\s+security|passwords?|api\s+keys?|access\s+tokens?)\b",
        r"\braw\s+database\b",
        r"\bdatabase\s+(?:dump|access|credentials?|password)\b",
        r"\bdump\s+(?:the\s+)?(?:database|table|user\s+data)\b",
        r"\brun\s+(?:a\s+|this\s+)?sql\b",
        r"\bselect\s+\*\s+from\b",
    ),
    suggested_response=(
        "I can't provide personal data about individual customers or raw data store "
        "access. I can summarise anonymised, aggregated feedback themes instead."
    ),
)

INPUT_RULES: tuple[GuardrailRule, ...] = (
    _PROMPT_INJECTION,
    _SYSTEM_PROMPT_DISCLOSURE,
    _CONTEXT_OVERRIDE,
    _DATA_ACCESS,
)
"""Ordered rules applied to user input before retrieval."""

_CONTEXT_RULES: tuple[GuardrailRule, ...] = (_PROMPT_INJECTION, _SYSTEM_PROMPT_DISCLOSURE)
"""Rules applied to retrieved chunk text to catch indirect prompt injection."""


def check_input(question: str) -> GuardrailDecision:
    """Run all input guardrail rules against a user question (pre-retrieval gate).

    Returns an ``allowed=False`` decision with a safe ``suggested_response``
    when the question is empty or matches a documented unsafe pattern.
    """
    if not question.strip():
        return GuardrailDecision(
            allowed=False,
            reason="empty_query: question is empty or whitespace-only",
            severity="low",
            suggested_response=("Please provide a non-empty question about the customer feedback."),
        )
    for rule in INPUT_RULES:
        matched = rule.first_match(question)
        if matched is not None:
            return GuardrailDecision(
                allowed=False,
                reason=f"{rule.name}: matched pattern {matched.pattern!r}",
                severity=rule.severity,
                suggested_response=rule.suggested_response,
            )
    return GuardrailDecision(
        allowed=True,
        reason="input passed all guardrail checks",
        severity="low",
    )


def is_suspicious_context(text: str) -> bool:
    """Return True when retrieved chunk text contains instruction-override content."""
    return any(rule.first_match(text) is not None for rule in _CONTEXT_RULES)


def check_context(texts: Sequence[str]) -> GuardrailDecision:
    """Check retrieved context for indirect prompt injection (pre-generation gate).

    The decision is ``allowed=False`` when any chunk carries injection-style
    content; callers should drop the suspicious chunks (identified per chunk
    with :func:`is_suspicious_context`) and answer from the clean remainder.
    """
    flagged = sum(1 for text in texts if is_suspicious_context(text))
    if flagged:
        return GuardrailDecision(
            allowed=False,
            reason=(
                f"context_injection: {flagged} retrieved chunk(s) contain "
                "instruction-override patterns"
            ),
            severity="medium",
        )
    return GuardrailDecision(
        allowed=True,
        reason="retrieved context passed all guardrail checks",
        severity="low",
    )
