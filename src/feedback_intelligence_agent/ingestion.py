"""Feedback ingestion utilities."""

from __future__ import annotations

from pathlib import Path

from feedback_intelligence_agent.data_contracts import (
    REQUIRED_COLUMNS,
    DataContractError,
    validate_feedback_csv,
)
from feedback_intelligence_agent.schemas import FeedbackRecord
from feedback_intelligence_agent.telemetry import Telemetry

__all__ = ["REQUIRED_COLUMNS", "FeedbackIngestionError", "load_feedback_csv"]


class FeedbackIngestionError(ValueError):
    """Raised when input feedback data cannot be loaded safely."""


def load_feedback_csv(
    path: str | Path,
    *,
    strict: bool = True,
    telemetry: Telemetry | None = None,
) -> list[FeedbackRecord]:
    """Load feedback records from a CSV file, validating the data contract first.

    Args:
        path: Path to a CSV file containing the required feedback columns.
        strict: When True (default), any contract violation aborts ingestion.
            When False, invalid rows are skipped and the valid rows are returned.
        telemetry: Optional telemetry emitter; emits ``ingestion_started`` and
            ``ingestion_finished`` events around the load.

    Returns:
        Validated feedback records.

    Raises:
        FileNotFoundError: If the file does not exist.
        FeedbackIngestionError: In strict mode, if required columns are missing
            or any row violates the data contract.
    """
    telemetry = telemetry or Telemetry()
    correlation_id = telemetry.new_correlation_id()
    with telemetry.span(
        "ingestion_started",
        "ingestion_finished",
        correlation_id=correlation_id,
        metadata={"path": str(path), "strict": strict},
    ) as span:
        try:
            _, records = validate_feedback_csv(path, strict=strict)
        except DataContractError as exc:
            raise FeedbackIngestionError(str(exc)) from exc
        span["records"] = len(records)
    return records
