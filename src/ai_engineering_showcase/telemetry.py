"""Structured logging and OpenTelemetry-style telemetry.

The module provides two layers:

1. Lightweight JSON logging helpers (:func:`configure_logging`, :func:`log_event`)
   that keep runtime logs consistent and easy to redirect to systems such as
   CloudWatch, Datadog, or OpenTelemetry collectors.
2. A structured telemetry pipeline (:class:`Telemetry` plus pluggable sinks)
   that records spans and events with timestamps, durations, and correlation
   IDs. Telemetry is a no-op unless a sink is attached, so the default code
   path stays free of side effects.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

LOGGER_NAME = "ai_engineering_showcase"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for scripts and local API usage."""
    logging.basicConfig(level=level, format="%(message)s")


def get_logger() -> logging.Logger:
    """Return the package logger."""
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, payload: Mapping[str, Any] | None = None) -> None:
    """Emit a JSON log event.

    Args:
        event: Stable event name.
        payload: Optional structured attributes.
    """
    logger = get_logger()
    body = {"event": event, "payload": dict(payload or {})}
    logger.info(json.dumps(body, sort_keys=True, default=str))


@dataclass(frozen=True)
class TelemetryEvent:
    """A single structured telemetry event.

    Attributes:
        name: Stable event name, e.g. ``retrieval_finished``.
        timestamp: ISO-8601 UTC timestamp of when the event was emitted.
        correlation_id: Identifier shared by all events of one logical operation.
        duration_ms: Elapsed time in milliseconds for ``*_finished`` events.
        metadata: Structured attributes describing the event.
    """

    name: str
    timestamp: str
    correlation_id: str
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the event."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


class TelemetrySink(Protocol):
    """Protocol implemented by telemetry sinks."""

    def emit(self, event: TelemetryEvent) -> None:
        """Persist or forward one telemetry event."""
        ...


class InMemoryTelemetrySink:
    """Sink that keeps events in a list. Intended for tests and inspection."""

    def __init__(self) -> None:
        """Initialise the empty event buffer."""
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        """Append the event to the in-memory buffer."""
        self.events.append(event)

    def event_names(self) -> list[str]:
        """Return the names of captured events in emission order."""
        return [event.name for event in self.events]


class JsonlTelemetrySink:
    """Sink that appends one JSON object per line to a local JSONL file."""

    def __init__(self, path: str | Path) -> None:
        """Create the sink and ensure the parent directory exists."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TelemetryEvent) -> None:
        """Append the event as a single JSON line."""
        line = json.dumps(event.to_dict(), sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class Telemetry:
    """Telemetry emitter that writes structured events to an optional sink.

    Without a sink the emitter is a cheap no-op, so instrumented code never
    needs to guard telemetry calls. Inject a sink (in-memory for tests, JSONL
    for local traces) to capture events.
    """

    def __init__(self, sink: TelemetrySink | None = None) -> None:
        """Attach an optional sink; ``None`` disables telemetry."""
        self.sink = sink

    @property
    def enabled(self) -> bool:
        """True when a sink is attached and events are recorded."""
        return self.sink is not None

    def new_correlation_id(self) -> str:
        """Return a fresh identifier used to correlate events of one operation."""
        return uuid.uuid4().hex

    def emit(
        self,
        name: str,
        *,
        correlation_id: str,
        duration_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one telemetry event to the configured sink, if any."""
        if self.sink is None:
            return
        event = TelemetryEvent(
            name=name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        self.sink.emit(event)

    @contextmanager
    def span(
        self,
        started_name: str,
        finished_name: str,
        *,
        correlation_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Emit a started/finished event pair around a block of work.

        Yields a mutable metadata dictionary so callers can attach result
        attributes (e.g. counts, scores) that should appear on the finished
        event. On error the finished event carries ``status: error`` and the
        exception message, and the exception is re-raised.
        """
        span_metadata: dict[str, Any] = dict(metadata or {})
        self.emit(started_name, correlation_id=correlation_id, metadata=span_metadata)
        start = time.perf_counter()
        try:
            yield span_metadata
        except Exception as exc:
            span_metadata["status"] = "error"
            span_metadata["error"] = str(exc)
            self.emit(
                finished_name,
                correlation_id=correlation_id,
                duration_ms=_elapsed_ms(start),
                metadata=span_metadata,
            )
            raise
        span_metadata.setdefault("status", "ok")
        self.emit(
            finished_name,
            correlation_id=correlation_id,
            duration_ms=_elapsed_ms(start),
            metadata=span_metadata,
        )


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``time.perf_counter`` start value."""
    return round((time.perf_counter() - start) * 1000.0, 3)
