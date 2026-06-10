"""Small structured logging helpers.

The goal is not to replace full observability tooling. The module keeps runtime
logs consistent and easy to redirect to systems such as CloudWatch, Datadog, or
OpenTelemetry collectors.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

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
