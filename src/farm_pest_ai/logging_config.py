"""Structured logging setup.

Provides a console handler for humans and an optional JSON Lines file handler
for machine-readable run logs. Training and evaluation scripts write their JSONL
log beside the other artifacts so that metrics can be reconstructed after the
fact without parsing prose.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "JsonLinesFormatter",
    "configure_logging",
    "get_logger",
    "log_event",
]

#: Attributes present on every ``LogRecord``; anything else is treated as extra
#: context and included in the JSON payload.
_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JsonLinesFormatter(logging.Formatter):
    """Format records as one JSON object per line.

    Any keyword passed through ``extra=`` is merged into the object, so
    ``logger.info("epoch done", extra={"epoch": 3, "macro_f1": 0.41})`` yields a
    directly parseable metrics record.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102 - see class
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def _coerce_level(level: int | str) -> int:
    """Translate a level name or number into a logging level integer."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(
    level: int | str = "INFO",
    *,
    json_file: Path | str | None = None,
    console: bool = True,
    force: bool = True,
    logger_name: str = "farm_pest_ai",
) -> logging.Logger:
    """Configure the project logger.

    Args:
        level: Console/file threshold; the ``FPA_LOG_LEVEL`` environment
            variable takes precedence when set.
        json_file: Optional path for a JSON Lines log. Parent directories are
            created as needed.
        console: Whether to emit human-readable output on stderr.
        force: Whether to drop handlers already attached to the logger, which
            keeps repeated CLI invocations from duplicating output.
        logger_name: Logger to configure; defaults to the package root.

    Returns:
        The configured logger.
    """
    resolved = _coerce_level(os.environ.get("FPA_LOG_LEVEL") or level)
    logger = logging.getLogger(logger_name)
    logger.setLevel(resolved)
    # The package logger owns its output; do not also bubble to the root.
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if console:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setLevel(resolved)
        stream.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
        logger.addHandler(stream)

    if json_file is not None:
        path = Path(json_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(resolved)
        file_handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger.

    Args:
        name: Dotted suffix, typically ``__name__``. The ``farm_pest_ai``
            prefix is added when absent so all project logs share a root.
    """
    if not name or name == "farm_pest_ai":
        return logging.getLogger("farm_pest_ai")
    if name.startswith("farm_pest_ai."):
        return logging.getLogger(name)
    return logging.getLogger(f"farm_pest_ai.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    /,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured event.

    Args:
        logger: Target logger.
        event: Machine-readable event name, e.g. ``"epoch_end"``.
        level: Logging level.
        **fields: Extra key/value pairs recorded in the JSON payload.
    """
    logger.log(level, event, extra={"event": event, **fields})


def summarise(mapping: Mapping[str, Any], limit: int = 8) -> str:
    """Render a short ``key=value`` summary for console messages."""
    items = list(mapping.items())[:limit]
    rendered = " ".join(f"{k}={v}" for k, v in items)
    if len(mapping) > limit:
        rendered += f" (+{len(mapping) - limit} more)"
    return rendered
