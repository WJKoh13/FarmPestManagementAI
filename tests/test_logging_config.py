"""Tests for structured logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from farm_pest_ai.logging_config import (
    JsonLinesFormatter,
    configure_logging,
    get_logger,
    log_event,
)


def test_get_logger_namespaces_under_package() -> None:
    assert get_logger("data.audit").name == "farm_pest_ai.data.audit"
    assert get_logger("farm_pest_ai.x").name == "farm_pest_ai.x"
    assert get_logger().name == "farm_pest_ai"


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.LogRecord(
        "farm_pest_ai.test", logging.INFO, __file__, 1, "epoch done", None, None
    )
    record.epoch = 3
    record.macro_f1 = 0.41
    payload = json.loads(JsonLinesFormatter().format(record))
    assert payload["message"] == "epoch done"
    assert payload["level"] == "INFO"
    assert payload["epoch"] == 3
    assert payload["macro_f1"] == 0.41


def test_jsonl_file_is_written(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "run.jsonl"
    logger = configure_logging("INFO", json_file=log_file, console=False,
                               logger_name="farm_pest_ai.test_jsonl")
    log_event(logger, "epoch_end", epoch=1, macro_f1=0.25, scope="rice10")
    for handler in logger.handlers:
        handler.flush()

    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "epoch_end"
    assert payload["epoch"] == 1
    assert payload["scope"] == "rice10"


def test_reconfigure_does_not_duplicate_handlers(tmp_path: Path) -> None:
    name = "farm_pest_ai.test_dup"
    for _ in range(3):
        logger = configure_logging(
            "INFO", json_file=tmp_path / "a.jsonl", console=False, logger_name=name
        )
    assert len(logger.handlers) == 1


def test_level_is_respected(tmp_path: Path) -> None:
    log_file = tmp_path / "run.jsonl"
    logger = configure_logging("WARNING", json_file=log_file, console=False,
                               logger_name="farm_pest_ai.test_level")
    logger.info("suppressed")
    logger.warning("kept")
    for handler in logger.handlers:
        handler.flush()

    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "kept"


def test_env_var_overrides_level(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FPA_LOG_LEVEL", "ERROR")
    logger = configure_logging("DEBUG", console=False, logger_name="farm_pest_ai.test_env")
    assert logger.level == logging.ERROR
