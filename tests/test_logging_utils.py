"""Unit tests for shared logging setup."""

from __future__ import annotations

from pathlib import Path

from src.utils import setup_logger


def test_setup_logger_creates_log_file(tmp_path: Path) -> None:
    logger, log_path = setup_logger("unit_test", tmp_path, "INFO")
    logger.info("hello from logger test")

    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()
    assert Path(log_path).parent == tmp_path / "logs"
    assert "hello from logger test" in log_path.read_text(encoding="utf-8")
