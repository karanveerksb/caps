"""Logging setup shared across scripts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

from src.config import DEFAULT_LOG_LEVEL


def _resolve_level(log_level: str) -> int:
    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {log_level}")
    return level


def setup_logger(
    script_name: str,
    artifacts_dir: str | Path,
    log_level: str = DEFAULT_LOG_LEVEL,
) -> Tuple[logging.Logger, Path]:
    """Create a console + file logger for one script run."""

    artifacts_path = Path(artifacts_dir).expanduser().resolve()
    logs_dir = artifacts_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{script_name}_{timestamp}.log"

    logger_name = f"multimodal_deepfake.{script_name}.{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(_resolve_level(log_level))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_resolve_level(log_level))
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(_resolve_level(log_level))
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_path
