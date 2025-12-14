"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger

from src.config import get_settings


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
    """
    settings = get_settings()

    log_level = level or settings.log_level
    file_path = log_file or settings.log_file

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler
    if file_path:
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    logger.info(f"Logging configured - Level: {log_level}")
