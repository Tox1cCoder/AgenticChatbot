"""
Logging configuration for the client backend.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from client_backend.core.config import client_settings


def setup_logging() -> logging.Logger:
    """
    Configure logging for the client backend.

    Returns:
        The root logger configured for the client backend.
    """
    log_level = getattr(logging, client_settings.log_level, logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Root logger for client_backend
    root_logger = logging.getLogger("client_backend")
    root_logger.setLevel(log_level)

    # Close replaced file streams as well as removing handlers. Repeated app
    # lifespans (including tests) otherwise leak descriptors.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if enabled)
    if client_settings.log_to_file:
        log_dir = Path(client_settings.profile_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Rotating file name with date
        log_file = log_dir / f"client_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_file}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: The module name (typically __name__).

    Returns:
        A logger instance.
    """
    if not name.startswith("client_backend"):
        name = f"client_backend.{name}"
    return logging.getLogger(name)
