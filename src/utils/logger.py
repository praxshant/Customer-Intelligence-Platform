# logger.py
# Purpose: Setup and manage logging across the application

import logging
import os
import sys
from datetime import datetime
from typing import Optional

try:
    import structlog
except Exception:  # structlog is optional at runtime
    structlog = None  # type: ignore


def _ensure_log_dir(path: str = "logs") -> str:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def _configure_std_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def _configure_structlog(level: str) -> None:
    if structlog is None:
        return
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Return a configured logger. If structlog is available, return a wrapped logger.

    This remains backward compatible for modules importing a stdlib logger.
    """
    _ensure_log_dir("logs")
    _configure_std_logging(log_level)
    _configure_structlog(log_level)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Attach file handler (rotated daily by filename)
    timestamp = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join("logs", f"{name}_{timestamp}.log")
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Ensure console handler exists
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger