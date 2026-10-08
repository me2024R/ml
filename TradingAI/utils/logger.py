from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create and configure a logger for the trading pipeline."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
