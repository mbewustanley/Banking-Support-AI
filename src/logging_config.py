
import logging
import os
import sys


def configure_logging() -> None:
    """Configure consistent application-wide logging."""

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, None)

    if not isinstance(log_level, int):
        raise ValueError(
            f"Invalid LOG_LEVEL: {log_level_name}. "
            "Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
