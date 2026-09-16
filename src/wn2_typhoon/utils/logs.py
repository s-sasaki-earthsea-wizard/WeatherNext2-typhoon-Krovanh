"""Logging setup for the CLI scripts.

The tracker logs every cyclogenesis candidate, every pruning decision and the
full interpolation result at each lead time. That is useful when debugging one
track and unreadable for 8 members over 40 steps, so the scripts keep their own
logger at INFO and leave everything else at WARNING.
"""

from __future__ import annotations

import logging


def configure(name: str, verbose: bool = False) -> logging.Logger:
    """Set up logging for a script and return its logger.

    Args:
        name: Logger name, usually the script's own.
        verbose: Keep third-party INFO records, including the tracker's
            per-candidate commentary.

    Returns:
        The script's logger, at INFO.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(message)s",
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
