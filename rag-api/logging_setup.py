"""Single-call logger configuration.

Idempotent. Logs ``logfmt``-ish key=value pairs to stderr — readable in
``docker logs`` and trivially grep-able.
"""
from __future__ import annotations

import logging
import sys

from ragcore import config


_CONFIGURED = False


def configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
