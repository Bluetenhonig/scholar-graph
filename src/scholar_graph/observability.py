"""Structured logging with a run-scoped correlation id.

Every log line emitted during a research run carries ``run_id``, so a single
grep reconstructs one run out of interleaved concurrent traffic.
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)

NOISY_LOGGERS = (
    "httpx",  # logs every request at INFO
    "httpcore",
    "autogen_core",  # publishes full message envelopes at DEBUG
    "autogen_core.events",
    "autogen_agentchat",
)


def _inject_run_id(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    run_id = _run_id.get()
    if run_id is not None:
        event_dict.setdefault("run_id", run_id)
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Apply logging configuration. Safe to call repeatedly.

    Deliberately re-appliable: entry points import modules (which build
    loggers) before they have read settings, so the first configuration is
    always a default one and the settings-driven call has to be able to
    override it.
    """
    renderer: Any = (
        structlog.dev.ConsoleRenderer() if fmt == "console" else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _inject_run_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        # Not cached: a cached bound logger keeps the level it was created
        # with, which would make a later reconfiguration silently ineffective.
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=level, format="%(message)s", force=True)

    # Third-party chatter, suppressed unless we are actually debugging.
    floor = max(logging.WARNING, logging.getLevelNamesMapping()[level])
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(floor)


def get_logger(name: str) -> Any:
    """Return a lazily-bound logger.

    No configuration happens here on purpose. Modules build loggers at import
    time, long before settings are loaded; configuring from here would lock in
    a default level that :func:`configure_logging` could never override.
    """
    return structlog.get_logger(name)


@contextmanager
def run_context(run_id: str) -> Iterator[None]:
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


@contextmanager
def timed(logger: Any, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log ``event`` on exit with a ``duration_ms`` field, success or failure."""
    start = time.perf_counter()
    extra: dict[str, Any] = {}
    try:
        yield extra
    except Exception as exc:
        logger.error(
            event,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
            outcome="error",
            error=str(exc),
            **fields,
            **extra,
        )
        raise
    else:
        logger.info(
            event,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
            outcome="ok",
            **fields,
            **extra,
        )
