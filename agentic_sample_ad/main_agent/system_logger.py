from __future__ import annotations

from typing import Any, Mapping

from agentic_sample_ad.system_logger import (
    enable_a2a_package_logging,
    initialize_process_logging,
    log_event,
    log_exception,
)


def initialize_main_logging() -> None:
    initialize_process_logging()


def log_main_event(
    action: str,
    details: Mapping[str, Any] | None = None,
    *,
    direction: str = "internal",
    level: str = "INFO",
) -> None:
    log_event("ad.main_agent", action, details or {}, direction=direction, level=level)


def log_main_exception(
    action: str,
    error: Exception,
    details: Mapping[str, Any] | None = None,
) -> None:
    log_exception("ad.main_agent", action, error, details or {})


__all__ = [
    "enable_a2a_package_logging",
    "initialize_main_logging",
    "log_main_event",
    "log_main_exception",
]


