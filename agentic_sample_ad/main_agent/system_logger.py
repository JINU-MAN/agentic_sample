from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agentic_sample_ad.system_logger import (
    ScopedSessionLogger,
    enable_a2a_package_logging,
    finalize_process_logging as _global_finalize_process_logging,
    initialize_process_logging as _global_initialize_process_logging,
    log_event as _global_log_event,
    log_exception as _global_log_exception,
    start_new_logging_session as _global_start_new_logging_session,
)

_MAIN_SESSION_LOGGER = ScopedSessionLogger(Path(__file__).resolve().parent / "log")


def _should_record_main_session_event(component: str, action: str) -> bool:
    component_key = str(component or "").strip()
    action_key = str(action or "").strip().lower()
    if component_key == "ad.main_agent" and action_key in {"run_started", "run_completed", "run_failed"}:
        return True
    if component_key == "event_manager.agent_message":
        return True
    if component_key.startswith("tool."):
        return True
    return False


def initialize_main_logging() -> None:
    _MAIN_SESSION_LOGGER.initialize()
    _global_initialize_process_logging()


def start_main_logging_session(*, reset_files: bool = True) -> str:
    _MAIN_SESSION_LOGGER.start_new_session(reset_files=reset_files)
    return _global_start_new_logging_session(reset_files=reset_files)


def finalize_main_logging() -> None:
    _MAIN_SESSION_LOGGER.finalize()
    _global_finalize_process_logging()


def log_event(
    component: str,
    action: str,
    details: Mapping[str, Any] | None = None,
    *,
    direction: str = "internal",
    level: str = "INFO",
) -> None:
    payload = details or {}
    if not _should_record_main_session_event(component, action):
        return
    _MAIN_SESSION_LOGGER.log_event(component, action, payload, direction=direction, level=level)
    _global_log_event(component, action, payload, direction=direction, level=level)


def log_exception(
    component: str,
    action: str,
    error: Exception,
    details: Mapping[str, Any] | None = None,
) -> None:
    payload = details or {}
    if not _should_record_main_session_event(component, action):
        return
    _MAIN_SESSION_LOGGER.log_exception(component, action, error, payload)
    _global_log_exception(component, action, error, payload)


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
    "finalize_main_logging",
    "initialize_main_logging",
    "log_event",
    "log_exception",
    "log_main_event",
    "log_main_exception",
    "start_main_logging_session",
]
