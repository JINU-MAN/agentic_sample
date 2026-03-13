from __future__ import annotations

from pathlib import Path
from typing import Mapping

from agentic_sample_ad.system_logger import ScopedSessionLogger

_LOGGER = ScopedSessionLogger(Path(__file__).resolve().parent / "log")


def _should_record_agent_session_event(component: str, action: str) -> bool:
    component_key = str(component or "").strip()
    action_key = str(action or "").strip().lower()
    if component_key == "ad.sns_agent.event_manager" and action_key in {"task_started", "task_completed", "task_failed"}:
        return True
    if component_key.startswith("a2a.agent_server.") and action_key in {
        "rpc_request_received",
        "agent_execution_started",
        "agent_execution_completed",
        "rpc_request_completed",
        "rpc_request_failed",
    }:
        return True
    if component_key.startswith("tool."):
        return True
    return False


def initialize_process_logging() -> None:
    _LOGGER.initialize()


def start_new_logging_session(*, reset_files: bool = True) -> None:
    _LOGGER.start_new_session(reset_files=reset_files)


def finalize_process_logging() -> None:
    _LOGGER.finalize()


def log_event(
    component: str,
    action: str,
    details: Mapping[str, object] | None = None,
    *,
    direction: str = "internal",
    level: str = "INFO",
) -> None:
    if _should_record_agent_session_event(component, action):
        _LOGGER.log_event(component, action, details or {}, direction=direction, level=level)


def log_exception(
    component: str,
    action: str,
    error: Exception,
    details: Mapping[str, object] | None = None,
) -> None:
    if _should_record_agent_session_event(component, action):
        _LOGGER.log_exception(component, action, error, details or {})


__all__ = [
    "finalize_process_logging",
    "initialize_process_logging",
    "log_event",
    "log_exception",
    "start_new_logging_session",
]
