from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
COMPONENT_LOG_DIR = LOG_DIR / "components"
SYSTEM_LOG_FILE = LOG_DIR / "system_events.jsonl"
TEST_LOG_FILE = LOG_DIR / "test_log.jsonl"

_LOCK = threading.Lock()
_MAX_DEPTH = 6
_MAX_ITEMS = 80
_MAX_STR_LEN = 8000
_A2A_LOGGING_ENABLED = False
_A2A_HANDLER_NAME = "system_logger_a2a_bridge"
_PROCESS_LOG_INITIALIZED = False


def _parse_log_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    value = str(level).strip().upper()
    if not value:
        return logging.INFO
    return getattr(logging, value, logging.INFO)


class _A2ABridgeHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            details: dict[str, Any] = {
                "logger": record.name,
                "message": message,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            if record.exc_info:
                details["exception"] = self.formatter.formatException(record.exc_info) if self.formatter else str(record.exc_info)
            log_event(
                "a2a.package",
                "python_log",
                details,
                direction="internal",
                level=record.levelname,
            )
        except Exception:
            return


def ensure_log_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    COMPONENT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def initialize_process_logging() -> None:
    """
    Initialize per-process logging artifacts once.
    - test_log.jsonl is truncated at process start.
    """
    global _PROCESS_LOG_INITIALIZED
    with _LOCK:
        if _PROCESS_LOG_INITIALIZED:
            return
        ensure_log_dirs()
        TEST_LOG_FILE.write_text("", encoding="utf-8")
        _PROCESS_LOG_INITIALIZED = True


def _sanitize_component_name(component: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", component.strip())
    return cleaned or "unknown"


def _truncate_text(text: str) -> str:
    if len(text) <= _MAX_STR_LEN:
        return text
    return text[:_MAX_STR_LEN] + "...(truncated)"


def _normalize_value(value: Any, depth: int = 0) -> Any:
    if depth >= _MAX_DEPTH:
        return "<max_depth_reached>"

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, bytes):
        return _truncate_text(value.decode("utf-8", errors="replace"))

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json", exclude_none=True)
            return _normalize_value(dumped, depth + 1)
        except Exception:
            return _truncate_text(str(value))

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= _MAX_ITEMS:
                normalized["..."] = f"trimmed after {_MAX_ITEMS} keys"
                break
            normalized[str(key)] = _normalize_value(item, depth + 1)
        return normalized

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        normalized_list = [_normalize_value(item, depth + 1) for item in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            normalized_list.append(f"... trimmed after {_MAX_ITEMS} items")
        return normalized_list

    return _truncate_text(str(value))


def _write_line(path: Path, payload: Mapping[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def log_event(
    component: str,
    action: str,
    details: Mapping[str, Any] | None = None,
    *,
    direction: str = "internal",
    level: str = "INFO",
) -> None:
    try:
        ensure_log_dirs()
        safe_component = _sanitize_component_name(component)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "component": safe_component,
            "action": action,
            "direction": direction,
            "details": _normalize_value(details or {}, depth=0),
        }

        component_file = COMPONENT_LOG_DIR / f"{safe_component}.jsonl"
        with _LOCK:
            _write_line(SYSTEM_LOG_FILE, payload)
            _write_line(component_file, payload)
            _write_line(TEST_LOG_FILE, payload)
    except Exception:
        # Logging must never break primary execution.
        return


def log_exception(
    component: str,
    action: str,
    error: Exception,
    details: Mapping[str, Any] | None = None,
) -> None:
    merged = dict(details or {})
    merged["error_type"] = type(error).__name__
    merged["error"] = str(error)
    log_event(component, action, merged, direction="internal", level="ERROR")


def enable_a2a_package_logging(level: int | str = "INFO") -> None:
    """
    Bridge Python `a2a` package logs into JSONL system logs.
    """
    global _A2A_LOGGING_ENABLED
    target_level = _parse_log_level(level)
    logger = logging.getLogger("a2a")

    if not any(getattr(handler, "name", "") == _A2A_HANDLER_NAME for handler in logger.handlers):
        bridge = _A2ABridgeHandler(level=target_level)
        bridge.name = _A2A_HANDLER_NAME
        bridge.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(bridge)

    logger.setLevel(target_level)
    _A2A_LOGGING_ENABLED = True
    log_event(
        "a2a.package",
        "logging_enabled",
        {
            "logger": "a2a",
            "level": logging.getLevelName(target_level),
            "enabled": _A2A_LOGGING_ENABLED,
        },
    )
