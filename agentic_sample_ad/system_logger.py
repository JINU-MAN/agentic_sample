from __future__ import annotations

import atexit
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
COMPONENT_LOG_DIR = LOG_DIR / "components"
SYSTEM_LOG_FILE = LOG_DIR / "system_events.jsonl"
SESSION_LOG_FILE = LOG_DIR / "session_log.jsonl"
SESSION_ID_ENV_KEY = "AGENTIC_SESSION_ID"
SESSION_SEQ_FILE_NAME = ".session_sequence"
SESSION_SEQ_FILE = LOG_DIR / SESSION_SEQ_FILE_NAME
SESSION_SEQ_LOCK_FILE_NAME = ".session_sequence.lock"
SESSION_SEQ_LOCK_FILE = LOG_DIR / SESSION_SEQ_LOCK_FILE_NAME
SESSION_ARCHIVE_PREFIX = "session_log_ex_"
SESSION_SEQ_LOCK_TIMEOUT_SEC = 5.0
SESSION_SEQ_LOCK_STALE_SEC = 30.0
FUNCTION_TRACE_ENV_KEY = "AGENTIC_LOG_FUNCTION_CALLS"
FUNCTION_TRACE_COMPONENT = "function_trace"
FUNCTION_TRACE_ACTION = "function_called"

_LOCK = threading.Lock()
_MAX_DEPTH = 6
_MAX_ITEMS = 80
_MAX_STR_LEN = 8000
_A2A_LOGGING_ENABLED = False
_A2A_HANDLER_NAME = "system_logger_a2a_bridge"
_PROCESS_LOG_INITIALIZED = False
_PROCESS_EXIT_HOOK_REGISTERED = False
_PROCESS_LOG_FINALIZED = False
_SESSION_ID: str | None = None
_SESSION_OWNER = False
_FUNCTION_TRACE_ENABLED = False
_TRACE_EMIT_GUARD = threading.local()


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


def _event_file_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S_%fZ")


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "disable", "disabled"}


def _is_traceable_source_file(filename: str) -> bool:
    if not filename or filename.startswith("<"):
        return False
    try:
        resolved = Path(filename).resolve()
    except Exception:
        return False
    if resolved == Path(__file__).resolve():
        # Skip logger internals to avoid recursive call tracing.
        return False
    try:
        resolved.relative_to(BASE_DIR)
        return True
    except Exception:
        return False


def _relative_source_path(filename: str) -> str:
    if not filename:
        return ""
    try:
        resolved = Path(filename).resolve()
        return str(resolved.relative_to(BASE_DIR)).replace("\\", "/")
    except Exception:
        return str(filename)


def _frame_callable_name(frame: Any) -> str:
    name = str(getattr(frame.f_code, "co_name", "") or "").strip() or "<unknown>"
    try:
        if "self" in frame.f_locals:
            cls_name = frame.f_locals["self"].__class__.__name__
            return f"{cls_name}.{name}"
        if "cls" in frame.f_locals and isinstance(frame.f_locals["cls"], type):
            cls_name = frame.f_locals["cls"].__name__
            return f"{cls_name}.{name}"
    except Exception:
        pass
    return name


def _profile_function_calls(frame: Any, event: str, arg: Any) -> None:
    if event != "call":
        return
    if getattr(_TRACE_EMIT_GUARD, "active", False):
        return

    code = getattr(frame, "f_code", None)
    if code is None:
        return
    source_file = str(getattr(code, "co_filename", "") or "")
    if not _is_traceable_source_file(source_file):
        return

    try:
        details: dict[str, Any] = {
            "module": str(frame.f_globals.get("__name__", "")).strip(),
            "function": _frame_callable_name(frame),
            "source": _relative_source_path(source_file),
            "line": int(getattr(code, "co_firstlineno", 0) or 0),
        }
        caller = getattr(frame, "f_back", None)
        if caller is not None:
            caller_code = getattr(caller, "f_code", None)
            if caller_code is not None:
                caller_source = str(getattr(caller_code, "co_filename", "") or "")
                details["caller_module"] = str(caller.f_globals.get("__name__", "")).strip()
                details["caller_function"] = _frame_callable_name(caller)
                details["caller_source"] = _relative_source_path(caller_source)
                details["caller_line"] = int(getattr(caller_code, "co_firstlineno", 0) or 0)

        _TRACE_EMIT_GUARD.active = True
        log_event(
            FUNCTION_TRACE_COMPONENT,
            FUNCTION_TRACE_ACTION,
            details,
            direction="internal",
            level="DEBUG",
        )
    except Exception:
        return
    finally:
        _TRACE_EMIT_GUARD.active = False


def _enable_function_call_tracing_if_needed() -> None:
    # Function-level call tracing is intentionally disabled.
    return


def _reset_active_session_log_files() -> None:
    SESSION_LOG_FILE.write_text("", encoding="utf-8")
    SESSION_SEQ_FILE.write_text("0", encoding="utf-8")
    try:
        SESSION_SEQ_LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _next_session_archive_dir() -> Path:
    max_seq = 0
    for candidate in LOG_DIR.glob(f"{SESSION_ARCHIVE_PREFIX}*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name[len(SESSION_ARCHIVE_PREFIX):]
        if suffix.isdigit():
            max_seq = max(max_seq, int(suffix))
    return LOG_DIR / f"{SESSION_ARCHIVE_PREFIX}{max_seq + 1:010d}"


def _parse_event_timestamp(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _session_event_file_name(payload: Mapping[str, Any], fallback_seq: int) -> str:
    seq_value = payload.get("session_seq", fallback_seq)
    try:
        seq_num = int(seq_value)
    except Exception:
        seq_num = fallback_seq

    if seq_num < 0:
        seq_num = fallback_seq

    component = _sanitize_component_name(str(payload.get("component", "") or "unknown"))
    action = _sanitize_component_name(str(payload.get("action", "") or "event"))
    ts_token = _event_file_timestamp(_parse_event_timestamp(payload.get("ts")))
    return f"{seq_num:010d}_{component}_{action}_{ts_token}.json"


def _archive_previous_session_log() -> None:
    if not SESSION_LOG_FILE.exists():
        return

    try:
        lines = [
            line.strip()
            for line in SESSION_LOG_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        lines = []

    if not lines:
        return

    archive_dir = _next_session_archive_dir()
    archive_dir.mkdir(parents=True, exist_ok=True)

    for idx, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except Exception:
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "component": "system_logger",
                "action": "invalid_session_jsonl_line",
                "direction": "internal",
                "details": {"line": _truncate_text(line)},
                "session_seq": idx,
            }

        if not isinstance(payload, dict):
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "component": "system_logger",
                "action": "invalid_session_jsonl_payload",
                "direction": "internal",
                "details": {"payload": _normalize_value(payload, depth=0)},
                "session_seq": idx,
            }

        filename = _session_event_file_name(payload, fallback_seq=idx)
        event_file = archive_dir / filename
        if event_file.exists():
            stem = event_file.stem
            suffix = 1
            while event_file.exists():
                event_file = archive_dir / f"{stem}_{suffix}.json"
                suffix += 1

        event_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _ensure_session_id() -> str:
    global _SESSION_ID, _SESSION_OWNER
    if _SESSION_ID:
        return _SESSION_ID

    inherited = str(os.getenv(SESSION_ID_ENV_KEY, "")).strip()
    if inherited:
        _SESSION_ID = _sanitize_component_name(inherited)
        _SESSION_OWNER = False
        return _SESSION_ID

    created = _event_file_timestamp(datetime.now(timezone.utc))
    _SESSION_ID = created
    _SESSION_OWNER = True
    os.environ[SESSION_ID_ENV_KEY] = created
    return _SESSION_ID


def start_new_logging_session(*, reset_files: bool = True) -> str:
    """
    Start a new logical logging session for the current process tree.
    Intended entrypoint: `start_agentic` startup.

    - Assigns fresh `AGENTIC_SESSION_ID`.
    - Optionally archives the previous `session_log.jsonl` to
      `log/session_log_ex_{num}/` and then resets active session files.
    """
    global _SESSION_ID, _SESSION_OWNER, _PROCESS_LOG_FINALIZED
    with _LOCK:
        ensure_log_dirs()
        created = _event_file_timestamp(datetime.now(timezone.utc))
        normalized = _sanitize_component_name(created)
        _SESSION_ID = normalized
        _SESSION_OWNER = True
        _PROCESS_LOG_FINALIZED = False
        os.environ[SESSION_ID_ENV_KEY] = normalized
        if reset_files:
            _archive_previous_session_log()
            _reset_active_session_log_files()
        return normalized


def _next_session_event_sequence() -> int:
    seq_file = SESSION_SEQ_FILE
    lock_file = SESSION_SEQ_LOCK_FILE
    deadline = time.monotonic() + SESSION_SEQ_LOCK_TIMEOUT_SEC
    lock_fd: int | None = None

    while True:
        try:
            lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age_sec = time.time() - lock_file.stat().st_mtime
                if age_sec >= SESSION_SEQ_LOCK_STALE_SEC:
                    lock_file.unlink(missing_ok=True)
                    continue
            except Exception:
                pass
            if time.monotonic() >= deadline:
                # Fallback to timestamp-based sequence token when lock acquisition fails.
                return int(datetime.now(timezone.utc).timestamp() * 1_000_000)
            time.sleep(0.01)

    try:
        current = 0
        if seq_file.exists():
            try:
                current = int(seq_file.read_text(encoding="utf-8").strip() or "0")
            except Exception:
                current = 0

        next_seq = current + 1
        seq_file.write_text(str(next_seq), encoding="utf-8")
        return next_seq
    finally:
        try:
            if lock_fd is not None:
                os.close(lock_fd)
        finally:
            try:
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass


def _finalize_logging_on_exit() -> None:
    try:
        finalize_process_logging()
    except Exception:
        return


def _register_exit_hook_if_needed() -> None:
    global _PROCESS_EXIT_HOOK_REGISTERED
    if _PROCESS_EXIT_HOOK_REGISTERED:
        return
    atexit.register(_finalize_logging_on_exit)
    _PROCESS_EXIT_HOOK_REGISTERED = True


def initialize_process_logging() -> None:
    """
    Initialize per-process logging artifacts once.
    - prepares shared session jsonl + sequence state under log/.
    """
    global _PROCESS_LOG_INITIALIZED
    with _LOCK:
        if _PROCESS_LOG_INITIALIZED:
            return
        ensure_log_dirs()
        _ensure_session_id()
        _register_exit_hook_if_needed()
        _enable_function_call_tracing_if_needed()
        _PROCESS_LOG_INITIALIZED = True


def finalize_process_logging() -> None:
    """
    Finalize current logging session.
    - no session folder is created.
    - keeps only JSONL logs (`system_events.jsonl`, `components/*.jsonl`, `session_log.jsonl`).
    - runs once per process; safe to call multiple times.
    """
    global _PROCESS_LOG_FINALIZED
    with _LOCK:
        if _PROCESS_LOG_FINALIZED:
            return
        if not _SESSION_OWNER:
            _PROCESS_LOG_FINALIZED = True
            return

        _ensure_session_id()
        _PROCESS_LOG_FINALIZED = True


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
        event_ts = datetime.now(timezone.utc)
        safe_component = _sanitize_component_name(component)
        payload = {
            "ts": event_ts.isoformat(),
            "level": level.upper(),
            "component": safe_component,
            "action": action,
            "direction": direction,
            "details": _normalize_value(details or {}, depth=0),
        }

        component_file = COMPONENT_LOG_DIR / f"{safe_component}.jsonl"
        with _LOCK:
            if _PROCESS_LOG_FINALIZED:
                return
            _register_exit_hook_if_needed()
            _ensure_session_id()
            session_seq = _next_session_event_sequence()
            payload["session_seq"] = int(session_seq)
            _write_line(SYSTEM_LOG_FILE, payload)
            _write_line(component_file, payload)
            _write_line(SESSION_LOG_FILE, payload)
    except Exception:
        # Logging must never break primary execution.
        return


def _exception_snapshot(
    error: BaseException,
    *,
    depth: int = 0,
    max_depth: int = 3,
    max_children: int = 8,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "error_type": type(error).__name__,
        "error": str(error),
    }

    try:
        tb_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        if tb_text.strip():
            snapshot["traceback"] = tb_text
    except Exception:
        pass

    if depth >= max_depth:
        return snapshot

    children = getattr(error, "exceptions", None)
    if isinstance(children, tuple) and children:
        child_summaries = [
            _exception_snapshot(
                child,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
            )
            for child in children[:max_children]
        ]
        if len(children) > max_children:
            child_summaries.append({"trimmed": len(children) - max_children})
        snapshot["sub_exceptions"] = child_summaries

    return snapshot


def log_exception(
    component: str,
    action: str,
    error: Exception,
    details: Mapping[str, Any] | None = None,
) -> None:
    merged = dict(details or {})
    snapshot = _exception_snapshot(error)
    merged["error_type"] = str(snapshot.get("error_type", type(error).__name__))
    merged["error"] = str(snapshot.get("error", str(error)))

    traceback_text = snapshot.get("traceback")
    if isinstance(traceback_text, str) and traceback_text.strip():
        merged["traceback"] = traceback_text

    sub_exceptions = snapshot.get("sub_exceptions")
    if isinstance(sub_exceptions, list) and sub_exceptions:
        merged["sub_exceptions"] = sub_exceptions
        merged["is_exception_group"] = True

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

