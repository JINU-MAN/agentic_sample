from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Any, Awaitable, Callable, Dict, List, Mapping, TypeVar

import httpx

from agentic_sample_ad.system_logger import log_event


T = TypeVar("T")

_RETRY_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_MESSAGE_TOKENS = (
    "connection aborted",
    "connection error",
    "connection refused",
    "connection reset",
    "dns",
    "econnreset",
    "high demand",
    "name or service not known",
    "network is unreachable",
    "rate limit",
    "remote protocol error",
    "service unavailable",
    "temporarily unavailable",
    "temporary failure in name resolution",
    "timed out",
    "timeout",
    "too many requests",
    "transport error",
    "unavailable",
)
_RETRYABLE_TYPE_NAMES = {
    "apiconnectionerror",
    "connecterror",
    "connectionerror",
    "connecttimeout",
    "pooltimeout",
    "protocolerror",
    "ratelimiterror",
    "readerror",
    "readtimeout",
    "remoteprotocolerror",
    "servererror",
    "serviceunavailableerror",
    "timeouterror",
    "transporterror",
    "writeerror",
    "writetimeout",
}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, minimum)


def network_retry_attempts() -> int:
    return _env_int("AGENTIC_NETWORK_RETRY_ATTEMPTS", 4, 1)


def network_retry_base_delay_sec() -> float:
    return _env_float("AGENTIC_NETWORK_RETRY_BASE_DELAY_SEC", 1.0, 0.1)


def network_retry_max_delay_sec() -> float:
    return _env_float("AGENTIC_NETWORK_RETRY_MAX_DELAY_SEC", 8.0, 0.1)


def _walk_exception_chain(exc: BaseException) -> List[BaseException]:
    chain: List[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None
    return chain


def is_retryable_network_error(exc: BaseException) -> bool:
    for current in _walk_exception_chain(exc):
        if isinstance(
            current,
            (
                asyncio.TimeoutError,
                ConnectionAbortedError,
                ConnectionError,
                ConnectionRefusedError,
                ConnectionResetError,
                TimeoutError,
                httpx.TimeoutException,
                httpx.TransportError,
                socket.gaierror,
            ),
        ):
            return True

        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int) and (status_code in _RETRY_STATUS_CODES or 500 <= status_code <= 599):
            return True

        error_type = type(current).__name__.strip().lower()
        if error_type in _RETRYABLE_TYPE_NAMES:
            return True

        message = f"{type(current).__name__}: {current}".strip().lower()
        if any(token in message for token in _RETRYABLE_MESSAGE_TOKENS):
            return True

    return False


def _normalize_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude_none=True)
        except Exception:
            dumped = None
        if dumped is not None:
            return _normalize_jsonable(dumped)
    if isinstance(value, dict):
        normalized = {
            str(key): _normalize_jsonable(item)
            for key, item in value.items()
            if item is not None
        }
        return {key: item for key, item in normalized.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(item) for item in value if item is not None]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_chunk(payload: Any) -> str:
    normalized = _normalize_jsonable(payload)
    try:
        return json.dumps(normalized, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(normalized), ensure_ascii=False)


def _extract_non_text_chunks_from_parts(parts: List[Any]) -> tuple[List[str], List[str]]:
    chunks: List[str] = []
    part_types: List[str] = []
    for part in parts:
        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            part_types.append("function_response")
            nested_parts = list(getattr(function_response, "parts", []) or [])
            nested_text = "".join(str(getattr(item, "text", "") or "") for item in nested_parts).strip()
            response_payload = _normalize_jsonable(getattr(function_response, "response", None))
            if isinstance(response_payload, dict) and response_payload:
                chunks.append(_json_chunk(response_payload))
            elif nested_text:
                chunks.append(nested_text)
            elif nested_parts:
                nested_chunks, _ = _extract_non_text_chunks_from_parts(nested_parts)
                chunks.extend(nested_chunks)
            else:
                chunks.append(
                    _json_chunk(
                        {
                            "function_response": {
                                "name": getattr(function_response, "name", None),
                                "id": getattr(function_response, "id", None),
                                "response": response_payload,
                                "will_continue": getattr(function_response, "will_continue", None),
                            }
                        }
                    )
                )
            continue

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            part_types.append("function_call")
            chunks.append(
                _json_chunk(
                    {
                        "function_call": {
                            "name": getattr(function_call, "name", None),
                            "id": getattr(function_call, "id", None),
                            "args": _normalize_jsonable(getattr(function_call, "args", None)),
                            "will_continue": getattr(function_call, "will_continue", None),
                        }
                    }
                )
            )
            continue

        code_execution_result = getattr(part, "code_execution_result", None)
        if code_execution_result is not None:
            part_types.append("code_execution_result")
            output = str(getattr(code_execution_result, "output", "") or "").strip()
            if output:
                chunks.append(output)
            else:
                chunks.append(
                    _json_chunk(
                        {
                            "code_execution_result": {
                                "outcome": getattr(code_execution_result, "outcome", None),
                                "output": output,
                            }
                        }
                    )
                )
            continue

        executable_code = getattr(part, "executable_code", None)
        if executable_code is not None:
            part_types.append("executable_code")
            chunks.append(
                _json_chunk(
                    {
                        "executable_code": {
                            "language": getattr(executable_code, "language", None),
                            "code": getattr(executable_code, "code", None),
                        }
                    }
                )
            )
            continue

        file_data = getattr(part, "file_data", None)
        if file_data is not None:
            part_types.append("file_data")
            chunks.append(
                _json_chunk(
                    {
                        "file_data": {
                            "mime_type": getattr(file_data, "mime_type", None),
                            "file_uri": getattr(file_data, "file_uri", None),
                            "display_name": getattr(file_data, "display_name", None),
                        }
                    }
                )
            )
            continue

        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None:
            part_types.append("inline_data")
            raw_data = getattr(inline_data, "data", None)
            size_bytes = len(raw_data) if isinstance(raw_data, (bytes, bytearray)) else None
            chunks.append(
                _json_chunk(
                    {
                        "inline_data": {
                            "mime_type": getattr(inline_data, "mime_type", None),
                            "display_name": getattr(inline_data, "display_name", None),
                            "size_bytes": size_bytes,
                        }
                    }
                )
            )
            continue

        video_metadata = getattr(part, "video_metadata", None)
        if video_metadata is not None:
            part_types.append("video_metadata")
            chunks.append(_json_chunk({"video_metadata": _normalize_jsonable(video_metadata)}))
            continue

        thought = getattr(part, "thought", None)
        if thought:
            part_types.append("thought")
            chunks.append(_json_chunk({"thought": True}))
            continue

        part_types.append("unknown_non_text")
        chunks.append(_json_chunk({"part": "unknown_non_text"}))

    deduped_types: List[str] = []
    seen_types: set[str] = set()
    for token in part_types:
        lowered = str(token).strip().lower()
        if not lowered or lowered in seen_types:
            continue
        seen_types.add(lowered)
        deduped_types.append(token)
    return chunks, deduped_types


def _dedupe_tokens(tokens: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = str(token).strip().lower()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(str(token))
    return deduped


def _normalize_part_payload(part: Any) -> Dict[str, Any]:
    text = str(getattr(part, "text", "") or "").strip()
    if text:
        return {"kind": "text", "text": text}

    function_response = getattr(part, "function_response", None)
    if function_response is not None:
        nested_parts = list(getattr(function_response, "parts", []) or [])
        return {
            "kind": "function_response",
            "name": getattr(function_response, "name", None),
            "id": getattr(function_response, "id", None),
            "response": _normalize_jsonable(getattr(function_response, "response", None)),
            "will_continue": getattr(function_response, "will_continue", None),
            "parts": [_normalize_part_payload(item) for item in nested_parts],
        }

    function_call = getattr(part, "function_call", None)
    if function_call is not None:
        return {
            "kind": "function_call",
            "name": getattr(function_call, "name", None),
            "id": getattr(function_call, "id", None),
            "args": _normalize_jsonable(getattr(function_call, "args", None)),
            "will_continue": getattr(function_call, "will_continue", None),
        }

    code_execution_result = getattr(part, "code_execution_result", None)
    if code_execution_result is not None:
        return {
            "kind": "code_execution_result",
            "outcome": getattr(code_execution_result, "outcome", None),
            "output": str(getattr(code_execution_result, "output", "") or "").strip(),
        }

    executable_code = getattr(part, "executable_code", None)
    if executable_code is not None:
        return {
            "kind": "executable_code",
            "language": getattr(executable_code, "language", None),
            "code": getattr(executable_code, "code", None),
        }

    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        return {
            "kind": "file_data",
            "mime_type": getattr(file_data, "mime_type", None),
            "file_uri": getattr(file_data, "file_uri", None),
            "display_name": getattr(file_data, "display_name", None),
        }

    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        raw_data = getattr(inline_data, "data", None)
        size_bytes = len(raw_data) if isinstance(raw_data, (bytes, bytearray)) else None
        return {
            "kind": "inline_data",
            "mime_type": getattr(inline_data, "mime_type", None),
            "display_name": getattr(inline_data, "display_name", None),
            "size_bytes": size_bytes,
        }

    video_metadata = getattr(part, "video_metadata", None)
    if video_metadata is not None:
        return {
            "kind": "video_metadata",
            "value": _normalize_jsonable(video_metadata),
        }

    thought = getattr(part, "thought", None)
    if thought:
        return {"kind": "thought", "value": True}

    thought_signature = getattr(part, "thought_signature", None)
    if thought_signature is not None:
        return {
            "kind": "thought_signature",
            "value": _normalize_jsonable(thought_signature),
        }

    return {"kind": "unknown_non_text"}


def _normalize_part_entries(parts: List[Any], *, event_index: int, author: str) -> List[Dict[str, Any]]:
    normalized_entries: List[Dict[str, Any]] = []
    for part_index, part in enumerate(parts, start=1):
        payload = _normalize_part_payload(part)
        normalized_entries.append(
            {
                "event_index": event_index,
                "part_index": part_index,
                "author": author,
                **payload,
            }
        )
    return normalized_entries


async def run_with_network_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    component: str,
    operation_name: str,
    retry_details: Mapping[str, Any] | None = None,
    log_event_fn: Callable[..., None] = log_event,
    max_attempts: int | None = None,
    base_delay_sec: float | None = None,
    max_delay_sec: float | None = None,
) -> T:
    attempts = max_attempts or network_retry_attempts()
    base_delay = base_delay_sec or network_retry_base_delay_sec()
    max_delay = max_delay_sec or network_retry_max_delay_sec()
    details = dict(retry_details or {})

    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= attempts or not is_retryable_network_error(exc):
                raise

            delay_sec = min(max_delay, base_delay * (2 ** (attempt - 1)))
            log_event_fn(
                component,
                "network_retry_scheduled",
                {
                    **details,
                    "operation": operation_name,
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "max_attempts": attempts,
                    "delay_sec": round(delay_sec, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                direction="internal",
                level="WARNING",
            )
            await asyncio.sleep(delay_sec)

    raise RuntimeError(f"Retry loop exhausted without returning: {operation_name}")


async def collect_response_parts_with_network_retry(
    *,
    runner: Any,
    user_id: str,
    new_message: Any,
    component: str,
    operation_name: str,
    retry_details: Mapping[str, Any] | None = None,
    on_text: Callable[[str, Any], None] | None = None,
    log_event_fn: Callable[..., None] = log_event,
) -> Dict[str, Any]:
    async def _run_once() -> Dict[str, Any]:
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
        )
        text_chunks: List[str] = []
        non_text_chunks: List[str] = []
        non_text_part_types: List[str] = []
        normalized_parts: List[Dict[str, Any]] = []
        event_index = 0
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=new_message,
        ):
            if event.content and event.content.parts:
                event_index += 1
                parts = list(event.content.parts)
                author = str(getattr(event, "author", "unknown") or "unknown")
                normalized_parts.extend(_normalize_part_entries(parts, event_index=event_index, author=author))
                text = "".join(str(getattr(part, "text", "") or "") for part in parts).strip()
                if text:
                    text_chunks.append(text)
                    if on_text is not None:
                        on_text(text, event)
                    continue
                fallback_chunks, part_types = _extract_non_text_chunks_from_parts(parts)
                if fallback_chunks:
                    non_text_chunks.extend(fallback_chunks)
                if part_types:
                    non_text_part_types.extend(part_types)

        used_non_text_fallback = not text_chunks and bool(non_text_chunks)
        deduped_types = _dedupe_tokens(non_text_part_types)
        if used_non_text_fallback:
            log_event_fn(
                component,
                "non_text_response_used",
                {
                    **dict(retry_details or {}),
                    "operation": operation_name,
                    "non_text_part_types": deduped_types,
                    "chunk_count": len(non_text_chunks),
                    "normalized_parts": normalized_parts,
                },
                direction="inbound",
                level="WARNING",
            )
        return {
            "chunks": text_chunks if text_chunks else non_text_chunks,
            "text_chunks": text_chunks,
            "non_text_chunks": non_text_chunks,
            "normalized_parts": normalized_parts,
            "non_text_part_types": deduped_types,
            "used_non_text_fallback": used_non_text_fallback,
        }

    return await run_with_network_retry(
        _run_once,
        component=component,
        operation_name=operation_name,
        retry_details=retry_details,
        log_event_fn=log_event_fn,
    )


async def collect_text_response_with_network_retry(
    *,
    runner: Any,
    user_id: str,
    new_message: Any,
    component: str,
    operation_name: str,
    retry_details: Mapping[str, Any] | None = None,
    on_text: Callable[[str, Any], None] | None = None,
    log_event_fn: Callable[..., None] = log_event,
) -> List[str]:
    collected = await collect_response_parts_with_network_retry(
        runner=runner,
        user_id=user_id,
        new_message=new_message,
        component=component,
        operation_name=operation_name,
        retry_details=retry_details,
        on_text=on_text,
        log_event_fn=log_event_fn,
    )
    return list(collected.get("chunks", []))
