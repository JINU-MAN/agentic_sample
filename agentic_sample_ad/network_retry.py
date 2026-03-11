from __future__ import annotations

import asyncio
import os
import socket
from typing import Any, Awaitable, Callable, List, Mapping, TypeVar

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


async def run_with_network_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    component: str,
    operation_name: str,
    retry_details: Mapping[str, Any] | None = None,
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
            log_event(
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


async def collect_text_response_with_network_retry(
    *,
    runner: Any,
    user_id: str,
    new_message: Any,
    component: str,
    operation_name: str,
    retry_details: Mapping[str, Any] | None = None,
    on_text: Callable[[str, Any], None] | None = None,
) -> List[str]:
    async def _run_once() -> List[str]:
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
        )
        chunks: List[str] = []
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=new_message,
        ):
            if event.content and event.content.parts:
                text = "".join(part.text or "" for part in event.content.parts).strip()
                if text:
                    chunks.append(text)
                    if on_text is not None:
                        on_text(text, event)
        return chunks

    return await run_with_network_retry(
        _run_once,
        component=component,
        operation_name=operation_name,
        retry_details=retry_details,
    )
