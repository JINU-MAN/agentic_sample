from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any, Dict, Iterable, Tuple


EMPTY_TEXT_RESPONSE = "(No text response emitted.)"
EMPTY_RESPONSE_SENTINELS = {
    "",
    EMPTY_TEXT_RESPONSE,
}


def run_coroutine_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Any = None
    error: Exception | None = None

    def _target() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover
            error = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result


def extract_json_object_with_source(text: str) -> Tuple[Dict[str, Any] | None, str]:
    stripped = str(text or "").strip()
    if not stripped:
        return None, "empty"

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed, "plain_json"
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, "fenced_json"
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, "substring"
        except json.JSONDecodeError:
            return None, "substring_invalid"

    return None, "not_found"


def extract_json_object(text: str) -> Dict[str, Any] | None:
    parsed, _ = extract_json_object_with_source(text)
    return parsed


def finalize_text_response(chunks: Iterable[str]) -> str:
    return "\n".join(str(chunk) for chunk in chunks).strip() or EMPTY_TEXT_RESPONSE
