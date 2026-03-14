from __future__ import annotations

import json
from typing import Any, Dict, List


TOOL_OUTPUT_CONTRACT: Dict[str, Any] = {
    "ok": "boolean success flag",
    "tool_name": "callable tool name",
    "summary": "required concise description of what the tool returned",
    "content_type": "collection | object | memory | delivery | error",
    "items": "list of normalized records when the tool returns multiple items",
    "data": "normalized object payload for structured details",
    "errors": "list of human-readable error messages when something is wrong",
    "metadata": "optional lightweight call metadata",
}


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(item)
            for key, item in value.items()
            if str(key).strip()
        }
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    return str(value)


def build_tool_output_payload(
    *,
    tool_name: str,
    ok: bool,
    summary: str,
    content_type: str,
    items: List[Any] | None = None,
    data: Dict[str, Any] | None = None,
    errors: List[str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_items = []
    for item in items or []:
        normalized_items.append(_normalize_json_value(item))

    normalized_data = _normalize_json_value(data or {})
    if not isinstance(normalized_data, dict):
        normalized_data = {"value": normalized_data}

    normalized_errors = [
        _compact_text(item, max_chars=280)
        for item in (errors or [])
        if str(item).strip()
    ]
    normalized_metadata = _normalize_json_value(metadata or {})
    if not isinstance(normalized_metadata, dict):
        normalized_metadata = {"value": normalized_metadata}

    return {
        "ok": bool(ok),
        "tool_name": str(tool_name or "").strip(),
        "summary": _compact_text(summary, max_chars=320),
        "content_type": str(content_type or "").strip() or "object",
        "items": normalized_items,
        "data": normalized_data,
        "errors": normalized_errors,
        "metadata": normalized_metadata,
    }


def render_tool_output(
    *,
    tool_name: str,
    ok: bool,
    summary: str,
    content_type: str,
    items: List[Any] | None = None,
    data: Dict[str, Any] | None = None,
    errors: List[str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> str:
    payload = build_tool_output_payload(
        tool_name=tool_name,
        ok=ok,
        summary=summary,
        content_type=content_type,
        items=items,
        data=data,
        errors=errors,
        metadata=metadata,
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["TOOL_OUTPUT_CONTRACT", "build_tool_output_payload", "render_tool_output"]
