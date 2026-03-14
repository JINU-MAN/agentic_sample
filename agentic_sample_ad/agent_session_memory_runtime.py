from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from agentic_sample_ad.tool_output_utils import render_tool_output


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_usage_rules(items: Any, *, max_items: int) -> List[str]:
    if not isinstance(items, list):
        return []
    return [_compact_text(item, max_chars=220) for item in items[:max_items] if str(item).strip()]


def _normalize_tools(items: Any, *, max_items: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "name": str(item.get("name", "")).strip(),
                "kind": str(item.get("kind", "")).strip(),
                "description": _compact_text(item.get("description", ""), max_chars=220),
            }
        )
    return normalized


def _normalize_public_contract(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "name": str(item.get("name", "")).strip(),
        "type": str(item.get("type", "")).strip(),
        "role": str(item.get("role", "")).strip(),
        "description": _compact_text(item.get("description", ""), max_chars=220),
        "capabilities": [str(value).strip() for value in item.get("capabilities", []) if str(value).strip()][:12],
        "ownership": _compact_text(item.get("ownership", ""), max_chars=220),
        "instruction_preview": _compact_text(item.get("instruction_preview", ""), max_chars=240),
    }


def _normalize_known_sub_agents(items: Any, *, max_items: int, include_tools: bool) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        entry: Dict[str, Any] = {
            "name": str(item.get("name", "")).strip(),
            "public_contract": _normalize_public_contract(item.get("public_contract")),
            "callable_by_main_agent": bool(item.get("callable_by_main_agent")),
            "usage_rule": _compact_text(item.get("usage_rule", ""), max_chars=220),
        }
        if include_tools:
            entry["private_tool_inventory"] = _normalize_tools(item.get("private_tool_inventory"), max_items=max_items)
        normalized.append(entry)
    return normalized


def _normalize_handoff_contract(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "structured_output_fields": [
            str(value).strip() for value in item.get("structured_output_fields", []) if str(value).strip()
        ][:10],
        "artifact_fields": [str(value).strip() for value in item.get("artifact_fields", []) if str(value).strip()][:12],
        "notes": _normalize_usage_rules(item.get("notes"), max_items=8),
    }


def _resolve_section(section: str, query: str) -> str:
    normalized = str(section or "").strip().lower()
    if normalized in {"overview", "tools", "contracts", "handoff", "all"}:
        return normalized

    lowered = str(query or "").strip().lower()
    if any(token in lowered for token in ("tool", "inventory", "callable", "function", "use")):
        return "tools"
    if any(token in lowered for token in ("contract", "ownership", "capability", "delegate", "agent", "card")):
        return "contracts"
    if any(token in lowered for token in ("handoff", "artifact", "need", "deliverable", "output")):
        return "handoff"
    return "overview"


def _load_session_memory_payload(memory_path: Path) -> Dict[str, Any]:
    if not memory_path.exists() or not memory_path.is_file():
        return {}
    try:
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_session_memory_view(memory_path: str | Path, *, section: str = "", query: str = "", max_items: int = 6) -> Dict[str, Any]:
    resolved = Path(memory_path).resolve()
    payload = _load_session_memory_payload(resolved)
    if not payload:
        return {
            "ok": False,
            "memory_kind": "agent_session_memory",
            "selected_section": _resolve_section(section, query),
            "message": f"No session memory is available at {resolved}.",
            "path": str(resolved),
        }

    max_items = max(1, min(int(max_items or 6), 12))
    selected_section = _resolve_section(section, query)

    result: Dict[str, Any] = {
        "memory_kind": str(payload.get("memory_kind", "")).strip() or "agent_session_memory",
        "selected_section": selected_section,
        "agent": _normalize_public_contract(payload.get("agent")),
        "usage_rules": _normalize_usage_rules(payload.get("usage_rules"), max_items=max_items),
    }
    if query:
        result["query"] = _compact_text(query, max_chars=180)

    if selected_section in {"overview", "all"}:
        if "own_tools" in payload:
            result["own_tools"] = _normalize_tools(payload.get("own_tools"), max_items=max_items)
        if "runtime_tools" in payload:
            result["runtime_tools"] = _normalize_tools(payload.get("runtime_tools"), max_items=max_items)
        if "known_sub_agents" in payload:
            result["known_sub_agents"] = _normalize_known_sub_agents(
                payload.get("known_sub_agents"),
                max_items=max_items,
                include_tools=False,
            )
        handoff_contract = _normalize_handoff_contract(payload.get("handoff_contract"))
        if handoff_contract:
            result["handoff_contract"] = handoff_contract

    if selected_section == "tools":
        if "own_tools" in payload:
            result["own_tools"] = _normalize_tools(payload.get("own_tools"), max_items=max_items)
        if "runtime_tools" in payload:
            result["runtime_tools"] = _normalize_tools(payload.get("runtime_tools"), max_items=max_items)
        if "known_sub_agents" in payload:
            result["known_sub_agents"] = _normalize_known_sub_agents(
                payload.get("known_sub_agents"),
                max_items=max_items,
                include_tools=True,
            )

    if selected_section == "contracts":
        if "known_sub_agents" in payload:
            result["known_sub_agents"] = _normalize_known_sub_agents(
                payload.get("known_sub_agents"),
                max_items=max_items,
                include_tools=False,
            )

    if selected_section == "handoff":
        handoff_contract = _normalize_handoff_contract(payload.get("handoff_contract"))
        if handoff_contract:
            result["handoff_contract"] = handoff_contract

    result["ok"] = True
    result["path"] = str(resolved)
    return result


def render_session_memory(memory_path: str | Path, *, section: str = "", query: str = "", max_items: int = 6) -> str:
    view = build_session_memory_view(memory_path, section=section, query=query, max_items=max_items)
    return json.dumps(view, ensure_ascii=False, indent=2)


def build_load_session_memory_tool(
    *,
    agent_name: str,
    memory_path: str | Path,
    log_event_fn: Callable[..., None] | None = None,
    log_exception_fn: Callable[..., None] | None = None,
):
    resolved_path = Path(memory_path).resolve()

    def load_session_memory(section: str = "", query: str = "", max_items: int = 6) -> str:
        """Load static agent session memory for this agent.

        Use this when you need the current tool inventory, owned capabilities,
        known agent contracts, or handoff schema for the current agent.
        Use `section="tools"`, `section="contracts"`, or `section="handoff"`
        to focus the result. Do not use it for per-run workflow state.
        """

        try:
            normalized_max_items = max(1, min(int(max_items or 6), 12))
        except (TypeError, ValueError):
            normalized_max_items = 6
        if log_event_fn is not None:
            log_event_fn(
                "tool.load_session_memory",
                "call_started",
                {
                    "agent": agent_name,
                    "section": section,
                    "query": query,
                    "max_items": normalized_max_items,
                },
                direction="outbound",
            )
        try:
            view = build_session_memory_view(
                resolved_path,
                section=section,
                query=query,
                max_items=normalized_max_items,
            )
            rendered = render_tool_output(
                tool_name="load_session_memory",
                ok=bool(view.get("ok")),
                summary=(
                    f"Loaded session memory section '{view.get('selected_section', 'overview')}'."
                    if bool(view.get("ok"))
                    else str(view.get("message", "Session memory is unavailable."))
                ),
                content_type="memory",
                data=view,
                errors=[] if bool(view.get("ok")) else [str(view.get("message", "session_memory_unavailable"))],
                metadata={
                    "agent": agent_name,
                    "section": view.get("selected_section", section or "overview"),
                    "max_items": normalized_max_items,
                },
            )
            if log_event_fn is not None:
                log_event_fn(
                    "tool.load_session_memory",
                    "call_completed",
                    {
                        "agent": agent_name,
                        "section": section,
                        "query": query,
                        "max_items": normalized_max_items,
                    },
                    direction="inbound",
                )
            return rendered
        except Exception as e:
            if log_exception_fn is not None:
                log_exception_fn(
                    "tool.load_session_memory",
                    "call_failed",
                    e,
                    {
                        "agent": agent_name,
                        "section": section,
                        "query": query,
                        "max_items": normalized_max_items,
                    },
                )
            raise

    load_session_memory.__name__ = "load_session_memory"
    load_session_memory.__qualname__ = "load_session_memory"
    load_session_memory.__doc__ = (
        f"Load static session memory for {agent_name}. "
        "Use this when you need the current tool inventory, owned capabilities, known agent contracts, "
        "or handoff schema. Use `section=\"tools\"`, `\"contracts\"`, or `\"handoff\"` to focus the result. "
        "Do not use it for per-run workflow state."
    )
    return load_session_memory


__all__ = ["build_load_session_memory_tool", "build_session_memory_view", "render_session_memory"]
