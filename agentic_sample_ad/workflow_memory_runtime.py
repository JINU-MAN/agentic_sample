from __future__ import annotations

import json
from contextvars import ContextVar, Token
from typing import Any, Dict, List


_ACTIVE_WORKFLOW_MEMORY: ContextVar[Dict[str, Any] | None] = ContextVar(
    "agentic_active_workflow_memory",
    default=None,
)


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_completed_steps(items: Any, *, max_items: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items[-max_items:]:
        if not isinstance(item, dict):
            continue
        entry: Dict[str, Any] = {
            "workflow_step": item.get("workflow_step"),
            "agent": str(item.get("agent", "")).strip(),
            "goal": _compact_text(item.get("goal", ""), max_chars=220),
            "ok": bool(item.get("ok")),
        }
        response = str(item.get("raw_response") or item.get("response") or item.get("error") or "").strip()
        if response:
            entry["response"] = _compact_text(response, max_chars=2400)
        artifact_ids = item.get("artifact_ids")
        if isinstance(artifact_ids, list) and artifact_ids:
            entry["artifact_ids"] = [str(value).strip() for value in artifact_ids[:8] if str(value).strip()]
        normalized.append(entry)
    return normalized


def _normalize_artifacts(items: Any, *, max_items: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items[-max_items:]:
        if not isinstance(item, dict):
            continue
        entry: Dict[str, Any] = {
            "id": str(item.get("id", "")).strip(),
            "type": str(item.get("type", "")).strip() or "note",
            "title": _compact_text(item.get("title", ""), max_chars=180),
            "summary": _compact_text(item.get("summary", ""), max_chars=320),
            "source_agent": str(item.get("source_agent", "")).strip(),
            "workflow_step": item.get("workflow_step"),
        }
        url = str(item.get("url", "")).strip()
        if url:
            entry["url"] = _compact_text(url, max_chars=320)
        identifiers = item.get("identifiers")
        if isinstance(identifiers, dict) and identifiers:
            entry["identifiers"] = {
                str(key): _compact_text(value, max_chars=120)
                for key, value in identifiers.items()
                if str(key).strip() and str(value).strip()
            }
        normalized.append(entry)
    return normalized


def _normalize_pending_steps(items: Any, *, max_items: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "agent": str(item.get("agent", "")).strip(),
                "goal": _compact_text(item.get("goal", ""), max_chars=220),
                "deliverable": _compact_text(item.get("deliverable", ""), max_chars=180),
            }
        )
    return normalized


def _normalize_open_needs(items: Any, *, max_items: int) -> List[str]:
    if not isinstance(items, list):
        return []
    return [_compact_text(item, max_chars=220) for item in items[:max_items] if str(item).strip()]


def _normalize_agent_cards(items: Any, *, max_items: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "name": str(item.get("name", "")).strip(),
                "role": str(item.get("role", "")).strip() or "worker",
                "description": _compact_text(item.get("description", ""), max_chars=220),
                "capability_summary": _compact_text(item.get("capability_summary", ""), max_chars=220),
                "ownership": _compact_text(item.get("ownership", ""), max_chars=220),
            }
        )
    return normalized


def normalize_workflow_memory_snapshot(snapshot: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    return {
        "workflow_id": str(snapshot.get("workflow_id", "")).strip(),
        "current_step": snapshot.get("current_step"),
        "current_agent": str(snapshot.get("current_agent", "")).strip(),
        "user_request": _compact_text(snapshot.get("user_request", ""), max_chars=500),
        "completed_steps": _normalize_completed_steps(snapshot.get("completed_steps"), max_items=8),
        "artifacts": _normalize_artifacts(snapshot.get("artifacts"), max_items=10),
        "open_needs": _normalize_open_needs(snapshot.get("open_needs"), max_items=10),
        "pending_steps": _normalize_pending_steps(snapshot.get("pending_steps"), max_items=6),
        "activated_agents": _normalize_agent_cards(snapshot.get("activated_agents"), max_items=8),
    }


def set_active_workflow_memory(snapshot: Dict[str, Any] | None) -> Token:
    return _ACTIVE_WORKFLOW_MEMORY.set(normalize_workflow_memory_snapshot(snapshot))


def reset_active_workflow_memory(token: Token) -> None:
    _ACTIVE_WORKFLOW_MEMORY.reset(token)


def get_active_workflow_memory() -> Dict[str, Any]:
    return normalize_workflow_memory_snapshot(_ACTIVE_WORKFLOW_MEMORY.get())


def render_active_workflow_memory(query: str = "", max_items: int = 6) -> str:
    snapshot = get_active_workflow_memory()
    if not snapshot or not snapshot.get("workflow_id"):
        return "No active workflow memory is available for this step."

    max_items = max(1, min(int(max_items or 6), 12))
    lowered = str(query or "").strip().lower()
    focused_on_artifacts = any(token in lowered for token in ["artifact", "url", "source", "search result", "result"])
    focused_on_needs = any(token in lowered for token in ["need", "blocker", "missing", "context"])

    payload: Dict[str, Any] = {
        "workflow_id": snapshot.get("workflow_id"),
        "current_step": snapshot.get("current_step"),
        "current_agent": snapshot.get("current_agent"),
        "user_request": snapshot.get("user_request"),
    }
    if lowered:
        payload["query"] = _compact_text(query, max_chars=180)

    completed_steps = list(snapshot.get("completed_steps", []))
    artifacts = list(snapshot.get("artifacts", []))
    open_needs = list(snapshot.get("open_needs", []))
    pending_steps = list(snapshot.get("pending_steps", []))
    activated_agents = list(snapshot.get("activated_agents", []))

    if focused_on_artifacts:
        payload["artifacts"] = artifacts[-max_items:]
        payload["completed_steps"] = completed_steps[-max_items:]
    elif focused_on_needs:
        payload["open_needs"] = open_needs[:max_items]
        payload["pending_steps"] = pending_steps[:max_items]
        payload["completed_steps"] = completed_steps[-max_items:]
    else:
        payload["completed_steps"] = completed_steps[-max_items:]
        payload["artifacts"] = artifacts[-max_items:]
        payload["open_needs"] = open_needs[:max_items]
        payload["pending_steps"] = pending_steps[:max_items]

    if activated_agents:
        payload["activated_agents"] = activated_agents[:max_items]
    return json.dumps(payload, ensure_ascii=False, indent=2)
