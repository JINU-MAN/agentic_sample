from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

from agentic_sample_ad.system_logger import log_event, log_exception


BASE_DIR = Path(__file__).resolve().parents[2]
AGENT_CARDS_DIR = BASE_DIR / "agent_cards"
DEFAULT_WORKFLOW_ID = "default"


def _run_coroutine_sync(coro: Any) -> Any:
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
        except Exception as e:  # pragma: no cover
            error = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _http_timeout(*, connect_sec: float, read_sec: float, write_sec: float, pool_sec: float) -> httpx.Timeout:
    return httpx.Timeout(connect=connect_sec, read=read_sec, write=write_sec, pool=pool_sec)


def _load_cards() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    if not AGENT_CARDS_DIR.exists():
        return cards

    for path in sorted(AGENT_CARDS_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                cards.extend(item for item in data if isinstance(item, dict))
            elif isinstance(data, dict):
                cards.append(data)
        except Exception as e:
            log_exception(
                "tool.delegate_to_agent_via_a2a",
                "agent_cards_load_failed",
                e,
                {"path": str(path)},
            )
    return cards


def _index_a2a_agents(cards: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    indexed: Dict[str, Dict[str, str]] = {}
    for card in cards:
        name = str(card.get("name", "")).strip()
        base_url = str(card.get("base_url", "")).strip()
        agent_type = str(card.get("type", "")).strip().lower()
        if not name or not base_url:
            continue
        if agent_type and agent_type != "a2a":
            continue
        key = name.lower()
        if key in indexed:
            continue
        indexed[key] = {"name": name, "base_url": base_url}
    return indexed


def _extract_text_from_parts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    chunks: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks).strip()


def _extract_a2a_text(payload: Dict[str, Any]) -> str:
    result = payload.get("result")
    if isinstance(result, dict):
        direct = _extract_text_from_parts(result.get("parts"))
        if direct:
            return direct
        status = result.get("status")
        if isinstance(status, dict):
            status_message = status.get("message")
            if isinstance(status_message, dict):
                status_text = _extract_text_from_parts(status_message.get("parts"))
                if status_text:
                    return status_text

    root = payload.get("root")
    if isinstance(root, dict):
        root_text = _extract_a2a_text(root)
        if root_text:
            return root_text

    error = payload.get("error")
    if isinstance(error, dict):
        error_message = error.get("message")
        if isinstance(error_message, str) and error_message.strip():
            return error_message.strip()
    return ""


async def _async_call_a2a_agent(base_url: str, user_message: str) -> Dict[str, Any]:
    card_url = f"{str(base_url).rstrip('/')}/.well-known/agent-card.json"
    connect_timeout_sec = _env_float("A2A_CONNECT_TIMEOUT_SEC", 3.0, 0.2)
    card_timeout_sec = _env_float("A2A_CARD_TIMEOUT_SEC", 6.0, 0.5)
    request_timeout_sec = _env_float("A2A_REQUEST_TIMEOUT_SEC", 120.0, 2.0)
    write_timeout_sec = _env_float("A2A_WRITE_TIMEOUT_SEC", 30.0, 1.0)
    pool_timeout_sec = _env_float("A2A_POOL_TIMEOUT_SEC", 30.0, 1.0)
    card_retry_count = _env_int("A2A_CARD_RETRY_COUNT", 3, 1)
    card_retry_delay_sec = _env_float("A2A_CARD_RETRY_DELAY_SEC", 0.35, 0.05)

    base_timeout = _http_timeout(
        connect_sec=connect_timeout_sec,
        read_sec=request_timeout_sec,
        write_sec=write_timeout_sec,
        pool_sec=pool_timeout_sec,
    )
    card_timeout = _http_timeout(
        connect_sec=connect_timeout_sec,
        read_sec=card_timeout_sec,
        write_sec=write_timeout_sec,
        pool_sec=pool_timeout_sec,
    )

    async with httpx.AsyncClient(timeout=base_timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = None
        last_card_error: Exception | None = None
        for attempt in range(1, card_retry_count + 1):
            try:
                agent_card = await resolver.get_agent_card(http_kwargs={"timeout": card_timeout})
                break
            except Exception as e:  # pragma: no cover
                last_card_error = e
                if attempt >= card_retry_count:
                    raise
                await asyncio.sleep(card_retry_delay_sec * attempt)
        if agent_card is None and last_card_error is not None:
            raise last_card_error

        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        payload: Dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": user_message}],
                "messageId": uuid4().hex,
            }
        }
        request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
        response = await client.send_message(
            request,
            http_kwargs={
                "timeout": _http_timeout(
                    connect_sec=connect_timeout_sec,
                    read_sec=request_timeout_sec,
                    write_sec=write_timeout_sec,
                    pool_sec=pool_timeout_sec,
                )
            },
        )
        return response.model_dump(mode="json", exclude_none=True)


def _normalize_workflow_id(workflow_id: str) -> str:
    value = str(workflow_id or "").strip()
    if not value:
        return DEFAULT_WORKFLOW_ID
    return value[:80]


def _build_delegate_message(
    *,
    caller_name: str,
    target_name: str,
    workflow_id: str,
    task: str,
) -> str:
    return (
        f"Delegation request from {caller_name} to {target_name}.\n"
        f"Workflow ID: {workflow_id}\n\n"
        "Please handle only the delegated task and return handoff-ready output.\n"
        "Include key evidence and source URLs when possible.\n\n"
        "Delegated task:\n"
        f"{task.strip()}\n\n"
        "Output format:\n"
        "1) Main response for delegated task.\n"
        "2) Additional Needs section:\n"
        "   - `Additional Needs: none` or\n"
        "   - `Additional Needs:` with bullets `[TargetAgentName] request`."
    )


def build_delegate_to_agent_tool(caller_agent_name: str):
    caller = str(caller_agent_name or "").strip() or "UnknownAgent"

    def delegate_to_agent_via_a2a(
        target_agent_name: str,
        task: str,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
    ) -> str:
        """
        Delegate a sub-task to another A2A agent and return its response as JSON.
        """
        target = str(target_agent_name or "").strip()
        safe_task = str(task or "").strip()
        safe_workflow_id = _normalize_workflow_id(workflow_id)
        log_event(
            "tool.delegate_to_agent_via_a2a",
            "call_started",
            {
                "caller_agent": caller,
                "target_agent": target,
                "workflow_id": safe_workflow_id,
                "task": safe_task,
            },
            direction="outbound",
        )

        try:
            if not target:
                payload = {
                    "ok": False,
                    "caller_agent": caller,
                    "target_agent": "",
                    "workflow_id": safe_workflow_id,
                    "error": "target_agent_name is required.",
                }
                return json.dumps(payload, ensure_ascii=False, indent=2)
            if not safe_task:
                payload = {
                    "ok": False,
                    "caller_agent": caller,
                    "target_agent": target,
                    "workflow_id": safe_workflow_id,
                    "error": "task is required.",
                }
                return json.dumps(payload, ensure_ascii=False, indent=2)

            if caller.lower() == target.lower():
                payload = {
                    "ok": False,
                    "caller_agent": caller,
                    "target_agent": target,
                    "workflow_id": safe_workflow_id,
                    "error": "Self delegation is blocked.",
                }
                return json.dumps(payload, ensure_ascii=False, indent=2)

            cards = _load_cards()
            indexed = _index_a2a_agents(cards)
            target_meta = indexed.get(target.lower())
            if target_meta is None:
                payload = {
                    "ok": False,
                    "caller_agent": caller,
                    "target_agent": target,
                    "workflow_id": safe_workflow_id,
                    "error": "Target agent not found in A2A cards.",
                    "available_agents": [item.get("name", "") for item in indexed.values()],
                }
                return json.dumps(payload, ensure_ascii=False, indent=2)

            resolved_target_name = str(target_meta.get("name", target))
            base_url = str(target_meta.get("base_url", "")).strip()
            delegated_message = _build_delegate_message(
                caller_name=caller,
                target_name=resolved_target_name,
                workflow_id=safe_workflow_id,
                task=safe_task,
            )

            raw_result = _run_coroutine_sync(_async_call_a2a_agent(base_url=base_url, user_message=delegated_message))
            response_text = _extract_a2a_text(raw_result).strip()
            payload = {
                "ok": True,
                "caller_agent": caller,
                "target_agent": resolved_target_name,
                "workflow_id": safe_workflow_id,
                "base_url": base_url,
                "delegated_task": safe_task,
                "response": response_text or json.dumps(raw_result, ensure_ascii=False),
                "raw_a2a_result": raw_result,
            }
            log_event(
                "tool.delegate_to_agent_via_a2a",
                "call_completed",
                {
                    "caller_agent": caller,
                    "target_agent": resolved_target_name,
                    "workflow_id": safe_workflow_id,
                    "ok": True,
                },
                direction="inbound",
            )
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception as e:
            log_exception(
                "tool.delegate_to_agent_via_a2a",
                "call_failed",
                e,
                {
                    "caller_agent": caller,
                    "target_agent": target,
                    "workflow_id": safe_workflow_id,
                    "task": safe_task,
                },
            )
            payload = {
                "ok": False,
                "caller_agent": caller,
                "target_agent": target,
                "workflow_id": safe_workflow_id,
                "error_type": type(e).__name__,
                "error": str(e),
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

    delegate_to_agent_via_a2a.__name__ = "delegate_to_agent_via_a2a"
    delegate_to_agent_via_a2a.__doc__ = (
        "Delegate a sub-task to another agent through A2A. "
        "Arguments: target_agent_name, task, workflow_id(optional). "
        "Returns JSON string with delegated response."
    )
    return delegate_to_agent_via_a2a


def build_list_delegatable_agents_tool(caller_agent_name: str):
    caller = str(caller_agent_name or "").strip() or "UnknownAgent"

    def list_delegatable_agents() -> str:
        """
        List available A2A agents that can receive delegated tasks.
        """
        cards = _load_cards()
        indexed = _index_a2a_agents(cards)
        rows: List[Dict[str, str]] = []
        for item in indexed.values():
            name = str(item.get("name", "")).strip()
            if not name or name.lower() == caller.lower():
                continue
            rows.append(
                {
                    "name": name,
                    "base_url": str(item.get("base_url", "")).strip(),
                }
            )
        payload = {
            "ok": True,
            "caller_agent": caller,
            "agents": rows,
            "count": len(rows),
        }
        log_event(
            "tool.list_delegatable_agents",
            "call_completed",
            {"caller_agent": caller, "count": len(rows)},
            direction="inbound",
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    list_delegatable_agents.__name__ = "list_delegatable_agents"
    list_delegatable_agents.__doc__ = (
        "List A2A agents available for delegation from this agent's perspective."
    )
    return list_delegatable_agents


__all__ = [
    "build_delegate_to_agent_tool",
    "build_list_delegatable_agents_tool",
]

