from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from google.adk.runners import InMemoryRunner
from google.genai import types

from agentic_sample_ad.agent_metadata_utils import (
    agent_capability_keys as _agent_capability_keys,
    agent_capability_summary_text as _agent_capability_summary_text,
    agent_ownership_text as _agent_ownership_text,
    format_available_agents_for_prompt,
)
from agentic_sample_ad.network_retry import (
    collect_response_parts_with_network_retry,
    collect_text_response_with_network_retry,
)
from agentic_sample_ad.main_agent.system_logger import log_event, log_exception
from agentic_sample_ad.runtime_utils import (
    EMPTY_RESPONSE_SENTINELS,
    extract_json_object as _extract_json_object,
    extract_json_object_with_source as _extract_json_object_with_source,
    finalize_text_response,
    run_coroutine_sync as _run_coroutine_sync,
)
from agentic_sample_ad.handoff_contract_tool import recover_handoff_contract_from_parts
from agentic_sample_ad.tool_output_utils import TOOL_OUTPUT_CONTRACT
from agentic_sample_ad.workflow_memory_runtime import (
    reset_active_workflow_memory,
    set_active_workflow_memory,
)


AGENT_MESSAGE_COMPONENT = "event_manager.agent_message"
PROGRESS_REVIEW_BLOCKER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:cannot|can't|unable to|do not have|don't have|missing|not found|insufficient)\b",
        r"\b(?:need(?:s)?|require(?:s|d)?)\b.{0,60}\b(?:context|results|artifacts|workflow memory|channel|clarification|access)\b",
        r"\bchannel_not_found\b",
        r"\bskills?\b.{0,40}\bnot found\b",
    )
]
STRICT_JSON_FENCE_ONLY_RE = re.compile(r"\A\s*```(?:json)?\s*(\{.*\})\s*```\s*\Z", re.DOTALL)
INTERNAL_BOOTSTRAP_TEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bload(?:ing|ed)?\b.{0,40}\bskill\b",
        r"\bskill\b.{0,40}\b(?:not loaded|needs to be loaded|must be loaded|required to be loaded)\b",
        r"\bload(?:ing|ed)?\b.{0,40}\bsession memory\b",
        r"\bsession memory\b.{0,40}\b(?:not loaded|needs to be loaded|must be loaded)\b",
        r"스킬.{0,24}(?:로드|불러)",
        r"(?:로드|불러).{0,24}스킬",
        r"세션 메모리.{0,24}(?:로드|불러)",
        r"(?:로드|불러).{0,24}세션 메모리",
    )
]
INTERNAL_BOOTSTRAP_CAPABILITY_KEYS = {"load_skill", "load_session_memory"}


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value < minimum:
        return minimum
    return value


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < minimum:
        return minimum
    return value


def _resolve_same_task_review_threshold() -> int:
    return _env_int("AGENTIC_SAME_TASK_REVIEW_THRESHOLD", default=2, minimum=1)


def _format_available_agents_for_review(available_agents: List[Dict[str, Any]]) -> str:
    return format_available_agents_for_prompt(available_agents, empty_text="(none)")


def _resolve_collaboration_max_steps(initial_steps: int) -> int:
    """
    Resolve collaboration max-steps with a generous default for complex workflows.
    Environment override:
      - COLLAB_MAX_STEPS
    """
    base = max(1, int(initial_steps))
    default_steps = max(40, base * 6 + 12)
    return _env_int("COLLAB_MAX_STEPS", default_steps, 8)


def _a2a_http_timeout(
    *,
    connect_timeout_sec: float,
    read_timeout_sec: float,
    write_timeout_sec: float,
    pool_timeout_sec: float,
) -> httpx.Timeout:
    return httpx.Timeout(
        connect=connect_timeout_sec,
        read=read_timeout_sec,
        write=write_timeout_sec,
        pool=pool_timeout_sec,
    )


def _is_a2a_agent(agent_meta: Dict[str, Any]) -> bool:
    return str(agent_meta.get("type", "")).strip().lower() == "a2a"


def _is_local_agent(agent_meta: Dict[str, Any]) -> bool:
    return str(agent_meta.get("type", "")).strip().lower() == "local"


def _message_preview(text: str, max_chars: int = 1200) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "...(truncated)"


def _log_agent_message(
    *,
    action: str,
    from_agent: str,
    to_agent: str,
    message: str,
    channel: str,
    workflow_id: str = "",
    workflow_step: int | None = None,
    ok: bool | None = None,
    direction: str = "internal",
) -> None:
    raw = str(message or "")
    payload: Dict[str, Any] = {
        "from_agent": str(from_agent or "").strip() or "UnknownAgent",
        "to_agent": str(to_agent or "").strip() or "UnknownAgent",
        "channel": str(channel or "").strip() or "unknown",
        "message_length": len(raw),
        "message": raw,
        "message_preview": _message_preview(raw),
    }
    if workflow_id.strip():
        payload["workflow_id"] = workflow_id.strip()
    if isinstance(workflow_step, int):
        payload["workflow_step"] = workflow_step
    if isinstance(ok, bool):
        payload["ok"] = ok

    log_event(AGENT_MESSAGE_COMPONENT, action, payload, direction=direction)


async def _async_call_a2a_agent(
    base_url: str,
    user_message: str,
) -> Dict[str, Any]:
    card_url = f"{str(base_url).rstrip('/')}/.well-known/agent-card.json"
    connect_timeout_sec = _env_float("A2A_CONNECT_TIMEOUT_SEC", 3.0, 0.2)
    card_timeout_sec = _env_float("A2A_CARD_TIMEOUT_SEC", 6.0, 0.5)
    request_timeout_sec = _env_float("A2A_REQUEST_TIMEOUT_SEC", 120.0, 2.0)
    write_timeout_sec = _env_float("A2A_WRITE_TIMEOUT_SEC", 30.0, 1.0)
    pool_timeout_sec = _env_float("A2A_POOL_TIMEOUT_SEC", 30.0, 1.0)
    card_retry_count = _env_int("A2A_CARD_RETRY_COUNT", 3, 1)
    card_retry_delay_sec = _env_float("A2A_CARD_RETRY_DELAY_SEC", 0.35, 0.05)
    log_event(
        "event_manager.a2a",
        "request_started",
        {
            "base_url": base_url,
            "card_url": card_url,
            "user_message": user_message,
            "timeouts": {
                "connect_sec": connect_timeout_sec,
                "card_read_sec": card_timeout_sec,
                "request_read_sec": request_timeout_sec,
                "write_sec": write_timeout_sec,
                "pool_sec": pool_timeout_sec,
            },
            "card_retry_count": card_retry_count,
        },
        direction="outbound",
    )
    base_timeout = _a2a_http_timeout(
        connect_timeout_sec=connect_timeout_sec,
        read_timeout_sec=request_timeout_sec,
        write_timeout_sec=write_timeout_sec,
        pool_timeout_sec=pool_timeout_sec,
    )
    card_timeout = _a2a_http_timeout(
        connect_timeout_sec=connect_timeout_sec,
        read_timeout_sec=card_timeout_sec,
        write_timeout_sec=write_timeout_sec,
        pool_timeout_sec=pool_timeout_sec,
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
                log_exception(
                    "event_manager.a2a",
                    "agent_card_fetch_retry",
                    e,
                    {
                        "base_url": base_url,
                        "card_url": card_url,
                        "attempt": attempt,
                        "max_attempts": card_retry_count,
                    },
                )
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

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload),
        )
        log_event(
            "event_manager.a2a",
            "request_payload_built",
            {"base_url": base_url, "payload": payload},
            direction="outbound",
        )
        response = await client.send_message(
            request,
            http_kwargs={
                "timeout": _a2a_http_timeout(
                    connect_timeout_sec=connect_timeout_sec,
                    read_timeout_sec=request_timeout_sec,
                    write_timeout_sec=write_timeout_sec,
                    pool_timeout_sec=pool_timeout_sec,
                )
            },
        )
        dumped = response.model_dump(mode="json", exclude_none=True)
        log_event(
            "event_manager.a2a",
            "response_received",
            {"base_url": base_url, "response": dumped},
            direction="inbound",
        )
        return dumped


def _collect_text_fragments_from_payload(value: Any, out: List[str]) -> None:
    if len(out) >= 80:
        return

    if isinstance(value, str):
        return

    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            out.append(text.strip())
        for key, item in value.items():
            if key == "text":
                continue
            _collect_text_fragments_from_payload(item, out)
        return

    if isinstance(value, list):
        for item in value:
            _collect_text_fragments_from_payload(item, out)


def _extract_text_from_message_parts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    chunks: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            token = text.strip()
            if token:
                chunks.append(token)
    return "\n".join(chunks).strip()


def _extract_a2a_message_text(payload: Dict[str, Any]) -> str:
    result = payload.get("result")
    if isinstance(result, dict):
        direct = _extract_text_from_message_parts(result.get("parts"))
        if direct:
            return direct

        status = result.get("status")
        if isinstance(status, dict):
            status_message = status.get("message")
            if isinstance(status_message, dict):
                from_status = _extract_text_from_message_parts(status_message.get("parts"))
                if from_status:
                    return from_status

    root = payload.get("root")
    if isinstance(root, dict):
        from_root = _extract_a2a_message_text(root)
        if from_root:
            return from_root

    error = payload.get("error")
    if isinstance(error, dict):
        error_message = error.get("message")
        if isinstance(error_message, str) and error_message.strip():
            return error_message.strip()

    return ""


def _extract_a2a_error_text(payload: Dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = str(error.get("message", "")).strip()
        code = error.get("code")
        if message and code is not None:
            return f"[code={code}] {message}"
        if message:
            return message

    root = payload.get("root")
    if isinstance(root, dict):
        from_root = _extract_a2a_error_text(root)
        if from_root:
            return from_root

    return ""


def _extract_a2a_response_text(payload: Dict[str, Any]) -> str:
    direct = _extract_a2a_message_text(payload)
    if direct:
        return direct

    fragments: List[str] = []
    _collect_text_fragments_from_payload(payload, fragments)

    deduped: List[str] = []
    seen: set[str] = set()
    for item in fragments:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= 40:
            break

    return "\n".join(deduped).strip()


def _extract_a2a_response_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = payload.get("result")
    if isinstance(result, dict):
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            return metadata
    root = payload.get("root")
    if isinstance(root, dict):
        return _extract_a2a_response_metadata(root)
    return {}


def _a2a_card_url(base_url: str) -> str:
    return f"{str(base_url).rstrip('/')}/.well-known/agent-card.json"


def _execute_single_a2a_agent(agent_meta: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    name = str(agent_meta.get("name", "UnknownA2AAgent")).strip() or "UnknownA2AAgent"
    base_url = agent_meta.get("base_url")
    card_url = _a2a_card_url(str(base_url or ""))
    started_at = time.monotonic()
    if not base_url:
        log_event(
            "event_manager.a2a",
            "request_skipped",
            {"agent_meta": agent_meta, "reason": "missing_base_url", "card_url": card_url},
            level="ERROR",
        )
        return {
            "ok": False,
            "agent": name,
            "error": "A2A agent has no base_url configured.",
        }

    try:
        raw_result = _run_coroutine_sync(_async_call_a2a_agent(base_url=base_url, user_message=user_input))
        elapsed_sec = round(time.monotonic() - started_at, 3)
        error_text = _extract_a2a_error_text(raw_result)
        if error_text:
            result = {
                "ok": False,
                "agent": name,
                "error": error_text,
                "raw_a2a_result": raw_result,
                "elapsed_sec": elapsed_sec,
            }
            log_event(
                "event_manager.a2a",
                "request_failed",
                {"base_url": base_url, "result": result},
                direction="inbound",
                level="ERROR",
            )
            return result

        response_text = _extract_a2a_response_text(raw_result)
        if not response_text:
            response_text = json.dumps(raw_result, ensure_ascii=False)
        response_metadata = _extract_a2a_response_metadata(raw_result)
        result = {
            "ok": True,
            "agent": name,
            "response": response_text,
            "raw_a2a_result": raw_result,
            "elapsed_sec": elapsed_sec,
            "normalized_response_parts": list(response_metadata.get("normalized_response_parts", []))
            if isinstance(response_metadata, dict)
            else [],
            "non_text_part_types": list(response_metadata.get("non_text_part_types", []))
            if isinstance(response_metadata, dict)
            else [],
            "used_non_text_fallback": bool(response_metadata.get("used_non_text_fallback"))
            if isinstance(response_metadata, dict)
            else False,
        }
        log_event("event_manager.a2a", "request_completed", {"base_url": base_url, "result": result})
        return result
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).strip()
        error_lower = error_msg.lower()
        is_timeout = "timeout" in error_type.lower() or "timed out" in error_lower
        is_card_fetch_error = "agent card" in error_lower or "/.well-known/agent-card.json" in error_lower
        request_timeout_sec = _env_float("A2A_REQUEST_TIMEOUT_SEC", 120.0, 2.0)
        elapsed_sec = round(time.monotonic() - started_at, 3)

        if is_card_fetch_error:
            error_text = (
                f"A2A agent card fetch failed ({error_type}): {error_msg}. "
                f"Card endpoint: {card_url}. "
                "Agent server may not be ready or has stopped. "
                "Run `python start_agentic.py` again, or `python scripts/start_a2a_agents.py` and verify the card endpoint."
            )
        elif is_timeout:
            error_text = (
                f"A2A agent request timed out ({error_type}): {error_msg}. "
                f"Agent base URL: {base_url}. "
                "The agent server received the request but did not finish within timeout. "
                "Increase `A2A_REQUEST_TIMEOUT_SEC` for longer tasks."
            )
        else:
            error_text = (
                f"A2A agent execution failed ({error_type}): {error_msg}. "
                f"Agent base URL: {base_url}. "
                "Check agent server health and network access."
            )
        log_exception(
            "event_manager.a2a",
            "request_failed",
            e,
            {"base_url": base_url, "user_input": user_input, "agent": name},
        )
        return {
            "ok": False,
            "agent": name,
            "error": error_text,
            "error_type": error_type,
            "error_kind": "timeout" if is_timeout else ("agent_card_fetch" if is_card_fetch_error else "execution"),
            "is_timeout": is_timeout,
            "timeout_sec": request_timeout_sec if is_timeout else None,
            "elapsed_sec": elapsed_sec,
        }


async def _async_run_local_agent(
    agent_obj: Any,
    agent_name: str,
    user_input: str,
) -> Dict[str, Any]:
    log_event(
        "event_manager.local_agent",
        "execution_started",
        {"agent": agent_name, "user_input": user_input},
        direction="outbound",
    )
    runner = InMemoryRunner(agent=agent_obj, app_name=f"local-{agent_name or 'agent'}")
    try:
        new_message = types.Content(role="user", parts=[types.Part(text=user_input)])
        def _on_text(text: str, event: Any) -> None:
            author = str(getattr(event, "author", "unknown"))
            log_event(
                "event_manager.local_agent",
                "message_chunk",
                {"agent": agent_name, "author": author, "text": text},
                direction="inbound",
            )

        collected = await collect_response_parts_with_network_retry(
            runner=runner,
            user_id="event-manager-user",
            new_message=new_message,
            component="event_manager.local_agent",
            operation_name=f"local_agent:{agent_name}",
            retry_details={"agent": agent_name, "user_input": user_input},
            on_text=_on_text,
            log_event_fn=log_event,
        )
        chunks = list(collected.get("chunks", []))

        response_text = finalize_text_response(chunks)
        if collected.get("used_non_text_fallback"):
            recovered = recover_handoff_contract_from_parts(
                list(collected.get("normalized_parts", []))
            )
            if recovered:
                response_text = recovered
        log_event(
            "event_manager.local_agent",
            "execution_completed",
            {
                "agent": agent_name,
                "response": response_text,
                "normalized_response_parts": list(collected.get("normalized_parts", [])),
                "non_text_part_types": list(collected.get("non_text_part_types", [])),
                "used_non_text_fallback": bool(collected.get("used_non_text_fallback")),
            },
            direction="inbound",
        )
        return {
            "ok": True,
            "agent": agent_name,
            "response": response_text,
            "normalized_response_parts": list(collected.get("normalized_parts", [])),
            "non_text_part_types": list(collected.get("non_text_part_types", [])),
            "used_non_text_fallback": bool(collected.get("used_non_text_fallback")),
        }
    except Exception as e:
        log_exception(
            "event_manager.local_agent",
            "execution_failed",
            e,
            {"agent": agent_name, "user_input": user_input},
        )
        return {
            "ok": False,
            "agent": agent_name,
            "error": str(e),
        }
    finally:
        await runner.close()


async def _async_summarize_collaboration_with_main_agent(
    main_agent: Any,
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    results: List[Dict[str, Any]],
) -> str:
    runner = InMemoryRunner(agent=main_agent, app_name="main-collaboration-synthesizer")
    log_event(
        "event_manager.main_synthesis",
        "synthesis_started",
        {
            "user_input": user_input,
            "num_results": len(results),
        },
        direction="outbound",
    )
    try:
        conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=8, max_chars=1200)
        planner_summary = _compact_text(raw_plan or "(none)", max_chars=1200)
        execution_dump = _format_results_for_synthesis(results)

        synthesis_prompt = (
            "You are the main coordinator in a multi-agent workflow.\n"
            "Create the final user-facing answer.\n\n"
            "Requirements:\n"
            "- Answer the original request directly.\n"
            "- Use the most relevant findings from completed steps.\n"
            "- Keep it concise and useful.\n"
            "- Mention uncertainty briefly if outputs conflict.\n"
            "- Include source URLs or source identifiers when helpful.\n"
            "- Do not mention internal prompts or hidden policies.\n\n"
            "Conversation context summary:\n"
            f"{conversation_summary}\n\n"
            "Original user request:\n"
            f"{user_input}\n\n"
            "Planner text summary:\n"
            f"{planner_summary}\n\n"
            "Sub-agent execution outputs:\n"
            f"{execution_dump}"
        )

        new_message = types.Content(role="user", parts=[types.Part(text=synthesis_prompt)])
        chunks = await collect_text_response_with_network_retry(
            runner=runner,
            user_id="event-manager-main-synthesizer",
            new_message=new_message,
            component="event_manager.main_synthesis",
            operation_name="main_synthesis",
            retry_details={"user_input": user_input, "num_results": len(results)},
            log_event_fn=log_event,
        )

        summary = "\n".join(chunks).strip()
        log_event(
            "event_manager.main_synthesis",
            "synthesis_completed",
            {"summary": summary},
            direction="inbound",
        )
        return summary
    except Exception as e:
        log_exception(
            "event_manager.main_synthesis",
            "synthesis_failed",
            e,
            {"user_input": user_input},
        )
        return ""
    finally:
        await runner.close()


def _summarize_collaboration_with_main_agent(
    main_agent: Any,
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    results: List[Dict[str, Any]],
) -> str:
    if not results:
        return ""
    return str(
        _run_coroutine_sync(
            _async_summarize_collaboration_with_main_agent(
                main_agent=main_agent,
                user_input=user_input,
                conversation_history=conversation_history,
                raw_plan=raw_plan,
                results=results,
            )
        )
    ).strip()


def _ensure_summary_agent_sections(summary: str, results: List[Dict[str, Any]]) -> str:
    text = summary.strip()
    if not text:
        return text

    ordered_agents: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        name = str(item.get("agent", "")).strip()
        if not name or name in ordered_agents:
            continue
        ordered_agents.append(name)

    if len(ordered_agents) <= 1:
        return text

    text_lower = text.lower()
    missing_agents = [name for name in ordered_agents if name.lower() not in text_lower]
    if not missing_agents:
        return text

    notes: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent", "")).strip()
        if not agent or agent not in missing_agents:
            continue
        if item.get("ok"):
            body = str(item.get("response", "")).strip()
        else:
            body = str(item.get("error", "Unknown error")).strip()
        excerpt = " ".join(body.split())[:280]
        notes.append(f"- {agent}: {excerpt or '(no details)'}")

    if not notes:
        return text
    return text + "\n\nAgent Coverage:\n" + "\n".join(notes)


def _execute_single_local_agent(
    agent_meta: Dict[str, Any],
    user_input: str,
    *,
    workflow_memory_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    name = str(agent_meta.get("name", "UnknownLocalAgent")).strip() or "UnknownLocalAgent"
    runtime_agent_obj = agent_meta.get("agent_obj")
    log_event(
        "event_manager.local_agent",
        "metadata_loaded",
        {
            "agent": name,
            "has_runtime_agent_obj": runtime_agent_obj is not None,
        },
    )

    if runtime_agent_obj is None:
        error = "Local agent runtime object is missing."
        log_event("event_manager.local_agent", "metadata_invalid", {"agent": name, "reason": error}, level="ERROR")
        return {
            "ok": False,
            "agent": name,
            "error": error,
        }

    workflow_memory_token = set_active_workflow_memory(workflow_memory_snapshot)
    try:
        result = _run_coroutine_sync(
            _async_run_local_agent(agent_obj=runtime_agent_obj, agent_name=name, user_input=user_input)
        )
        log_event(
            "event_manager.local_agent",
            "execution_returned",
            {"agent": name, "result": result, "execution_source": "runtime_agent_obj"},
        )
        return result
    except Exception as e:
        log_exception(
            "event_manager.local_agent",
            "execution_crashed",
            e,
            {"agent": name, "execution_source": "runtime_agent_obj"},
        )
        return {
            "ok": False,
            "agent": name,
            "error": str(e),
        }
    finally:
        reset_active_workflow_memory(workflow_memory_token)


def _execute_single_agent(
    agent_meta: Dict[str, Any],
    user_input: str,
    *,
    workflow_memory_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if _is_local_agent(agent_meta):
        return _execute_single_local_agent(
            agent_meta,
            user_input,
            workflow_memory_snapshot=workflow_memory_snapshot,
        )
    if _is_a2a_agent(agent_meta):
        return _execute_single_a2a_agent(agent_meta, user_input)

    name = str(agent_meta.get("name", "UnknownAgent")).strip() or "UnknownAgent"
    error = f"Unsupported agent type: {agent_meta.get('type')}"
    log_event(
        "event_manager",
        "agent_execution_skipped",
        {"agent": name, "reason": error, "agent_meta": agent_meta},
        level="ERROR",
    )
    return {"ok": False, "agent": name, "error": error}


def _format_execution_output(raw_plan: str, results: List[Dict[str, Any]]) -> str:
    parts: List[str] = []

    if raw_plan.strip():
        parts.append("=== Plan ===\n" + raw_plan.strip())

    parts.append("=== Execution Results ===")
    for result in results:
        agent_name = str(result.get("agent", "UnknownAgent"))
        step_index = result.get("workflow_step")
        goal = str(result.get("goal", "")).strip()
        if isinstance(step_index, int):
            header = f"[Step {step_index} - {agent_name}]"
        else:
            header = f"[{agent_name}]"
        if result.get("ok"):
            artifact_ids = [str(item) for item in result.get("artifact_ids", []) if str(item).strip()]
            artifact_line = f"\nArtifacts: {len(artifact_ids)}" if artifact_ids else ""
            if goal:
                parts.append(f"{header}\nGoal: {goal}{artifact_line}\n{result.get('response', '')}")
            else:
                parts.append(f"{header}{artifact_line}\n{result.get('response', '')}")
        else:
            if goal:
                parts.append(f"{header} ERROR\nGoal: {goal}\n{result.get('error', 'Unknown error')}")
            else:
                parts.append(f"{header} ERROR\n{result.get('error', 'Unknown error')}")

    return "\n\n".join(parts).strip()


def _index_agents(agents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for agent in agents:
        name = str(agent.get("name", "")).strip()
        if name and name.lower() not in indexed:
            indexed[name.lower()] = agent
    return indexed


def _extract_collaboration_steps(
    collaboration_plan: Any,
    available_agents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(collaboration_plan, dict):
        return []

    raw_steps = collaboration_plan.get("steps", [])
    if not isinstance(raw_steps, list):
        return []

    available_index = _index_agents(available_agents)
    resolved_steps: List[Dict[str, Any]] = []

    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        agent_name = str(item.get("agent", "")).strip()
        if not agent_name:
            continue

        goal = str(item.get("goal", "")).strip()
        deliverable = str(item.get("deliverable", "")).strip()
        resolved_agent_meta = available_index.get(agent_name.lower())
        if not resolved_agent_meta:
            continue
        if not goal:
            goal = "Handle this step and provide a handoff-ready output."
        if _is_internal_management_step(goal, deliverable):
            log_event(
                "event_manager.collaboration",
                "internal_management_step_ignored",
                {
                    "agent": agent_name,
                    "goal": goal,
                    "deliverable": deliverable,
                },
                level="WARNING",
            )
            continue

        resolved_steps.append(
            {
                "agent": str(resolved_agent_meta.get("name", "")).strip() or agent_name,
                "goal": goal[:1000],
                "deliverable": deliverable[:1000],
                "agent_meta": resolved_agent_meta,
            }
        )

    return resolved_steps


def _normalize_replanned_steps(
    raw_steps: Any,
    available_agents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(raw_steps, list):
        return []
    return _extract_collaboration_steps({"steps": raw_steps}, available_agents)


def _step_signature(step: Dict[str, Any]) -> str:
    agent = str(step.get("agent", "")).strip().lower()
    goal = " ".join(str(step.get("goal", "")).split()).strip().lower()
    return f"{agent}|{goal}"


def _step_signature_list(steps: List[Dict[str, Any]]) -> List[str]:
    signatures: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        signatures.append(_step_signature(step))
    return signatures


def _compact_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[:max_chars]
    return text[: max_chars - 14] + "...(truncated)"


def _extract_first_text_value(*values: Any, max_chars: int = 500) -> str:
    for value in values:
        if isinstance(value, str):
            compact = _compact_text(value, max_chars=max_chars)
            if compact:
                return compact
    return ""


def _normalize_artifact_item(
    raw_item: Any,
    *,
    source_agent: str,
    workflow_id: str,
    workflow_step: int,
    index: int,
) -> Dict[str, Any]:
    if isinstance(raw_item, str):
        item: Dict[str, Any] = {"title": raw_item, "summary": raw_item}
    elif isinstance(raw_item, dict):
        item = dict(raw_item)
    else:
        return {}

    raw_type = _extract_first_text_value(item.get("type"), item.get("kind"), item.get("category"), max_chars=80)
    normalized_type = re.sub(r"[^a-z0-9_]+", "_", raw_type.lower()).strip("_") or "note"
    title = _extract_first_text_value(
        item.get("title"),
        item.get("name"),
        item.get("label"),
        item.get("filename"),
        max_chars=220,
    )
    summary = _extract_first_text_value(
        item.get("summary"),
        item.get("description"),
        item.get("snippet"),
        item.get("abstract"),
        item.get("excerpt"),
        item.get("why_relevant"),
        item.get("content"),
        item.get("result"),
        max_chars=520,
    )
    url = _extract_first_text_value(item.get("url"), item.get("link"), item.get("href"), max_chars=320)
    content = _extract_first_text_value(item.get("content"), item.get("excerpt"), item.get("abstract"), max_chars=900)
    tags = [
        _compact_text(str(tag), max_chars=48)
        for tag in item.get("tags", [])
        if isinstance(tag, (str, int, float)) and str(tag).strip()
    ][:8]

    identifiers: Dict[str, str] = {}
    nested_identifiers = item.get("identifiers")
    if isinstance(nested_identifiers, dict):
        for key, value in nested_identifiers.items():
            compact = _extract_first_text_value(value, max_chars=180)
            if compact:
                identifiers[str(key).strip()] = compact
    for key in ["doi", "arxiv_id", "pmid", "pmcid", "path", "filename"]:
        compact = _extract_first_text_value(item.get(key), max_chars=220)
        if compact:
            identifiers[key] = compact

    metadata: Dict[str, Any] = {}
    reserved_keys = {
        "id",
        "type",
        "kind",
        "category",
        "title",
        "name",
        "label",
        "summary",
        "description",
        "snippet",
        "abstract",
        "excerpt",
        "content",
        "result",
        "url",
        "link",
        "href",
        "tags",
        "identifiers",
        "doi",
        "arxiv_id",
        "pmid",
        "pmcid",
        "path",
        "filename",
        "why_relevant",
    }
    for key, value in item.items():
        if key in reserved_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[str(key)] = _compact_text(str(value), max_chars=180)
        elif isinstance(value, list):
            compact_items = [
                _compact_text(str(entry), max_chars=80)
                for entry in value
                if isinstance(entry, (str, int, float, bool)) and str(entry).strip()
            ][:6]
            if compact_items:
                metadata[str(key)] = compact_items

    artifact_id = _extract_first_text_value(item.get("id"), max_chars=160)
    if not artifact_id:
        artifact_id = f"{workflow_id}:{workflow_step}:{source_agent}:{index}:{normalized_type}"

    if not title:
        title = _extract_first_text_value(summary, url, identifiers.get("doi"), identifiers.get("arxiv_id"), max_chars=220)
    if not summary:
        summary = _extract_first_text_value(content, title, max_chars=520)
    if not any([title, summary, url, identifiers]):
        return {}

    normalized: Dict[str, Any] = {
        "id": artifact_id,
        "type": normalized_type,
        "title": title or normalized_type,
        "summary": summary,
        "source_agent": source_agent,
        "workflow_step": workflow_step,
    }
    if url:
        normalized["url"] = url
    if content and content != summary:
        normalized["content"] = content
    if tags:
        normalized["tags"] = tags
    if identifiers:
        normalized["identifiers"] = identifiers
    if metadata:
        normalized["metadata"] = metadata
    return normalized


def _artifact_identity_key(artifact: Dict[str, Any]) -> str:
    artifact_id = str(artifact.get("id", "")).strip().lower()
    if artifact_id:
        return f"id:{artifact_id}"
    url = str(artifact.get("url", "")).strip().lower()
    if url:
        return f"url:{url}"
    identifiers = artifact.get("identifiers", {})
    if isinstance(identifiers, dict):
        for key in ["doi", "arxiv_id", "pmid", "pmcid", "path", "filename"]:
            value = str(identifiers.get(key, "")).strip().lower()
            if value:
                return f"{key}:{value}"
    artifact_type = str(artifact.get("type", "")).strip().lower()
    title = str(artifact.get("title", "")).strip().lower()
    if artifact_type or title:
        return f"{artifact_type}|{title}"
    return ""


def _extract_artifacts_from_agent_output(
    text: str,
    *,
    source_agent: str,
    workflow_id: str,
    workflow_step: int,
) -> List[Dict[str, Any]]:
    parsed = _extract_json_object(str(text or "").strip())
    if not isinstance(parsed, dict):
        return []
    return _normalize_artifacts_from_payload(
        parsed.get("artifacts"),
        source_agent=source_agent,
        workflow_id=workflow_id,
        workflow_step=workflow_step,
    )


def _normalize_artifacts_from_payload(
    raw_artifacts: Any,
    *,
    source_agent: str,
    workflow_id: str,
    workflow_step: int,
) -> List[Dict[str, Any]]:
    if isinstance(raw_artifacts, dict):
        raw_items: List[Any] = [raw_artifacts]
    elif isinstance(raw_artifacts, list):
        raw_items = list(raw_artifacts)
    else:
        raw_items = []

    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_items, start=1):
        artifact = _normalize_artifact_item(
            item,
            source_agent=source_agent,
            workflow_id=workflow_id,
            workflow_step=workflow_step,
            index=index,
        )
        if not artifact:
            continue
        identity = _artifact_identity_key(artifact)
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        normalized.append(artifact)
        if len(normalized) >= 12:
            break
    return normalized


def _extract_agent_output_summary_from_payload(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""

    summary = _extract_first_text_value(
        payload.get("summary"),
        payload.get("text_response"),
        payload.get("result"),
        payload.get("response"),
        payload.get("answer"),
        payload.get("message"),
        max_chars=3200,
    )
    if summary:
        return summary

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        titles: List[str] = []
        for item in artifacts[:3]:
            if isinstance(item, dict):
                title = _extract_first_text_value(item.get("title"), item.get("name"), max_chars=120)
            elif isinstance(item, str):
                title = _compact_text(item, max_chars=120)
            else:
                title = ""
            if title:
                titles.append(title)
        if titles:
            return f"Reusable artifacts prepared: {', '.join(titles)}"
        return f"Reusable artifacts prepared: {len(artifacts)} items"

    return ""


def _extract_agent_output_text_response_from_payload(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    return _extract_first_text_value(
        payload.get("text_response"),
        payload.get("details"),
        payload.get("narrative"),
        max_chars=12000,
    )


def _extract_strict_contract_payload(raw_text: str) -> tuple[Dict[str, Any] | None, str, List[str]]:
    parsed, source = _extract_json_object_with_source(raw_text)
    if source in {"plain_json", "substring", "fenced_json"} and isinstance(parsed, dict):
        return parsed, source, []

    fence_match = STRICT_JSON_FENCE_ONLY_RE.fullmatch(raw_text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            fenced_parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None, "fenced_json_invalid", ["invalid_json_payload"]
        if isinstance(fenced_parsed, dict):
            return fenced_parsed, "fenced_json", []
        return None, "fenced_json_invalid", ["top_level_json_object_required"]

    violations: List[str] = ["response_must_be_single_json_object"]
    if source == "substring":
        violations.append("plain_prose_outside_json_not_allowed")
    elif source == "substring_invalid":
        violations.append("invalid_json_payload")
    elif source == "empty":
        violations.append("empty_response")
    elif source == "not_found":
        violations.append("json_payload_not_found")
    return None, source, violations


def _validate_worker_contract_payload(payload: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    tool_contract_keys = {"ok", "tool_name", "summary", "content_type", "items", "data", "errors", "metadata"}
    if "tool_name" in payload and tool_contract_keys.issubset(set(payload.keys())):
        violations.append("raw_tool_output_contract_not_allowed")

    status = str(payload.get("status", "")).strip().lower()
    if status not in {"completed", "partial", "blocked", "failed"}:
        violations.append("status_required")

    summary = _extract_agent_output_summary_from_payload(payload)
    if not summary:
        violations.append("summary_required")

    raw_artifacts = payload.get("artifacts", [])
    if raw_artifacts is not None and not isinstance(raw_artifacts, (list, dict)):
        violations.append("artifacts_must_be_array")

    raw_needs = payload.get("needs", [])
    if raw_needs is not None and not isinstance(raw_needs, (list, dict)):
        violations.append("needs_must_be_array")

    raw_text_response = payload.get("text_response")
    if raw_text_response is not None and not isinstance(raw_text_response, (str, int, float, bool)):
        violations.append("text_response_must_be_string")

    return violations


def _collect_called_tool_names(normalized_parts: Any) -> List[str]:
    names: List[str] = []
    if not isinstance(normalized_parts, list):
        return names
    for item in normalized_parts:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"function_call", "function_response"}:
            continue
        name = str(item.get("name", "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def _is_bootstrap_only_worker_output(
    *,
    structured_output: Dict[str, Any],
    normalized_parts: Any,
) -> bool:
    tool_names = _collect_called_tool_names(normalized_parts)
    if not tool_names:
        return False
    if any(name not in INTERNAL_BOOTSTRAP_CAPABILITY_KEYS for name in tool_names):
        return False
    if structured_output.get("artifacts") or structured_output.get("needs"):
        return False

    raw_text = str(structured_output.get("raw_response", "")).strip()
    summary = str(structured_output.get("summary", "")).strip()
    text_response = str(structured_output.get("text_response", "")).strip()
    combined = "\n".join(part for part in [summary, text_response, raw_text] if part).strip()
    if not combined:
        return True
    return any(pattern.search(combined) for pattern in INTERNAL_BOOTSTRAP_TEXT_PATTERNS)


def _apply_bootstrap_only_violation(
    *,
    structured_output: Dict[str, Any] | None,
    normalized_parts: Any,
) -> bool:
    if not isinstance(structured_output, dict):
        return False
    if not _is_bootstrap_only_worker_output(
        structured_output=structured_output,
        normalized_parts=normalized_parts,
    ):
        return False
    violations = list(structured_output.get("contract_violations", []))
    violations.append("bootstrap_only_output_not_allowed")
    structured_output["contract_violations"] = list(dict.fromkeys(violations))
    structured_output["contract_valid"] = False
    current_status = str(structured_output.get("status", "")).strip().lower()
    if current_status == "completed":
        structured_output["status"] = "blocked"
    return True


def _normalize_worker_output_status(
    value: Any,
    *,
    has_summary: bool,
    has_artifacts: bool,
    has_needs: bool,
    raw_text: str,
) -> str:
    token = str(value or "").strip().lower()
    if token in {"completed", "complete", "done", "success", "ok"}:
        return "completed"
    if token in {"partial", "in_progress", "deferred"}:
        return "partial"
    if token in {"blocked", "needs_input", "failed", "error"}:
        return "blocked"
    if raw_text in EMPTY_RESPONSE_SENTINELS:
        return "blocked"
    if has_needs and not (has_summary or has_artifacts):
        return "blocked"
    if has_needs:
        return "partial"
    return "completed"


def _extract_structured_agent_output(
    text: str,
    *,
    source_agent: str,
    workflow_id: str,
    workflow_step: int,
) -> Dict[str, Any]:
    raw_text = str(text or "").strip()
    parsed, contract_source, contract_violations = _extract_strict_contract_payload(raw_text)
    summary = _extract_agent_output_summary_from_payload(parsed) if isinstance(parsed, dict) else ""
    text_response = _extract_agent_output_text_response_from_payload(parsed) if isinstance(parsed, dict) else ""
    artifacts = _normalize_artifacts_from_payload(
        parsed.get("artifacts") if isinstance(parsed, dict) else [],
        source_agent=source_agent,
        workflow_id=workflow_id,
        workflow_step=workflow_step,
    )
    needs = _normalize_need_entries(
        parsed.get("needs") if isinstance(parsed, dict) else [],
        source_agent=source_agent,
        workflow_step=workflow_step,
    )
    if isinstance(parsed, dict):
        contract_violations.extend(_validate_worker_contract_payload(parsed))
    has_summary = bool(summary and summary not in EMPTY_RESPONSE_SENTINELS)
    has_artifacts = bool(artifacts)
    has_needs = bool(needs)
    status = _normalize_worker_output_status(
        parsed.get("status") if isinstance(parsed, dict) else "",
        has_summary=has_summary,
        has_artifacts=has_artifacts,
        has_needs=has_needs,
        raw_text=raw_text,
    )
    if not summary:
        if has_artifacts:
            summary = f"Reusable artifacts prepared: {len(artifacts)} item(s)"
        elif has_needs:
            summary = _need_display_text(needs[0])
        else:
            summary = raw_text
    contract_valid = isinstance(parsed, dict) and not contract_violations
    return {
        "status": status,
        "summary": summary,
        "text_response": text_response,
        "artifacts": artifacts,
        "needs": needs,
        "raw_response": raw_text,
        "contract_valid": contract_valid,
        "contract_source": contract_source,
        "contract_violations": list(dict.fromkeys(contract_violations)),
    }


def _build_artifact_context_entry(artifact: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "id": str(artifact.get("id", "")).strip(),
        "type": str(artifact.get("type", "")).strip() or "note",
        "title": _compact_text(str(artifact.get("title", "")).strip(), max_chars=180),
        "summary": _compact_text(str(artifact.get("summary", "")).strip(), max_chars=320),
        "source_agent": str(artifact.get("source_agent", "")).strip(),
        "workflow_step": artifact.get("workflow_step"),
    }
    url = str(artifact.get("url", "")).strip()
    if url:
        entry["url"] = _compact_text(url, max_chars=220)
    identifiers = artifact.get("identifiers", {})
    if isinstance(identifiers, dict) and identifiers:
        compact_identifiers: Dict[str, str] = {}
        for key, value in identifiers.items():
            compact = _compact_text(str(value).strip(), max_chars=120)
            if compact:
                compact_identifiers[str(key)] = compact
        if compact_identifiers:
            entry["identifiers"] = compact_identifiers
    tags = artifact.get("tags", [])
    if isinstance(tags, list):
        compact_tags = [_compact_text(str(tag), max_chars=40) for tag in tags if str(tag).strip()][:6]
        if compact_tags:
            entry["tags"] = compact_tags
    return entry


def _select_input_artifacts_for_step(
    *,
    artifact_store: List[Dict[str, Any]],
    max_items: int = 6,
) -> List[Dict[str, Any]]:
    if not artifact_store:
        return []

    recent_items = artifact_store[-max_items:]
    recent_items.reverse()
    return [_build_artifact_context_entry(item) for item in recent_items]


def _summarize_conversation_history(
    conversation_history: str,
    *,
    max_turn_lines: int = 6,
    max_chars: int = 900,
) -> str:
    raw = str(conversation_history or "").strip()
    if not raw:
        return "(none)"
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return "(none)"
    tail = lines[-max_turn_lines:]
    return _compact_text("\n".join(tail), max_chars=max_chars)


def _build_delegation_options(
    *,
    available_agents: List[Dict[str, Any]],
    current_agent_name: str,
    max_items: int = 6,
) -> List[Dict[str, Any]]:
    current_key = str(current_agent_name).strip().lower()
    options: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _push(agent_meta: Dict[str, Any]) -> None:
        token = str(agent_meta.get("name", "")).strip()
        key = token.lower()
        if not token or key in seen:
            return
        seen.add(key)
        options.append(
            {
                "name": token,
                "role": str(agent_meta.get("role", "")).strip() or "worker",
                "description": str(agent_meta.get("description", "")).strip(),
                "capabilities": _agent_capability_keys(agent_meta),
                "capability_summary": _agent_capability_summary_text(agent_meta),
                "ownership": _agent_ownership_text(agent_meta),
            }
        )

    for agent in available_agents:
        name = str(agent.get("name", "")).strip()
        if not name:
            continue
        if name.strip().lower() == current_key:
            continue
        _push(agent)
        if len(options) >= max_items:
            break
    return options[:max_items]


def _format_remaining_steps(
    steps: List[Dict[str, Any]],
    *,
    max_items: int = 5,
    max_goal_chars: int = 180,
) -> str:
    if not steps:
        return "(none)"
    lines: List[str] = []
    for idx, step in enumerate(steps[:max_items], start=1):
        agent = str(step.get("agent", "UnknownAgent"))
        goal = _compact_text(str(step.get("goal", "")).strip(), max_chars=max_goal_chars)
        if goal:
            lines.append(f"{idx}. {agent} - {goal}")
        else:
            lines.append(f"{idx}. {agent}")
    if len(steps) > max_items:
        lines.append(f"... ({len(steps) - max_items} more pending steps)")
    return "\n".join(lines).strip()


def _format_prior_results_for_handoff(
    results: List[Dict[str, Any]],
    *,
    max_steps: int = 4,
    max_item_chars: int = 380,
    max_total_chars: int = 2200,
) -> str:
    if not results:
        return "(none)"

    parts: List[str] = []
    selected = results[-max_steps:]
    for item in selected:
        agent_name = str(item.get("agent", "UnknownAgent"))
        step_index = item.get("workflow_step")
        prefix = f"Step {step_index} - {agent_name}" if isinstance(step_index, int) else agent_name
        if item.get("ok"):
            body = _compact_text(str(item.get("response", "")).strip(), max_chars=max_item_chars)
            parts.append(f"[{prefix}] {body}")
        else:
            body = _compact_text(str(item.get("error", "Unknown error")).strip(), max_chars=max_item_chars)
            parts.append(f"[{prefix}] ERROR {body}")
    if len(results) > max_steps:
        parts.insert(0, f"(showing last {max_steps} of {len(results)} completed steps)")
    return _compact_text("\n".join(parts).strip(), max_chars=max_total_chars)


def _format_results_for_synthesis(
    results: List[Dict[str, Any]],
    *,
    max_steps: int = 8,
    max_item_chars: int = 520,
    max_total_chars: int = 5200,
) -> str:
    if not results:
        return "(no execution output)"

    selected = results[-max_steps:]
    blocks: List[str] = []
    for item in selected:
        agent_name = str(item.get("agent", "UnknownAgent"))
        step_idx = item.get("workflow_step")
        prefix = f"Step {step_idx} - {agent_name}" if isinstance(step_idx, int) else agent_name
        if item.get("ok"):
            body = _compact_text(str(item.get("response", "")).strip(), max_chars=max_item_chars)
            blocks.append(f"[{prefix}] {body}")
        else:
            body = _compact_text(str(item.get("error", "Unknown error")).strip(), max_chars=max_item_chars)
            blocks.append(f"[{prefix}] ERROR {body}")

    if len(results) > max_steps:
        blocks.insert(0, f"(showing last {max_steps} of {len(results)} execution outputs)")
    return _compact_text("\n".join(blocks).strip(), max_chars=max_total_chars)


def _build_agent_card_snapshot(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    name = str(agent_meta.get("name", "")).strip()
    if not name:
        return {}

    return {
        "name": name,
        "type": str(agent_meta.get("type", "")).strip() or "a2a",
        "role": str(agent_meta.get("role", "")).strip() or "worker",
        "description": str(agent_meta.get("description", "")).strip(),
        "capability_summary": _agent_capability_summary_text(agent_meta),
        "ownership": _agent_ownership_text(agent_meta),
        "instruction_preview": str(agent_meta.get("instruction_preview", "")).strip(),
    }


def _build_workflow_memory_result_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    response = str(item.get("raw_response") or item.get("response") or item.get("error") or "").strip()
    entry: Dict[str, Any] = {
        "workflow_step": item.get("workflow_step"),
        "agent": str(item.get("agent", "")).strip(),
        "goal": _compact_text(str(item.get("goal", "")).strip(), max_chars=220),
        "ok": bool(item.get("ok")),
    }
    if response:
        entry["response"] = _compact_text(response, max_chars=2800)
    artifact_ids = item.get("artifact_ids")
    if isinstance(artifact_ids, list) and artifact_ids:
        entry["artifact_ids"] = [str(value).strip() for value in artifact_ids[:8] if str(value).strip()]
    return entry


def _build_workflow_memory_snapshot(
    *,
    workflow_id: str,
    user_input: str,
    current_agent: str,
    current_step: int,
    results: List[Dict[str, Any]],
    artifact_store: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
    pending_steps: List[Dict[str, Any]],
    activated_agent_cards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "current_step": current_step,
        "current_agent": str(current_agent).strip(),
        "user_request": _compact_text(user_input, max_chars=500),
        "completed_steps": [
            _build_workflow_memory_result_entry(item)
            for item in results[-8:]
            if isinstance(item, dict)
        ],
        "artifacts": _select_input_artifacts_for_step(
            artifact_store=artifact_store,
            max_items=10,
        ),
        "open_needs": [
            _build_need_context_entry(item)
            for item in open_needs[:10]
            if isinstance(item, dict)
        ],
        "pending_steps": [
            {
                "agent": str(step.get("agent", "")).strip(),
                "goal": _compact_text(str(step.get("goal", "")).strip(), max_chars=220),
                "deliverable": _compact_text(str(step.get("deliverable", "")).strip(), max_chars=180),
            }
            for step in pending_steps[:6]
            if isinstance(step, dict)
        ],
        "activated_agents": [
            _build_agent_card_snapshot(card)
            for card in activated_agent_cards[:8]
            if isinstance(card, dict)
        ],
    }


def _normalize_need_text(value: str) -> str:
    compact = " ".join(value.split()).strip()
    compact = re.sub(r"^[-*]\s*", "", compact)
    compact = re.sub(r"^\d+\.\s*", "", compact)
    return compact[:300]


def _normalize_need_kind(value: str) -> str:
    token = re.sub(r"[^a-z0-9_/-]+", "_", str(value or "").strip().lower()).strip("_")
    return token[:60]


def _normalize_need_capabilities(raw_value: Any) -> List[str]:
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    normalized: List[str] = []
    seen: set[str] = set()
    for item in values:
        token = str(item or "").strip()
        lowered = token.lower()
        if not token or lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token[:80])
    return normalized[:8]


def _normalize_need_entry(
    value: Any,
    *,
    source_agent: str = "",
    workflow_step: int | None = None,
) -> Dict[str, Any] | None:
    request = ""
    reason = ""
    kind = ""
    required_capabilities: List[str] = []
    blocking: bool | None = None

    if isinstance(value, dict):
        request = _normalize_need_text(
            str(
                value.get("request")
                or value.get("task")
                or value.get("message")
                or value.get("need")
                or ""
            )
        )
        reason = _compact_text(
            str(value.get("reason") or value.get("why") or "").strip(),
            max_chars=220,
        )
        kind = _normalize_need_kind(str(value.get("kind") or value.get("type") or ""))
        required_capabilities = _normalize_need_capabilities(
            value.get("required_capabilities")
            or value.get("capabilities")
            or value.get("required_capability")
            or []
        )
        if "blocking" in value:
            blocking = bool(value.get("blocking"))
    elif isinstance(value, str):
        token = _normalize_need_text(value)
        if not token:
            return None
        request = token
    else:
        return None

    if not request or request.lower() in {"none", "n/a", "no", "null", "없음"}:
        return None

    entry: Dict[str, Any] = {"request": request[:300]}
    if reason:
        entry["reason"] = reason
    if kind:
        entry["kind"] = kind
    if required_capabilities:
        entry["required_capabilities"] = required_capabilities
    if blocking is not None:
        entry["blocking"] = bool(blocking)
    if source_agent:
        entry["source_agent"] = str(source_agent).strip()
    if isinstance(workflow_step, int):
        entry["workflow_step"] = workflow_step
    return entry


def _need_display_text(need: Dict[str, Any] | str) -> str:
    if isinstance(need, str):
        entry = _normalize_need_entry(need)
    else:
        entry = need if isinstance(need, dict) else None
    if not isinstance(entry, dict):
        return ""
    request = _normalize_need_text(str(entry.get("request", "")).strip())
    if not request:
        return ""
    return request[:300]


def _need_identity_key(need: Dict[str, Any] | str) -> str:
    if isinstance(need, str):
        entry = _normalize_need_entry(need)
    else:
        entry = need if isinstance(need, dict) else None
    if not isinstance(entry, dict):
        return ""
    kind = _normalize_need_kind(str(entry.get("kind", ""))).lower()
    request = _normalize_need_text(str(entry.get("request", "")).strip()).lower()
    capability_key = ",".join(
        token.lower()
        for token in _normalize_need_capabilities(entry.get("required_capabilities", []))
    )
    if not request:
        return ""
    return f"{kind}|{capability_key}|{request}"


def _build_need_context_entry(need: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "request": _compact_text(str(need.get("request", "")).strip(), max_chars=220),
    }
    reason = _compact_text(str(need.get("reason", "")).strip(), max_chars=180)
    if reason:
        entry["reason"] = reason
    kind = _normalize_need_kind(str(need.get("kind", "")).strip())
    if kind:
        entry["kind"] = kind
    required_capabilities = _normalize_need_capabilities(need.get("required_capabilities", []))
    if required_capabilities:
        entry["required_capabilities"] = required_capabilities
    if "blocking" in need:
        entry["blocking"] = bool(need.get("blocking"))
    source_agent = str(need.get("source_agent", "")).strip()
    if source_agent:
        entry["source_agent"] = source_agent
    workflow_step = need.get("workflow_step")
    if isinstance(workflow_step, int):
        entry["workflow_step"] = workflow_step
    return entry


def _format_open_needs_for_prompt(needs: List[Dict[str, Any]], *, max_items: int = 8) -> str:
    if not needs:
        return "(none)"
    lines: List[str] = []
    for need in needs[:max_items]:
        if not isinstance(need, dict):
            continue
        text = _need_display_text(need)
        if not text:
            continue
        kind = _normalize_need_kind(str(need.get("kind", "")).strip())
        reason = _compact_text(str(need.get("reason", "")).strip(), max_chars=120)
        source_agent = str(need.get("source_agent", "")).strip()
        required_capabilities = _normalize_need_capabilities(need.get("required_capabilities", []))
        suffix_parts: List[str] = []
        if kind:
            suffix_parts.append(f"kind: {kind}")
        if required_capabilities:
            suffix_parts.append(f"capabilities: {', '.join(required_capabilities)}")
        if reason:
            suffix_parts.append(f"reason: {reason}")
        if source_agent:
            suffix_parts.append(f"source: {source_agent}")
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        lines.append(f"- {text}{suffix}")
    if not lines:
        return "(none)"
    if len(needs) > max_items:
        lines.append(f"... ({len(needs) - max_items} more open needs)")
    return "\n".join(lines)


def _normalize_need_entries(
    raw_value: Any,
    *,
    source_agent: str = "",
    workflow_step: int | None = None,
    max_items: int = 12,
) -> List[Dict[str, Any]]:
    if raw_value is None:
        return []
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        entry = _normalize_need_entry(item, source_agent=source_agent, workflow_step=workflow_step)
        if not entry:
            continue
        key = _need_identity_key(entry)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(entry)
        if len(normalized) >= max_items:
            break
    return normalized


def _extract_needs_from_agent_output(
    text: str,
    *,
    source_agent: str,
    workflow_step: int,
) -> List[Dict[str, Any]]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    parsed = _extract_json_object(raw_text)
    if isinstance(parsed, dict):
        normalized = _normalize_need_entries(
            parsed.get("needs"),
            source_agent=source_agent,
            workflow_step=workflow_step,
        )
        if normalized:
            return normalized

        raw_events = parsed.get("workflow_events")
        if isinstance(raw_events, list):
            event_needs: List[Dict[str, Any]] = []
            for item in raw_events:
                if not isinstance(item, dict):
                    continue
                event_type = str(item.get("type", "")).strip().lower()
                if event_type not in {"need_request", "delegate_request", "handoff_request"}:
                    continue
                event_needs.append(
                    {
                        "request": str(item.get("request") or item.get("payload") or item.get("message") or "").strip(),
                        "reason": str(item.get("reason") or "").strip(),
                    }
                )
            normalized = _normalize_need_entries(
                event_needs,
                source_agent=source_agent,
                workflow_step=workflow_step,
            )
            if normalized:
                return normalized

    lines = raw_text.splitlines()
    marker_index = -1
    inline_value = ""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("needs:"):
            marker_index = idx
            inline_value = stripped[len("needs:") :].strip()
            break
        if lowered in {"needs", "needs:"}:
            marker_index = idx
            break

    if marker_index < 0:
        return []

    fallback_values: List[str] = []
    if inline_value:
        fallback_values.append(inline_value)

    for line in lines[marker_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if fallback_values:
                break
            continue
        if fallback_values and re.match(r"^[A-Za-z][A-Za-z0-9 _/-]{0,40}:$", stripped):
            break
        fallback_values.append(stripped)
        if len(fallback_values) >= 12:
            break

    return _normalize_need_entries(
        fallback_values,
        source_agent=source_agent,
        workflow_step=workflow_step,
    )


def _is_user_clarification_need(need: Dict[str, Any] | str) -> bool:
    request = ""
    kind = ""
    required_capabilities: List[str] = []
    if isinstance(need, dict):
        request = _normalize_need_text(str(need.get("request", "")).strip())
        kind = _normalize_need_kind(str(need.get("kind", "")).strip())
        required_capabilities = _normalize_need_capabilities(need.get("required_capabilities", []))
    else:
        request = _normalize_need_text(need)

    required_capability_keys = {token.lower() for token in required_capabilities}
    if kind == "user_clarification" or "user_clarification_routing" in required_capability_keys:
        return True

    text = str(request or _need_display_text(need)).strip()
    if not text:
        return False

    lowered = text.lower()
    explicit_clarification_tokens = [
        "ask user",
        "ask the user",
        "user clarification",
        "need user input",
        "need user confirmation",
        "please ask",
        "please specify",
        "clarify",
        "사용자 입력",
        "사용자 확인",
        "사용자에게",
        "유저에게",
        "질문 필요",
        "확인 필요",
    ]
    is_slack_channel_question = (
        ("슬랙" in text or "slack" in lowered)
        and ("채널" in text or "channel" in lowered)
        and any(token in lowered for token in ["ask", "need", "provide", "id", "name", "알려", "확인", "입력"])
    )
    return ("?" in text) or any(token in lowered for token in explicit_clarification_tokens) or is_slack_channel_question


def _is_internal_bootstrap_text(text: str) -> bool:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in INTERNAL_BOOTSTRAP_TEXT_PATTERNS)


def _is_internal_bootstrap_need(need: Dict[str, Any] | str) -> bool:
    if isinstance(need, dict):
        request = _normalize_need_text(str(need.get("request", "")).strip())
        reason = _compact_text(str(need.get("reason", "")).strip(), max_chars=220)
        capabilities = {
            token.lower()
            for token in _normalize_need_capabilities(need.get("required_capabilities", []))
        }
    else:
        request = _normalize_need_text(str(need or "").strip())
        reason = ""
        capabilities = set()
    if capabilities & INTERNAL_BOOTSTRAP_CAPABILITY_KEYS:
        return True
    combined = " ".join(part for part in [request, reason] if part).strip()
    return _is_internal_bootstrap_text(combined)


def _first_user_clarification_request(needs: List[Dict[str, Any]]) -> str:
    for need in needs:
        if not _is_user_clarification_need(need):
            continue
        request = _normalize_need_text(str(need.get("request", "")).strip())
        if request:
            return request
        normalized = _need_display_text(need)
        lowered = normalized.lower()
        if ("슬랙" in normalized or "slack" in lowered) and ("채널" in normalized or "channel" in lowered):
            return "슬랙 게시를 위해 정확한 채널 이름 또는 채널 ID(예: C12345678)를 알려주세요."
        return normalized
    return ""


def _agent_matches_need(agent_meta: Dict[str, Any], need: Dict[str, Any]) -> int:
    if not isinstance(agent_meta, dict) or not isinstance(need, dict):
        return 0

    required_capabilities = {
        token.lower()
        for token in _normalize_need_capabilities(need.get("required_capabilities", []))
    }
    agent_capabilities = {
        token.lower()
        for token in _agent_capability_keys(agent_meta)
    }
    if required_capabilities:
        overlap = len(required_capabilities & agent_capabilities)
        if overlap <= 0:
            return 0
        coordinator_bonus = 0 if str(agent_meta.get("role", "")).strip().lower() == "coordinator" else 10
        return overlap * 100 + coordinator_bonus

    return 0


def _is_internal_management_step(goal: str, deliverable: str = "") -> bool:
    combined = " ".join(part for part in [str(goal or "").strip(), str(deliverable or "").strip()] if part).strip()
    return _is_internal_bootstrap_text(combined)


def _select_agent_for_need(
    *,
    need: Dict[str, Any],
    available_agents: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    best_agent: Dict[str, Any] | None = None
    best_score = 0
    for agent_meta in available_agents:
        score = _agent_matches_need(agent_meta, need)
        if score > best_score:
            best_score = score
            best_agent = agent_meta
    return best_agent


def _build_indirect_delegation_fallback_steps(
    *,
    open_needs: List[Dict[str, Any]],
    available_agents: List[Dict[str, Any]],
    pending_steps: List[Dict[str, Any]],
    max_steps: int = 3,
) -> Dict[str, Any]:
    available_index = _index_agents(available_agents)
    existing_signatures: set[str] = set()
    for step in pending_steps:
        step_agent = str(step.get("agent", "")).strip().lower()
        step_goal = str(step.get("goal", "")).strip().lower()
        if step_agent and step_goal:
            existing_signatures.add(f"{step_agent}|{step_goal}")

    added_steps: List[Dict[str, Any]] = []
    consumed_need_keys: set[str] = set()
    local_signatures: set[str] = set()

    for need in open_needs:
        if not isinstance(need, dict):
            continue
        request = _normalize_need_text(str(need.get("request", "")).strip())
        need_kind = _normalize_need_kind(str(need.get("kind", "")).strip())
        if not request:
            continue
        if need_kind == "user_clarification":
            continue

        agent_meta = _select_agent_for_need(need=need, available_agents=available_agents)
        if agent_meta is None:
            continue
        agent_name = str(agent_meta.get("name", "")).strip() or "UnknownAgent"
        agent_key = agent_name.lower()

        canonical_need = f"{agent_key}|{request.lower()}"
        if canonical_need in local_signatures:
            continue

        goal = (
            "Handle this unresolved follow-up need and return a usable result.\n"
            f"Requested need: {request}"
        )
        signature = f"{agent_key}|{goal.lower()}"
        if signature in existing_signatures:
            consumed_need_keys.add(_need_identity_key(need))
            continue

        added_steps.append(
            {
                "agent": agent_name,
                "goal": goal[:1000],
                "deliverable": f"Concrete response/evidence addressing: {request}"[:1000],
                "agent_meta": agent_meta,
            }
        )
        local_signatures.add(canonical_need)
        existing_signatures.add(signature)
        consumed_need_keys.add(_need_identity_key(need))

        if len(added_steps) >= max_steps:
            break

    return {"steps": added_steps, "consumed_need_keys": sorted(consumed_need_keys)}


def _result_has_actionable_evidence(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict) or not bool(result.get("ok")):
        return False
    if result.get("artifacts_emitted"):
        return True
    agent_name = str(result.get("agent", "")).strip().lower()
    if agent_name == "mainagent":
        return False
    output_status = str(result.get("output_status", "")).strip().lower()
    if output_status == "blocked":
        return False
    response = str(result.get("response", "")).strip()
    if not response or response in EMPTY_RESPONSE_SENTINELS:
        return False
    if _is_internal_bootstrap_text(response):
        return False
    return True


def _is_delivery_step(step: Dict[str, Any]) -> bool:
    if not isinstance(step, dict):
        return False
    agent_meta = step.get("agent_meta", {})
    capability_keys = {
        token.lower()
        for token in _agent_capability_keys(agent_meta if isinstance(agent_meta, dict) else {})
    }
    if "comm.slack.post" not in capability_keys:
        return False
    combined = " ".join(
        part for part in [str(step.get("goal", "")).strip(), str(step.get("deliverable", "")).strip()] if part
    ).lower()
    delivery_tokens = ["slack", "channel", "post", "deliver", "send", "message", "게시", "전송", "메시지"]
    return any(token in combined for token in delivery_tokens)


def _delivery_gate_error(
    *,
    step: Dict[str, Any],
    open_needs: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    artifact_store: List[Dict[str, Any]],
) -> str:
    if not _is_delivery_step(step):
        return ""
    unresolved_blocking = [need for need in open_needs if isinstance(need, dict) and bool(need.get("blocking"))]
    if unresolved_blocking:
        return (
            "Delivery is blocked because unresolved workflow needs remain. "
            "Resolve the missing evidence or clarification before posting a final message."
        )
    if artifact_store:
        return ""
    if any(_result_has_actionable_evidence(item) for item in results):
        return ""
    return (
        "Delivery is blocked because no reusable evidence or specialist findings are available yet. "
        "Gather or synthesize evidence before posting to Slack."
    )


def _pending_step_log_entries(steps: List[Dict[str, Any]], *, max_items: int = 8) -> List[Dict[str, str]]:
    return [
        {
            "agent": str(step.get("agent", "")),
            "goal": str(step.get("goal", "")),
        }
        for step in steps[:max_items]
        if isinstance(step, dict)
    ]


def _append_unique_open_needs(
    *,
    open_needs: List[Dict[str, Any]],
    new_needs: List[Dict[str, Any]],
    seen_need_keys: set[str],
) -> List[Dict[str, Any]]:
    added_needs: List[Dict[str, Any]] = []
    for need in new_needs:
        if not isinstance(need, dict):
            continue
        if _is_internal_bootstrap_need(need):
            log_event(
                "workflow.need",
                "need_ignored_internal_bootstrap",
                {
                    "request": _normalize_need_text(str(need.get("request", "")).strip()),
                    "reason": _compact_text(str(need.get("reason", "")).strip(), max_chars=180),
                    "source_agent": str(need.get("source_agent", "")).strip(),
                    "required_capabilities": _normalize_need_capabilities(need.get("required_capabilities", [])),
                },
                level="WARNING",
            )
            continue
        key = _need_identity_key(need)
        if not key or key in seen_need_keys:
            continue
        seen_need_keys.add(key)
        open_needs.append(need)
        added_needs.append(need)
    return added_needs


def _resolve_open_needs_for_completed_agent(
    *,
    open_needs: List[Dict[str, Any]],
    previous_open_needs: List[Dict[str, Any]],
    agent_name: str,
    agent_meta: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    previous_need_keys = {
        _need_identity_key(item)
        for item in previous_open_needs
        if isinstance(item, dict)
    }
    resolved: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []

    for need in open_needs:
        if not isinstance(need, dict):
            continue
        need_key = _need_identity_key(need)
        if need_key not in previous_need_keys:
            remaining.append(need)
            continue
        if _agent_matches_need(agent_meta, need) > 0:
            resolved.append(need)
            continue
        remaining.append(need)

    return remaining, resolved


def _replace_pending_steps_with_log(
    *,
    candidate_steps: List[Dict[str, Any]],
    event_name: str,
    log_payload: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    new_pending_steps = list(candidate_steps)
    payload = dict(log_payload or {})
    payload["new_pending_steps"] = _pending_step_log_entries(new_pending_steps)
    log_event("event_manager.collaboration", event_name, payload)
    return new_pending_steps


def _register_new_artifacts(
    *,
    artifact_store: List[Dict[str, Any]],
    new_artifacts: List[Dict[str, Any]],
    seen_artifact_keys: set[str],
    source_agent: str,
    workflow_step: int,
) -> List[Dict[str, Any]]:
    added_artifacts: List[Dict[str, Any]] = []
    for artifact in new_artifacts:
        artifact_key = _artifact_identity_key(artifact)
        if artifact_key and artifact_key in seen_artifact_keys:
            continue
        if artifact_key:
            seen_artifact_keys.add(artifact_key)
        artifact_store.append(artifact)
        added_artifacts.append(artifact)
    if added_artifacts:
        log_event(
            "workflow.artifact",
            "artifacts_registered",
            {
                "source_agent": source_agent,
                "workflow_step": workflow_step,
                "artifact_count": len(added_artifacts),
                "artifact_ids": [str(item.get("id", "")) for item in added_artifacts],
                "artifact_types": [str(item.get("type", "")) for item in added_artifacts],
                "store_size": len(artifact_store),
            },
            direction="inbound",
        )
    return added_artifacts


def _log_emitted_needs(
    *,
    source_agent: str,
    workflow_id: str,
    workflow_step: int,
    needs: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
) -> None:
    for need in needs:
        request = _normalize_need_text(str(need.get("request", "")).strip())
        required_capabilities = _normalize_need_capabilities(need.get("required_capabilities", []))
        log_event(
            "workflow.need",
            "need_emitted",
            {
                "source_agent": source_agent,
                "required_capabilities": required_capabilities,
                "kind": _normalize_need_kind(str(need.get("kind", "")).strip()),
                "request": request or _need_display_text(need),
                "reason": str(need.get("reason", "")).strip(),
                "workflow_step": workflow_step,
            },
        )
        _log_agent_message(
            action="need_requested",
            from_agent=source_agent,
            to_agent="MainAgent",
            message=request or _need_display_text(need),
            channel="needs",
            workflow_id=workflow_id,
            workflow_step=workflow_step,
            direction="outbound",
        )

    if needs:
        log_event(
            "event_manager.collaboration",
            "open_needs_updated_from_agent_output",
            {
                "added_needs": [_build_need_context_entry(item) for item in needs],
                "open_needs": [_build_need_context_entry(item) for item in open_needs],
            },
            direction="inbound",
        )


def _mark_workflow_paused_for_user_input(
    *,
    result: Dict[str, Any],
    step: int,
    agent: str,
    request: str,
    source: str,
) -> None:
    result["workflow_paused"] = True
    result["pause_reason"] = "awaiting_user_clarification"
    result["pause_request"] = request
    log_event(
        "event_manager.collaboration",
        "workflow_paused_for_user_input",
        {
            "step": step,
            "agent": agent,
            "request": request,
            "source": source,
        },
    )


def _append_error_detail(result: Dict[str, Any], heading: str, detail: str) -> None:
    normalized = str(detail).strip()
    if not normalized:
        return
    current_error = str(result.get("error", "Unknown error")).strip()
    result["error"] = f"{current_error}\n\n{heading}: {normalized}"


def _append_coordinator_message(result: Dict[str, Any], user_message: str) -> None:
    normalized = str(user_message).strip()
    if not normalized:
        return
    current_error = str(result.get("error", "Unknown error")).strip()
    result["error"] = f"{current_error}\n\nCoordinator Message: {normalized}"


def _maybe_replace_pending_steps_from_review(
    *,
    should_replace: bool,
    updated_steps: Any,
    event_name: str,
    failed_step: int,
    failed_agent: str,
    reason: str,
) -> List[Dict[str, Any]] | None:
    if not should_replace or not updated_steps:
        return None
    return _replace_pending_steps_with_log(
        candidate_steps=list(updated_steps),
        event_name=event_name,
        log_payload={
            "failed_step": failed_step,
            "failed_agent": failed_agent,
            "reason": reason,
        },
    )


def _default_progress_review_result(reason: str) -> Dict[str, Any]:
    return {
        "needs": [],
        "should_update_plan": False,
        "updated_steps": [],
        "reason": reason,
    }


def _run_progress_review_or_skip(
    *,
    review_trigger_reasons: List[str],
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    results: List[Dict[str, Any]],
    enriched: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
    step_counter: int,
    agent_name: str,
) -> Dict[str, Any]:
    if review_trigger_reasons:
        return _review_collaboration_progress_with_main_agent(
            main_agent=main_agent,
            available_agents=available_agents,
            user_input=user_input,
            conversation_history=conversation_history,
            raw_plan=raw_plan,
            completed_results=results,
            latest_result=enriched,
            pending_steps=pending_steps,
            open_needs=open_needs,
            trigger_reasons=review_trigger_reasons,
        )

    review = _default_progress_review_result("review_skipped_no_trigger")
    log_event(
        "event_manager.collaboration",
        "replan_review_skipped",
        {
            "step": step_counter,
            "agent": agent_name,
            "pending_count": len(pending_steps),
            "open_needs_count": len(open_needs),
            "trigger_reasons": review_trigger_reasons,
        },
    )
    return review


def _apply_progress_review_plan_update(
    *,
    review: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    completed_agent: str,
    completed_goal: str,
) -> tuple[bool, List[Dict[str, Any]]]:
    if not (review.get("should_update_plan") and review.get("updated_steps")):
        return False, pending_steps

    candidate_steps = list(review["updated_steps"])
    candidate_signatures = _step_signature_list(candidate_steps)
    pending_signatures = _step_signature_list(pending_steps)
    completed_signature = _step_signature({"agent": completed_agent, "goal": completed_goal})
    needs_in_review = bool(review.get("needs"))

    no_progress_reasons: List[str] = []
    if candidate_signatures and all(sig == completed_signature for sig in candidate_signatures):
        no_progress_reasons.append("repeats_completed_step")
    if candidate_signatures and candidate_signatures == pending_signatures:
        no_progress_reasons.append("identical_to_existing_pending")

    if no_progress_reasons and not needs_in_review:
        log_event(
            "event_manager.collaboration",
            "plan_update_rejected_no_progress",
            {
                "reason": str(review.get("reason", "")),
                "reasons": no_progress_reasons,
                "candidate_signatures": candidate_signatures,
                "pending_signatures": pending_signatures,
            },
        )
        return False, pending_steps

    return True, _replace_pending_steps_with_log(
        candidate_steps=candidate_steps,
        event_name="plan_updated",
        log_payload={"reason": str(review.get("reason", ""))},
    )


def _build_collaboration_step_input(
    *,
    workflow_id: str,
    user_input: str,
    prior_results: List[Dict[str, Any]],
    input_artifacts: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
    remaining_steps: List[Dict[str, Any]],
    available_agents: List[Dict[str, Any]],
    step: Dict[str, Any],
    step_index: int,
    total_steps_hint: int,
) -> str:
    step_goal = _compact_text(str(step.get("goal", "")).strip(), max_chars=320)
    deliverable = _compact_text(str(step.get("deliverable", "")).strip(), max_chars=220)

    request_brief = _compact_text(user_input, max_chars=320)
    prior_text = _format_prior_results_for_handoff(
        prior_results,
        max_steps=2,
        max_item_chars=260,
        max_total_chars=1100,
    )
    remaining_text = _format_remaining_steps(
        remaining_steps,
        max_items=2,
        max_goal_chars=140,
    )
    delegation_options = _build_delegation_options(
        available_agents=available_agents,
        current_agent_name=str(step.get("agent", "")),
        max_items=6,
    )

    context_packet: Dict[str, Any] = {
        "workflow": {
            "workflow_id": workflow_id,
            "current_step": step_index,
            "assigned_agent": str(step.get("agent", "UnknownAgent")),
        },
        "task": {
            "goal": step_goal or "(no explicit goal provided)",
            "expected_deliverable": deliverable or "(not specified)",
        },
        "request": {"user_request_brief": request_brief},
        "state": {
            "prior_results": prior_text,
            "input_artifacts": input_artifacts,
            "open_needs": [
                _build_need_context_entry(item)
                for item in open_needs[:4]
                if isinstance(item, dict)
            ],
            "remaining_steps_hint": remaining_text,
        },
        "delegation_options": delegation_options,
        "total_steps_hint": total_steps_hint,
        "handoff_support": {
            "input_artifacts": True,
            "artifacts": ["type", "title", "summary", "url", "identifiers"],
            "tool_output_contract": TOOL_OUTPUT_CONTRACT,
            "output_contract": {
                "status": "completed | partial | blocked | failed",
                "summary": "required concise handoff-ready result",
                "text_response": "optional fuller explanation kept inside JSON",
                "artifacts": ["optional reusable structured items"],
                "needs": ["optional follow-up needs using the needs schema below"],
            },
            "needs": {
                "supported": True,
                "schema": {
                    "needs": [
                        {
                            "kind": "missing_evidence | user_clarification | missing_context",
                            "request": "what is needed next",
                            "required_capabilities": ["capability_key"],
                            "reason": "why this handoff is needed",
                            "blocking": True,
                        }
                    ]
                },
            },
        },
    }
    context_json = json.dumps(context_packet, ensure_ascii=False)

    return (
        "You are handling one step in a multi-agent workflow.\n"
        "Use the context packet below as working context, not as a rigid script.\n"
        "Solve the step directly when it matches your specialization and tools.\n"
        "If another available agent is a better fit, or important workflow context is missing, say that clearly and return the strongest partial result or handoff you can.\n"
        "Your direct callable tools return JSON using this internal tool contract:\n"
        "{\n"
        '  "ok": true,\n'
        '  "tool_name": "callable tool name",\n'
        '  "summary": "required concise tool-result summary",\n'
        '  "content_type": "collection | object | memory | delivery | error",\n'
        '  "items": ["optional normalized list records"],\n'
        '  "data": {"optional structured payload": "..."},\n'
        '  "errors": ["optional error strings"],\n'
        '  "metadata": {"optional call metadata": "..."}\n'
        "}\n"
        "Use that tool contract internally, but do not return a raw tool output contract as your final workflow-step answer.\n"
        "Return JSON only using this contract:\n"
        "{\n"
        '  "status": "completed | partial | blocked | failed",\n'
        '  "summary": "required concise handoff-ready result",\n'
        '  "text_response": "optional fuller explanation kept inside JSON",\n'
        '  "artifacts": [\n'
        "    {\n"
        '      "type": "artifact type",\n'
        '      "title": "short title",\n'
        '      "summary": "what this artifact contains",\n'
        '      "url": "optional source url",\n'
        '      "identifiers": {"doi": "...", "arxiv_id": "..."}\n'
        "    }\n"
        "  ],\n"
        '  "needs": [\n'
        "    {\n"
        '      "kind": "missing_evidence | user_clarification | missing_context",\n'
        '      "request": "what is needed next",\n'
        '      "required_capabilities": ["capability_key"],\n'
        '      "reason": "why this handoff is needed",\n'
        '      "blocking": true\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- `summary` is always required, even when blocked.\n"
        "- Put any longer narrative or explanation inside `text_response`, not outside the JSON object.\n"
        "- Use your private tools, skills, and session memory directly before declaring a blocker.\n"
        "- Read internal tool results from their `summary`, `items`, `data`, and `errors` fields, then synthesize the task result in the workflow-step contract.\n"
        "- Do not treat `load_skill`, `load_session_memory`, or other internal bootstrap output as the final answer to the workflow step.\n"
        "- Do not emit `needs` asking the coordinator to load your own skills, tools, or session memory.\n"
        "- Use `needs` only for missing evidence, missing shared workflow context, or user clarification that another agent or the user must provide.\n"
        "- If you are blocked, explain the blocker in `summary` and include `needs` when coordinator follow-up is required.\n"
        "- If you have reusable evidence, include it in `artifacts` and keep `summary` concise.\n"
        "- Do not return plain prose outside the JSON object.\n"
        "Keep the response focused on advancing the workflow.\n\n"
        "Workflow Context Packet:\n"
        f"{context_json}"
    )


def _build_contract_repair_input(
    *,
    previous_response: str,
    violations: List[str],
) -> str:
    violations_text = ", ".join(str(item).strip() for item in violations if str(item).strip()) or "invalid_output_contract"
    return (
        "Your previous workflow-step response violated the required output contract.\n"
        "Do NOT write JSON manually. Do NOT output plain prose.\n\n"
        "Steps to repair:\n"
        "1. Read the previous response below and extract the key content: what was found (status, summary), any reusable items (artifacts), any outstanding needs.\n"
        "2. Call `format_handoff_contract(status=..., summary=..., text_response=..., artifacts_json=..., needs_json=...)` with that content.\n"
        "3. After the tool returns, write its return value as your next message — copy the returned string exactly as a text message, with no prose before or after it.\n"
        "4. If `format_handoff_contract` returns a line starting with 'format_handoff_contract failed', fix the arguments and call it again.\n\n"
        "Argument rules:\n"
        "- `status`: one of 'completed', 'partial', 'blocked', 'failed'.\n"
        "- `summary`: 1–3 sentences describing what was found or done. Must not be empty.\n"
        "- `text_response`: detailed narrative if needed, otherwise pass empty string ''.\n"
        "- `artifacts_json`: JSON array string — each item needs 'title' and 'summary'. "
        "Add 'url', 'doi', 'arxiv_id' when available. Pass empty string '' if none.\n"
        "- `needs_json`: JSON array string — each item needs 'request'. "
        "Add 'required_capabilities' and 'blocking' when relevant. Pass empty string '' if none.\n\n"
        "Do not return raw internal tool contracts (ok/tool_name/items/data/errors/metadata).\n"
        "Do not include bootstrap outputs such as `load_skill` / `load_session_memory` results.\n"
        "Do not ask the coordinator to load your private skills, tools, or session memory.\n\n"
        f"Violations detected: {violations_text}\n\n"
        "Previous response to repair:\n"
        f"{previous_response}"
    )


def _attempt_worker_output_contract_repair(
    *,
    agent_meta: Dict[str, Any],
    agent_name: str,
    workflow_id: str,
    workflow_step: int,
    invalid_response: str,
    violations: List[str],
    workflow_memory_snapshot: Dict[str, Any] | None,
) -> Dict[str, Any]:
    repair_prompt = _build_contract_repair_input(
        previous_response=invalid_response,
        violations=violations,
    )
    log_event(
        "event_manager.collaboration",
        "worker_output_contract_repair_started",
        {
            "agent": agent_name,
            "workflow_step": workflow_step,
            "violations": violations,
        },
        level="WARNING",
    )
    _log_agent_message(
        action="sent",
        from_agent="MainAgent",
        to_agent=agent_name,
        message=repair_prompt,
        channel="contract_repair",
        workflow_id=workflow_id,
        workflow_step=workflow_step,
        direction="outbound",
    )
    repair_result = _execute_single_agent(
        agent_meta,
        repair_prompt,
        workflow_memory_snapshot=workflow_memory_snapshot,
    )
    repair_raw_output = (
        str(repair_result.get("response", "")).strip()
        if repair_result.get("ok")
        else str(repair_result.get("error", "")).strip()
    )
    _log_agent_message(
        action="received",
        from_agent=agent_name,
        to_agent="MainAgent",
        message=repair_raw_output,
        channel="contract_repair",
        workflow_id=workflow_id,
        workflow_step=workflow_step,
        ok=bool(repair_result.get("ok")),
        direction="inbound",
    )
    structured_output: Dict[str, Any] | None = None
    if repair_result.get("ok"):
        structured_output = _extract_structured_agent_output(
            repair_raw_output,
            source_agent=agent_name,
            workflow_id=workflow_id,
            workflow_step=workflow_step,
        )
        _apply_bootstrap_only_violation(
            structured_output=structured_output,
            normalized_parts=repair_result.get("normalized_response_parts", []),
        )
    success = bool(repair_result.get("ok")) and isinstance(structured_output, dict) and bool(
        structured_output.get("contract_valid")
    )
    log_event(
        "event_manager.collaboration",
        "worker_output_contract_repair_completed",
        {
            "agent": agent_name,
            "workflow_step": workflow_step,
            "success": success,
            "violations": violations,
            "repair_violations": list(structured_output.get("contract_violations", []))
            if isinstance(structured_output, dict)
            else [],
        },
        direction="inbound" if success else "outbound",
        level="INFO" if success else "WARNING",
    )
    return {
        "result": repair_result,
        "structured_output": structured_output or {},
        "success": success,
        "raw_output": repair_raw_output,
    }


def _result_text_for_progress_review(latest_result: Dict[str, Any]) -> str:
    raw = str(latest_result.get("raw_response") or latest_result.get("response") or latest_result.get("error") or "").strip()
    if raw in EMPTY_RESPONSE_SENTINELS:
        return raw
    return " ".join(raw.split()).strip()


def _collect_progress_review_reasons(
    *,
    latest_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
) -> List[str]:
    reasons: List[str] = []

    if open_needs:
        reasons.append("open_needs")
    if not pending_steps:
        reasons.append("pending_exhausted")
    if bool(latest_result.get("workflow_paused")):
        reasons.append("workflow_paused")
    if not bool(latest_result.get("ok")):
        reasons.append("latest_step_failed")

    response_text = _result_text_for_progress_review(latest_result)
    if response_text in EMPTY_RESPONSE_SENTINELS:
        reasons.append("empty_output")
    else:
        for pattern in PROGRESS_REVIEW_BLOCKER_PATTERNS:
            if pattern.search(response_text):
                reasons.append("blocker_language")
                break

    emitted_artifacts = latest_result.get("artifacts_emitted")
    parsed_needs = latest_result.get("parsed_needs")
    if (
        bool(latest_result.get("ok"))
        and pending_steps
        and str(latest_result.get("agent", "")).strip().lower() != "mainagent"
        and not emitted_artifacts
        and not parsed_needs
        and response_text in EMPTY_RESPONSE_SENTINELS
    ):
        reasons.append("missing_handoff_material")

    repeat_count = latest_result.get("same_task_attempt_count")
    repeat_threshold = latest_result.get("same_task_attempt_threshold")
    if isinstance(repeat_count, int) and isinstance(repeat_threshold, int) and repeat_count > repeat_threshold:
        reasons.append(f"same_task_threshold_exceeded:{repeat_count}>{repeat_threshold}")

    seen: set[str] = set()
    deduped: List[str] = []
    for reason in reasons:
        key = str(reason).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(str(reason).strip())
    return deduped


async def _async_review_collaboration_progress_with_main_agent(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    completed_results: List[Dict[str, Any]],
    latest_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
    trigger_reasons: List[str],
) -> Dict[str, Any]:
    runner = InMemoryRunner(agent=main_agent, app_name="main-collaboration-replanner")
    log_event(
        "event_manager.collaboration",
        "replan_review_started",
        {
            "completed_count": len(completed_results),
            "pending_count": len(pending_steps),
            "open_needs_count": len(open_needs),
            "trigger_reasons": trigger_reasons,
        },
        direction="outbound",
    )
    try:
        agents_desc = _format_available_agents_for_review(available_agents)

        latest_step = latest_result.get("workflow_step")
        latest_agent = str(latest_result.get("agent", "UnknownAgent"))
        latest_status = "ok" if latest_result.get("ok") else "error"
        if latest_result.get("ok"):
            latest_text = _compact_text(str(latest_result.get("response", "")).strip(), max_chars=420)
        else:
            latest_text = _compact_text(str(latest_result.get("error", "Unknown error")).strip(), max_chars=420)

        completed_text = _format_prior_results_for_handoff(completed_results)
        pending_text = _format_remaining_steps(pending_steps)
        needs_text = _format_open_needs_for_prompt(open_needs)
        conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=8, max_chars=1200)
        planner_summary = _compact_text(raw_plan or "(none)", max_chars=1200)
        trigger_text = ", ".join(trigger_reasons) if trigger_reasons else "(none)"

        prompt = (
            "You are the main coordinator reviewing multi-agent progress.\n"
            "Decide whether the remaining plan should change.\n"
            "Return JSON only.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "needs": [\n'
            "    {\n"
            '      "kind": "missing_evidence | user_clarification | missing_context",\n'
            '      "request": "what is needed next",\n'
            '      "required_capabilities": ["capability_key"],\n'
            '      "reason": "why this handoff or follow-up is needed",\n'
            '      "blocking": true\n'
            "    }\n"
            "  ],\n"
            '  "should_update_plan": true,\n'
            '  "updated_steps": [\n'
            "    {\n"
            '      "agent": "AgentName",\n'
            '      "goal": "what to do next",\n'
            '      "deliverable": "expected output"\n'
            "    }\n"
            "  ],\n"
            '  "reason": "short reason"\n'
            "}\n\n"
            "Guidelines:\n"
            "- needs should include only unresolved concrete follow-up needs.\n"
            "- Do not request or plan agent-private skill/tool/session-memory loading; those are internal runtime concerns.\n"
            "- If no plan update is needed, set should_update_plan=false and updated_steps=[].\n"
            "- updated_steps should contain only future steps and use names from Available agents.\n"
            "- Reassign work only when another agent is clearly a better fit based on capability, ownership, and description.\n"
            "- Respect explicit user constraints.\n\n"
            f"Conversation context summary:\n{conversation_summary}\n\n"
            f"User request:\n{user_input}\n\n"
            f"Original planner text summary:\n{planner_summary}\n\n"
            f"Review was triggered because:\n{trigger_text}\n\n"
            f"Latest completed step: {latest_step} ({latest_agent}, {latest_status})\n"
            f"Latest step output:\n{latest_text or '(none)'}\n\n"
            f"Completed outputs so far:\n{completed_text}\n\n"
            f"Current open needs:\n{needs_text}\n\n"
            f"Current pending steps:\n{pending_text}\n\n"
            f"Available agents:\n{agents_desc}\n"
        )

        new_message = types.Content(role="user", parts=[types.Part(text=prompt)])
        chunks = await collect_text_response_with_network_retry(
            runner=runner,
            user_id="event-manager-main-replanner",
            new_message=new_message,
            component="event_manager.collaboration",
            operation_name="collaboration_replan_review",
            retry_details={
                "pending_count": len(pending_steps),
                "open_needs_count": len(open_needs),
            },
            log_event_fn=log_event,
        )

        raw_text = "\n".join(chunks).strip()
        parsed = _extract_json_object(raw_text) or {}

        review_needs = _normalize_need_entries(parsed.get("needs"), source_agent="MainAgent", max_items=12)

        updated_steps = _normalize_replanned_steps(parsed.get("updated_steps"), available_agents)
        should_update = bool(parsed.get("should_update_plan")) and bool(updated_steps)
        reason = str(parsed.get("reason", "")).strip()

        result = {
            "needs": review_needs,
            "should_update_plan": should_update,
            "updated_steps": updated_steps,
            "reason": reason,
        }
        log_event("event_manager.collaboration", "replan_review_completed", result, direction="inbound")
        return result
    except Exception as e:
        log_exception(
            "event_manager.collaboration",
            "replan_review_failed",
            e,
            {"pending_count": len(pending_steps), "open_needs_count": len(open_needs)},
        )
        return {
            "needs": [],
            "should_update_plan": False,
            "updated_steps": [],
            "reason": "replan_review_failed",
        }
    finally:
        await runner.close()


def _review_collaboration_progress_with_main_agent(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    completed_results: List[Dict[str, Any]],
    latest_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
    trigger_reasons: List[str],
) -> Dict[str, Any]:
    return _run_coroutine_sync(
        _async_review_collaboration_progress_with_main_agent(
            main_agent=main_agent,
            available_agents=available_agents,
            user_input=user_input,
            conversation_history=conversation_history,
            raw_plan=raw_plan,
            completed_results=completed_results,
            latest_result=latest_result,
            pending_steps=pending_steps,
            open_needs=open_needs,
            trigger_reasons=trigger_reasons,
        )
    )


async def _async_handle_collaboration_failure_with_main_agent(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    results_so_far: List[Dict[str, Any]],
    failed_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    runner = InMemoryRunner(agent=main_agent, app_name="main-collaboration-failure-handler")
    failed_step = failed_result.get("workflow_step")
    failed_agent = str(failed_result.get("agent", "UnknownAgent"))
    failed_error = _compact_text(str(failed_result.get("error", "Unknown error")).strip(), max_chars=420)
    log_event(
        "event_manager.collaboration",
        "failure_review_started",
        {
            "failed_step": failed_step,
            "failed_agent": failed_agent,
            "pending_count": len(pending_steps),
            "open_needs_count": len(open_needs),
        },
        direction="outbound",
    )
    try:
        agents_desc = _format_available_agents_for_review(available_agents)

        completed_text = _format_prior_results_for_handoff(results_so_far)
        pending_text = _format_remaining_steps(pending_steps)
        needs_text = _format_open_needs_for_prompt(open_needs)
        conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=8, max_chars=1200)
        planner_summary = _compact_text(raw_plan or "(none)", max_chars=1200)

        prompt = (
            "You are the main coordinator handling an interrupted workflow.\n"
            "A collaboration step failed. Decide whether to replan the remaining work or stop.\n"
            "Return JSON only.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "decision": "replan" | "abort",\n'
            '  "root_cause": "what caused the interruption",\n'
            '  "user_message": "concise message to user",\n'
            '  "updated_steps": [\n'
            "    {\n"
            '      "agent": "AgentName",\n'
            '      "goal": "what to do next",\n'
            '      "deliverable": "expected output"\n'
            "    }\n"
            "  ],\n"
            '  "reason": "short reason for decision"\n'
            "}\n\n"
            "Guidelines:\n"
            "- If replan is feasible this turn, set decision=replan and provide updated_steps.\n"
            "- If not feasible, set decision=abort and provide a clear user_message.\n"
            "- Do not create steps for loading agent-private skills, tools, or session memory.\n"
            "- updated_steps must contain only future steps and use names from Available agents.\n"
            "- Use capability, ownership, and description to decide whether to retry, switch agents, or stop.\n"
            "- Respect explicit user constraints.\n\n"
            f"Conversation context summary:\n{conversation_summary}\n\n"
            f"User request:\n{user_input}\n\n"
            f"Original planner text summary:\n{planner_summary}\n\n"
            f"Failed step: {failed_step} ({failed_agent})\n"
            f"Failure detail:\n{failed_error or '(none)'}\n\n"
            f"Execution output so far:\n{completed_text}\n\n"
            f"Current open needs:\n{needs_text}\n\n"
            f"Current pending steps:\n{pending_text}\n\n"
            f"Available agents:\n{agents_desc}\n"
        )

        new_message = types.Content(role="user", parts=[types.Part(text=prompt)])
        chunks = await collect_text_response_with_network_retry(
            runner=runner,
            user_id="event-manager-main-failure-handler",
            new_message=new_message,
            component="event_manager.collaboration",
            operation_name="collaboration_failure_review",
            retry_details={"failed_step": failed_step, "failed_agent": failed_agent},
            log_event_fn=log_event,
        )

        raw_text = "\n".join(chunks).strip()
        parsed = _extract_json_object(raw_text) or {}

        decision = str(parsed.get("decision", "abort")).strip().lower()
        root_cause = str(parsed.get("root_cause", "")).strip()
        user_message = str(parsed.get("user_message", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
        updated_steps = _normalize_replanned_steps(parsed.get("updated_steps"), available_agents)
        should_replan = decision == "replan" and bool(updated_steps)

        result = {
            "decision": "replan" if should_replan else "abort",
            "should_replan": should_replan,
            "updated_steps": updated_steps if should_replan else [],
            "root_cause": root_cause,
            "user_message": user_message,
            "reason": reason,
        }
        log_event("event_manager.collaboration", "failure_review_completed", result, direction="inbound")
        return result
    except Exception as e:
        log_exception(
            "event_manager.collaboration",
            "failure_review_failed",
            e,
            {"failed_step": failed_step, "failed_agent": failed_agent},
        )
        return {
            "decision": "abort",
            "should_replan": False,
            "updated_steps": [],
            "root_cause": "Failure analysis could not be completed.",
            "user_message": "작업 중 오류가 발생해 진행을 중단했습니다. 잠시 후 다시 시도해 주세요.",
            "reason": "failure_review_failed",
        }
    finally:
        await runner.close()


def _handle_collaboration_failure_with_main_agent(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    results_so_far: List[Dict[str, Any]],
    failed_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return _run_coroutine_sync(
        _async_handle_collaboration_failure_with_main_agent(
            main_agent=main_agent,
            available_agents=available_agents,
            user_input=user_input,
            conversation_history=conversation_history,
            raw_plan=raw_plan,
            results_so_far=results_so_far,
            failed_result=failed_result,
            pending_steps=pending_steps,
            open_needs=open_needs,
        )
    )


def _default_timeout_control_result(reason: str) -> Dict[str, Any]:
    return {
        "decision": "abort",
        "should_continue": False,
        "next_step_policy": "resume_pending",
        "replace_pending": False,
        "updated_steps": [],
        "status_summary": "Workflow paused on timeout and could not be safely resumed.",
        "root_cause": "Timeout control review did not produce a valid decision payload.",
        "user_message": "작업이 시간 제한을 넘어 중단되었습니다. 현재 상태를 보존하고 실행을 멈췄습니다.",
        "reason": reason,
    }


def _normalize_timeout_control_result(
    parsed: Dict[str, Any],
    available_agents: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    decision = str(parsed.get("decision", "abort")).strip().lower()
    next_step_policy = str(parsed.get("next_step_policy", "resume_pending")).strip().lower()
    status_summary = str(parsed.get("status_summary", "")).strip()
    root_cause = str(parsed.get("root_cause", "")).strip()
    user_message = str(parsed.get("user_message", "")).strip()
    reason = str(parsed.get("reason", "")).strip()
    raw_updated_steps = parsed.get("updated_steps", [])

    if decision not in {"continue", "abort"}:
        return None
    if next_step_policy not in {"resume_pending", "replace_pending"}:
        return None
    if not isinstance(raw_updated_steps, list):
        raw_updated_steps = []

    status_summary = status_summary or reason or "Workflow paused after a timeout."
    root_cause = root_cause or reason or "A step exceeded its timeout budget."
    user_message = user_message or status_summary
    reason = reason or root_cause

    updated_steps = _normalize_replanned_steps(raw_updated_steps, available_agents)

    if decision == "abort":
        return {
            "decision": "abort",
            "should_continue": False,
            "next_step_policy": "resume_pending",
            "replace_pending": False,
            "updated_steps": [],
            "status_summary": status_summary,
            "root_cause": root_cause,
            "user_message": user_message,
            "reason": reason,
        }

    if next_step_policy == "resume_pending":
        return {
            "decision": "continue",
            "should_continue": True,
            "next_step_policy": next_step_policy,
            "replace_pending": False,
            "updated_steps": [],
            "status_summary": status_summary,
            "root_cause": root_cause,
            "user_message": user_message,
            "reason": reason,
        }

    if not updated_steps:
        return {
            "decision": "continue",
            "should_continue": True,
            "next_step_policy": "resume_pending",
            "replace_pending": False,
            "updated_steps": [],
            "status_summary": status_summary,
            "root_cause": root_cause,
            "user_message": user_message,
            "reason": reason,
        }
    return {
        "decision": "continue",
        "should_continue": True,
        "next_step_policy": next_step_policy,
        "replace_pending": True,
        "updated_steps": updated_steps,
        "status_summary": status_summary,
        "root_cause": root_cause,
        "user_message": user_message,
        "reason": reason,
    }


def _build_timeout_control_packet(
    *,
    workflow_id: str,
    step_counter: int,
    failed_result: Dict[str, Any],
    completed_results: List[Dict[str, Any]],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    failed_agent = str(failed_result.get("agent", "UnknownAgent")).strip() or "UnknownAgent"
    failed_error = str(failed_result.get("error", "Unknown error")).strip()
    error_type = str(failed_result.get("error_type", "")).strip()
    timeout_sec = failed_result.get("timeout_sec")
    elapsed_sec = failed_result.get("elapsed_sec")

    timeout_payload: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "failed_step": step_counter,
        "failed_agent": failed_agent,
        "error": failed_error,
        "error_type": error_type,
        "is_timeout": bool(failed_result.get("is_timeout")),
        "completed_count": len(completed_results),
        "pending_count": len(pending_steps),
        "open_needs_count": len(open_needs),
        "completed_results_summary": _format_prior_results_for_handoff(completed_results),
        "pending_steps_summary": _format_remaining_steps(pending_steps),
        "open_needs": [
            _build_need_context_entry(item)
            for item in open_needs[:30]
            if isinstance(item, dict)
        ],
    }
    if isinstance(timeout_sec, (int, float)):
        timeout_payload["timeout_sec"] = float(timeout_sec)
    if isinstance(elapsed_sec, (int, float)):
        timeout_payload["elapsed_sec"] = float(elapsed_sec)
    return timeout_payload


async def _async_handle_timeout_with_main_agent(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    timeout_packet: Dict[str, Any],
) -> Dict[str, Any]:
    runner = InMemoryRunner(agent=main_agent, app_name="main-timeout-control")
    log_event(
        "event_manager.collaboration",
        "timeout_control_review_started",
        {
            "failed_step": timeout_packet.get("failed_step"),
            "failed_agent": timeout_packet.get("failed_agent"),
            "pending_count": timeout_packet.get("pending_count"),
            "open_needs_count": timeout_packet.get("open_needs_count"),
        },
        direction="outbound",
    )
    try:
        agents_desc = _format_available_agents_for_review(available_agents)
        timeout_packet_json = json.dumps(timeout_packet, ensure_ascii=False, indent=2)
        conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=8, max_chars=1200)
        planner_summary = _compact_text(raw_plan or "(none)", max_chars=1200)

        prompt = (
            "You are the main coordinator handling a timeout pause.\n"
            "A workflow step exceeded its timeout budget and execution is paused.\n"
            "Decide whether to continue from current state or abort.\n"
            "Return JSON only.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "decision": "continue" | "abort",\n'
            '  "next_step_policy": "resume_pending" | "replace_pending",\n'
            '  "status_summary": "one-sentence status of current workflow",\n'
            '  "root_cause": "what caused timeout in practical terms",\n'
            '  "user_message": "concise user-facing status update",\n'
            '  "reason": "short decision reason",\n'
            '  "updated_steps": [\n'
            "    {\n"
            '      "agent": "AgentName",\n'
            '      "goal": "what to do next",\n'
            '      "deliverable": "expected output"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Guidelines:\n"
            "- Use replace_pending only when the current pending steps should be replaced.\n"
            "- Do not create steps for loading agent-private skills, tools, or session memory.\n"
            "- updated_steps must use names from Available agents.\n"
            "- Use capability, ownership, and description to choose agents.\n"
            "- Keep the decision grounded in the current workflow status packet.\n"
            "- Respect explicit user constraints.\n\n"
            f"Conversation context summary:\n{conversation_summary}\n\n"
            f"User request:\n{user_input}\n\n"
            f"Original planner text summary:\n{planner_summary}\n\n"
            f"Timeout status packet:\n{timeout_packet_json}\n\n"
            f"Available agents:\n{agents_desc}\n"
        )

        new_message = types.Content(role="user", parts=[types.Part(text=prompt)])
        chunks = await collect_text_response_with_network_retry(
            runner=runner,
            user_id="event-manager-main-timeout-control",
            new_message=new_message,
            component="event_manager.collaboration",
            operation_name="timeout_control_review",
            retry_details={
                "failed_step": timeout_packet.get("failed_step"),
                "failed_agent": timeout_packet.get("failed_agent"),
            },
            log_event_fn=log_event,
        )

        raw_text = "\n".join(chunks).strip()
        parsed = _extract_json_object(raw_text)
        if not isinstance(parsed, dict):
            log_event(
                "event_manager.collaboration",
                "timeout_control_invalid_json",
                {"raw_text": raw_text[:2000]},
                level="ERROR",
            )
            return _default_timeout_control_result("invalid_json")

        normalized = _normalize_timeout_control_result(parsed, available_agents)
        if not isinstance(normalized, dict):
            log_event(
                "event_manager.collaboration",
                "timeout_control_invalid_schema",
                {"parsed": parsed},
                level="ERROR",
            )
            return _default_timeout_control_result("invalid_schema")

        log_event("event_manager.collaboration", "timeout_control_review_completed", normalized, direction="inbound")
        return normalized
    except Exception as e:
        log_exception(
            "event_manager.collaboration",
            "timeout_control_review_failed",
            e,
            {
                "failed_step": timeout_packet.get("failed_step"),
                "failed_agent": timeout_packet.get("failed_agent"),
            },
        )
        return _default_timeout_control_result("timeout_control_review_failed")
    finally:
        await runner.close()


def _handle_timeout_with_main_agent(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    timeout_packet: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_coroutine_sync(
        _async_handle_timeout_with_main_agent(
            main_agent=main_agent,
            available_agents=available_agents,
            user_input=user_input,
            conversation_history=conversation_history,
            raw_plan=raw_plan,
            timeout_packet=timeout_packet,
        )
    )


def _run_collaboration_workflow(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    workflow_id: str,
    raw_plan: str,
    steps: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    initial_results: List[Dict[str, Any]] | None = None,
    initial_artifact_store: List[Dict[str, Any]] | None = None,
    initial_open_needs: List[Dict[str, Any]] | None = None,
    initial_step_counter: int = 0,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = list(initial_results or [])
    artifact_store: List[Dict[str, Any]] = list(initial_artifact_store or [])
    seen_artifact_keys: set[str] = set()
    for item in artifact_store:
        key = _artifact_identity_key(item)
        if key:
            seen_artifact_keys.add(key)
    pending_steps: List[Dict[str, Any]] = list(steps)
    open_needs: List[Dict[str, Any]] = list(initial_open_needs or [])
    seen_need_keys: set[str] = set()
    for item in open_needs:
        key = _need_identity_key(item)
        if key:
            seen_need_keys.add(key)
    step_attempt_counts: Dict[str, int] = {}
    activated_agent_cards_map: Dict[str, Dict[str, Any]] = {}
    step_counter = initial_step_counter
    max_steps = _resolve_collaboration_max_steps(len(pending_steps) + len(results))
    same_task_review_threshold = _resolve_same_task_review_threshold()

    while pending_steps and step_counter < max_steps:
        step_counter += 1
        step = pending_steps.pop(0)
        agent_name = str(step.get("agent", "UnknownAgent"))
        goal = str(step.get("goal", "")).strip()
        step_signature = _step_signature(step)
        same_task_attempt_count = step_attempt_counts.get(step_signature, 0) + 1
        step_attempt_counts[step_signature] = same_task_attempt_count
        pre_step_open_needs = list(open_needs)
        current_agent_meta = step.get("agent_meta", {})
        workflow_memory_snapshot: Dict[str, Any] | None = None
        if isinstance(current_agent_meta, dict):
            snapshot = _build_agent_card_snapshot(current_agent_meta)
            snapshot_name = str(snapshot.get("name", "")).strip().lower()
            if snapshot_name:
                activated_agent_cards_map[snapshot_name] = snapshot
        activated_agent_cards = list(activated_agent_cards_map.values())
        total_steps_hint = step_counter + len(pending_steps)
        input_artifacts = _select_input_artifacts_for_step(
            artifact_store=artifact_store,
        )
        log_event(
            "event_manager.collaboration",
            "step_started",
            {
                "step": step_counter,
                "total_steps_hint": total_steps_hint,
                "agent": agent_name,
                "goal": goal,
                "task_signature": step_signature,
                "same_task_attempt_count": same_task_attempt_count,
                "same_task_attempt_threshold": same_task_review_threshold,
                "open_needs": [
                    _build_need_context_entry(item)
                    for item in open_needs
                    if isinstance(item, dict)
                ],
                "input_artifact_count": len(input_artifacts),
                "input_artifact_ids": [str(item.get("id", "")) for item in input_artifacts],
                "activated_agent_cards": [str(item.get("name", "")) for item in activated_agent_cards],
            },
            direction="outbound",
        )
        delivery_gate_error = _delivery_gate_error(
            step=step,
            open_needs=open_needs,
            results=results,
            artifact_store=artifact_store,
        )
        if delivery_gate_error:
            log_event(
                "event_manager.collaboration",
                "delivery_step_blocked_by_gate",
                {
                    "step": step_counter,
                    "agent": agent_name,
                    "goal": goal,
                    "reason": delivery_gate_error,
                },
                level="WARNING",
            )
            result = {
                "ok": False,
                "agent": agent_name,
                "error": delivery_gate_error,
                "error_type": "delivery_gate",
                "skipped": True,
            }
        else:
            step_input = _build_collaboration_step_input(
                workflow_id=workflow_id,
                user_input=user_input,
                prior_results=results,
                input_artifacts=input_artifacts,
                open_needs=open_needs,
                remaining_steps=pending_steps,
                available_agents=available_agents,
                step=step,
                step_index=step_counter,
                total_steps_hint=total_steps_hint,
            )
            _log_agent_message(
                action="sent",
                from_agent="MainAgent",
                to_agent=agent_name,
                message=step_input,
                channel="collaboration_step",
                workflow_id=workflow_id,
                workflow_step=step_counter,
                direction="outbound",
            )

            workflow_memory_snapshot = _build_workflow_memory_snapshot(
                workflow_id=workflow_id,
                user_input=user_input,
                current_agent=agent_name,
                current_step=step_counter,
                results=results,
                artifact_store=artifact_store,
                open_needs=open_needs,
                pending_steps=pending_steps,
                activated_agent_cards=activated_agent_cards,
            )
            result = _execute_single_agent(
                step["agent_meta"],
                step_input,
                workflow_memory_snapshot=workflow_memory_snapshot,
            )
        enriched = dict(result)
        enriched["workflow_step"] = step_counter
        enriched["goal"] = goal
        enriched["task_signature"] = step_signature
        enriched["same_task_attempt_count"] = same_task_attempt_count
        enriched["same_task_attempt_threshold"] = same_task_review_threshold
        enriched["input_artifacts"] = input_artifacts
        raw_agent_output = (
            str(enriched.get("response", "")).strip()
            if enriched.get("ok")
            else str(enriched.get("error", "")).strip()
        )
        if enriched.get("ok"):
            structured_output = _extract_structured_agent_output(
                raw_agent_output,
                source_agent=agent_name,
                workflow_id=workflow_id,
                workflow_step=step_counter,
            )
            bootstrap_only_detected = _apply_bootstrap_only_violation(
                structured_output=structured_output,
                normalized_parts=enriched.get("normalized_response_parts", []),
            )
            if bootstrap_only_detected:
                log_event(
                    "event_manager.collaboration",
                    "worker_output_bootstrap_only_detected",
                    {
                        "step": step_counter,
                        "agent": agent_name,
                        "normalized_response_parts": list(enriched.get("normalized_response_parts", []))
                        if isinstance(enriched.get("normalized_response_parts", []), list)
                        else [],
                    },
                    level="WARNING",
                )
            enriched["raw_response"] = structured_output["raw_response"]
            enriched["structured_output"] = structured_output
            enriched["output_status"] = structured_output["status"]
            enriched["response"] = (
                str(structured_output.get("text_response", "")).strip()
                or str(structured_output.get("summary", "")).strip()
            )
            if not bool(structured_output.get("contract_valid")):
                violations = list(structured_output.get("contract_violations", []))
                log_event(
                    "event_manager.collaboration",
                    "worker_output_contract_violated",
                    {
                        "step": step_counter,
                        "agent": agent_name,
                        "violations": violations,
                        "contract_source": structured_output.get("contract_source"),
                    },
                    level="WARNING",
                )
                repair = _attempt_worker_output_contract_repair(
                    agent_meta=step["agent_meta"],
                    agent_name=agent_name,
                    workflow_id=workflow_id,
                    workflow_step=step_counter,
                    invalid_response=raw_agent_output,
                    violations=violations,
                    workflow_memory_snapshot=workflow_memory_snapshot,
                )
                enriched["contract_repair_attempted"] = True
                enriched["contract_repair_success"] = bool(repair.get("success"))
                enriched["contract_repair_raw_response"] = str(repair.get("raw_output", "")).strip()
                if repair.get("success"):
                    repaired_result = dict(repair.get("result", {}))
                    repaired_structured = dict(repair.get("structured_output", {}))
                    for key in [
                        "elapsed_sec",
                        "timeout_sec",
                        "is_timeout",
                        "error_type",
                        "repair_count",
                    ]:
                        if key in repaired_result:
                            enriched[key] = repaired_result.get(key)
                    enriched["raw_response"] = str(repaired_structured.get("raw_response", "")).strip()
                    enriched["structured_output"] = repaired_structured
                    enriched["output_status"] = str(repaired_structured.get("status", "")).strip()
                    enriched["response"] = (
                        str(repaired_structured.get("text_response", "")).strip()
                        or str(repaired_structured.get("summary", "")).strip()
                    )
                else:
                    final_violations = list(
                        dict.fromkeys(
                            violations
                            + list(
                                dict(repair.get("structured_output", {})).get("contract_violations", [])
                                if isinstance(repair.get("structured_output", {}), dict)
                                else []
                            )
                        )
                    )
                    contract_error = (
                        "Worker output contract violated after one repair retry: "
                        + (", ".join(final_violations) if final_violations else "invalid structured response")
                    )
                    log_event(
                        "event_manager.collaboration",
                        "worker_output_contract_repair_failed",
                        {
                            "step": step_counter,
                            "agent": agent_name,
                            "violations": final_violations,
                        },
                        level="ERROR",
                    )
                    enriched["ok"] = False
                    enriched["error"] = contract_error
                    enriched["error_type"] = "worker_output_contract_violation"
                    enriched["response"] = ""
        _log_agent_message(
            action="received",
            from_agent=agent_name,
            to_agent="MainAgent",
            message=raw_agent_output,
            channel="collaboration_step",
            workflow_id=workflow_id,
            workflow_step=step_counter,
            ok=bool(enriched.get("ok")),
            direction="inbound",
        )
        if enriched.get("ok"):
            structured_output = enriched.get("structured_output", {})
            parsed_needs = list(structured_output.get("needs", [])) if isinstance(structured_output, dict) else []
            parsed_artifacts = list(structured_output.get("artifacts", [])) if isinstance(structured_output, dict) else []
        else:
            parsed_needs = []
            parsed_artifacts = []
        enriched["parsed_needs"] = parsed_needs
        enriched["artifacts_emitted"] = parsed_artifacts
        results.append(enriched)

        added_artifacts = _register_new_artifacts(
            artifact_store=artifact_store,
            new_artifacts=parsed_artifacts,
            seen_artifact_keys=seen_artifact_keys,
            source_agent=agent_name,
            workflow_step=step_counter,
        )
        if added_artifacts:
            enriched["artifact_ids"] = [str(item.get("id", "")) for item in added_artifacts]

        added_needs = _append_unique_open_needs(
            open_needs=open_needs,
            new_needs=parsed_needs,
            seen_need_keys=seen_need_keys,
        )
        if added_needs:
            _log_emitted_needs(
                needs=added_needs,
                source_agent=agent_name,
                workflow_id=workflow_id,
                workflow_step=step_counter,
                open_needs=open_needs,
            )

        if enriched.get("ok"):
            clarification_request = _first_user_clarification_request(added_needs)
            clarification_source = "agent_output_needs"
            if not clarification_request:
                clarification_request = _first_user_clarification_request(open_needs)
                clarification_source = "agent_output_open_needs"
            if clarification_request:
                _mark_workflow_paused_for_user_input(
                    result=enriched,
                    request=clarification_request,
                    step=step_counter,
                    agent=agent_name,
                    source=clarification_source,
                )
                break

        if enriched.get("ok") and pre_step_open_needs:
            open_needs, resolved_by_step = _resolve_open_needs_for_completed_agent(
                open_needs=open_needs,
                previous_open_needs=pre_step_open_needs,
                agent_name=agent_name,
                agent_meta=step.get("agent_meta", {}),
            )
            for need in resolved_by_step:
                request = _normalize_need_text(str(need.get("request", "")).strip())
                log_event(
                    "workflow.need",
                    "need_resolved_by_step_completion",
                    {
                        "resolved_by_agent": agent_name,
                        "request": request or _need_display_text(need),
                        "workflow_step": step_counter,
                    },
                )
            if resolved_by_step:
                log_event(
                    "event_manager.collaboration",
                    "open_needs_resolved_by_step",
                    {
                        "resolved_needs": [_build_need_context_entry(item) for item in resolved_by_step],
                        "remaining_open_needs": [_build_need_context_entry(item) for item in open_needs],
                    },
                    direction="inbound",
                )

        log_event(
            "event_manager.collaboration",
            "step_completed",
            {
                "step": step_counter,
                "total_steps_hint": total_steps_hint,
                "agent": agent_name,
                "ok": bool(enriched.get("ok")),
                "task_signature": step_signature,
                "same_task_attempt_count": same_task_attempt_count,
                "same_task_attempt_threshold": same_task_review_threshold,
            },
            direction="inbound",
        )

        if same_task_attempt_count > same_task_review_threshold:
            log_event(
                "event_manager.collaboration",
                "same_task_threshold_exceeded",
                {
                    "step": step_counter,
                    "agent": agent_name,
                    "goal": goal,
                    "task_signature": step_signature,
                    "same_task_attempt_count": same_task_attempt_count,
                    "same_task_attempt_threshold": same_task_review_threshold,
                },
                level="WARNING",
            )

        if not enriched.get("ok"):
            if bool(enriched.get("is_timeout")):
                timeout_packet = _build_timeout_control_packet(
                    workflow_id=workflow_id,
                    step_counter=step_counter,
                    failed_result=enriched,
                    completed_results=results,
                    pending_steps=pending_steps,
                    open_needs=open_needs,
                )
                timeout_packet_json = json.dumps(timeout_packet, ensure_ascii=False, indent=2)
                log_event(
                    "event_manager.collaboration",
                    "workflow_paused_on_timeout",
                    {
                        "step": step_counter,
                        "agent": agent_name,
                        "timeout_sec": enriched.get("timeout_sec"),
                        "elapsed_sec": enriched.get("elapsed_sec"),
                    },
                )
                _log_agent_message(
                    action="sent",
                    from_agent="WorkflowEngine",
                    to_agent="MainAgent",
                    message=timeout_packet_json,
                    channel="timeout_control",
                    workflow_id=workflow_id,
                    workflow_step=step_counter,
                    direction="outbound",
                )
                timeout_review = _handle_timeout_with_main_agent(
                    main_agent=main_agent,
                    available_agents=available_agents,
                    user_input=user_input,
                    conversation_history=conversation_history,
                    raw_plan=raw_plan,
                    timeout_packet=timeout_packet,
                )
                _log_agent_message(
                    action="received",
                    from_agent="MainAgent",
                    to_agent="WorkflowEngine",
                    message=json.dumps(timeout_review, ensure_ascii=False),
                    channel="timeout_control",
                    workflow_id=workflow_id,
                    workflow_step=step_counter,
                    ok=bool(timeout_review.get("should_continue")),
                    direction="inbound",
                )

                decision = str(timeout_review.get("decision", "abort")).strip().lower() or "abort"
                reason = str(timeout_review.get("reason", "")).strip()
                root_cause = str(timeout_review.get("root_cause", "")).strip()
                user_message = str(timeout_review.get("user_message", "")).strip()
                status_summary = str(timeout_review.get("status_summary", "")).strip()

                _append_error_detail(enriched, "Timeout Review", root_cause)

                enriched["timeout_recovery"] = {
                    "decision": decision,
                    "reason": reason,
                    "root_cause": root_cause,
                    "user_message": user_message,
                    "status_summary": status_summary,
                    "next_step_policy": str(timeout_review.get("next_step_policy", "resume_pending")).strip(),
                }

                if timeout_review.get("should_continue"):
                    replaced_pending = _maybe_replace_pending_steps_from_review(
                        should_replace=bool(timeout_review.get("replace_pending")),
                        updated_steps=timeout_review.get("updated_steps"),
                        event_name="plan_updated_after_timeout",
                        failed_step=step_counter,
                        failed_agent=agent_name,
                        reason=reason,
                    )
                    if replaced_pending is not None:
                        pending_steps = replaced_pending
                    log_event(
                        "event_manager.collaboration",
                        "workflow_resumed_after_timeout",
                        {
                            "failed_step": step_counter,
                            "agent": agent_name,
                            "decision": decision,
                            "reason": reason,
                        },
                    )
                    continue

                _append_coordinator_message(enriched, user_message)

                log_event(
                    "event_manager.collaboration",
                    "workflow_stopped_on_timeout",
                    {
                        "failed_step": step_counter,
                        "agent": agent_name,
                        "decision": decision,
                        "reason": reason,
                        "root_cause": root_cause,
                    },
                    level="ERROR",
                )
                break

            failure_review = _handle_collaboration_failure_with_main_agent(
                main_agent=main_agent,
                available_agents=available_agents,
                user_input=user_input,
                conversation_history=conversation_history,
                raw_plan=raw_plan,
                results_so_far=results,
                failed_result=enriched,
                pending_steps=pending_steps,
                open_needs=open_needs,
            )

            root_cause = str(failure_review.get("root_cause", "")).strip()
            user_message = str(failure_review.get("user_message", "")).strip()
            decision = str(failure_review.get("decision", "abort")).strip().lower() or "abort"
            reason = str(failure_review.get("reason", "")).strip()

            _append_error_detail(enriched, "Failure Analysis", root_cause)

            enriched["failure_recovery"] = {
                "decision": decision,
                "reason": reason,
                "root_cause": root_cause,
                "user_message": user_message,
            }

            replaced_pending = _maybe_replace_pending_steps_from_review(
                should_replace=bool(failure_review.get("should_replan")),
                updated_steps=failure_review.get("updated_steps"),
                event_name="plan_recovered_from_error",
                failed_step=step_counter,
                failed_agent=agent_name,
                reason=reason,
            )
            if replaced_pending is not None:
                pending_steps = replaced_pending
                continue

            _append_coordinator_message(enriched, user_message)

            log_event(
                "event_manager.collaboration",
                "workflow_stopped_on_error",
                {
                    "failed_step": step_counter,
                    "agent": agent_name,
                    "decision": decision,
                    "reason": reason,
                    "root_cause": root_cause,
                },
                level="ERROR",
            )
            break

        review_trigger_reasons = _collect_progress_review_reasons(
            latest_result=enriched,
            pending_steps=pending_steps,
            open_needs=open_needs,
        )
        review = _run_progress_review_or_skip(
            review_trigger_reasons=review_trigger_reasons,
            main_agent=main_agent,
            available_agents=available_agents,
            user_input=user_input,
            conversation_history=conversation_history,
            raw_plan=raw_plan,
            results=results,
            enriched=enriched,
            pending_steps=pending_steps,
            open_needs=open_needs,
            step_counter=step_counter,
            agent_name=agent_name,
        )

        review_added_needs = _append_unique_open_needs(
            open_needs=open_needs,
            new_needs=list(review.get("needs", [])),
            seen_need_keys=seen_need_keys,
        )
        if review.get("needs"):
            log_event(
                "event_manager.collaboration",
                "open_needs_updated",
                {"open_needs": [_build_need_context_entry(item) for item in open_needs]},
                direction="inbound",
            )

        clarification_request = _first_user_clarification_request(review_added_needs)
        clarification_source = "review_needs"
        if not clarification_request:
            clarification_request = _first_user_clarification_request(open_needs)
            clarification_source = "review_open_needs"
        if clarification_request:
            _mark_workflow_paused_for_user_input(
                result=enriched,
                request=clarification_request,
                step=step_counter,
                agent=agent_name,
                source=clarification_source,
            )
            break

        review_updated_plan, pending_steps = _apply_progress_review_plan_update(
            review=review,
            pending_steps=pending_steps,
            completed_agent=agent_name,
            completed_goal=goal,
        )
        if not review_updated_plan:
            fallback_plan = _build_indirect_delegation_fallback_steps(
                open_needs=open_needs,
                available_agents=available_agents,
                pending_steps=pending_steps,
            )
            fallback_steps = [
                item
                for item in fallback_plan.get("steps", [])
                if isinstance(item, dict)
            ]
            consumed_need_keys = {
                str(item).strip().lower()
                for item in fallback_plan.get("consumed_need_keys", [])
                if str(item).strip()
            }
            if fallback_steps:
                pending_steps = fallback_steps + pending_steps
                if consumed_need_keys:
                    removed_needs = [
                        need
                        for need in open_needs
                        if _need_identity_key(need) in consumed_need_keys
                    ]
                    open_needs = [
                        need
                        for need in open_needs
                        if _need_identity_key(need) not in consumed_need_keys
                    ]
                    for removed in removed_needs:
                        request = _normalize_need_text(str(removed.get("request", "")).strip())
                        log_event(
                            "workflow.need",
                            "need_resolved_by_plan_augmentation",
                            {
                                "required_capabilities": _normalize_need_capabilities(removed.get("required_capabilities", [])),
                                "request": request or _need_display_text(removed),
                            },
                        )
                log_event(
                    "event_manager.collaboration",
                    "plan_augmented_from_open_needs",
                    {
                        "added_steps": _pending_step_log_entries(fallback_steps),
                        "remaining_open_needs": [_build_need_context_entry(item) for item in open_needs],
                    },
                )

    if step_counter >= max_steps and pending_steps:
        clarification_request = _first_user_clarification_request(open_needs)
        if clarification_request:
            pause_entry: Dict[str, Any] = {
                "ok": True,
                "agent": "MainAgent",
                "workflow_step": step_counter,
                "goal": "사용자 입력이 필요한 후속 단계",
                "response": "추가 정보가 필요하여 진행을 일시 중단하고 사용자 응답을 기다립니다.",
                "workflow_paused": True,
                "pause_reason": "awaiting_user_clarification",
                "pause_request": clarification_request,
                "parsed_needs": [],
            }
            results.append(pause_entry)
            log_event(
                "event_manager.collaboration",
                "workflow_paused_for_user_input",
                {
                    "step": step_counter,
                    "agent": "MainAgent",
                    "request": clarification_request,
                    "source": "max_steps_guard_open_needs",
                    "max_steps": max_steps,
                    "remaining_steps": len(pending_steps),
                },
            )
        else:
            log_event(
                "event_manager.collaboration",
                "workflow_stopped_on_max_steps",
                {"max_steps": max_steps, "remaining_steps": len(pending_steps)},
                level="ERROR",
            )

    return {
        "workflow_id": workflow_id,
        "results": results,
        "artifact_store": artifact_store,
        "open_needs": open_needs,
        "pending_steps": pending_steps,
        "step_counter": step_counter,
    }


def _extract_pause_request_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for item in reversed(results):
        if not isinstance(item, dict):
            continue
        if not bool(item.get("workflow_paused")):
            continue
        request = str(item.get("pause_request", "")).strip()
        if not request:
            request = "추가 정보가 필요합니다. 필요한 범위/조건을 알려주세요."
        return {
            "workflow_step": item.get("workflow_step"),
            "agent": str(item.get("agent", "MainAgent")).strip() or "MainAgent",
            "request": request,
        }
    return None


def _serialize_pending_steps_for_snapshot(pending_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for step in pending_steps:
        if not isinstance(step, dict):
            continue
        agent = str(step.get("agent", "")).strip()
        goal = str(step.get("goal", "")).strip()
        deliverable = str(step.get("deliverable", "")).strip()
        if not agent:
            continue
        serialized.append(
            {
                "agent": agent,
                "goal": goal[:1000],
                "deliverable": deliverable[:1000],
            }
        )
    return serialized


def _copy_open_needs_for_snapshot(open_needs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    for need in open_needs:
        if not isinstance(need, dict):
            continue
        copied.append(
            {
                "kind": _normalize_need_kind(str(need.get("kind", "")).strip()),
                "request": _normalize_need_text(str(need.get("request", "")).strip()),
                "required_capabilities": _normalize_need_capabilities(need.get("required_capabilities", [])),
                "reason": str(need.get("reason", "")).strip(),
                "blocking": bool(need.get("blocking")),
                "source_agent": str(need.get("source_agent", "")).strip(),
                "workflow_step": need.get("workflow_step"),
            }
        )
    return copied


def _copy_artifacts_for_snapshot(artifact_store: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    for artifact in artifact_store:
        if not isinstance(artifact, dict):
            continue
        copied.append(dict(artifact))
    return copied


def _build_paused_workflow_snapshot(
    *,
    workflow_id: str,
    raw_plan: str,
    collaboration_plan: Any,
    original_user_input: str,
    conversation_history: str,
    workflow_state: Dict[str, Any],
    pause_payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "raw_plan": raw_plan,
        "collaboration_plan": collaboration_plan if isinstance(collaboration_plan, dict) else {},
        "original_user_input": str(original_user_input or ""),
        "conversation_history": str(conversation_history or ""),
        "pause_request": str(pause_payload.get("request", "")).strip(),
        "pause_agent": str(pause_payload.get("agent", "MainAgent")).strip() or "MainAgent",
        "pause_step": pause_payload.get("workflow_step"),
        "pending_steps": _serialize_pending_steps_for_snapshot(list(workflow_state.get("pending_steps", []))),
        "open_needs": _copy_open_needs_for_snapshot(list(workflow_state.get("open_needs", []))),
        "artifact_store": _copy_artifacts_for_snapshot(list(workflow_state.get("artifact_store", []))),
        "completed_results": [dict(item) for item in list(workflow_state.get("results", [])) if isinstance(item, dict)],
    }


def _merge_user_input_with_clarification(
    *,
    original_user_input: str,
    pause_request: str,
    clarification_response: str,
) -> str:
    merged_parts = [str(original_user_input or "").strip()]
    request = str(pause_request or "").strip()
    response = str(clarification_response or "").strip()
    if request and response:
        merged_parts.append(f"User clarification for '{request}': {response}")
    elif response:
        merged_parts.append(f"User clarification: {response}")
    return "\n\n".join(part for part in merged_parts if part).strip()


def _build_user_clarification_artifact(
    *,
    workflow_id: str,
    clarification_request: str,
    clarification_response: str,
) -> Dict[str, Any]:
    summary = _compact_text(str(clarification_response or "").strip(), max_chars=240)
    title = _compact_text(str(clarification_request or "User clarification").strip(), max_chars=120)
    return {
        "id": f"{workflow_id}:clarification:{uuid4().hex[:8]}",
        "type": "user_clarification",
        "title": title or "User clarification",
        "summary": summary or "(empty clarification response)",
        "source_agent": "User",
        "workflow_step": 0,
        "request": str(clarification_request or "").strip(),
        "response": str(clarification_response or "").strip(),
    }


def _build_agent_input(user_input: str, conversation_history: str) -> str:
    request_brief = _compact_text(user_input, max_chars=700)
    if request_brief:
        return request_brief
    if conversation_history.strip():
        return _summarize_conversation_history(conversation_history, max_turn_lines=3, max_chars=320)
    return ""


def _fallback_collaboration_steps_from_agents(
    executable_agents: List[Dict[str, Any]],
    user_input: str,
) -> List[Dict[str, Any]]:
    if len(executable_agents) != 1:
        return []

    only_agent = executable_agents[0]
    agent_name = str(only_agent.get("name", "UnknownAgent")).strip() or "UnknownAgent"
    return [
        {
            "agent": agent_name,
            "goal": (
                "Handle the user request directly and provide a handoff-ready output.\n"
                f"User request: {user_input}"
            )[:1000],
            "deliverable": "Direct result with the most relevant evidence or action.",
            "agent_meta": only_agent,
        }
    ]


def execute_plan_detailed(
    plan: Dict[str, Any],
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    raw_plan = str(plan.get("raw_plan", ""))
    user_input = str(plan.get("meta", {}).get("user_input", ""))
    collaboration_plan = plan.get("meta", {}).get("collaboration_plan", {})
    context_map = context or {}
    conversation_history = str(context_map.get("conversation_history", ""))
    session_id = str(context_map.get("session_id", "")).strip() or "default"
    workflow_id = f"{session_id}-{uuid4().hex[:8]}"
    agent_input = _build_agent_input(user_input=user_input, conversation_history=conversation_history)
    log_event(
        "event_manager",
        "execute_plan_started",
        {
            "raw_plan": raw_plan,
            "user_input": user_input,
            "session_id": session_id,
            "workflow_id": workflow_id,
            "collaboration_plan": collaboration_plan if isinstance(collaboration_plan, dict) else {},
            "num_available_agents": len(available_agents),
        },
    )

    executable_agents = [agent for agent in available_agents if _is_local_agent(agent) or _is_a2a_agent(agent)]
    collaboration_steps = _extract_collaboration_steps(
        collaboration_plan=collaboration_plan,
        available_agents=executable_agents,
    )
    if not collaboration_steps and user_input:
        collaboration_steps = _fallback_collaboration_steps_from_agents(executable_agents, user_input)
    if collaboration_steps and user_input:
        log_event(
            "event_manager.collaboration",
            "workflow_selected",
            {
                "source": "planner_collaboration_plan" if collaboration_plan else "single_agent_fallback",
                "steps": [
                    {
                        "step": idx + 1,
                        "agent": str(step.get("agent", "")),
                        "goal": str(step.get("goal", "")),
                    }
                    for idx, step in enumerate(collaboration_steps)
                ],
            },
        )
        workflow_state = _run_collaboration_workflow(
            main_agent=main_agent,
            available_agents=executable_agents,
            workflow_id=workflow_id,
            raw_plan=raw_plan,
            steps=collaboration_steps,
            user_input=user_input,
            conversation_history=conversation_history,
        )
        results = list(workflow_state.get("results", []))
        pause_payload = _extract_pause_request_from_results(results)
        if pause_payload:
            paused_workflow = _build_paused_workflow_snapshot(
                workflow_id=workflow_id,
                raw_plan=raw_plan,
                collaboration_plan=collaboration_plan,
                original_user_input=user_input,
                conversation_history=conversation_history,
                workflow_state=workflow_state,
                pause_payload=pause_payload,
            )
            log_event(
                "event_manager.collaboration",
                "workflow_paused_response_returned",
                pause_payload,
            )
            request = str(pause_payload.get("request", "")).strip()
            return {
                "output_text": (
                    "진행을 일시 중단하고 사용자 응답을 기다립니다.\n\n"
                    f"{request}"
                ),
                "raw_plan": raw_plan,
                "workflow_id": workflow_id,
                "results": results,
                "paused_workflow": paused_workflow,
            }
        formatted = _format_execution_output(raw_plan=raw_plan, results=results)
        final_summary = _summarize_collaboration_with_main_agent(
            main_agent=main_agent,
            user_input=user_input,
            conversation_history=conversation_history,
            raw_plan=raw_plan,
            results=results,
        )
        final_summary = _ensure_summary_agent_sections(final_summary, results)
        if final_summary:
            formatted = f"{formatted}\n\n=== Final Summary ===\n{final_summary}"
        log_event("event_manager", "collaboration_execution_completed", {"results": results})
        return {
            "output_text": formatted,
            "raw_plan": raw_plan,
            "workflow_id": workflow_id,
            "results": results,
            "paused_workflow": None,
        }

    a2a_agents = [agent for agent in available_agents if _is_a2a_agent(agent)]
    if len(a2a_agents) == 1 and user_input:
        direct_agent_name = str(a2a_agents[0].get("name", "UnknownA2AAgent")).strip() or "UnknownA2AAgent"
        log_event(
            "event_manager",
            "a2a_execution_selected",
            {"agent": direct_agent_name, "base_url": a2a_agents[0].get("base_url", "")},
        )
        _log_agent_message(
            action="sent",
            from_agent="MainAgent",
            to_agent=direct_agent_name,
            message=agent_input,
            channel="direct_a2a",
            workflow_id=workflow_id,
            direction="outbound",
        )
        direct_result = _execute_single_a2a_agent(a2a_agents[0], agent_input)
        direct_message = (
            str(direct_result.get("response", "")).strip()
            if direct_result.get("ok")
            else str(direct_result.get("error", "")).strip()
        )
        _log_agent_message(
            action="received",
            from_agent=direct_agent_name,
            to_agent="MainAgent",
            message=direct_message,
            channel="direct_a2a",
            workflow_id=workflow_id,
            ok=bool(direct_result.get("ok")),
            direction="inbound",
        )
        return {
            "output_text": str(direct_result),
            "raw_plan": raw_plan,
            "workflow_id": workflow_id,
            "results": [direct_result] if isinstance(direct_result, dict) else [],
            "paused_workflow": None,
        }

    fallback = raw_plan or "No plan was generated."
    log_event("event_manager", "execute_plan_fallback", {"result": fallback})
    return {
        "output_text": fallback,
        "raw_plan": raw_plan,
        "workflow_id": workflow_id,
        "results": [],
        "paused_workflow": None,
    }


def _resume_paused_workflow_acknowledgement(
    *,
    pause_request: str,
    clarification_response: str,
) -> str:
    request = str(pause_request or "").strip()
    response = str(clarification_response or "").strip()
    if request and response:
        return f"확인했습니다. '{request}'에 대해 '{response}'라고 답해 주셔서 해당 확인 단계를 종료했습니다."
    if response:
        return f"확인했습니다. 추가로 주신 답변 '{response}'을(를) 반영했습니다."
    return "확인했습니다. 사용자 응답을 반영해 일시중단된 확인 단계를 종료했습니다."


def resume_paused_workflow_detailed(
    *,
    paused_workflow: Dict[str, Any],
    clarification_response: str,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    snapshot = dict(paused_workflow or {})
    workflow_id = str(snapshot.get("workflow_id", "")).strip() or f"default-{uuid4().hex[:8]}"
    raw_plan = str(snapshot.get("raw_plan", "")).strip()
    original_user_input = str(snapshot.get("original_user_input", "")).strip()
    pause_request = str(snapshot.get("pause_request", "")).strip()
    context_map = context or {}
    conversation_history = str(context_map.get("conversation_history", "") or snapshot.get("conversation_history", "")).strip()

    executable_agents = [agent for agent in available_agents if _is_local_agent(agent) or _is_a2a_agent(agent)]
    pending_steps = _normalize_replanned_steps(snapshot.get("pending_steps"), executable_agents)
    open_needs = _normalize_need_entries(snapshot.get("open_needs"), source_agent="MainAgent", max_items=20)
    open_needs = [need for need in open_needs if not _is_user_clarification_need(need)]
    completed_results = [
        dict(item)
        for item in snapshot.get("completed_results", [])
        if isinstance(item, dict)
    ]
    artifact_store = _copy_artifacts_for_snapshot(list(snapshot.get("artifact_store", [])))
    clarification_artifact = _build_user_clarification_artifact(
        workflow_id=workflow_id,
        clarification_request=pause_request,
        clarification_response=clarification_response,
    )
    artifact_store.append(clarification_artifact)

    merged_user_input = _merge_user_input_with_clarification(
        original_user_input=original_user_input,
        pause_request=pause_request,
        clarification_response=clarification_response,
    )

    log_event(
        "event_manager.collaboration",
        "workflow_resume_started",
        {
            "workflow_id": workflow_id,
            "pending_count": len(pending_steps),
            "open_needs_count": len(open_needs),
            "completed_count": len(completed_results),
        },
    )

    if not pending_steps and not open_needs:
        output_text = _resume_paused_workflow_acknowledgement(
            pause_request=pause_request,
            clarification_response=clarification_response,
        )
        log_event(
            "event_manager.collaboration",
            "workflow_resumed_without_replan",
            {"workflow_id": workflow_id, "reason": "no_remaining_work_after_clarification"},
        )
        return {
            "output_text": output_text,
            "raw_plan": raw_plan,
            "workflow_id": workflow_id,
            "results": completed_results,
            "paused_workflow": None,
        }

    latest_result = {
        "ok": True,
        "agent": "MainAgent",
        "workflow_step": len(completed_results),
        "goal": "사용자 응답을 반영하여 일시중단된 workflow를 재개합니다.",
        "response": _compact_text(
            f"User clarification received for '{pause_request}': {clarification_response}",
            max_chars=420,
        ),
        "raw_response": _compact_text(
            f"User clarification received for '{pause_request}': {clarification_response}",
            max_chars=420,
        ),
        "parsed_needs": [],
        "artifacts_emitted": [clarification_artifact],
        "workflow_paused": False,
    }
    review = _review_collaboration_progress_with_main_agent(
        main_agent=main_agent,
        available_agents=executable_agents,
        user_input=merged_user_input,
        conversation_history=conversation_history,
        raw_plan=raw_plan,
        completed_results=completed_results,
        latest_result=latest_result,
        pending_steps=pending_steps,
        open_needs=open_needs,
        trigger_reasons=["resumed_after_user_clarification"],
    )
    if review.get("needs"):
        review_needs = _normalize_need_entries(review.get("needs"), source_agent="MainAgent", max_items=12)
        seen_need_keys = {_need_identity_key(item) for item in open_needs if _need_identity_key(item)}
        _append_unique_open_needs(
            open_needs=open_needs,
            new_needs=review_needs,
            seen_need_keys=seen_need_keys,
        )

    clarification_request = _first_user_clarification_request(open_needs)
    if clarification_request:
        paused_snapshot = {
            "workflow_id": workflow_id,
            "raw_plan": raw_plan,
            "collaboration_plan": snapshot.get("collaboration_plan", {}),
            "original_user_input": merged_user_input,
            "conversation_history": conversation_history,
            "pause_request": clarification_request,
            "pause_agent": "MainAgent",
            "pause_step": latest_result.get("workflow_step"),
            "pending_steps": _serialize_pending_steps_for_snapshot(pending_steps),
            "open_needs": _copy_open_needs_for_snapshot(open_needs),
            "artifact_store": _copy_artifacts_for_snapshot(artifact_store),
            "completed_results": completed_results,
        }
        return {
            "output_text": "진행을 일시 중단하고 사용자 응답을 기다립니다.\n\n" + clarification_request,
            "raw_plan": raw_plan,
            "workflow_id": workflow_id,
            "results": completed_results,
            "paused_workflow": paused_snapshot,
        }

    review_updated_plan, pending_steps = _apply_progress_review_plan_update(
        review=review,
        pending_steps=pending_steps,
        completed_agent="MainAgent",
        completed_goal=str(latest_result.get("goal", "")),
    )
    if not review_updated_plan:
        fallback_plan = _build_indirect_delegation_fallback_steps(
            open_needs=open_needs,
            available_agents=executable_agents,
            pending_steps=pending_steps,
        )
        fallback_steps = [item for item in fallback_plan.get("steps", []) if isinstance(item, dict)]
        consumed_need_keys = {
            str(item).strip().lower()
            for item in fallback_plan.get("consumed_need_keys", [])
            if str(item).strip()
        }
        if fallback_steps:
            pending_steps = fallback_steps + pending_steps
            if consumed_need_keys:
                open_needs = [
                    need
                    for need in open_needs
                    if _need_identity_key(need) not in consumed_need_keys
                ]

    resumed_state = _run_collaboration_workflow(
        main_agent=main_agent,
        available_agents=executable_agents,
        workflow_id=workflow_id,
        raw_plan=raw_plan,
        steps=pending_steps,
        user_input=merged_user_input,
        conversation_history=conversation_history,
        initial_results=completed_results,
        initial_artifact_store=artifact_store,
        initial_open_needs=open_needs,
        initial_step_counter=len(completed_results),
    )
    resumed_results = list(resumed_state.get("results", []))
    pause_payload = _extract_pause_request_from_results(resumed_results)
    if pause_payload:
        paused_snapshot = _build_paused_workflow_snapshot(
            workflow_id=workflow_id,
            raw_plan=raw_plan,
            collaboration_plan=snapshot.get("collaboration_plan", {}),
            original_user_input=merged_user_input,
            conversation_history=conversation_history,
            workflow_state=resumed_state,
            pause_payload=pause_payload,
        )
        return {
            "output_text": "진행을 일시 중단하고 사용자 응답을 기다립니다.\n\n" + str(pause_payload.get("request", "")).strip(),
            "raw_plan": raw_plan,
            "workflow_id": workflow_id,
            "results": resumed_results,
            "paused_workflow": paused_snapshot,
        }

    formatted = _format_execution_output(raw_plan=raw_plan, results=resumed_results)
    final_summary = _summarize_collaboration_with_main_agent(
        main_agent=main_agent,
        user_input=merged_user_input,
        conversation_history=conversation_history,
        raw_plan=raw_plan,
        results=resumed_results,
    )
    final_summary = _ensure_summary_agent_sections(final_summary, resumed_results)
    if final_summary:
        formatted = f"{formatted}\n\n=== Final Summary ===\n{final_summary}"
    log_event(
        "event_manager.collaboration",
        "workflow_resumed_completed",
        {"workflow_id": workflow_id, "results_count": len(resumed_results)},
    )
    return {
        "output_text": formatted,
        "raw_plan": raw_plan,
        "workflow_id": workflow_id,
        "results": resumed_results,
        "paused_workflow": None,
    }


def execute_plan(
    plan: Dict[str, Any],
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
) -> str:
    return str(
        execute_plan_detailed(
            plan=plan,
            main_agent=main_agent,
            available_agents=available_agents,
            context=context,
        ).get("output_text", "")
    )


__all__ = ["execute_plan", "execute_plan_detailed", "resume_paused_workflow_detailed"]
