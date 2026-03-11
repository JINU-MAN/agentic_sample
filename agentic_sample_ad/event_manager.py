from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from typing import Any, Dict, List
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from google.adk.runners import InMemoryRunner
from google.genai import types

from agentic_sample_ad.network_retry import collect_text_response_with_network_retry
from agentic_sample_ad.system_logger import log_event, log_exception


CAPABILITY_POLICIES: List[Dict[str, str]] = [
    {
        "capability": "comm.slack.post",
        "tool": "slack_post_message",
        "owner": "MainAgent",
    }
]
TOOL_OWNER_OVERRIDES: Dict[str, str] = {
    str(item.get("tool", "")).strip().lower(): str(item.get("owner", "")).strip()
    for item in CAPABILITY_POLICIES
    if str(item.get("tool", "")).strip() and str(item.get("owner", "")).strip()
}
AGENT_MESSAGE_COMPONENT = "event_manager.agent_message"


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


def _run_coroutine_sync(coro: Any) -> Any:
    """
    Run a coroutine from sync code.

    - If no running loop exists in this thread: use asyncio.run.
    - If a loop already exists: run in a dedicated thread.
    """
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


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def _extract_strict_json_object(text: str) -> Dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    candidate = stripped
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


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
        result = {
            "ok": True,
            "agent": name,
            "response": response_text,
            "raw_a2a_result": raw_result,
            "elapsed_sec": elapsed_sec,
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


def _normalize_hint_keywords(routing_hint: Dict[str, Any]) -> List[str]:
    raw = routing_hint.get("keywords", [])
    if not isinstance(raw, list):
        return []

    keywords: List[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        token = item.strip().lower()
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _agent_search_blob(agent_meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(str(agent_meta.get("name", "")))
    parts.append(str(agent_meta.get("description", "")))
    parts.extend(str(item) for item in agent_meta.get("capabilities", []))

    for tool in agent_meta.get("tools", []):
        if isinstance(tool, dict):
            parts.append(str(tool.get("name", "")))
            parts.append(str(tool.get("description", "")))
        elif isinstance(tool, str):
            parts.append(tool)

    return " ".join(parts).lower()


def _select_executable_agents(
    candidate_agents: List[Dict[str, Any]],
    raw_plan: str,
    user_input: str,
    routing_hint: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    log_event(
        "event_manager.routing",
        "local_agent_selection_started",
        {
            "local_agent_names": [str(agent.get("name", "")) for agent in candidate_agents],
            "raw_plan": raw_plan,
            "user_input": user_input,
            "routing_hint": routing_hint or {},
        },
    )
    if not candidate_agents:
        log_event("event_manager.routing", "local_agent_selection_result", {"selected": []})
        return []

    plan_lower = raw_plan.lower()
    user_lower = user_input.lower()
    hint = routing_hint or {}

    name_map: Dict[str, Dict[str, Any]] = {}
    for agent_meta in candidate_agents:
        name = str(agent_meta.get("name", "")).strip()
        if name:
            name_map[name.lower()] = agent_meta

    selected_names = hint.get("selected_agents", [])
    if isinstance(selected_names, list) and selected_names:
        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in selected_names:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if not key or key in seen:
                continue
            match = name_map.get(key)
            if match is None:
                continue
            seen.add(key)
            selected.append(match)
        if selected:
            log_event(
                "event_manager.routing",
                "local_agent_selection_result",
                {
                    "method": "routing_hint.selected_agents",
                    "selected": [str(item.get("name", "")) for item in selected],
                },
            )
            return selected

    hint_keywords = _normalize_hint_keywords(hint)

    scored: List[tuple[int, Dict[str, Any]]] = []
    for agent_meta in candidate_agents:
        score = 0
        name = str(agent_meta.get("name", "")).lower()
        caps = [str(c).lower() for c in agent_meta.get("capabilities", [])]
        blob = _agent_search_blob(agent_meta)

        if name and name in plan_lower:
            score += 5
        if name and name in user_lower:
            score += 4

        for cap in caps:
            if cap and cap in plan_lower:
                score += 2
            if cap and cap in user_lower:
                score += 2
            for token in [t for t in cap.split("_") if t]:
                if len(token) >= 3 and token in user_lower:
                    score += 1

        if "slack" in user_lower and ("slack_post" in caps or "slack" in blob):
            score += 1

        for keyword in hint_keywords:
            if keyword in user_lower:
                score += 2
            if keyword in plan_lower:
                score += 1
            if keyword in name:
                score += 1
            if any(keyword in cap for cap in caps):
                score += 1
            if keyword in blob:
                score += 2

        if score > 0:
            scored.append((score, agent_meta))

    if scored:
        max_score = max(score for score, _ in scored)
        selected_by_score = [meta for score, meta in scored if score == max_score]
        log_event(
            "event_manager.routing",
            "local_agent_selection_result",
            {
                "method": "score",
                "max_score": max_score,
                "selected": [str(item.get("name", "")) for item in selected_by_score],
            },
        )
        return selected_by_score

    if len(candidate_agents) == 1:
        log_event(
            "event_manager.routing",
            "local_agent_selection_result",
            {
                "method": "single_local_agent",
                "selected": [str(candidate_agents[0].get("name", ""))],
            },
        )
        return candidate_agents

    # Final fallback: pick the first candidate agent.
    fallback = [candidate_agents[0]]
    log_event(
        "event_manager.routing",
        "local_agent_selection_result",
        {
            "method": "fallback_first",
            "selected": [str(fallback[0].get("name", ""))],
        },
    )
    return fallback


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

        chunks = await collect_text_response_with_network_retry(
            runner=runner,
            user_id="event-manager-user",
            new_message=new_message,
            component="event_manager.local_agent",
            operation_name=f"local_agent:{agent_name}",
            retry_details={"agent": agent_name, "user_input": user_input},
            on_text=_on_text,
        )

        response_text = "\n".join(chunks).strip() or "(No text response emitted.)"
        log_event(
            "event_manager.local_agent",
            "execution_completed",
            {"agent": agent_name, "response": response_text},
            direction="inbound",
        )
        return {
            "ok": True,
            "agent": agent_name,
            "response": response_text,
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


def _execute_single_local_agent(agent_meta: Dict[str, Any], user_input: str) -> Dict[str, Any]:
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


def _execute_single_agent(agent_meta: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    if _is_local_agent(agent_meta):
        return _execute_single_local_agent(agent_meta, user_input)
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


def _agent_tool_names(agent_meta: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for tool in agent_meta.get("tools", []):
        token = ""
        if isinstance(tool, dict):
            token = str(tool.get("name", "")).strip()
        elif isinstance(tool, str):
            token = tool.strip()
        key = token.lower()
        if token and key not in seen:
            seen.add(key)
            names.append(token)
    return names


def _resolve_policy_owner_for_hint(
    *,
    tool_hint: str,
    available_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    normalized = str(tool_hint).strip().lower()
    if not normalized:
        return None

    owner_name = TOOL_OWNER_OVERRIDES.get(normalized, "")
    if owner_name:
        owner_meta = available_index.get(owner_name.lower())
        if owner_meta is not None:
            return owner_meta
    return None


def _apply_step_owner_policy(
    *,
    step_agent_name: str,
    step_goal: str,
    tool_hints: List[str],
    available_index: Dict[str, Dict[str, Any]],
) -> tuple[str, Dict[str, Any]]:
    assigned_name = str(step_agent_name).strip()
    assigned_meta = available_index.get(assigned_name.lower())

    target_meta: Dict[str, Any] | None = None
    owner_hint = ""
    for hint in tool_hints:
        owner = _resolve_policy_owner_for_hint(tool_hint=hint, available_index=available_index)
        if owner is None:
            continue
        owner_name = str(owner.get("name", "")).strip()
        if owner_name and owner_name.lower() != assigned_name.lower():
            target_meta = owner
            owner_hint = hint
            break

    if target_meta is None:
        goal_lower = str(step_goal or "").lower()
        if "slack" in goal_lower and "mainagent" in available_index and assigned_name.lower() != "mainagent":
            target_meta = available_index["mainagent"]
            owner_hint = "goal_contains_slack"

    if target_meta is None:
        if assigned_meta is None:
            return assigned_name, {}
        return assigned_name, assigned_meta

    rerouted_name = str(target_meta.get("name", "")).strip() or assigned_name
    log_event(
        "event_manager.policy",
        "step_rerouted_by_owner_policy",
        {
            "from_agent": assigned_name,
            "to_agent": rerouted_name,
            "tool_hint": owner_hint,
            "goal_preview": " ".join(str(step_goal or "").split())[:240],
        },
    )
    return rerouted_name, target_meta


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
        tool_hints: List[str] = []
        raw_tool_hints = item.get("tool_hints", [])
        if isinstance(raw_tool_hints, list):
            for hint in raw_tool_hints:
                if not isinstance(hint, str):
                    continue
                token = hint.strip()
                if token and token not in tool_hints:
                    tool_hints.append(token)
                if len(tool_hints) >= 8:
                    break
        resolved_agent_name, resolved_agent_meta = _apply_step_owner_policy(
            step_agent_name=agent_name,
            step_goal=goal,
            tool_hints=tool_hints,
            available_index=available_index,
        )
        if not resolved_agent_meta:
            continue
        if not goal:
            goal = "Handle this step and provide a handoff-ready output."

        resolved_steps.append(
            {
                "agent": str(resolved_agent_meta.get("name", "")).strip() or resolved_agent_name,
                "goal": goal[:1000],
                "deliverable": deliverable[:1000],
                "tool_hints": tool_hints,
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

    raw_artifacts = parsed.get("artifacts")
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


def _extract_agent_output_summary(text: str) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""

    parsed = _extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        return raw_text

    summary = _extract_first_text_value(
        parsed.get("summary"),
        parsed.get("result"),
        parsed.get("response"),
        parsed.get("answer"),
        parsed.get("message"),
        max_chars=3200,
    )
    if summary:
        return summary

    artifacts = parsed.get("artifacts")
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

    return raw_text


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


def _build_delegation_targets(
    *,
    available_agents: List[Dict[str, Any]],
    current_agent_name: str,
    max_items: int = 6,
) -> List[str]:
    current_key = str(current_agent_name).strip().lower()
    names: List[str] = []
    seen: set[str] = set()

    def _push(name: str) -> None:
        token = str(name).strip()
        key = token.lower()
        if not token or key in seen:
            return
        seen.add(key)
        names.append(token)

    _push("MainAgent")
    for agent in available_agents:
        name = str(agent.get("name", "")).strip()
        if not name:
            continue
        if name.strip().lower() == current_key:
            continue
        _push(name)
        if len(names) >= max_items:
            break
    return names[:max_items]


def _format_remaining_steps(
    steps: List[Dict[str, Any]],
    *,
    max_items: int = 5,
    max_goal_chars: int = 180,
    max_hint_items: int = 4,
) -> str:
    if not steps:
        return "(none)"
    lines: List[str] = []
    for idx, step in enumerate(steps[:max_items], start=1):
        agent = str(step.get("agent", "UnknownAgent"))
        goal = _compact_text(str(step.get("goal", "")).strip(), max_chars=max_goal_chars)
        hints = [str(item).strip() for item in step.get("tool_hints", []) if isinstance(item, str) and str(item).strip()]
        hints = hints[:max_hint_items]
        hint_text = f" (tool_hints: {', '.join(hints)})" if hints else ""
        if goal:
            lines.append(f"{idx}. {agent} - {goal}{hint_text}")
        else:
            lines.append(f"{idx}. {agent}{hint_text}")
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

    capabilities = [str(item).strip() for item in agent_meta.get("capabilities", []) if str(item).strip()]
    tool_entries: List[str] = []
    for tool in agent_meta.get("tools", []):
        if isinstance(tool, dict):
            tool_name = str(tool.get("name", "")).strip()
            tool_desc = str(tool.get("description", "")).strip()
            if tool_name and tool_desc:
                tool_entries.append(f"{tool_name}: {tool_desc}")
            elif tool_name:
                tool_entries.append(tool_name)
        elif isinstance(tool, str):
            token = tool.strip()
            if token:
                tool_entries.append(token)

    return {
        "name": name,
        "type": str(agent_meta.get("type", "")).strip() or "a2a",
        "role": str(agent_meta.get("role", "")).strip() or "worker",
        "description": str(agent_meta.get("description", "")).strip(),
        "capabilities": capabilities,
        "tools": tool_entries,
        "instruction_preview": str(agent_meta.get("instruction_preview", "")).strip(),
    }


def _format_agent_card_snapshots(cards: List[Dict[str, Any]]) -> str:
    if not cards:
        return "(none)"
    lines: List[str] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        name = str(card.get("name", "UnknownAgent")).strip() or "UnknownAgent"
        agent_type = str(card.get("type", "a2a")).strip() or "a2a"
        role = str(card.get("role", "worker")).strip() or "worker"
        desc = str(card.get("description", "")).strip()
        caps = ", ".join(str(item) for item in card.get("capabilities", []) if str(item).strip()) or "(none)"
        tools = "; ".join(str(item) for item in card.get("tools", []) if str(item).strip()) or "(none)"
        instruction_preview = str(card.get("instruction_preview", "")).strip() or "(not provided)"
        lines.append(
            f"- name: {name}\n"
            f"  type: {agent_type}\n"
            f"  role: {role}\n"
            f"  description: {desc}\n"
            f"  capabilities: {caps}\n"
            f"  tools: {tools}\n"
            f"  instruction_preview: {instruction_preview}"
        )
    return "\n".join(lines).strip() or "(none)"


def _normalize_need_text(value: str) -> str:
    compact = " ".join(value.split()).strip()
    compact = re.sub(r"^[-*]\s*", "", compact)
    compact = re.sub(r"^\d+\.\s*", "", compact)
    return compact[:300]


def _canonicalize_need_text(value: str) -> str:
    token = _normalize_need_text(value)
    if not token:
        return ""
    if token.lower() in {"none", "n/a", "no", "null", "없음"}:
        return ""

    target_agent, request = _parse_targeted_need(token)
    if target_agent and request:
        return f"[{target_agent}] {request}"[:300]
    return token[:300]


def _extract_additional_needs_from_agent_output(text: str) -> List[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    parsed = _extract_json_object(raw_text)
    if isinstance(parsed, dict):
        raw_needs = parsed.get("additional_needs")
        if isinstance(raw_needs, list):
            collected: List[str] = []
            seen: set[str] = set()
            for item in raw_needs:
                if not isinstance(item, str):
                    continue
                need = _canonicalize_need_text(item)
                if not need:
                    continue
                key = need.lower()
                if key in seen:
                    continue
                seen.add(key)
                collected.append(need)
                if len(collected) >= 12:
                    break
            return collected
        if isinstance(raw_needs, str):
            token = _canonicalize_need_text(raw_needs)
            if token:
                return [token]

        normalized_from_structured: List[str] = []
        seen_structured: set[str] = set()

        def _push_structured_need(raw_value: str) -> None:
            token = _canonicalize_need_text(raw_value)
            key = token.lower()
            if not token:
                return
            if key in seen_structured:
                return
            seen_structured.add(key)
            normalized_from_structured.append(token)

        raw_needs = parsed.get("needs")
        if isinstance(raw_needs, list):
            for item in raw_needs:
                if isinstance(item, str):
                    _push_structured_need(item)
                elif isinstance(item, dict):
                    target = str(
                        item.get("target")
                        or item.get("agent")
                        or item.get("to")
                        or ""
                    ).strip()
                    request = str(
                        item.get("request")
                        or item.get("task")
                        or item.get("message")
                        or item.get("need")
                        or ""
                    ).strip()
                    if target and request:
                        _push_structured_need(f"[{target}] {request}")
                    elif request:
                        _push_structured_need(request)
                if len(normalized_from_structured) >= 12:
                    break

        raw_events = parsed.get("workflow_events")
        if isinstance(raw_events, list) and len(normalized_from_structured) < 12:
            for item in raw_events:
                if not isinstance(item, dict):
                    continue
                event_type = str(item.get("type", "")).strip().lower()
                if event_type not in {"need_request", "delegate_request", "handoff_request"}:
                    continue
                target = str(item.get("target") or item.get("agent") or "").strip()
                request = str(item.get("request") or item.get("payload") or item.get("message") or "").strip()
                if target and request:
                    _push_structured_need(f"[{target}] {request}")
                elif request:
                    _push_structured_need(request)
                if len(normalized_from_structured) >= 12:
                    break

        if normalized_from_structured:
            return normalized_from_structured[:12]

    lines = raw_text.splitlines()
    marker_index = -1
    inline_value = ""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("additional needs:"):
            marker_index = idx
            inline_value = stripped[len("additional needs:") :].strip()
            break
        if lowered in {"additional needs", "additional needs:"}:
            marker_index = idx
            break

    if marker_index < 0:
        return []

    needs: List[str] = []
    seen_keys: set[str] = set()

    if inline_value:
        token = _canonicalize_need_text(inline_value)
        if token:
            seen_keys.add(token.lower())
            needs.append(token)

    for line in lines[marker_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if needs:
                break
            continue

        # Stop at a likely next section heading.
        if needs and re.match(r"^[A-Za-z][A-Za-z0-9 _/-]{0,40}:$", stripped):
            break

        token = _canonicalize_need_text(stripped)
        if not token:
            continue
        key = token.lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        needs.append(token)
        if len(needs) >= 12:
            break

    return needs


def _parse_targeted_need(need: str) -> tuple[str, str]:
    token = _normalize_need_text(need)
    if not token:
        return "", ""
    bracket_match = re.match(r"^\[(?P<agent>[^\[\]]{1,80})\]\s*(?P<request>.+)$", token)
    if bracket_match:
        target_agent = str(bracket_match.group("agent") or "").strip()
        request = str(bracket_match.group("request") or "").strip()
        return target_agent, request

    colon_match = re.match(r"^(?P<agent>[A-Za-z][A-Za-z0-9_-]{1,79})\s*:\s*(?P<request>.+)$", token)
    if colon_match:
        target_agent = str(colon_match.group("agent") or "").strip()
        request = str(colon_match.group("request") or "").strip()
        return target_agent, request

    return "", ""


def _is_user_clarification_need(need: str) -> bool:
    target_agent, request = _parse_targeted_need(need)
    target_lower = target_agent.strip().lower()
    if target_lower and target_lower != "mainagent":
        return False

    text = str(request or need).strip()
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
    if target_lower == "mainagent":
        return ("?" in text) or any(token in lowered for token in explicit_clarification_tokens) or is_slack_channel_question
    return False


def _first_user_clarification_request(needs: List[str]) -> str:
    for need in needs:
        if not _is_user_clarification_need(need):
            continue
        target_agent, request = _parse_targeted_need(need)
        if target_agent and request:
            return request
        normalized = _normalize_need_text(need)
        lowered = normalized.lower()
        if ("슬랙" in normalized or "slack" in lowered) and ("채널" in normalized or "channel" in lowered):
            return "슬랙 게시를 위해 정확한 채널 이름 또는 채널 ID(예: C12345678)를 알려주세요."
        return normalized
    return ""


def _tool_hints_from_agent_meta(agent_meta: Dict[str, Any], max_hints: int = 4) -> List[str]:
    return []


def _build_indirect_delegation_fallback_steps(
    *,
    open_needs: List[str],
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
        need_text = _normalize_need_text(need)
        target_agent, request = _parse_targeted_need(need_text)
        if not target_agent or not request:
            continue

        target_key = target_agent.lower()
        agent_meta = available_index.get(target_key)
        if agent_meta is None:
            continue

        canonical_need = f"{target_key}|{request.lower()}"
        if canonical_need in local_signatures:
            continue

        goal = (
            "Handle this unresolved follow-up need and return a usable result.\n"
            f"Requested need: {request}"
        )
        signature = f"{target_key}|{goal.lower()}"
        if signature in existing_signatures:
            consumed_need_keys.add(need_text.lower())
            continue

        added_steps.append(
            {
                "agent": str(agent_meta.get("name", "")).strip() or target_agent,
                "goal": goal[:1000],
                "deliverable": f"Concrete response/evidence addressing: {request}"[:1000],
                "tool_hints": [],
                "agent_meta": agent_meta,
            }
        )
        local_signatures.add(canonical_need)
        existing_signatures.add(signature)
        consumed_need_keys.add(need_text.lower())

        if len(added_steps) >= max_steps:
            break

    return {"steps": added_steps, "consumed_need_keys": sorted(consumed_need_keys)}


def _format_delegate_agent_names(available_agents: List[Dict[str, Any]]) -> str:
    names: List[str] = []
    for agent in available_agents:
        name = str(agent.get("name", "")).strip()
        if name and name not in names:
            names.append(name)
    if "MainAgent" not in names:
        names.append("MainAgent")
    if not names:
        return "(none)"
    return ", ".join(names)


def _format_delegate_agent_profiles(
    available_agents: List[Dict[str, Any]],
    current_agent_name: str,
) -> str:
    lines: List[str] = []
    current_lower = str(current_agent_name or "").strip().lower()
    for agent in available_agents:
        name = str(agent.get("name", "")).strip()
        if not name:
            continue
        if current_lower and name.lower() == current_lower:
            continue

        caps = [str(item).strip() for item in agent.get("capabilities", []) if str(item).strip()]
        cap_text = ", ".join(caps) if caps else "(none)"
        tool_names: List[str] = []
        for tool in agent.get("tools", []):
            if isinstance(tool, dict):
                token = str(tool.get("name", "")).strip()
            else:
                token = str(tool).strip()
            if token and token not in tool_names:
                tool_names.append(token)
        tool_text = ", ".join(tool_names) if tool_names else "(none)"
        lines.append(f"- {name}: capabilities={cap_text}; tools={tool_text}")

    if "mainagent" not in {line.lower().split(":")[0].replace("- ", "").strip() for line in lines}:
        lines.append("- MainAgent: coordination, replanning, user-clarification routing")
    return "\n".join(lines).strip() or "(none)"


def _build_agent_catalog_for_context(
    *,
    available_agents: List[Dict[str, Any]],
    current_agent_name: str,
) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    current_key = str(current_agent_name).strip().lower()
    for agent in available_agents:
        name = str(agent.get("name", "")).strip()
        if not name:
            continue
        caps = [str(item).strip() for item in agent.get("capabilities", []) if str(item).strip()][:6]
        tools = _agent_tool_names(agent)[:4]
        catalog.append(
            {
                "name": name,
                "role": str(agent.get("role", "")).strip() or "worker",
                "type": str(agent.get("type", "")).strip() or "local",
                "is_current": bool(current_key and name.lower() == current_key),
                "description": _compact_text(str(agent.get("description", "")).strip(), max_chars=180),
                "capabilities": caps,
                "tools": tools,
            }
        )
    return catalog


def _build_collaboration_step_input(
    *,
    workflow_id: str,
    user_input: str,
    prior_results: List[Dict[str, Any]],
    input_artifacts: List[Dict[str, Any]],
    open_needs: List[str],
    remaining_steps: List[Dict[str, Any]],
    available_agents: List[Dict[str, Any]],
    step: Dict[str, Any],
    step_index: int,
    total_steps_hint: int,
) -> str:
    step_goal = _compact_text(str(step.get("goal", "")).strip(), max_chars=320)
    deliverable = _compact_text(str(step.get("deliverable", "")).strip(), max_chars=220)
    raw_tool_hints = step.get("tool_hints", [])
    tool_hints = [str(item).strip() for item in raw_tool_hints if isinstance(item, str) and str(item).strip()][:5]

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
        max_hint_items=2,
    )
    delegation_targets = _build_delegation_targets(
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
            "tool_hints": tool_hints,
        },
        "request": {"user_request_brief": request_brief},
        "state": {
            "prior_results": prior_text,
            "input_artifacts": input_artifacts,
            "open_needs": [
                _compact_text(str(item).strip(), max_chars=180)
                for item in open_needs
                if str(item).strip()
            ][:4],
            "remaining_steps_hint": remaining_text,
        },
        "delegation_targets": delegation_targets,
        "total_steps_hint": total_steps_hint,
        "runtime_hints": {
            "input_artifacts_supported": True,
            "structured_handoff_supported": True,
            "artifact_handoff_fields": ["type", "title", "summary", "url", "identifiers"],
            "needs_handoff_supported": True,
        },
    }
    context_json = json.dumps(context_packet, ensure_ascii=False)

    return (
        "You are handling one step in a multi-agent workflow.\n"
        "Use the context packet below as task context for this step.\n"
        "Your own agent instruction and tools remain the primary contract for this step.\n"
        "Work autonomously and choose the concrete approach yourself.\n"
        "Treat goal, deliverable, tool_hints, and runtime_hints as guidance rather than rigid instructions.\n"
        "Do not call other agents directly.\n"
        "Return the best handoff-ready result you can for this step.\n\n"
        "Workflow Context Packet:\n"
        f"{context_json}"
    )


async def _async_review_collaboration_progress_with_main_agent(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    activated_agent_cards: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    completed_results: List[Dict[str, Any]],
    latest_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[str],
) -> Dict[str, Any]:
    runner = InMemoryRunner(agent=main_agent, app_name="main-collaboration-replanner")
    log_event(
        "event_manager.collaboration",
        "replan_review_started",
        {
            "completed_count": len(completed_results),
            "pending_count": len(pending_steps),
            "open_needs_count": len(open_needs),
        },
        direction="outbound",
    )
    try:
        lines: List[str] = []
        for agent in available_agents:
            name = str(agent.get("name", "UnknownAgent"))
            role = str(agent.get("role", "")).strip() or "worker"
            desc = str(agent.get("description", "")).strip()
            caps = ", ".join(str(c) for c in agent.get("capabilities", []))
            tool_entries: List[str] = []
            for tool in agent.get("tools", []):
                if isinstance(tool, dict):
                    tool_name = str(tool.get("name", "")).strip()
                    tool_desc = str(tool.get("description", "")).strip()
                    if tool_name and tool_desc:
                        tool_entries.append(f"{tool_name}: {tool_desc}")
                    elif tool_name:
                        tool_entries.append(tool_name)
                elif isinstance(tool, str):
                    token = tool.strip()
                    if token:
                        tool_entries.append(token)
            tools_text = "; ".join(tool_entries) if tool_entries else "(not provided)"
            instruction_preview = str(agent.get("instruction_preview", "")).strip() or "(not provided)"
            lines.append(
                f"- name: {name}\n"
                f"  role: {role}\n"
                f"  description: {desc}\n"
                f"  capabilities: {caps}\n"
                f"  tools: {tools_text}\n"
                f"  instruction_preview: {instruction_preview}"
            )
        agents_desc = "\n".join(lines) or "(none)"

        latest_step = latest_result.get("workflow_step")
        latest_agent = str(latest_result.get("agent", "UnknownAgent"))
        latest_status = "ok" if latest_result.get("ok") else "error"
        if latest_result.get("ok"):
            latest_text = _compact_text(str(latest_result.get("response", "")).strip(), max_chars=420)
        else:
            latest_text = _compact_text(str(latest_result.get("error", "Unknown error")).strip(), max_chars=420)

        completed_text = _format_prior_results_for_handoff(completed_results)
        pending_text = _format_remaining_steps(pending_steps)
        needs_text = "\n".join(f"- {item}" for item in open_needs) if open_needs else "(none)"
        activated_cards_text = _format_agent_card_snapshots(activated_agent_cards)
        conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=8, max_chars=1200)
        planner_summary = _compact_text(raw_plan or "(none)", max_chars=1200)

        prompt = (
            "You are the main coordinator reviewing multi-agent progress.\n"
            "Decide whether the remaining plan should change.\n"
            "Return JSON only.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "additional_needs": ["need1", "need2"],\n'
            '  "should_update_plan": true,\n'
            '  "updated_steps": [\n'
            "    {\n"
            '      "agent": "AgentName",\n'
            '      "goal": "what to do next",\n'
            '      "deliverable": "expected output",\n'
            '      "tool_hints": ["tool_or_strategy_1", "tool_or_strategy_2"]\n'
            "    }\n"
            "  ],\n"
            '  "reason": "short reason"\n'
            "}\n\n"
            "Guidelines:\n"
            "- Use only names from Available agents.\n"
            "- additional_needs should include only unresolved concrete needs.\n"
            "- If no plan update is needed, set should_update_plan=false and updated_steps=[].\n"
            "- updated_steps should contain only future steps.\n"
            "- Leave tool_hints empty unless they are clearly useful.\n"
            "- Respect explicit user constraints.\n\n"
            f"Conversation context summary:\n{conversation_summary}\n\n"
            f"User request:\n{user_input}\n\n"
            f"Original planner text summary:\n{planner_summary}\n\n"
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
        )

        raw_text = "\n".join(chunks).strip()
        parsed = _extract_json_object(raw_text) or {}

        additional_needs: List[str] = []
        seen_needs: set[str] = set()
        for item in parsed.get("additional_needs", []):
            if not isinstance(item, str):
                continue
            token = _canonicalize_need_text(item)
            key = token.lower()
            if len(token) < 2 or key in seen_needs:
                continue
            seen_needs.add(key)
            additional_needs.append(token)
            if len(additional_needs) >= 12:
                break

        updated_steps = _normalize_replanned_steps(parsed.get("updated_steps"), available_agents)
        should_update = bool(parsed.get("should_update_plan")) and bool(updated_steps)
        reason = str(parsed.get("reason", "")).strip()

        result = {
            "additional_needs": additional_needs,
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
            "additional_needs": [],
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
    activated_agent_cards: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    completed_results: List[Dict[str, Any]],
    latest_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[str],
) -> Dict[str, Any]:
    return _run_coroutine_sync(
        _async_review_collaboration_progress_with_main_agent(
            main_agent=main_agent,
            available_agents=available_agents,
            activated_agent_cards=activated_agent_cards,
            user_input=user_input,
            conversation_history=conversation_history,
            raw_plan=raw_plan,
            completed_results=completed_results,
            latest_result=latest_result,
            pending_steps=pending_steps,
            open_needs=open_needs,
        )
    )


async def _async_handle_collaboration_failure_with_main_agent(
    *,
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    activated_agent_cards: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    results_so_far: List[Dict[str, Any]],
    failed_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[str],
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
        lines: List[str] = []
        for agent in available_agents:
            name = str(agent.get("name", "UnknownAgent"))
            role = str(agent.get("role", "")).strip() or "worker"
            desc = str(agent.get("description", "")).strip()
            caps = ", ".join(str(c) for c in agent.get("capabilities", []))
            tool_entries: List[str] = []
            for tool in agent.get("tools", []):
                if isinstance(tool, dict):
                    tool_name = str(tool.get("name", "")).strip()
                    tool_desc = str(tool.get("description", "")).strip()
                    if tool_name and tool_desc:
                        tool_entries.append(f"{tool_name}: {tool_desc}")
                    elif tool_name:
                        tool_entries.append(tool_name)
                elif isinstance(tool, str):
                    token = tool.strip()
                    if token:
                        tool_entries.append(token)
            tools_text = "; ".join(tool_entries) if tool_entries else "(not provided)"
            instruction_preview = str(agent.get("instruction_preview", "")).strip() or "(not provided)"
            lines.append(
                f"- name: {name}\n"
                f"  role: {role}\n"
                f"  description: {desc}\n"
                f"  capabilities: {caps}\n"
                f"  tools: {tools_text}\n"
                f"  instruction_preview: {instruction_preview}"
            )
        agents_desc = "\n".join(lines) or "(none)"

        completed_text = _format_prior_results_for_handoff(results_so_far)
        pending_text = _format_remaining_steps(pending_steps)
        needs_text = "\n".join(f"- {item}" for item in open_needs) if open_needs else "(none)"
        activated_cards_text = _format_agent_card_snapshots(activated_agent_cards)
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
            '      "deliverable": "expected output",\n'
            '      "tool_hints": ["tool_or_strategy_1", "tool_or_strategy_2"]\n'
            "    }\n"
            "  ],\n"
            '  "reason": "short reason for decision"\n'
            "}\n\n"
            "Guidelines:\n"
            "- Use only names from Available agents.\n"
            "- If replan is feasible this turn, set decision=replan and provide updated_steps.\n"
            "- If not feasible, set decision=abort and provide a clear user_message.\n"
            "- updated_steps must contain only future steps.\n"
            "- Leave tool_hints empty unless they are clearly useful.\n"
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
    activated_agent_cards: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    results_so_far: List[Dict[str, Any]],
    failed_result: Dict[str, Any],
    pending_steps: List[Dict[str, Any]],
    open_needs: List[str],
) -> Dict[str, Any]:
    return _run_coroutine_sync(
        _async_handle_collaboration_failure_with_main_agent(
            main_agent=main_agent,
            available_agents=available_agents,
            activated_agent_cards=activated_agent_cards,
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
    open_needs: List[str],
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
        "open_needs": [str(item).strip() for item in open_needs if str(item).strip()][:30],
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
    activated_agent_cards: List[Dict[str, Any]],
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
        lines: List[str] = []
        for agent in available_agents:
            name = str(agent.get("name", "UnknownAgent"))
            role = str(agent.get("role", "")).strip() or "worker"
            desc = str(agent.get("description", "")).strip()
            caps = ", ".join(str(c) for c in agent.get("capabilities", []))
            tool_entries: List[str] = []
            for tool in agent.get("tools", []):
                if isinstance(tool, dict):
                    tool_name = str(tool.get("name", "")).strip()
                    tool_desc = str(tool.get("description", "")).strip()
                    if tool_name and tool_desc:
                        tool_entries.append(f"{tool_name}: {tool_desc}")
                    elif tool_name:
                        tool_entries.append(tool_name)
                elif isinstance(tool, str):
                    token = tool.strip()
                    if token:
                        tool_entries.append(token)
            tools_text = "; ".join(tool_entries) if tool_entries else "(not provided)"
            instruction_preview = str(agent.get("instruction_preview", "")).strip() or "(not provided)"
            lines.append(
                f"- name: {name}\n"
                f"  role: {role}\n"
                f"  description: {desc}\n"
                f"  capabilities: {caps}\n"
                f"  tools: {tools_text}\n"
                f"  instruction_preview: {instruction_preview}"
            )
        agents_desc = "\n".join(lines) or "(none)"
        activated_cards_text = _format_agent_card_snapshots(activated_agent_cards)
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
            '      "deliverable": "expected output",\n'
            '      "tool_hints": ["tool_or_strategy_1", "tool_or_strategy_2"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Guidelines:\n"
            "- Use only names from Available agents in updated_steps.\n"
            "- Use replace_pending only when the current pending steps should be replaced.\n"
            "- Leave tool_hints empty unless they are clearly useful.\n"
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
    activated_agent_cards: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    timeout_packet: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_coroutine_sync(
        _async_handle_timeout_with_main_agent(
            main_agent=main_agent,
            available_agents=available_agents,
            activated_agent_cards=activated_agent_cards,
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
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    artifact_store: List[Dict[str, Any]] = []
    seen_artifact_keys: set[str] = set()
    pending_steps: List[Dict[str, Any]] = list(steps)
    open_needs: List[str] = []
    seen_need_keys: set[str] = set()
    activated_agent_cards_map: Dict[str, Dict[str, Any]] = {}
    step_counter = 0
    max_steps = _resolve_collaboration_max_steps(len(steps))

    while pending_steps and step_counter < max_steps:
        step_counter += 1
        step = pending_steps.pop(0)
        agent_name = str(step.get("agent", "UnknownAgent"))
        goal = str(step.get("goal", "")).strip()
        tool_hints = [str(item).strip() for item in step.get("tool_hints", []) if isinstance(item, str) and str(item).strip()]
        pre_step_open_needs = list(open_needs)
        current_agent_meta = step.get("agent_meta", {})
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
        log_event(
            "event_manager.collaboration",
            "step_started",
            {
                "step": step_counter,
                "total_steps_hint": total_steps_hint,
                "agent": agent_name,
                "goal": goal,
                "tool_hints": tool_hints,
                "open_needs": open_needs,
                "input_artifact_count": len(input_artifacts),
                "input_artifact_ids": [str(item.get("id", "")) for item in input_artifacts],
                "activated_agent_cards": [str(item.get("name", "")) for item in activated_agent_cards],
            },
            direction="outbound",
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

        result = _execute_single_agent(step["agent_meta"], step_input)
        enriched = dict(result)
        enriched["workflow_step"] = step_counter
        enriched["goal"] = goal
        enriched["tool_hints"] = tool_hints
        enriched["input_artifacts"] = input_artifacts
        raw_agent_output = (
            str(enriched.get("response", "")).strip()
            if enriched.get("ok")
            else str(enriched.get("error", "")).strip()
        )
        if enriched.get("ok") and raw_agent_output:
            structured_summary = _extract_agent_output_summary(raw_agent_output)
            if structured_summary:
                enriched["raw_response"] = raw_agent_output
                enriched["response"] = structured_summary
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
        parsed_needs = _extract_additional_needs_from_agent_output(raw_agent_output)
        parsed_artifacts = (
            _extract_artifacts_from_agent_output(
                raw_agent_output,
                source_agent=agent_name,
                workflow_id=workflow_id,
                workflow_step=step_counter,
            )
            if enriched.get("ok")
            else []
        )
        enriched["parsed_additional_needs"] = parsed_needs
        enriched["artifacts_emitted"] = parsed_artifacts
        results.append(enriched)

        added_artifacts: List[Dict[str, Any]] = []
        for artifact in parsed_artifacts:
            artifact_key = _artifact_identity_key(artifact)
            if artifact_key and artifact_key in seen_artifact_keys:
                continue
            if artifact_key:
                seen_artifact_keys.add(artifact_key)
            artifact_store.append(artifact)
            added_artifacts.append(artifact)
        if added_artifacts:
            enriched["artifact_ids"] = [str(item.get("id", "")) for item in added_artifacts]
            log_event(
                "workflow.artifact",
                "artifacts_registered",
                {
                    "source_agent": agent_name,
                    "workflow_step": step_counter,
                    "artifact_count": len(added_artifacts),
                    "artifact_ids": [str(item.get("id", "")) for item in added_artifacts],
                    "artifact_types": [str(item.get("type", "")) for item in added_artifacts],
                    "store_size": len(artifact_store),
                },
                direction="inbound",
            )

        added_needs: List[str] = []
        if parsed_needs:
            for item in parsed_needs:
                need = _canonicalize_need_text(item)
                key = need.lower()
                if not need or key in seen_need_keys:
                    continue
                seen_need_keys.add(key)
                open_needs.append(need)
                added_needs.append(need)
                target_agent, request = _parse_targeted_need(need)
                log_event(
                    "workflow.need",
                    "need_emitted",
                    {
                        "source_agent": agent_name,
                        "target_agent": target_agent,
                        "request": request or need,
                        "workflow_step": step_counter,
                    },
                )
                _log_agent_message(
                    action="need_requested",
                    from_agent=agent_name,
                    to_agent=target_agent or "MainAgent",
                    message=request or need,
                    channel="additional_needs",
                    workflow_id=workflow_id,
                    workflow_step=step_counter,
                    direction="outbound",
                )
            if added_needs:
                log_event(
                    "event_manager.collaboration",
                    "open_needs_updated_from_agent_output",
                    {"added_needs": added_needs, "open_needs": open_needs},
                    direction="inbound",
                )

        if enriched.get("ok"):
            clarification_request = _first_user_clarification_request(added_needs)
            clarification_source = "agent_output_additional_needs"
            if not clarification_request:
                clarification_request = _first_user_clarification_request(open_needs)
                clarification_source = "agent_output_open_needs"
            if clarification_request:
                enriched["workflow_paused"] = True
                enriched["pause_reason"] = "awaiting_user_clarification"
                enriched["pause_request"] = clarification_request
                log_event(
                    "event_manager.collaboration",
                    "workflow_paused_for_user_input",
                    {
                        "step": step_counter,
                        "agent": agent_name,
                        "request": clarification_request,
                        "source": clarification_source,
                    },
                )
                break

        if enriched.get("ok") and pre_step_open_needs:
            current_agent_key = agent_name.strip().lower()
            pre_step_need_keys = {
                str(item).strip().lower()
                for item in pre_step_open_needs
                if str(item).strip()
            }
            resolved_by_step: List[str] = []
            remaining_open_needs: List[str] = []
            for need in open_needs:
                need_text = str(need).strip()
                need_key = need_text.lower()
                if need_key not in pre_step_need_keys:
                    remaining_open_needs.append(need)
                    continue
                target_agent, request = _parse_targeted_need(need_text)
                if target_agent and target_agent.strip().lower() == current_agent_key:
                    resolved_by_step.append(need_text)
                    log_event(
                        "workflow.need",
                        "need_resolved_by_step_completion",
                        {
                            "resolved_by_agent": agent_name,
                            "target_agent": target_agent,
                            "request": request or need_text,
                            "workflow_step": step_counter,
                        },
                    )
                    continue
                remaining_open_needs.append(need)

            if resolved_by_step:
                open_needs = remaining_open_needs
                log_event(
                    "event_manager.collaboration",
                    "open_needs_resolved_by_step",
                    {"resolved_needs": resolved_by_step, "remaining_open_needs": open_needs},
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
            },
            direction="inbound",
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
                    activated_agent_cards=activated_agent_cards,
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

                if root_cause:
                    current_error = str(enriched.get("error", "Unknown error")).strip()
                    enriched["error"] = f"{current_error}\n\nTimeout Review: {root_cause}"

                enriched["timeout_recovery"] = {
                    "decision": decision,
                    "reason": reason,
                    "root_cause": root_cause,
                    "user_message": user_message,
                    "status_summary": status_summary,
                    "next_step_policy": str(timeout_review.get("next_step_policy", "resume_pending")).strip(),
                }
                enriched["failure_recovery"] = dict(enriched["timeout_recovery"])

                if timeout_review.get("should_continue"):
                    if timeout_review.get("replace_pending") and timeout_review.get("updated_steps"):
                        pending_steps = list(timeout_review["updated_steps"])
                        log_event(
                            "event_manager.collaboration",
                            "plan_updated_after_timeout",
                            {
                                "failed_step": step_counter,
                                "failed_agent": agent_name,
                                "reason": reason,
                                "new_pending_steps": [
                                    {
                                        "agent": str(step_meta.get("agent", "")),
                                        "goal": str(step_meta.get("goal", "")),
                                    }
                                    for step_meta in pending_steps
                                ],
                            },
                        )
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

                if user_message:
                    current_error = str(enriched.get("error", "Unknown error")).strip()
                    enriched["error"] = f"{current_error}\n\nCoordinator Message: {user_message}"

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
                activated_agent_cards=activated_agent_cards,
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

            if root_cause:
                current_error = str(enriched.get("error", "Unknown error")).strip()
                enriched["error"] = f"{current_error}\n\nFailure Analysis: {root_cause}"

            enriched["failure_recovery"] = {
                "decision": decision,
                "reason": reason,
                "root_cause": root_cause,
                "user_message": user_message,
            }

            if failure_review.get("should_replan") and failure_review.get("updated_steps"):
                pending_steps = list(failure_review["updated_steps"])
                log_event(
                    "event_manager.collaboration",
                    "plan_recovered_from_error",
                    {
                        "failed_step": step_counter,
                        "failed_agent": agent_name,
                        "reason": reason,
                        "new_pending_steps": [
                            {
                                "agent": str(step_meta.get("agent", "")),
                                "goal": str(step_meta.get("goal", "")),
                            }
                            for step_meta in pending_steps
                        ],
                    },
                )
                continue

            if user_message:
                current_error = str(enriched.get("error", "Unknown error")).strip()
                enriched["error"] = f"{current_error}\n\nCoordinator Message: {user_message}"

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

        should_review = bool(open_needs or not pending_steps)
        if should_review:
            review = _review_collaboration_progress_with_main_agent(
                main_agent=main_agent,
                available_agents=available_agents,
                activated_agent_cards=activated_agent_cards,
                user_input=user_input,
                conversation_history=conversation_history,
                raw_plan=raw_plan,
                completed_results=results,
                latest_result=enriched,
                pending_steps=pending_steps,
                open_needs=open_needs,
            )
        else:
            review = {
                "additional_needs": [],
                "should_update_plan": False,
                "updated_steps": [],
                "reason": "review_skipped_no_open_needs",
            }
            log_event(
                "event_manager.collaboration",
                "replan_review_skipped",
                {
                    "step": step_counter,
                    "agent": agent_name,
                    "pending_count": len(pending_steps),
                    "open_needs_count": len(open_needs),
                },
            )

        review_added_needs: List[str] = []
        for item in review.get("additional_needs", []):
            if not isinstance(item, str):
                continue
            need = _canonicalize_need_text(item)
            key = need.lower()
            if not need or key in seen_need_keys:
                continue
            seen_need_keys.add(key)
            open_needs.append(need)
            review_added_needs.append(need)
        if review.get("additional_needs"):
            log_event(
                "event_manager.collaboration",
                "open_needs_updated",
                {"open_needs": open_needs},
                direction="inbound",
            )

        clarification_request = _first_user_clarification_request(review_added_needs)
        clarification_source = "review_additional_needs"
        if not clarification_request:
            clarification_request = _first_user_clarification_request(open_needs)
            clarification_source = "review_open_needs"
        if clarification_request:
            enriched["workflow_paused"] = True
            enriched["pause_reason"] = "awaiting_user_clarification"
            enriched["pause_request"] = clarification_request
            log_event(
                "event_manager.collaboration",
                "workflow_paused_for_user_input",
                {
                    "step": step_counter,
                    "agent": agent_name,
                    "request": clarification_request,
                    "source": clarification_source,
                },
            )
            break

        review_updated_plan = bool(review.get("should_update_plan") and review.get("updated_steps"))
        if review_updated_plan:
            candidate_steps = list(review["updated_steps"])
            candidate_signatures = _step_signature_list(candidate_steps)
            pending_signatures = _step_signature_list(pending_steps)
            completed_signature = _step_signature({"agent": agent_name, "goal": goal})
            additional_needs_in_review = bool(review.get("additional_needs"))

            no_progress_reasons: List[str] = []
            if candidate_signatures and all(sig == completed_signature for sig in candidate_signatures):
                no_progress_reasons.append("repeats_completed_step")
            if candidate_signatures and candidate_signatures == pending_signatures:
                no_progress_reasons.append("identical_to_existing_pending")

            if no_progress_reasons and not additional_needs_in_review:
                review_updated_plan = False
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
            else:
                pending_steps = candidate_steps
                log_event(
                    "event_manager.collaboration",
                    "plan_updated",
                    {
                        "reason": str(review.get("reason", "")),
                        "new_pending_steps": [
                            {
                                "agent": str(step_meta.get("agent", "")),
                                "goal": str(step_meta.get("goal", "")),
                            }
                            for step_meta in pending_steps
                        ],
                    },
                )
        else:
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
                        if str(need).strip().lower() in consumed_need_keys
                    ]
                    open_needs = [
                        need
                        for need in open_needs
                        if str(need).strip().lower() not in consumed_need_keys
                    ]
                    for removed in removed_needs:
                        target_agent, request = _parse_targeted_need(str(removed))
                        log_event(
                            "workflow.need",
                            "need_resolved_by_plan_augmentation",
                            {
                                "target_agent": target_agent,
                                "request": request or str(removed),
                            },
                        )
                log_event(
                    "event_manager.collaboration",
                    "plan_augmented_from_additional_needs",
                    {
                        "added_steps": [
                            {
                                "agent": str(step_meta.get("agent", "")),
                                "goal": str(step_meta.get("goal", "")),
                            }
                            for step_meta in fallback_steps
                        ],
                        "remaining_open_needs": open_needs,
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
                "parsed_additional_needs": [],
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

    return results


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


def _build_agent_input(user_input: str, conversation_history: str) -> str:
    request_brief = _compact_text(user_input, max_chars=700)
    if request_brief:
        return request_brief
    if conversation_history.strip():
        return _summarize_conversation_history(conversation_history, max_turn_lines=3, max_chars=320)
    return ""


def execute_plan(
    plan: Dict[str, Any],
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
) -> Any:
    raw_plan = str(plan.get("raw_plan", ""))
    user_input = str(plan.get("meta", {}).get("user_input", ""))
    routing_hint = plan.get("meta", {}).get("routing_hint", {})
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
            "routing_hint": routing_hint if isinstance(routing_hint, dict) else {},
            "collaboration_plan": collaboration_plan if isinstance(collaboration_plan, dict) else {},
            "num_available_agents": len(available_agents),
        },
    )

    executable_agents = [agent for agent in available_agents if _is_local_agent(agent) or _is_a2a_agent(agent)]
    selected_agents = _select_executable_agents(
        candidate_agents=executable_agents,
        raw_plan=raw_plan,
        user_input=user_input,
        routing_hint=routing_hint if isinstance(routing_hint, dict) else {},
    )
    log_event(
        "event_manager",
        "agents_selected",
        {"selected": [str(item.get("name", "")) for item in selected_agents]},
    )

    collaboration_steps = _extract_collaboration_steps(
        collaboration_plan=collaboration_plan,
        available_agents=executable_agents,
    )
    if collaboration_steps and user_input:
        log_event(
            "event_manager.collaboration",
            "workflow_selected",
            {
                "source": "planner_collaboration_plan",
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
        results = _run_collaboration_workflow(
            main_agent=main_agent,
            available_agents=executable_agents,
            workflow_id=workflow_id,
            raw_plan=raw_plan,
            steps=collaboration_steps,
            user_input=user_input,
            conversation_history=conversation_history,
        )
        pause_payload = _extract_pause_request_from_results(results)
        if pause_payload:
            log_event(
                "event_manager.collaboration",
                "workflow_paused_response_returned",
                pause_payload,
            )
            request = str(pause_payload.get("request", "")).strip()
            return (
                "진행을 일시 중단하고 사용자 응답을 기다립니다.\n\n"
                f"{request}"
            )
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
        return formatted

    if selected_agents and user_input:
        fallback_steps: List[Dict[str, Any]] = []
        for meta in selected_agents:
            fallback_steps.append(
                {
                    "agent": str(meta.get("name", "UnknownAgent")),
                    "goal": "Handle your part of the user request and provide handoff-ready output.",
                    "deliverable": "Concise result with key facts for the next step.",
                    "tool_hints": [],
                    "agent_meta": meta,
                }
            )
        log_event(
            "event_manager.collaboration",
            "workflow_selected",
            {
                "source": "selected_agents_fallback",
                "steps": [
                    {"step": idx + 1, "agent": str(step.get("agent", ""))}
                    for idx, step in enumerate(fallback_steps)
                ],
            },
        )
        results = _run_collaboration_workflow(
            main_agent=main_agent,
            available_agents=executable_agents,
            workflow_id=workflow_id,
            raw_plan=raw_plan,
            steps=fallback_steps,
            user_input=user_input,
            conversation_history=conversation_history,
        )
        pause_payload = _extract_pause_request_from_results(results)
        if pause_payload:
            log_event(
                "event_manager.collaboration",
                "workflow_paused_response_returned",
                pause_payload,
            )
            request = str(pause_payload.get("request", "")).strip()
            return (
                "진행을 일시 중단하고 사용자 응답을 기다립니다.\n\n"
                f"{request}"
            )
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
        log_event("event_manager", "local_execution_completed", {"results": results})
        return formatted

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
        return direct_result

    fallback = raw_plan or "No plan was generated."
    log_event("event_manager", "execute_plan_fallback", {"result": fallback})
    return fallback


__all__ = ["execute_plan"]

