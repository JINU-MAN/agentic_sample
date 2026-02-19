from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import threading
from typing import Any, Dict, List
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from google.adk.runners import InMemoryRunner
from google.genai import types

from system_logger import log_event, log_exception


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
    return agent_meta.get("type") == "a2a"


def _is_local_agent(agent_meta: Dict[str, Any]) -> bool:
    return agent_meta.get("type") == "local"


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
        response_text = _extract_a2a_response_text(raw_result)
        if not response_text:
            response_text = json.dumps(raw_result, ensure_ascii=False)
        result = {
            "ok": True,
            "agent": name,
            "response": response_text,
            "raw_a2a_result": raw_result,
        }
        log_event("event_manager.a2a", "request_completed", {"base_url": base_url, "result": result})
        return result
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).strip()
        error_lower = error_msg.lower()
        is_timeout = "timeout" in error_type.lower() or "timed out" in error_lower
        is_card_fetch_error = "agent card" in error_lower or "/.well-known/agent-card.json" in error_lower

        if is_card_fetch_error:
            error_text = (
                f"A2A agent card fetch failed ({error_type}): {error_msg}. "
                f"Card endpoint: {card_url}. "
                "Bridge server may not be ready or has stopped. "
                "Run `python start_agentic.py` again, or `python scripts/start_a2a_agents.py` and verify the card endpoint."
            )
        elif is_timeout:
            error_text = (
                f"A2A agent request timed out ({error_type}): {error_msg}. "
                f"Agent base URL: {base_url}. "
                "The bridge received the request but did not finish within timeout. "
                "Increase `A2A_REQUEST_TIMEOUT_SEC` for longer tasks."
            )
        else:
            error_text = (
                f"A2A agent execution failed ({error_type}): {error_msg}. "
                f"Agent base URL: {base_url}. "
                "Check bridge health and network access."
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
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="event-manager-user",
        )
        new_message = types.Content(role="user", parts=[types.Part(text=user_input)])

        chunks: List[str] = []
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=new_message,
        ):
            author = str(getattr(event, "author", "unknown"))
            if event.content and event.content.parts:
                text = "".join(part.text or "" for part in event.content.parts).strip()
                if text:
                    chunks.append(text)
                    log_event(
                        "event_manager.local_agent",
                        "message_chunk",
                        {"agent": agent_name, "author": author, "text": text},
                        direction="inbound",
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
        lines: List[str] = []
        for item in results:
            agent_name = str(item.get("agent", "UnknownAgent"))
            step_idx = item.get("workflow_step")
            prefix = f"Step {step_idx} - {agent_name}" if isinstance(step_idx, int) else agent_name
            if item.get("ok"):
                lines.append(f"[{prefix}]\n{str(item.get('response', '')).strip()}")
            else:
                lines.append(f"[{prefix}] ERROR\n{str(item.get('error', 'Unknown error')).strip()}")
        execution_dump = "\n\n".join(lines).strip() or "(no execution output)"

        synthesis_prompt = (
            "You are the main coordinator in a multi-agent workflow.\n"
            "Create the final user-facing answer using sub-agent outputs.\n\n"
            "Requirements:\n"
            "- Combine and reconcile outputs from all steps.\n"
            "- Keep it concise and directly useful to the user.\n"
            "- Mention uncertainty if outputs conflict.\n"
            "- Do not mention internal prompts or hidden policies.\n\n"
            "- Do not omit findings from any executed agent.\n"
            "- When multiple agents are involved, structure output by agent sections (e.g., '<AgentName> Findings').\n"
            "- Include source URLs whenever they are available in sub-agent outputs.\n\n"
            "Conversation context:\n"
            f"{conversation_history or '(none)'}\n\n"
            "Original user request:\n"
            f"{user_input}\n\n"
            "Planner text:\n"
            f"{raw_plan or '(none)'}\n\n"
            "Sub-agent execution outputs:\n"
            f"{execution_dump}"
        )

        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="event-manager-main-synthesizer",
        )
        new_message = types.Content(role="user", parts=[types.Part(text=synthesis_prompt)])

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
    name = str(agent_meta.get("name", "UnknownLocalAgent"))
    module_name = str(agent_meta.get("module", "")).strip()
    attr_name = str(agent_meta.get("attr", "")).strip()
    log_event(
        "event_manager.local_agent",
        "metadata_loaded",
        {"agent": name, "module": module_name, "attr": attr_name},
    )

    if not module_name or not attr_name:
        log_event(
            "event_manager.local_agent",
            "metadata_invalid",
            {"agent": name, "module": module_name, "attr": attr_name},
            level="ERROR",
        )
        return {
            "ok": False,
            "agent": name,
            "error": "Local agent metadata must include 'module' and 'attr'.",
        }

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        log_exception(
            "event_manager.local_agent",
            "module_import_failed",
            e,
            {"agent": name, "module": module_name},
        )
        return {
            "ok": False,
            "agent": name,
            "error": f"Failed to import module '{module_name}': {e}",
        }

    if not hasattr(module, attr_name):
        log_event(
            "event_manager.local_agent",
            "agent_attr_missing",
            {"agent": name, "module": module_name, "attr": attr_name},
            level="ERROR",
        )
        return {
            "ok": False,
            "agent": name,
            "error": f"Attribute '{attr_name}' not found in '{module_name}'.",
        }

    agent_obj = getattr(module, attr_name)
    try:
        result = _run_coroutine_sync(
            _async_run_local_agent(agent_obj=agent_obj, agent_name=name, user_input=user_input)
        )
        log_event("event_manager.local_agent", "execution_returned", {"agent": name, "result": result})
        return result
    except Exception as e:
        log_exception(
            "event_manager.local_agent",
            "execution_crashed",
            e,
            {"agent": name},
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
            if goal:
                parts.append(f"{header}\nGoal: {goal}\n{result.get('response', '')}")
            else:
                parts.append(f"{header}\n{result.get('response', '')}")
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

        agent_meta = available_index.get(agent_name.lower())
        if agent_meta is None:
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
        if not goal:
            goal = "Handle this step and provide a handoff-ready output."

        resolved_steps.append(
            {
                "agent": str(agent_meta.get("name", "")).strip() or agent_name,
                "goal": goal[:1000],
                "deliverable": deliverable[:1000],
                "tool_hints": tool_hints,
                "agent_meta": agent_meta,
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


def _format_remaining_steps(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "(none)"
    lines: List[str] = []
    for idx, step in enumerate(steps, start=1):
        agent = str(step.get("agent", "UnknownAgent"))
        goal = str(step.get("goal", "")).strip()
        hints = [str(item).strip() for item in step.get("tool_hints", []) if isinstance(item, str) and str(item).strip()]
        hint_text = f" (tool_hints: {', '.join(hints)})" if hints else ""
        if goal:
            lines.append(f"{idx}. {agent} - {goal}{hint_text}")
        else:
            lines.append(f"{idx}. {agent}{hint_text}")
    return "\n".join(lines).strip()


def _format_prior_results_for_handoff(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "(none)"

    parts: List[str] = []
    for item in results:
        agent_name = str(item.get("agent", "UnknownAgent"))
        step_index = item.get("workflow_step")
        prefix = f"Step {step_index} - {agent_name}" if isinstance(step_index, int) else agent_name
        if item.get("ok"):
            parts.append(f"[{prefix}]\n{str(item.get('response', '')).strip()}")
        else:
            parts.append(f"[{prefix}] ERROR\n{str(item.get('error', 'Unknown error')).strip()}")
    return "\n\n".join(parts).strip()


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
        "type": str(agent_meta.get("type", "")).strip() or "local",
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
        agent_type = str(card.get("type", "local")).strip() or "local"
        desc = str(card.get("description", "")).strip()
        caps = ", ".join(str(item) for item in card.get("capabilities", []) if str(item).strip()) or "(none)"
        tools = "; ".join(str(item) for item in card.get("tools", []) if str(item).strip()) or "(none)"
        instruction_preview = str(card.get("instruction_preview", "")).strip() or "(not provided)"
        lines.append(
            f"- name: {name}\n"
            f"  type: {agent_type}\n"
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
                need = _normalize_need_text(item)
                if not need:
                    continue
                key = need.lower()
                if key in {"none", "n/a", "no", "null", "없음"}:
                    return []
                if key in seen:
                    continue
                seen.add(key)
                collected.append(need)
                if len(collected) >= 12:
                    break
            return collected
        if isinstance(raw_needs, str):
            token = _normalize_need_text(raw_needs)
            if token.lower() in {"none", "n/a", "no", "null", "없음"}:
                return []
            if token:
                return [token]

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
        token = _normalize_need_text(inline_value)
        if token.lower() in {"none", "n/a", "no", "null", "없음"}:
            return []
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

        token = _normalize_need_text(stripped)
        if not token:
            continue
        key = token.lower()
        if key in {"none", "n/a", "no", "null", "없음"}:
            return []
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
    match = re.match(r"^\[(?P<agent>[^\[\]]{1,80})\]\s*(?P<request>.+)$", token)
    if not match:
        return "", ""
    target_agent = str(match.group("agent") or "").strip()
    request = str(match.group("request") or "").strip()
    return target_agent, request


def _tool_hints_from_agent_meta(agent_meta: Dict[str, Any], max_hints: int = 4) -> List[str]:
    hints: List[str] = []
    for tool in agent_meta.get("tools", []):
        if isinstance(tool, dict):
            token = str(tool.get("name", "")).strip()
        else:
            token = str(tool).strip()
        if token and token not in hints:
            hints.append(token)
        if len(hints) >= max_hints:
            break
    return hints


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
        if target_key == "mainagent":
            continue

        agent_meta = available_index.get(target_key)
        if agent_meta is None:
            continue

        canonical_need = f"{target_key}|{request.lower()}"
        if canonical_need in local_signatures:
            continue

        goal = (
            "Handle unresolved follow-up need from previous collaboration output.\n"
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
                "tool_hints": _tool_hints_from_agent_meta(agent_meta),
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


def _build_collaboration_step_input(
    *,
    workflow_id: str,
    user_input: str,
    conversation_history: str,
    prior_results: List[Dict[str, Any]],
    open_needs: List[str],
    remaining_steps: List[Dict[str, Any]],
    available_agents: List[Dict[str, Any]],
    step: Dict[str, Any],
    step_index: int,
    total_steps_hint: int,
) -> str:
    step_goal = str(step.get("goal", "")).strip()
    deliverable = str(step.get("deliverable", "")).strip()
    raw_tool_hints = step.get("tool_hints", [])
    tool_hints = [str(item).strip() for item in raw_tool_hints if isinstance(item, str) and str(item).strip()]
    tool_hints_text = "\n".join(f"- {item}" for item in tool_hints) if tool_hints else "(none)"

    agent_meta = step.get("agent_meta", {})
    available_tools: List[str] = []
    if isinstance(agent_meta, dict):
        for tool in agent_meta.get("tools", []):
            if isinstance(tool, dict):
                tool_name = str(tool.get("name", "")).strip()
                if tool_name:
                    available_tools.append(tool_name)
            elif isinstance(tool, str):
                token = tool.strip()
                if token:
                    available_tools.append(token)
    available_tools = list(dict.fromkeys(available_tools))
    available_tools_text = ", ".join(available_tools) if available_tools else "(not provided)"

    prior_text = _format_prior_results_for_handoff(prior_results)
    needs_text = "\n".join(f"- {item}" for item in open_needs) if open_needs else "(none)"
    remaining_text = _format_remaining_steps(remaining_steps)
    delegate_agent_names = _format_delegate_agent_names(available_agents)
    delegate_agent_profiles = _format_delegate_agent_profiles(
        available_agents=available_agents,
        current_agent_name=str(step.get("agent", "")),
    )

    return (
        "You are participating in a multi-agent collaboration workflow.\n"
        f"Current step: {step_index}/{total_steps_hint}\n"
        f"Assigned agent: {step.get('agent', 'UnknownAgent')}\n\n"
        f"Workflow ID: {workflow_id}\n\n"
        f"Step goal:\n{step_goal or '(no explicit goal provided)'}\n\n"
        f"Expected handoff deliverable:\n{deliverable or '(not specified)'}\n\n"
        f"Planner tool hints for this step:\n{tool_hints_text}\n\n"
        f"Agent available tools:\n{available_tools_text}\n\n"
        "Rules:\n"
        "- Focus only on this step.\n"
        "- Use previous step outputs as key input.\n"
        "- Decide your own approach and choose tools based on this step goal.\n"
        "- Use indirect delegation only: do not call other agents directly from this step.\n"
        "- If a specialist can improve quality, request it in `Additional Needs` as `[TargetAgentName] concrete request`.\n"
        "- MainAgent will convert unresolved additional needs into replanned steps when feasible.\n"
        "- Do not rely on fixed keyword templates unless user explicitly provided exact terms.\n"
        "- If local evidence is insufficient, request follow-up work from another agent.\n"
        "- If user clarification is required, ask MainAgent to request clarification.\n"
        "- Return output that the next step (or user) can directly consume.\n\n"
        f"Available agents for additional-need targeting:\n{delegate_agent_names}\n\n"
        f"Agent capability profiles:\n{delegate_agent_profiles}\n\n"
        "Conversation context (recent turns):\n"
        f"{conversation_history or '(none)'}\n\n"
        "Original user request:\n"
        f"{user_input}\n\n"
        "Shared progress context:\n"
        f"Previous step outputs:\n{prior_text}\n\n"
        f"Open additional needs:\n{needs_text}\n\n"
        f"Current remaining planned steps:\n{remaining_text}\n\n"
        "Output format:\n"
        "1) Main response for this step.\n"
        "2) Additional needs under heading 'Additional Needs:'.\n"
        "   - If none: `Additional Needs: none`\n"
        "   - If needed, use bullets with target prefix:\n"
        "     - [TargetAgentName] concrete request\n"
        "   - TargetAgentName must be one of the available agents listed above."
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
            latest_text = str(latest_result.get("response", "")).strip()
        else:
            latest_text = str(latest_result.get("error", "Unknown error")).strip()

        completed_text = _format_prior_results_for_handoff(completed_results)
        pending_text = _format_remaining_steps(pending_steps)
        needs_text = "\n".join(f"- {item}" for item in open_needs) if open_needs else "(none)"
        activated_cards_text = _format_agent_card_snapshots(activated_agent_cards)

        prompt = (
            "You are the main coordinator reviewing multi-agent progress.\n"
            "Decide whether to update the remaining plan based on latest output.\n"
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
            "Rules:\n"
            "- Use only names from Available agents.\n"
            "- Prefer steps that match each agent's available tools/capabilities.\n"
            "- Use activated agent cards as working memory for immediate replanning decisions.\n"
            "- additional_needs should include only unresolved concrete needs.\n"
            "- Treat `Current open needs` as high-priority unresolved work for replanning.\n"
            "- If a need is formatted as `[AgentName] request` and that agent exists, prefer adding/updating a step for that agent.\n"
            "- Follow indirect delegation model: specialists request via Additional Needs, MainAgent routes via updated_steps.\n"
            "- If no plan update is needed, set should_update_plan=false and updated_steps=[].\n"
            "- updated_steps should represent ONLY remaining future steps, not completed ones.\n"
            "- Respect explicit user constraints.\n\n"
            f"Conversation context:\n{conversation_history or '(none)'}\n\n"
            f"User request:\n{user_input}\n\n"
            f"Original planner text:\n{raw_plan or '(none)'}\n\n"
            f"Latest completed step: {latest_step} ({latest_agent}, {latest_status})\n"
            f"Latest step output:\n{latest_text or '(none)'}\n\n"
            f"Completed outputs so far:\n{completed_text}\n\n"
            f"Current open needs:\n{needs_text}\n\n"
            f"Current pending steps:\n{pending_text}\n\n"
            f"Activated agent cards so far:\n{activated_cards_text}\n\n"
            f"Available agents:\n{agents_desc}\n"
        )

        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="event-manager-main-replanner",
        )
        new_message = types.Content(role="user", parts=[types.Part(text=prompt)])

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

        raw_text = "\n".join(chunks).strip()
        parsed = _extract_json_object(raw_text) or {}

        additional_needs: List[str] = []
        seen_needs: set[str] = set()
        for item in parsed.get("additional_needs", []):
            if not isinstance(item, str):
                continue
            token = item.strip()
            key = token.lower()
            if len(token) < 2 or key in seen_needs:
                continue
            seen_needs.add(key)
            additional_needs.append(token[:300])
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
    failed_error = str(failed_result.get("error", "Unknown error")).strip()
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

        prompt = (
            "You are the main coordinator handling an interrupted workflow.\n"
            "A collaboration step failed. Analyze root cause and choose one action:\n"
            "1) replan remaining steps, or 2) abort and return user-facing error guidance.\n"
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
            "Rules:\n"
            "- Use only names from Available agents.\n"
            "- If replan is feasible this turn, set decision=replan and provide updated_steps.\n"
            "- If not feasible, set decision=abort and provide a clear user_message.\n"
            "- Treat `Current open needs` as unresolved work; if a need is `[AgentName] request`, prefer routing to that agent in updated_steps.\n"
            "- Follow indirect delegation model: specialists do not call each other directly, MainAgent handles rerouting.\n"
            "- updated_steps must contain only future steps.\n"
            "- Respect explicit user constraints.\n\n"
            f"Conversation context:\n{conversation_history or '(none)'}\n\n"
            f"User request:\n{user_input}\n\n"
            f"Original planner text:\n{raw_plan or '(none)'}\n\n"
            f"Failed step: {failed_step} ({failed_agent})\n"
            f"Failure detail:\n{failed_error or '(none)'}\n\n"
            f"Execution output so far:\n{completed_text}\n\n"
            f"Current open needs:\n{needs_text}\n\n"
            f"Current pending steps:\n{pending_text}\n\n"
            f"Activated agent cards so far:\n{activated_cards_text}\n\n"
            f"Available agents:\n{agents_desc}\n"
        )

        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="event-manager-main-failure-handler",
        )
        new_message = types.Content(role="user", parts=[types.Part(text=prompt)])

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
    pending_steps: List[Dict[str, Any]] = list(steps)
    open_needs: List[str] = []
    seen_need_keys: set[str] = set()
    activated_agent_cards_map: Dict[str, Dict[str, Any]] = {}
    step_counter = 0
    max_steps = max(8, len(steps) + 6)

    while pending_steps and step_counter < max_steps:
        step_counter += 1
        step = pending_steps.pop(0)
        agent_name = str(step.get("agent", "UnknownAgent"))
        goal = str(step.get("goal", "")).strip()
        tool_hints = [str(item).strip() for item in step.get("tool_hints", []) if isinstance(item, str) and str(item).strip()]
        current_agent_meta = step.get("agent_meta", {})
        if isinstance(current_agent_meta, dict):
            snapshot = _build_agent_card_snapshot(current_agent_meta)
            snapshot_name = str(snapshot.get("name", "")).strip().lower()
            if snapshot_name:
                activated_agent_cards_map[snapshot_name] = snapshot
        activated_agent_cards = list(activated_agent_cards_map.values())
        total_steps_hint = step_counter + len(pending_steps)
        step_input = _build_collaboration_step_input(
            workflow_id=workflow_id,
            user_input=user_input,
            conversation_history=conversation_history,
            prior_results=results,
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
                "activated_agent_cards": [str(item.get("name", "")) for item in activated_agent_cards],
            },
            direction="outbound",
        )

        result = _execute_single_agent(step["agent_meta"], step_input)
        enriched = dict(result)
        enriched["workflow_step"] = step_counter
        enriched["goal"] = goal
        enriched["tool_hints"] = tool_hints
        response_for_need_parse = (
            str(enriched.get("response", "")).strip()
            if enriched.get("ok")
            else str(enriched.get("error", "")).strip()
        )
        parsed_needs = _extract_additional_needs_from_agent_output(response_for_need_parse)
        enriched["parsed_additional_needs"] = parsed_needs
        results.append(enriched)

        if parsed_needs:
            added_needs: List[str] = []
            for item in parsed_needs:
                need = item.strip()
                key = need.lower()
                if not need or key in seen_need_keys:
                    continue
                seen_need_keys.add(key)
                open_needs.append(need)
                added_needs.append(need)
            if added_needs:
                log_event(
                    "event_manager.collaboration",
                    "open_needs_updated_from_agent_output",
                    {"added_needs": added_needs, "open_needs": open_needs},
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

        for item in review.get("additional_needs", []):
            if not isinstance(item, str):
                continue
            need = item.strip()
            key = need.lower()
            if not need or key in seen_need_keys:
                continue
            seen_need_keys.add(key)
            open_needs.append(need)
        if review.get("additional_needs"):
            log_event(
                "event_manager.collaboration",
                "open_needs_updated",
                {"open_needs": open_needs},
                direction="inbound",
            )

        review_updated_plan = bool(review.get("should_update_plan") and review.get("updated_steps"))
        if review_updated_plan:
            pending_steps = list(review["updated_steps"])
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
                    open_needs = [
                        need
                        for need in open_needs
                        if str(need).strip().lower() not in consumed_need_keys
                    ]
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
        log_event(
            "event_manager.collaboration",
            "workflow_stopped_on_max_steps",
            {"max_steps": max_steps, "remaining_steps": len(pending_steps)},
            level="ERROR",
        )

    return results


def _build_agent_input(user_input: str, conversation_history: str) -> str:
    if not conversation_history.strip():
        return user_input
    return (
        "Conversation context (recent turns):\n"
        f"{conversation_history}\n\n"
        f"Current user request:\n{user_input}"
    )


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

    executable_agents = [
        agent for agent in available_agents if _is_local_agent(agent) or _is_a2a_agent(agent)
    ]
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
                    "agent": str(meta.get("name", "UnknownLocalAgent")),
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
        log_event(
            "event_manager",
            "a2a_execution_selected",
            {"agent": str(a2a_agents[0].get("name", "")), "base_url": a2a_agents[0].get("base_url", "")},
        )
        return _execute_single_a2a_agent(a2a_agents[0], agent_input)

    fallback = raw_plan or "No plan was generated."
    log_event("event_manager", "execute_plan_fallback", {"result": fallback})
    return fallback


__all__ = ["execute_plan"]
