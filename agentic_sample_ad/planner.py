from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any, Dict, List

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from agentic_sample_ad.network_retry import collect_text_response_with_network_retry
from agentic_sample_ad.system_logger import log_event, log_exception


def _run_coroutine_sync(coro: Any) -> Any:
    """
    Run an async coroutine from sync code.

    - If no event loop is running in this thread, use asyncio.run.
    - If an event loop is already running, run in a separate thread.
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


async def _async_run_agent_prompt(agent: LlmAgent, prompt: str, task: str) -> str:
    runner = InMemoryRunner(agent=agent, app_name="main-planner")
    agent_name = getattr(agent, "name", "unknown")
    log_event(
        "planner",
        "prompt_dispatched",
        {
            "task": task,
            "agent_name": agent_name,
            "prompt": prompt,
        },
        direction="outbound",
    )
    try:
        new_message = types.Content(role="user", parts=[types.Part(text=prompt)])

        def _on_text(text: str, event: Any) -> None:
            author = str(getattr(event, "author", "unknown"))
            log_event(
                "planner",
                "agent_message_chunk",
                {
                    "task": task,
                    "author": author,
                    "text": text,
                },
                direction="inbound",
            )

        chunks = await collect_text_response_with_network_retry(
            runner=runner,
            user_id="planner-user",
            new_message=new_message,
            component="planner",
            operation_name=f"agent_prompt:{task}",
            retry_details={"task": task, "agent_name": agent_name},
            on_text=_on_text,
        )

        result = "\n".join(chunks).strip()
        log_event(
            "planner",
            "prompt_completed",
            {"task": task, "result": result},
            direction="inbound",
        )
        return result
    except Exception as e:
        log_exception(
            "planner",
            "prompt_failed",
            e,
            {"task": task, "agent_name": agent_name},
        )
        raise
    finally:
        await runner.close()


def _run_agent_prompt(agent: LlmAgent, prompt: str, task: str) -> str:
    return str(
        _run_coroutine_sync(_async_run_agent_prompt(agent=agent, prompt=prompt, task=task))
    )


def _compact_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[:max_chars]
    return text[: max_chars - 14] + "...(truncated)"


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


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        log_event("planner", "extract_json_empty", {})
        return None

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            log_event("planner", "extract_json_success", {"source": "plain_json"})
            return obj
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                log_event("planner", "extract_json_success", {"source": "fenced_json"})
                return obj
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                log_event("planner", "extract_json_success", {"source": "substring"})
                return obj
        except json.JSONDecodeError:
            log_event("planner", "extract_json_failed", {"source": "substring", "text": stripped})
            return None

    log_event("planner", "extract_json_failed", {"source": "all", "text": stripped})
    return None


def _normalize_routing_hint(
    raw_hint: Dict[str, Any] | None,
    available_agents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not raw_hint:
        log_event("planner", "routing_hint_missing", {})
        return {"selected_agents": [], "keywords": [], "reason": ""}

    name_map: Dict[str, str] = {}
    for agent in available_agents:
        name = str(agent.get("name", "")).strip()
        if name:
            name_map[name.lower()] = name

    selected_agents: List[str] = []
    seen_selected: set[str] = set()
    for item in raw_hint.get("selected_agents", []):
        if not isinstance(item, str):
            continue
        normalized = name_map.get(item.strip().lower())
        if not normalized or normalized in seen_selected:
            continue
        seen_selected.add(normalized)
        selected_agents.append(normalized)

    keywords: List[str] = []
    seen_keywords: set[str] = set()
    for item in raw_hint.get("keywords", []):
        if not isinstance(item, str):
            continue
        keyword = item.strip().lower()
        if len(keyword) < 2 or keyword in seen_keywords:
            continue
        seen_keywords.add(keyword)
        keywords.append(keyword)

    reason = str(raw_hint.get("reason", "")).strip()
    max_selected = min(max(len(available_agents), 1), 8)
    normalized = {
        "selected_agents": selected_agents[:max_selected],
        "keywords": keywords[:12],
        "reason": reason,
    }
    log_event("planner", "routing_hint_normalized", {"routing_hint": normalized})
    return normalized


def _normalize_collaboration_plan(
    raw_plan: Dict[str, Any] | None,
    available_agents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not raw_plan:
        return {"steps": [], "notes": ""}

    name_map: Dict[str, str] = {}
    for agent in available_agents:
        name = str(agent.get("name", "")).strip()
        if name:
            name_map[name.lower()] = name

    steps: List[Dict[str, Any]] = []
    for item in raw_plan.get("steps", []):
        if not isinstance(item, dict):
            continue

        raw_agent_name = str(item.get("agent", "")).strip()
        normalized_agent = name_map.get(raw_agent_name.lower())
        if not normalized_agent:
            continue

        goal = str(
            item.get("goal")
            or item.get("task")
            or item.get("objective")
            or ""
        ).strip()
        deliverable = str(item.get("deliverable") or item.get("output") or "").strip()
        tool_hints: List[str] = []
        raw_tool_hints = item.get("tool_hints")
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
            goal = "Handle this step with your specialization and provide a handoff-ready output."

        steps.append(
            {
                "agent": normalized_agent,
                "goal": goal[:800],
                "deliverable": deliverable[:800],
                "tool_hints": tool_hints,
            }
        )
        if len(steps) >= 8:
            break

    notes = str(raw_plan.get("notes") or raw_plan.get("reason") or "").strip()
    normalized = {"steps": steps, "notes": notes}
    log_event("planner", "collaboration_plan_normalized", {"collaboration_plan": normalized})
    return normalized


def _format_available_agents_for_prompt(available_agents: List[Dict[str, Any]]) -> str:
    if not available_agents:
        return "No sub-agents are currently configured."

    lines: List[str] = []
    for agent in available_agents:
        name = str(agent.get("name", "UnknownAgent")).strip() or "UnknownAgent"
        agent_type = str(agent.get("type", "")).strip() or "unknown"
        role = str(agent.get("role", "")).strip() or "worker"
        desc = str(agent.get("description", "")).strip()

        capabilities = [str(item).strip() for item in agent.get("capabilities", []) if str(item).strip()]
        caps_text = ", ".join(capabilities) if capabilities else "(none)"

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
        tools_text = "; ".join(tool_entries) if tool_entries else "(unknown or not provided)"

        instruction_preview = str(agent.get("instruction_preview", "")).strip()
        if not instruction_preview:
            instruction_preview = "(not provided)"

        lines.append(
            f"- name: {name}\n"
            f"  type: {agent_type}\n"
            f"  role: {role}\n"
            f"  description: {desc}\n"
            f"  capabilities: {caps_text}\n"
            f"  tools: {tools_text}\n"
            f"  instruction_preview: {instruction_preview}"
        )
    return "\n".join(lines)


def _derive_step_tool_hints(agent_meta: Dict[str, Any], max_hints: int = 3) -> List[str]:
    return []


def _build_specialist_step(agent_meta: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    name = str(agent_meta.get("name", "UnknownAgent")).strip() or "UnknownAgent"
    goal = (
        "Handle the part of the request that best matches your role and return a handoff-ready result.\n"
        f"User request: {user_input}"
    )
    deliverable = "Concise result with the most relevant evidence and next-useful details."
    return {
        "agent": name,
        "goal": goal[:800],
        "deliverable": deliverable[:800],
        "tool_hints": _derive_step_tool_hints(agent_meta),
    }


def _ensure_selected_agents_covered(
    collaboration_plan: Dict[str, Any],
    routing_hint: Dict[str, Any],
    available_agents: List[Dict[str, Any]],
    user_input: str,
) -> Dict[str, Any]:
    steps = [dict(item) for item in collaboration_plan.get("steps", []) if isinstance(item, dict)]
    if steps:
        return collaboration_plan

    selected = [
        str(item).strip()
        for item in routing_hint.get("selected_agents", [])
        if isinstance(item, str) and str(item).strip()
    ]
    if not selected:
        return collaboration_plan

    indexed: Dict[str, Dict[str, Any]] = {}
    for meta in available_agents:
        name = str(meta.get("name", "")).strip()
        if name:
            indexed[name.lower()] = meta

    added_steps: List[Dict[str, Any]] = []
    for name in selected:
        key = name.lower()
        meta = indexed.get(key)
        if meta is None:
            continue
        step = _build_specialist_step(meta, user_input=user_input)
        steps.append(step)
        added_steps.append(step)
        if len(steps) >= 8:
            break

    if not added_steps:
        return collaboration_plan

    notes = str(collaboration_plan.get("notes", "")).strip()
    updated = {
        "steps": steps[:8],
        "notes": notes or "Fallback steps were created from the selected agents.",
    }
    log_event(
        "planner",
        "collaboration_plan_fallback_steps_created",
        {
            "added_agents": [str(item.get("agent", "")) for item in added_steps],
            "step_count": len(updated["steps"]),
        },
    )
    return updated


def _fallback_collaboration_plan(
    available_agents: List[Dict[str, Any]],
    routing_hint: Dict[str, Any],
    user_input: str,
) -> Dict[str, Any]:
    name_map: Dict[str, Dict[str, Any]] = {}
    for agent in available_agents:
        name = str(agent.get("name", "")).strip()
        if name:
            name_map[name.lower()] = agent

    steps: List[Dict[str, str]] = []
    for item in routing_hint.get("selected_agents", []):
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if key not in name_map:
            continue
        steps.append(
            {
                "agent": str(name_map[key].get("name", "")).strip(),
                "goal": (
                    "Work on the user request with your specialization and provide "
                    "a result that the next step can directly use.\n"
                    f"User request: {user_input}"
                )[:800],
                "deliverable": "Concise handoff summary with actionable details.",
                "tool_hints": [],
            }
        )

    if not steps and len(available_agents) == 1:
        only = str(available_agents[0].get("name", "")).strip()
        if only:
            steps.append(
                {
                    "agent": only,
                    "goal": (
                        "Handle the user request directly and provide final-ready output.\n"
                        f"User request: {user_input}"
                    )[:800],
                    "deliverable": "Direct user-facing answer.",
                    "tool_hints": [],
                }
            )

    fallback = {
        "steps": steps,
        "notes": "Fallback collaboration plan derived from routing hint.",
    }
    log_event("planner", "collaboration_plan_fallback", {"collaboration_plan": fallback})
    return fallback


def _derive_routing_hint(
    main_agent: LlmAgent,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
) -> Dict[str, Any]:
    if not available_agents:
        log_event("planner", "routing_hint_skipped", {"reason": "no_available_agents"})
        return {"selected_agents": [], "keywords": [], "reason": ""}

    agents_desc = _format_available_agents_for_prompt(available_agents)
    conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=6, max_chars=900)
    plan_summary = _compact_text(raw_plan or "(none)", max_chars=1400)

    routing_prompt = (
        "Select which agents, if any, should work on this turn.\n"
        "Return JSON only.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "selected_agents": ["AgentName1", "AgentName2"],\n'
        '  "keywords": ["keyword1", "keyword2"],\n'
        '  "reason": "short reason"\n'
        "}\n\n"
        "Guidelines:\n"
        "- selected_agents must be names from Available agents.\n"
        "- keywords must be short routing terms for this turn.\n"
        "- Select only agents that materially improve the result.\n"
        "- Use capabilities, tools, and descriptions from metadata.\n"
        "- Include the coordinator only when coordination or channel delivery is needed.\n"
        "- Avoid hardcoded assumptions about specific agent names.\n\n"
        f"Recent conversation context summary:\n{conversation_summary}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Current plan text summary:\n{plan_summary}\n\n"
        f"Available agents:\n{agents_desc}\n"
    )

    raw = _run_agent_prompt(main_agent, routing_prompt, task="routing_hint")
    parsed = _extract_json_object(raw)
    normalized = _normalize_routing_hint(parsed, available_agents)
    return normalized


def _derive_collaboration_plan(
    main_agent: LlmAgent,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
    routing_hint: Dict[str, Any],
) -> Dict[str, Any]:
    if not available_agents:
        log_event("planner", "collaboration_plan_skipped", {"reason": "no_available_agents"})
        return {"steps": [], "notes": ""}

    agents_desc = _format_available_agents_for_prompt(available_agents)
    conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=6, max_chars=900)
    plan_summary = _compact_text(raw_plan or "(none)", max_chars=1400)

    routing_selected = ", ".join(
        [str(item) for item in routing_hint.get("selected_agents", []) if isinstance(item, str)]
    ) or "(none)"

    collaboration_prompt = (
        "Create the next collaboration workflow for this request.\n"
        "Return JSON only.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "steps": [\n'
        "    {\n"
        '      "agent": "AgentName",\n'
        '      "goal": "what this agent should do in this step",\n'
        '      "deliverable": "output format for handoff",\n'
        '      "tool_hints": ["tool_or_strategy_1", "tool_or_strategy_2"]\n'
        "    }\n"
        "  ],\n"
        '  "notes": "short note"\n'
        "}\n\n"
        "Guidelines:\n"
        "- Use only names from Available agents.\n"
        "- Create 0 to 5 practical future steps.\n"
        "- Each step should be useful on its own and easy for the assigned agent to execute.\n"
        "- Respect explicit user constraints.\n"
        "- Choose agents from capabilities/tools metadata.\n"
        "- Leave tool_hints empty unless they are genuinely helpful.\n"
        "- Use the coordinator only for coordination or owned delivery actions.\n\n"
        f"Recent conversation context summary:\n{conversation_summary}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Current plan text summary:\n{plan_summary}\n\n"
        f"Routing hint selected_agents:\n{routing_selected}\n\n"
        f"Available agents:\n{agents_desc}\n"
    )

    raw = _run_agent_prompt(main_agent, collaboration_prompt, task="collaboration_plan")
    parsed = _extract_json_object(raw)
    normalized = _normalize_collaboration_plan(parsed, available_agents)
    normalized = _ensure_selected_agents_covered(
        normalized,
        routing_hint=routing_hint,
        available_agents=available_agents,
        user_input=user_input,
    )
    if normalized.get("steps"):
        log_event("planner", "collaboration_plan_derived", {"collaboration_plan": normalized})
        return normalized

    return _fallback_collaboration_plan(
        available_agents=available_agents,
        routing_hint=routing_hint,
        user_input=user_input,
    )


def plan_with_main_agent(
    main_agent: LlmAgent,
    available_agents: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an execution plan text with the main LLM agent.
    """
    user_input = context.get("user_input", "")
    conversation_history = context.get("conversation_history", "")
    log_event(
        "planner",
        "planning_started",
        {
            "user_input": user_input,
            "num_available_agents": len(available_agents),
        },
    )

    agents_desc = _format_available_agents_for_prompt(available_agents)
    conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=6, max_chars=900)

    planning_prompt = (
        "You are the planner of a multi-agent system.\n"
        "Read the user request and outline a short practical plan.\n\n"
        "Requirements:\n"
        "1) Summarize the user goal in one sentence.\n"
        "2) Write a 'Plan:' section with short numbered steps.\n"
        "3) Mention which agent should handle each step when a specialist is useful.\n"
        "4) If no sub-agent is needed, say so briefly.\n"
        "5) Choose agents from available capabilities/tools metadata.\n"
        "6) Include the coordinator only when direct handling, coordination, or channel delivery is needed.\n\n"
        f"Recent conversation context summary:\n{conversation_summary}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Available agents:\n{agents_desc}\n"
    )

    raw_plan = _run_agent_prompt(main_agent, planning_prompt, task="planning")
    routing_hint = _derive_routing_hint(
        main_agent=main_agent,
        available_agents=available_agents,
        user_input=user_input,
        conversation_history=conversation_history,
        raw_plan=raw_plan,
    )
    collaboration_plan = _derive_collaboration_plan(
        main_agent=main_agent,
        available_agents=available_agents,
        user_input=user_input,
        conversation_history=conversation_history,
        raw_plan=raw_plan,
        routing_hint=routing_hint,
    )
    result = {
        "raw_plan": raw_plan,
        "meta": {
            "user_input": user_input,
            "num_available_agents": len(available_agents),
            "routing_hint": routing_hint,
            "collaboration_plan": collaboration_plan,
        },
    }
    log_event("planner", "planning_completed", {"result": result})
    return result


__all__ = ["plan_with_main_agent"]

