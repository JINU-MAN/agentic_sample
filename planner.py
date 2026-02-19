from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any, Dict, List

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from system_logger import log_event, log_exception


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
    log_event(
        "planner",
        "prompt_dispatched",
        {
            "task": task,
            "agent_name": getattr(agent, "name", "unknown"),
            "prompt": prompt,
        },
        direction="outbound",
    )
    try:
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="planner-user",
        )
        new_message = types.Content(role="user", parts=[types.Part(text=prompt)])

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
                        "planner",
                        "agent_message_chunk",
                        {
                            "task": task,
                            "author": author,
                            "text": text,
                        },
                        direction="inbound",
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
            {"task": task, "agent_name": getattr(agent, "name", "unknown")},
        )
        raise
    finally:
        await runner.close()


def _run_agent_prompt(agent: LlmAgent, prompt: str, task: str) -> str:
    return str(
        _run_coroutine_sync(_async_run_agent_prompt(agent=agent, prompt=prompt, task=task))
    )


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
            f"  description: {desc}\n"
            f"  capabilities: {caps_text}\n"
            f"  tools: {tools_text}\n"
            f"  instruction_preview: {instruction_preview}"
        )
    return "\n".join(lines)


def _agent_domain_tags(agent_meta: Dict[str, Any]) -> set[str]:
    blob_parts: List[str] = [
        str(agent_meta.get("name", "")),
        str(agent_meta.get("description", "")),
    ]
    blob_parts.extend(str(item) for item in agent_meta.get("capabilities", []))
    for tool in agent_meta.get("tools", []):
        if isinstance(tool, dict):
            blob_parts.append(str(tool.get("name", "")))
            blob_parts.append(str(tool.get("description", "")))
        elif isinstance(tool, str):
            blob_parts.append(tool)
    blob = " ".join(blob_parts).lower()

    domains: set[str] = set()
    if any(token in blob for token in ["paper", "papers", "pdf", "research"]):
        domains.add("paper")
    if any(token in blob for token in ["web", "article", "articles", "news", "search_web", "fetch_web"]):
        domains.add("web")
    if any(token in blob for token in ["sns", "social", "post", "posts"]):
        domains.add("sns")
    return domains


def _requested_domains_from_text(text: str) -> set[str]:
    lowered = str(text or "").lower()
    requested: set[str] = set()
    if any(token in lowered for token in ["paper", "papers", "pdf", "research"]):
        requested.add("paper")
    if any(token in lowered for token in ["web", "article", "articles", "news"]):
        requested.add("web")
    if any(token in lowered for token in ["sns", "social media", "social", "post", "posts"]):
        requested.add("sns")
    return requested


def _expand_routing_hint_for_accuracy(
    routing_hint: Dict[str, Any],
    available_agents: List[Dict[str, Any]],
    user_input: str,
    raw_plan: str,
) -> Dict[str, Any]:
    selected = [
        str(item).strip()
        for item in routing_hint.get("selected_agents", [])
        if isinstance(item, str) and str(item).strip()
    ]
    selected_set = {item.lower() for item in selected}

    requested_domains = _requested_domains_from_text(user_input)
    requested_domains.update(_requested_domains_from_text(raw_plan))
    if not requested_domains:
        return routing_hint

    domain_by_agent: Dict[str, set[str]] = {}
    name_by_key: Dict[str, str] = {}
    for meta in available_agents:
        name = str(meta.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        name_by_key[key] = name
        domain_by_agent[key] = _agent_domain_tags(meta)

    covered_domains: set[str] = set()
    for key in selected_set:
        covered_domains.update(domain_by_agent.get(key, set()))
    missing_domains = [domain for domain in requested_domains if domain not in covered_domains]
    if not missing_domains:
        return routing_hint

    added_agents: List[str] = []
    for domain in missing_domains:
        candidates: List[tuple[int, str]] = []
        for key, domains in domain_by_agent.items():
            if key in selected_set:
                continue
            if domain not in domains:
                continue
            overlap = len(requested_domains.intersection(domains))
            candidates.append((overlap, key))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen_key = candidates[0][1]
        chosen_name = name_by_key.get(chosen_key, chosen_key)
        selected.append(chosen_name)
        selected_set.add(chosen_key)
        covered_domains.update(domain_by_agent.get(chosen_key, set()))
        added_agents.append(chosen_name)

    if not added_agents:
        return routing_hint

    max_selected = min(max(len(available_agents), 1), 8)
    updated = dict(routing_hint)
    updated["selected_agents"] = selected[:max_selected]
    reason = str(updated.get("reason", "")).strip()
    suffix = (
        f" Coverage-first adjustment added: {', '.join(added_agents)} "
        f"(requested domains: {', '.join(sorted(requested_domains))})."
    )
    updated["reason"] = (reason + suffix).strip()
    log_event(
        "planner",
        "routing_hint_coverage_augmented",
        {
            "requested_domains": sorted(requested_domains),
            "added_agents": added_agents,
            "selected_agents": updated["selected_agents"],
        },
    )
    return updated


def _derive_step_tool_hints(agent_meta: Dict[str, Any], max_hints: int = 3) -> List[str]:
    hints: List[str] = []
    for tool in agent_meta.get("tools", []):
        token = ""
        if isinstance(tool, dict):
            token = str(tool.get("name", "")).strip()
        elif isinstance(tool, str):
            token = tool.strip()
        if token and token not in hints:
            hints.append(token)
        if len(hints) >= max_hints:
            break
    return hints


def _build_specialist_step(agent_meta: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    name = str(agent_meta.get("name", "UnknownAgent")).strip() or "UnknownAgent"
    tags = _agent_domain_tags(agent_meta)
    if "paper" in tags:
        goal = (
            "Lead paper/research evidence collection and analysis for this request. "
            "Search local paper DB first, and if coverage is insufficient, request WebSearchAnalyst follow-up via Additional Needs for MainAgent routing.\n"
            f"User request: {user_input}"
        )
        deliverable = "Research-focused summary with paper-level evidence and source paths/URLs."
    elif "web" in tags:
        goal = (
            "Lead external web/article evidence collection and verification for this request.\n"
            f"User request: {user_input}"
        )
        deliverable = "Web evidence summary with trustworthy source URLs."
    elif "sns" in tags:
        goal = (
            "Lead SNS/social-signal collection and meaning-level summarization for this request.\n"
            f"User request: {user_input}"
        )
        deliverable = "SNS signal summary with key posts and relevance rationale."
    else:
        goal = (
            "Handle your specialist part of the request and provide handoff-ready output.\n"
            f"User request: {user_input}"
        )
        deliverable = "Specialist output summary with key evidence."
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

    existing = {str(step.get("agent", "")).strip().lower() for step in steps}
    added_steps: List[Dict[str, Any]] = []
    for name in selected:
        key = name.lower()
        if key in existing:
            continue
        meta = indexed.get(key)
        if meta is None:
            continue
        step = _build_specialist_step(meta, user_input=user_input)
        steps.append(step)
        added_steps.append(step)
        existing.add(key)
        if len(steps) >= 8:
            break

    if not added_steps:
        return collaboration_plan

    notes = str(collaboration_plan.get("notes", "")).strip()
    addendum = (
        " Selected specialists were appended to ensure coverage-first execution "
        "for user-request domains."
    )
    updated = {
        "steps": steps[:8],
        "notes": (notes + addendum).strip(),
    }
    log_event(
        "planner",
        "collaboration_plan_specialist_coverage_added",
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
    fallback = _ensure_selected_agents_covered(
        fallback,
        routing_hint=routing_hint,
        available_agents=available_agents,
        user_input=user_input,
    )
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

    routing_prompt = (
        "You are selecting which agents to execute for the current turn.\n"
        "Return JSON only. No markdown, no explanation outside JSON.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "selected_agents": ["AgentName1", "AgentName2"],\n'
        '  "keywords": ["keyword1", "keyword2"],\n'
        '  "reason": "short reason"\n'
        "}\n\n"
        "Rules:\n"
        "- selected_agents must be names from Available agents.\n"
        "- keywords must be short routing terms for this turn.\n"
        "- Prioritize answer quality and user-request coverage over execution speed.\n"
        "- Prefer domain-specialized agents when a specialist is available.\n"
        "- Include every agent that materially improves accuracy/completeness.\n"
        "- Do not force single-agent selection when multiple evidence sources are requested.\n\n"
        "- Select based on user goal and available agent capabilities/tools.\n"
        "- Avoid hardcoded assumptions about specific agent names.\n\n"
        f"Recent conversation context:\n{conversation_history or '(none)'}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Current plan text:\n{raw_plan}\n\n"
        f"Available agents:\n{agents_desc}\n"
    )

    raw = _run_agent_prompt(main_agent, routing_prompt, task="routing_hint")
    parsed = _extract_json_object(raw)
    normalized = _normalize_routing_hint(parsed, available_agents)
    return _expand_routing_hint_for_accuracy(
        normalized,
        available_agents=available_agents,
        user_input=user_input,
        raw_plan=raw_plan,
    )


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

    routing_selected = ", ".join(
        [str(item) for item in routing_hint.get("selected_agents", []) if isinstance(item, str)]
    ) or "(none)"

    collaboration_prompt = (
        "You are creating an agent-to-agent collaboration workflow.\n"
        "Return JSON only. No markdown, no prose outside JSON.\n\n"
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
        "Rules:\n"
        "- Use only names from Available agents.\n"
        "- Keep steps practical and complete (1 to 6).\n"
        "- Steps should pass useful outputs to the next step.\n"
        "- Respect explicit user constraints.\n"
        "- Prioritize satisfying user intent and factual completeness over runtime speed.\n"
        "- Prefer specialist-owned steps rather than assigning specialist tasks to non-specialists.\n"
        "- Prefer multi-agent collaboration when it improves coverage or verification.\n\n"
        "- Choose agents and tool usage strategy based on available metadata, not hardcoded roles.\n"
        "- For each step, add tool_hints when useful.\n\n"
        f"Recent conversation context:\n{conversation_history or '(none)'}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Current plan text:\n{raw_plan}\n\n"
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

    planning_prompt = (
        "You are the planner of a multi-agent system.\n"
        "Read the user request and produce a practical execution plan.\n\n"
        "Requirements:\n"
        "1) First, summarize the user goal in one sentence.\n"
        "2) Then write a 'Plan:' section with numbered steps.\n"
        "3) For each step, specify which agent to use and what input to provide.\n"
        "4) If no sub-agent is needed, explain why and provide direct handling steps.\n"
        "5) Select agents and tool strategies from the available metadata; avoid hardcoded assumptions.\n"
        "6) Prioritize accuracy, evidence coverage, and user-request fulfillment over speed.\n"
        "7) When different evidence sources are requested, include matching specialists.\n\n"
        f"Recent conversation context:\n{conversation_history or '(none)'}\n\n"
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
