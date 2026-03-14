from __future__ import annotations

import re
from typing import Any, Dict, List

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from agentic_sample_ad.agent_metadata_utils import format_available_agents_for_prompt as _format_available_agents_for_prompt
from agentic_sample_ad.main_agent.system_logger import log_event, log_exception
from agentic_sample_ad.network_retry import collect_text_response_with_network_retry
from agentic_sample_ad.runtime_utils import (
    extract_json_object_with_source,
    run_coroutine_sync,
)


def _build_planning_agent_view(agent: LlmAgent) -> LlmAgent:
    return LlmAgent(
        name=str(getattr(agent, "name", "")).strip() or "PlanningAgent",
        model=getattr(agent, "model", ""),
        instruction=str(getattr(agent, "instruction", "") or "").strip(),
    )


def _is_internal_management_step(goal: str, deliverable: str = "") -> bool:
    combined = " ".join(part for part in [str(goal or "").strip(), str(deliverable or "").strip()] if part).strip()
    if not combined:
        return False
    return bool(
        re.search(r"\bload(?:ing|ed)?\b.{0,40}\bskill\b", combined, flags=re.IGNORECASE)
        or re.search(r"\bload(?:ing|ed)?\b.{0,40}\bsession memory\b", combined, flags=re.IGNORECASE)
        or re.search(r"스킬.{0,24}(?:로드|불러)", combined, flags=re.IGNORECASE)
        or re.search(r"세션 메모리.{0,24}(?:로드|불러)", combined, flags=re.IGNORECASE)
    )


async def _async_run_agent_prompt(agent: LlmAgent, prompt: str, task: str) -> str:
    runner = InMemoryRunner(agent=_build_planning_agent_view(agent), app_name="main-planner")
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
            log_event_fn=log_event,
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
        run_coroutine_sync(_async_run_agent_prompt(agent=agent, prompt=prompt, task=task))
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
    parsed, source = extract_json_object_with_source(text)
    stripped = str(text or "").strip()
    if parsed is not None:
        log_event("planner", "extract_json_success", {"source": source})
        return parsed
    if source == "empty":
        log_event("planner", "extract_json_empty", {})
    else:
        log_event("planner", "extract_json_failed", {"source": source, "text": stripped})
    return None


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
        if not goal:
            goal = "Handle this step with your specialization and provide a handoff-ready output."
        if _is_internal_management_step(goal, deliverable):
            log_event(
                "planner",
                "internal_management_step_ignored",
                {
                    "agent": normalized_agent,
                    "goal": goal,
                    "deliverable": deliverable,
                },
                level="WARNING",
            )
            continue

        steps.append(
            {
                "agent": normalized_agent,
                "goal": goal[:800],
                "deliverable": deliverable[:800],
            }
        )
        if len(steps) >= 8:
            break

    notes = str(raw_plan.get("notes") or raw_plan.get("reason") or "").strip()
    normalized = {"steps": steps, "notes": notes}
    log_event("planner", "collaboration_plan_normalized", {"collaboration_plan": normalized})
    return normalized


def _fallback_collaboration_plan(
    available_agents: List[Dict[str, Any]],
    user_input: str,
) -> Dict[str, Any]:
    steps: List[Dict[str, str]] = []
    if len(available_agents) == 1:
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
                }
            )

    fallback = {
        "steps": steps,
        "notes": "Fallback collaboration plan derived from available agents.",
    }
    log_event("planner", "collaboration_plan_fallback", {"collaboration_plan": fallback})
    return fallback


def _derive_collaboration_plan(
    main_agent: LlmAgent,
    available_agents: List[Dict[str, Any]],
    user_input: str,
    conversation_history: str,
    raw_plan: str,
) -> Dict[str, Any]:
    if not available_agents:
        log_event("planner", "collaboration_plan_skipped", {"reason": "no_available_agents"})
        return {"steps": [], "notes": ""}

    agents_desc = _format_available_agents_for_prompt(
        available_agents,
        empty_text="No sub-agents are currently configured.",
    )
    conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=6, max_chars=900)
    plan_summary = _compact_text(raw_plan or "(none)", max_chars=1400)

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
        "    }\n"
        "  ],\n"
        '  "notes": "short note"\n'
        "}\n\n"
        "Guidelines:\n"
        "- Use only names from Available agents.\n"
        "- Create practical future steps which are necessary to achieve goal.\n"
        "- Each step should be useful on its own and easy for the assigned agent to execute.\n"
        "- Do not create workflow steps for loading agent-private skills, tools, or session memory.\n"
        "- Respect explicit user constraints.\n"
        "- Choose agents from role, capability, ownership, and description metadata.\n"
        "- Prefer specialists for evidence gathering and domain work; use the coordinator for orchestration, synthesis, or delivery it owns.\n\n"
        f"Recent conversation context summary:\n{conversation_summary}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Current plan text summary:\n{plan_summary}\n\n"
        f"Available agents:\n{agents_desc}\n"
    )

    raw = _run_agent_prompt(main_agent, collaboration_prompt, task="collaboration_plan")
    parsed = _extract_json_object(raw)
    normalized = _normalize_collaboration_plan(parsed, available_agents)
    if normalized.get("steps"):
        log_event("planner", "collaboration_plan_derived", {"collaboration_plan": normalized})
        return normalized

    return _fallback_collaboration_plan(
        available_agents=available_agents,
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

    agents_desc = _format_available_agents_for_prompt(
        available_agents,
        empty_text="No sub-agents are currently configured.",
    )
    conversation_summary = _summarize_conversation_history(conversation_history, max_turn_lines=6, max_chars=900)

    planning_prompt = (
        "You are the planner of a multi-agent system.\n"
        "Read the user request and outline a short practical plan.\n\n"
        "Requirements:\n"
        "1) Summarize the user goal in one sentence.\n"
        "2) Write a 'Plan:' section with short numbered steps.\n"
        "3) Assign each step to the agent that owns the work or best matches the specialization.\n"
        "4) If no sub-agent is needed, say so briefly.\n"
        "5) Choose from role, capability, and ownership metadata below, not by raw tool names.\n"
        "6) Prefer specialists for research or analysis work, and use the coordinator for orchestration, synthesis, or delivery it owns.\n"
        "7) This phase is planning only. Do not execute tools or perform the plan.\n"
        "8) Do not create workflow steps for loading agent-private skills, tools, or session memory.\n\n"
        f"Recent conversation context summary:\n{conversation_summary}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Available agents:\n{agents_desc}\n"
    )

    raw_plan = _run_agent_prompt(main_agent, planning_prompt, task="planning")
    collaboration_plan = _derive_collaboration_plan(
        main_agent=main_agent,
        available_agents=available_agents,
        user_input=user_input,
        conversation_history=conversation_history,
        raw_plan=raw_plan,
    )
    result = {
        "raw_plan": raw_plan,
        "meta": {
            "user_input": user_input,
            "num_available_agents": len(available_agents),
            "collaboration_plan": collaboration_plan,
        },
    }
    log_event("planner", "planning_completed", {"result": result})
    return result


__all__ = ["plan_with_main_agent"]
