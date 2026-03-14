from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from google.adk.agents import LlmAgent

from agentic_sample_ad.agent_session_memory_runtime import build_load_session_memory_tool
from agentic_sample_ad.event_manager import execute_plan_detailed, resume_paused_workflow_detailed
from agentic_sample_ad.main_agent.slack_mcp_tool import slack_post_message
from agentic_sample_ad.main_agent.workflow_memory_tool import read_workflow_memory
from agentic_sample_ad.model_settings import resolve_agent_model
from agentic_sample_ad.planner import plan_with_main_agent
from agentic_sample_ad.skill_runtime import build_skill_toolset

from .card_registry import load_sub_agent_cards
from .session_memory import get_or_create_session
from .system_logger import (
    enable_a2a_package_logging,
    initialize_main_logging,
    log_event,
    log_exception,
    log_main_event,
    log_main_exception,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_MEMORY_PATH = PROJECT_ROOT / "main_agent" / "memory" / "session_memory.json"
MAIN_AGENT_CORE_CAPABILITIES = [
    "coordination",
    "workflow_replanning",
    "user_clarification_routing",
    "workflow_memory_read",
    "comm.slack.post",
]


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        log_main_event("env_file_missing", {"path": str(env_path)})
        return

    loaded_keys: List[str] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded_keys.append(key)
    log_main_event(
        "env_loaded",
        {"path": str(env_path), "loaded_count": len(loaded_keys), "loaded_keys": loaded_keys},
    )


def create_main_agent() -> LlmAgent:
    model_name = resolve_agent_model("MainAgent")
    load_session_memory = build_load_session_memory_tool(
        agent_name="MainAgent",
        memory_path=SESSION_MEMORY_PATH,
        log_event_fn=log_event,
        log_exception_fn=log_exception,
    )
    tools: List[Any] = [slack_post_message, read_workflow_memory, load_session_memory]
    skill_toolset = build_skill_toolset(PROJECT_ROOT / "main_agent" / "skills")
    if skill_toolset is not None:
        tools.append(skill_toolset)
    agent = LlmAgent(
        name="MainAgent",
        model=model_name,
        instruction=(
            "You are the coordinator of a multi-agent system. "
            "Understand the user request, decide what can be handled directly, and delegate specialist work when it improves the result. "
            "Own orchestration, replanning, user clarification, and final delivery actions that belong to the coordinator. "
            "Your direct tools return JSON with `ok`, `tool_name`, `summary`, `content_type`, `items`, `data`, `errors`, and `metadata`; use that structure internally and do not confuse raw tool output with final user-facing answers."
        ),
        tools=tools,
    )
    log_main_event("main_agent_created", {"name": "MainAgent", "model": model_name})
    return agent


def _doc_preview(value: str, max_len: int = 220) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "..."


def _extract_tool_metadata(agent_obj: Any) -> List[Dict[str, str]]:
    tools = getattr(agent_obj, "tools", []) or []
    extracted: List[Dict[str, str]] = []
    seen_names: set[str] = set()
    for tool in tools:
        name = str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        desc = str(getattr(tool, "description", "") or getattr(tool, "__doc__", "") or "").strip()
        extracted.append(
            {
                "name": name,
                "description": _doc_preview(desc, max_len=180) if desc else "",
            }
        )
    return extracted


def _derive_capabilities(
    *,
    existing: List[str] | None = None,
) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()

    for token in existing or []:
        value = str(token).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)

    return merged[:40]


def _normalize_sub_agent_cards_for_remote_execution(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for raw in cards:
        item = dict(raw)
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        item["name"] = name
        if name.lower() == "mainagent":
            continue

        if not str(item.get("role", "")).strip():
            item["role"] = "worker"
        item["type"] = "a2a"

        # Main runtime must delegate sub-agents through A2A only.
        item.pop("agent_obj", None)
        item.pop("module", None)
        item.pop("attr", None)
        item.pop("tools", None)

        existing_caps = [str(cap).strip() for cap in item.get("capabilities", []) if str(cap).strip()]
        item["capabilities"] = _derive_capabilities(
            existing=existing_caps,
        )
        normalized.append(item)

        log_main_event(
            "sub_agent_card_registered_for_a2a",
            {
                "name": name,
                "type": "a2a",
                "base_url": str(item.get("base_url", "")).strip(),
                "source_card_path": str(item.get("source_card_path", "")).strip(),
            },
        )
    return normalized


def _build_main_agent_registry_entry(main_agent: LlmAgent) -> Dict[str, Any]:
    name = str(getattr(main_agent, "name", "")).strip() or "MainAgent"
    instruction = str(getattr(main_agent, "instruction", "") or "").strip()
    tools = _extract_tool_metadata(main_agent)
    return {
        "name": name,
        "type": "local",
        "role": "coordinator",
        "description": "Coordinator for planning, replanning, and cross-agent handoff execution.",
        "capabilities": _derive_capabilities(
            existing=list(MAIN_AGENT_CORE_CAPABILITIES),
        ),
        "ownership": "Own planning, coordination, replanning, user clarification, and coordinator-controlled delivery actions.",
        "tools": tools,
        "instruction_preview": _doc_preview(instruction, max_len=320) if instruction else "",
        # Runtime object reference for local execution in event_manager.
        "agent_obj": main_agent,
    }


def _build_unified_agent_registry(
    *,
    main_agent: LlmAgent,
    sub_agents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registry: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    main_entry = _build_main_agent_registry_entry(main_agent)
    main_name_key = str(main_entry.get("name", "")).strip().lower()
    if main_name_key:
        seen_names.add(main_name_key)
        registry.append(main_entry)

    for item in sub_agents:
        entry = dict(item)
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        if not str(entry.get("role", "")).strip():
            entry["role"] = "worker"
        registry.append(entry)
    return registry


def run_main_agent(user_input: str, session_id: str = "default") -> str:
    initialize_main_logging()
    _load_env_file()
    enable_a2a_package_logging(os.getenv("A2A_PACKAGE_LOG_LEVEL", "INFO"))

    log_main_event(
        "run_started",
        {"session_id": session_id, "user_input": user_input},
        direction="inbound",
    )
    try:
        session = get_or_create_session(session_id)
        session.add_user_turn(user_input)

        main_agent = create_main_agent()
        sub_agents = _normalize_sub_agent_cards_for_remote_execution(load_sub_agent_cards())
        available_agents = _build_unified_agent_registry(
            main_agent=main_agent,
            sub_agents=sub_agents,
        )
        log_main_event(
            "available_agents_finalized",
            {
                "count": len(available_agents),
                "names": [str(item.get("name", "")) for item in available_agents],
                "roles": {
                    str(item.get("name", "")): str(item.get("role", "")).strip() or "worker"
                    for item in available_agents
                },
            },
        )

        planning_context: Dict[str, Any] = {
            "user_input": user_input,
            "conversation_history": session.history_as_text(),
            "session_id": session_id,
        }
        paused_workflow = session.get_paused_workflow()
        if paused_workflow:
            log_main_event(
                "paused_workflow_resume_started",
                {
                    "session_id": session_id,
                    "workflow_id": str(paused_workflow.get("workflow_id", "")).strip(),
                },
            )
            plan: Dict[str, Any] = {
                "raw_plan": str(paused_workflow.get("raw_plan", "")),
                "meta": {
                    "user_input": str(paused_workflow.get("original_user_input", "")).strip() or user_input,
                    "collaboration_plan": paused_workflow.get("collaboration_plan", {}),
                },
            }
            execution = resume_paused_workflow_detailed(
                paused_workflow=paused_workflow,
                clarification_response=user_input,
                main_agent=main_agent,
                available_agents=available_agents,
                context=planning_context,
            )
        else:
            plan = plan_with_main_agent(
                main_agent=main_agent,
                available_agents=available_agents,
                context=planning_context,
            )
            execution = execute_plan_detailed(
                plan=plan,
                main_agent=main_agent,
                available_agents=available_agents,
                context=planning_context,
            )
        result_text = str(execution.get("output_text", ""))

        paused_snapshot = execution.get("paused_workflow")
        if isinstance(paused_snapshot, dict) and paused_snapshot:
            session.set_paused_workflow(paused_snapshot)
            log_main_event(
                "paused_workflow_saved",
                {
                    "session_id": session_id,
                    "workflow_id": str(paused_snapshot.get("workflow_id", "")).strip(),
                    "pause_request": str(paused_snapshot.get("pause_request", "")).strip(),
                },
            )
        else:
            session.clear_paused_workflow()
            log_main_event("paused_workflow_cleared", {"session_id": session_id})

        session.add_workflow_context(
            {
                "raw_plan": str(plan.get("raw_plan", "")),
                "collaboration_plan": plan.get("meta", {}).get("collaboration_plan", {}),
                "execution_output": result_text,
                "workflow_id": str(execution.get("workflow_id", "")).strip(),
                "paused_workflow_active": bool(paused_snapshot),
            }
        )
        session.add_assistant_turn(result_text)
        log_main_event(
            "run_completed",
            {"session_id": session_id, "result": result_text},
            direction="outbound",
        )
        return result_text
    except Exception as e:
        log_main_exception(
            "run_failed",
            e,
            {"session_id": session_id, "user_input": user_input},
        )
        raise


__all__ = ["create_main_agent", "run_main_agent"]


