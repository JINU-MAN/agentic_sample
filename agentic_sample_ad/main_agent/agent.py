from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

from google.adk.agents import LlmAgent

from agentic_sample_ad.planner import plan_with_main_agent

from .card_registry import load_sub_agent_cards
from .event_manager import execute_plan
from .session_memory import get_or_create_session
from .system_logger import (
    enable_a2a_package_logging,
    initialize_main_logging,
    log_main_event,
    log_main_exception,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    agent = LlmAgent(
        name="MainAgent",
        model="gemini-2.0-flash",
        instruction=(
            "You are the coordinator of a multi-agent system. "
            "Understand the user request, create practical execution steps, "
            "and orchestrate specialist agents for best quality."
        ),
        tools=[],
    )
    log_main_event("main_agent_created", {"name": "MainAgent", "model": "gemini-2.0-flash"})
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
    agent_name: str = "",
    tool_names: List[str] | None = None,
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

    if agent_name:
        normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in agent_name)
        normalized = "_".join(part for part in normalized.split("_") if part)
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)

    for name in tool_names or []:
        value = str(name).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)

    return merged[:40]


def _enrich_cards_with_runtime_metadata(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for raw in cards:
        item = dict(raw)
        module_name = str(item.get("module", "")).strip()
        attr_name = str(item.get("attr", "")).strip()
        if not module_name or not attr_name:
            enriched.append(item)
            continue

        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, attr_name):
                enriched.append(item)
                continue
            agent_obj = getattr(module, attr_name)
            instruction = str(getattr(agent_obj, "instruction", "") or "").strip()
            tools = _extract_tool_metadata(agent_obj)
            tool_names = [
                str(tool.get("name", "")).strip()
                for tool in tools
                if isinstance(tool, Mapping) and str(tool.get("name", "")).strip()
            ]

            if not str(item.get("name", "")).strip():
                item["name"] = str(getattr(agent_obj, "name", "")).strip() or attr_name
            if not str(item.get("description", "")).strip():
                item["description"] = _doc_preview(instruction, max_len=180)
            if tools:
                item["tools"] = tools

            existing_caps = [str(cap).strip() for cap in item.get("capabilities", []) if str(cap).strip()]
            item["capabilities"] = _derive_capabilities(
                existing=existing_caps,
                agent_name=str(item.get("name", "")),
                tool_names=tool_names,
            )
            if instruction:
                item["instruction_preview"] = _doc_preview(instruction, max_len=320)

            log_main_event(
                "agent_runtime_metadata_enriched",
                {
                    "name": str(item.get("name", "")),
                    "module": module_name,
                    "attr": attr_name,
                    "tool_names": tool_names,
                },
            )
        except Exception as e:
            log_main_exception(
                "agent_runtime_metadata_enrich_failed",
                e,
                {"module": module_name, "attr": attr_name},
            )
        enriched.append(item)
    return enriched


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
        available_agents = _enrich_cards_with_runtime_metadata(load_sub_agent_cards())
        log_main_event(
            "available_agents_finalized",
            {
                "count": len(available_agents),
                "names": [str(item.get("name", "")) for item in available_agents],
            },
        )

        planning_context: Dict[str, Any] = {
            "user_input": user_input,
            "conversation_history": session.history_as_text(),
            "session_id": session_id,
        }
        plan = plan_with_main_agent(
            main_agent=main_agent,
            available_agents=available_agents,
            context=planning_context,
        )
        result = execute_plan(
            plan=plan,
            main_agent=main_agent,
            available_agents=available_agents,
            context=planning_context,
        )
        result_text = str(result)

        session.add_workflow_context(
            {
                "raw_plan": str(plan.get("raw_plan", "")),
                "routing_hint": plan.get("meta", {}).get("routing_hint", {}),
                "collaboration_plan": plan.get("meta", {}).get("collaboration_plan", {}),
                "execution_output": result_text,
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


