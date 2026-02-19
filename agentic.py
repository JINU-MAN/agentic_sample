from __future__ import annotations

import importlib
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from google.adk.agents import LlmAgent

from planner import plan_with_main_agent
from event_manager import execute_plan
from session_store import get_or_create_session
from system_logger import (
    enable_a2a_package_logging,
    initialize_process_logging,
    log_event,
    log_exception,
)


BASE_DIR = Path(__file__).parent


def _load_env_file() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        log_event("agentic", "env_file_missing", {"path": str(env_path)})
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

    log_event(
        "agentic",
        "env_loaded",
        {
            "path": str(env_path),
            "loaded_count": len(loaded_keys),
            "loaded_keys": loaded_keys,
        },
    )


def create_main_agent() -> LlmAgent:
    agent = LlmAgent(
        name="MainAgent",
        model="gemini-2.0-flash",
        instruction=(
            "당신은 멀티 에이전트 시스템의 메인 코디네이터입니다. "
            "사용자 요청을 이해하고 필요한 경우 다른 에이전트 활용 계획을 세웁니다."
        ),
        tools=[],
    )
    log_event("main_agent", "created", {"name": "MainAgent", "model": "gemini-2.0-flash"})
    return agent


def _load_available_agents_from_cards() -> List[Dict[str, Any]]:
    cards_dir = BASE_DIR / "agent_cards"
    default_cards_path = cards_dir / "agent_card.json"
    card_files = sorted(cards_dir.glob("*.json")) if cards_dir.exists() else []
    if not card_files and default_cards_path.exists():
        card_files = [default_cards_path]

    if not card_files:
        log_event("agentic", "agent_card_missing", {"path": str(default_cards_path)})
        return []

    loaded: List[Dict[str, Any]] = []
    for card_path in card_files:
        try:
            with card_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                loaded.extend(item for item in data if isinstance(item, dict))
                log_event("agentic", "agents_loaded", {"count": len(data), "path": str(card_path)})
                continue
            if isinstance(data, dict):
                loaded.append(data)
                log_event("agentic", "agents_loaded", {"count": 1, "path": str(card_path)})
                continue
            log_event(
                "agentic",
                "agent_card_invalid_shape",
                {"path": str(card_path), "type": str(type(data))},
                level="ERROR",
            )
        except Exception as e:
            log_exception("agentic", "agent_card_parse_failed", e, {"path": str(card_path)})
    return loaded


def _doc_preview(value: str, max_len: int = 220) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "..."


def _extract_tool_metadata(agent_obj: Any) -> List[Dict[str, str]]:
    tools = getattr(agent_obj, "tools", []) or []
    extracted: List[Dict[str, str]] = []
    seen_names: set[str] = set()
    for tool in tools:
        try:
            name = str(getattr(tool, "name", "")).strip()
            if not name:
                name = str(getattr(tool, "__name__", "")).strip()
            if not name:
                name = str(tool).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)

            doc = str(getattr(tool, "description", "") or "").strip()
            if not doc:
                doc = str(getattr(tool, "__doc__", "") or "").strip()
            extracted.append(
                {
                    "name": name,
                    "description": _doc_preview(doc) if doc else "",
                }
            )
        except Exception:
            continue
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
        key = value.lower()
        if not value or key in seen:
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
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        merged.append(value)

    return merged[:40]


def _build_runtime_local_agent_metadata(
    module_name: str,
    attr_name: str,
    agent_obj: Any,
) -> Dict[str, Any]:
    display_name = str(getattr(agent_obj, "name", "") or attr_name).strip() or attr_name
    instruction = str(getattr(agent_obj, "instruction", "") or "").strip()
    tools = _extract_tool_metadata(agent_obj)
    tool_names = [str(item.get("name", "")).strip() for item in tools if str(item.get("name", "")).strip()]
    description = _doc_preview(instruction, max_len=180) if instruction else (
        f"Discovered local agent from {module_name}.{attr_name}"
    )

    return {
        "name": display_name,
        "type": "local",
        "module": module_name,
        "attr": attr_name,
        "description": description,
        "tools": tools,
        "capabilities": _derive_capabilities(existing=[], agent_name=display_name, tool_names=tool_names),
        "instruction_preview": _doc_preview(instruction, max_len=300) if instruction else "",
    }


def _discover_local_agents_from_runtime() -> List[Dict[str, Any]]:
    agent_dir = BASE_DIR / "agent"
    if not agent_dir.exists():
        return []

    discovered: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, str]] = set()
    seen_names: set[str] = set()
    for path in sorted(agent_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"agent.{path.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            log_exception("agentic", "agent_runtime_discovery_import_failed", e, {"module": module_name})
            continue

        for attr_name, candidate in vars(module).items():
            if not isinstance(candidate, LlmAgent):
                continue
            pair = (module_name, attr_name)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            meta = _build_runtime_local_agent_metadata(module_name, attr_name, candidate)
            name_key = str(meta.get("name", "")).strip().lower()
            if name_key and name_key in seen_names:
                continue
            if name_key:
                seen_names.add(name_key)
            discovered.append(meta)

    log_event(
        "agentic",
        "agent_runtime_discovery_completed",
        {
            "discovered_count": len(discovered),
            "agent_names": [str(item.get("name", "")) for item in discovered],
        },
    )
    return discovered


def _merge_agent_metadata_with_runtime_discovery(
    card_agents: List[Dict[str, Any]],
    discovered_local_agents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = [dict(item) for item in card_agents]

    index_by_name: Dict[str, int] = {}
    index_by_module_attr: Dict[Tuple[str, str], int] = {}
    for idx, item in enumerate(merged):
        name = str(item.get("name", "")).strip().lower()
        if name and name not in index_by_name:
            index_by_name[name] = idx
        module_key = str(item.get("module", "")).strip()
        attr_key = str(item.get("attr", "")).strip()
        if module_key and attr_key and (module_key, attr_key) not in index_by_module_attr:
            index_by_module_attr[(module_key, attr_key)] = idx

    for runtime_item in discovered_local_agents:
        name_key = str(runtime_item.get("name", "")).strip().lower()
        module_key = str(runtime_item.get("module", "")).strip()
        attr_key = str(runtime_item.get("attr", "")).strip()
        match_idx = None
        if name_key in index_by_name:
            match_idx = index_by_name[name_key]
        elif (module_key, attr_key) in index_by_module_attr:
            match_idx = index_by_module_attr[(module_key, attr_key)]

        if match_idx is None:
            merged.append(dict(runtime_item))
            new_idx = len(merged) - 1
            if name_key:
                index_by_name[name_key] = new_idx
            if module_key and attr_key:
                index_by_module_attr[(module_key, attr_key)] = new_idx
            continue

        current = dict(merged[match_idx])
        if not str(current.get("type", "")).strip():
            current["type"] = "local"
        if not str(current.get("module", "")).strip():
            current["module"] = module_key
        if not str(current.get("attr", "")).strip():
            current["attr"] = attr_key
        if not str(current.get("description", "")).strip():
            current["description"] = str(runtime_item.get("description", "")).strip()
        if not current.get("tools"):
            current["tools"] = runtime_item.get("tools", [])
        if not str(current.get("instruction_preview", "")).strip():
            current["instruction_preview"] = str(runtime_item.get("instruction_preview", "")).strip()

        existing_caps = [str(item).strip() for item in current.get("capabilities", []) if str(item).strip()]
        runtime_tool_names: List[str] = []
        for tool in runtime_item.get("tools", []):
            if isinstance(tool, dict):
                name = str(tool.get("name", "")).strip()
                if name:
                    runtime_tool_names.append(name)
            elif isinstance(tool, str):
                token = tool.strip()
                if token:
                    runtime_tool_names.append(token)
        current["capabilities"] = _derive_capabilities(
            existing=existing_caps,
            agent_name=str(current.get("name", "")),
            tool_names=runtime_tool_names,
        )
        merged[match_idx] = current

    return merged


def _enrich_available_agents_with_runtime_metadata(
    available_agents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for agent_meta in available_agents:
        item = dict(agent_meta)
        agent_type = str(item.get("type", "")).strip().lower() or "local"
        if agent_type not in {"local", "a2a"}:
            enriched.append(item)
            continue

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
            runtime_meta = _build_runtime_local_agent_metadata(module_name, attr_name, agent_obj)

            item["name"] = str(item.get("name", "")).strip() or str(runtime_meta.get("name", "")).strip()
            if not str(item.get("description", "")).strip():
                item["description"] = str(runtime_meta.get("description", "")).strip()

            runtime_tools = runtime_meta.get("tools", [])
            if runtime_tools:
                item["tools"] = runtime_tools
            runtime_tool_names = [
                str(tool.get("name", "")).strip()
                for tool in runtime_tools
                if isinstance(tool, dict) and str(tool.get("name", "")).strip()
            ]
            existing_caps = [str(c).strip() for c in item.get("capabilities", []) if str(c).strip()]
            item["capabilities"] = _derive_capabilities(
                existing=existing_caps,
                agent_name=str(item.get("name", "")),
                tool_names=runtime_tool_names,
            )

            runtime_instruction_preview = str(runtime_meta.get("instruction_preview", "")).strip()
            if runtime_instruction_preview:
                item["instruction_preview"] = runtime_instruction_preview
            log_event(
                "agentic",
                "agent_runtime_metadata_enriched",
                {
                    "name": str(item.get("name", "")),
                    "type": agent_type,
                    "module": module_name,
                    "attr": attr_name,
                    "tool_names": runtime_tool_names,
                },
            )
        except Exception as e:
            log_exception(
                "agentic",
                "agent_runtime_metadata_enrich_failed",
                e,
                {"name": str(item.get("name", "")), "module": module_name, "attr": attr_name},
            )
        enriched.append(item)
    return enriched


def run_main_agent(user_input: str, session_id: str = "default") -> str:
    initialize_process_logging()
    _load_env_file()
    enable_a2a_package_logging(os.getenv("A2A_PACKAGE_LOG_LEVEL", "INFO"))
    log_event(
        "main_agent",
        "run_started",
        {"session_id": session_id, "user_input": user_input},
        direction="inbound",
    )
    try:
        session = get_or_create_session(session_id)
        session.add_user_turn(user_input)

        main_agent = create_main_agent()
        configured_agents = _load_available_agents_from_cards()
        discovered_local_agents = _discover_local_agents_from_runtime()
        available_agents = _merge_agent_metadata_with_runtime_discovery(
            card_agents=configured_agents,
            discovered_local_agents=discovered_local_agents,
        )
        available_agents = _enrich_available_agents_with_runtime_metadata(available_agents)
        log_event(
            "agentic",
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
        log_event("main_agent", "plan_generated", {"session_id": session_id, "plan": plan})

        result = execute_plan(
            plan=plan,
            main_agent=main_agent,
            available_agents=available_agents,
            context=planning_context,
        )

        result_text = str(result)
        log_event(
            "main_agent",
            "execution_completed",
            {"session_id": session_id, "result": result_text},
            direction="outbound",
        )
        session.add_assistant_turn(result_text)
        return result_text
    except Exception as e:
        log_exception(
            "main_agent",
            "run_failed",
            e,
            {"session_id": session_id, "user_input": user_input},
        )
        raise


__all__ = ["create_main_agent", "run_main_agent"]
