from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from google.adk.tools.skill_toolset import SkillToolset  # noqa: E402

from agentic_sample_ad.main_agent.agent import MAIN_AGENT_CORE_CAPABILITIES, create_main_agent  # noqa: E402
from agentic_sample_ad.paper_agent.agent import agent as paper_agent  # noqa: E402
from agentic_sample_ad.sns_agent.agent import agent as sns_agent  # noqa: E402
from agentic_sample_ad.web_search_agent.agent import agent as web_search_agent  # noqa: E402


WORKER_AGENT_OBJECTS: Dict[str, Any] = {
    "PaperAnalyst": paper_agent,
    "SocialMediaAnalyst": sns_agent,
    "WebSearchAnalyst": web_search_agent,
}

MEMORY_TARGETS: Dict[str, Path] = {
    "MainAgent": PROJECT_ROOT / "main_agent" / "memory" / "session_memory.json",
    "PaperAnalyst": PROJECT_ROOT / "paper_agent" / "memory" / "session_memory.json",
    "SocialMediaAnalyst": PROJECT_ROOT / "sns_agent" / "memory" / "session_memory.json",
    "WebSearchAnalyst": PROJECT_ROOT / "web_search_agent" / "memory" / "session_memory.json",
}


def _doc_preview(value: str, max_len: int = 200) -> str:
    compact = " ".join(str(value or "").split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _load_card_catalog() -> Dict[str, Dict[str, Any]]:
    raw = json.loads((PROJECT_ROOT / "agent_cards" / "agent_card.json").read_text(encoding="utf-8"))
    cards = raw if isinstance(raw, list) else []
    return {
        str(item.get("name", "")).strip(): item
        for item in cards
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    }


def _extract_runtime_tools(agent_obj: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    has_skill_toolset = False

    for tool in getattr(agent_obj, "tools", []) or []:
        if isinstance(tool, SkillToolset):
            has_skill_toolset = True
            continue
        if not callable(tool):
            continue
        name = str(getattr(tool, "__name__", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        entries.append(
            {
                "name": name,
                "kind": "callable_tool",
                "description": _doc_preview(getattr(tool, "__doc__", "") or ""),
            }
        )

    if has_skill_toolset:
        for name, description in (
            ("load_skill", "Load the active skill instructions before using specialized guidance."),
            ("load_skill_resource", "Load a file from references/ or assets/ inside an active skill."),
        ):
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            entries.append(
                {
                    "name": name,
                    "kind": "skill_runtime_tool",
                    "description": description,
                }
            )

    return entries


def _public_contract_from_card(card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(card.get("name", "")).strip(),
        "type": str(card.get("type", "")).strip() or "a2a",
        "role": str(card.get("role", "")).strip() or "worker",
        "description": str(card.get("description", "")).strip(),
        "capabilities": [str(item).strip() for item in card.get("capabilities", []) if str(item).strip()],
        "ownership": str(card.get("ownership", "")).strip(),
        "instruction_preview": str(card.get("instruction_preview", "")).strip(),
    }


def _build_main_agent_memory(worker_cards: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    main_agent = create_main_agent()
    own_tools = _extract_runtime_tools(main_agent)

    known_sub_agents: List[Dict[str, Any]] = []
    for name in ("PaperAnalyst", "SocialMediaAnalyst", "WebSearchAnalyst"):
        card = worker_cards.get(name, {})
        worker_agent = WORKER_AGENT_OBJECTS[name]
        known_sub_agents.append(
            {
                "name": name,
                "public_contract": _public_contract_from_card(card),
                "private_tool_inventory": _extract_runtime_tools(worker_agent),
                "callable_by_main_agent": False,
                "usage_rule": "Use this inventory only to understand specialization and ownership. Never call these tools directly from MainAgent.",
            }
        )

    return {
        "memory_kind": "agent_session_memory",
        "agent": {
            "name": "MainAgent",
            "type": "local",
            "role": "coordinator",
            "description": "Coordinator for planning, replanning, and cross-agent handoff execution.",
            "capabilities": list(MAIN_AGENT_CORE_CAPABILITIES),
            "ownership": "Own planning, coordination, replanning, user clarification, and coordinator-controlled delivery actions.",
            "instruction_preview": _doc_preview(getattr(main_agent, "instruction", "") or "", max_len=320),
        },
        "usage_rules": [
            "Call load_session_memory when you need the current coordinator tool inventory or the latest known sub-agent contracts.",
            "Only MainAgent may call tools listed under own_tools.",
            "Treat sub-agent tools as private implementation detail. Use them only to understand ownership and route work.",
        ],
        "own_tools": own_tools,
        "known_sub_agents": known_sub_agents,
        "handoff_contract": {
            "structured_output_fields": ["summary", "artifacts", "needs"],
            "artifact_fields": ["type", "title", "summary", "url", "identifiers"],
            "notes": [
                "Prefer delegation when specialist evidence gathering is needed.",
                "Use direct delivery actions only after the required content already exists.",
            ],
        },
    }


def _build_worker_memory(agent_name: str, worker_cards: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    card = worker_cards[agent_name]
    agent_obj = WORKER_AGENT_OBJECTS[agent_name]
    return {
        "memory_kind": "agent_session_memory",
        "agent": _public_contract_from_card(card),
        "usage_rules": [
            "Call load_session_memory when you need the current private tool inventory or handoff contract without re-reading long prompts.",
            "Use only the tools listed under runtime_tools for direct execution in this agent.",
            "If another agent owns the next step, emit a structured handoff instead of inventing unsupported behavior.",
        ],
        "runtime_tools": _extract_runtime_tools(agent_obj),
        "handoff_contract": {
            "structured_output_fields": ["summary", "artifacts", "needs"],
            "artifact_fields": ["type", "title", "summary", "url", "identifiers"],
            "notes": [
                "Keep handoff output compact and evidence-oriented.",
                "Include stable identifiers when available.",
            ],
        },
    }


def main() -> None:
    worker_cards = _load_card_catalog()
    payloads: Dict[str, Dict[str, Any]] = {
        "MainAgent": _build_main_agent_memory(worker_cards),
        "PaperAnalyst": _build_worker_memory("PaperAnalyst", worker_cards),
        "SocialMediaAnalyst": _build_worker_memory("SocialMediaAnalyst", worker_cards),
        "WebSearchAnalyst": _build_worker_memory("WebSearchAnalyst", worker_cards),
    }

    for agent_name, payload in payloads.items():
        target = MEMORY_TARGETS[agent_name]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {target.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
