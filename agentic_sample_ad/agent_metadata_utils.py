from __future__ import annotations

import re
from typing import Any, Dict, List


CAPABILITY_LABELS: Dict[str, str] = {
    "coordination": "coordination and orchestration",
    "workflow_replanning": "workflow replanning",
    "user_clarification_routing": "user clarification and routing",
    "workflow_memory_read": "shared workflow memory lookup",
    "comm.slack.post": "Slack channel delivery",
    "paper_search": "paper search in the local PDF corpus",
    "paper_memory": "workflow-scoped paper memory analysis",
    "external_paper_fetch": "external paper reference lookup",
    "paper_evidence": "paper evidence synthesis",
    "sns_search": "SNS post collection",
    "sns_summary": "social signal summarization",
    "social_signal_analysis": "social signal analysis",
    "web_search": "web research and current-information lookup",
    "web_evidence_summary": "citation-grounded web evidence synthesis",
    "web_research": "web research workflow ownership",
}


def humanize_capability(token: str) -> str:
    lowered = str(token).strip().lower()
    if lowered in CAPABILITY_LABELS:
        return CAPABILITY_LABELS[lowered]
    return " ".join(part for part in re.split(r"[._]+", str(token).strip()) if part)


def tool_name_keys(agent_meta: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for tool in agent_meta.get("tools", []):
        if isinstance(tool, dict):
            token = str(tool.get("name", "")).strip().lower()
        else:
            token = str(tool).strip().lower()
        if token:
            keys.add(token)
    return keys


def agent_capability_keys(agent_meta: Dict[str, Any]) -> List[str]:
    seen: set[str] = set()
    keys: List[str] = []
    for raw in agent_meta.get("capabilities", []):
        token = str(raw).strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        keys.append(token)
    return keys


def agent_capability_summary_text(agent_meta: Dict[str, Any]) -> str:
    desc = str(agent_meta.get("description", "")).strip()
    tool_keys = tool_name_keys(agent_meta)
    name_key = re.sub(r"[^a-z0-9]+", "", str(agent_meta.get("name", "")).lower())
    seen: set[str] = set()
    labels: List[str] = []
    for raw in agent_meta.get("capabilities", []):
        token = str(raw).strip()
        lowered = token.lower()
        compact = re.sub(r"[^a-z0-9]+", "", lowered)
        if not token or lowered in tool_keys or lowered.endswith("_with_mcp") or compact == name_key:
            continue
        label = humanize_capability(token).strip()
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return "; ".join(labels) if labels else (desc or "(not provided)")


def agent_ownership_text(agent_meta: Dict[str, Any]) -> str:
    explicit = str(agent_meta.get("ownership", "")).strip()
    if explicit:
        return explicit
    role = str(agent_meta.get("role", "")).strip().lower()
    if role == "coordinator":
        return (
            "Own coordinator-only work such as orchestration, replanning, direct user handling, "
            "and delivery actions that belong to this agent."
        )
    return (
        "Use this agent when the request benefits from its specialization. "
        "The coordinator can combine this agent's output or handle final delivery."
    )


def format_available_agents_for_prompt(
    available_agents: List[Dict[str, Any]],
    *,
    empty_text: str = "(none)",
) -> str:
    if not available_agents:
        return empty_text

    lines: List[str] = []
    for agent in available_agents:
        name = str(agent.get("name", "UnknownAgent")).strip() or "UnknownAgent"
        role = str(agent.get("role", "")).strip() or "worker"
        desc = str(agent.get("description", "")).strip()
        capability_summary = agent_capability_summary_text(agent)
        ownership = agent_ownership_text(agent)
        lines.append(
            f"- name: {name}\n"
            f"  role: {role}\n"
            f"  description: {desc}\n"
            f"  capability_summary: {capability_summary}\n"
            f"  ownership: {ownership}"
        )
    return "\n".join(lines) or empty_text
