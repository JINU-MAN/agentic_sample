from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent

from agentic_sample_ad.agent_session_memory_runtime import build_load_session_memory_tool
from agentic_sample_ad.model_settings import resolve_agent_model
from agentic_sample_ad.skill_runtime import build_skill_toolset

from .tool import scrape_sns_with_mcp


load_session_memory = build_load_session_memory_tool(
    agent_name="SocialMediaAnalyst",
    memory_path=Path(__file__).resolve().parent / "memory" / "session_memory.json",
)
_TOOLS = [scrape_sns_with_mcp, load_session_memory]
_SKILL_TOOLSET = build_skill_toolset(Path(__file__).resolve().parent / "skills")
if _SKILL_TOOLSET is not None:
    _TOOLS.append(_SKILL_TOOLSET)


sns_agent = LlmAgent(
    name="SocialMediaAnalyst",
    model=resolve_agent_model("SocialMediaAnalyst"),
    instruction=(
        "You are a social media research specialist. "
        "Own SNS evidence gathering and social-signal summarization. "
        "Decide your own search strategy from user intent and return the strongest signals with enough source detail for follow-up. "
        "If prior workflow context is missing, request the missing workflow memory from MainAgent through structured `needs` instead of asking the user to repeat internal step data. "
        "If you discover reusable posts, accounts, links, or signals for downstream agents, prefer a compact structured handoff with `summary`, `artifacts`, and `needs`. "
        "Summarize the strongest signals, explain why they matter, and include enough source detail for follow-up. "
        "Do not invent facts. If evidence is weak or missing, clearly say so and state what is needed next."
    ),
    tools=_TOOLS,
)

agent = sns_agent


__all__ = [
    "agent",
    "sns_agent",
    "load_session_memory",
    "scrape_sns_with_mcp",
]
