from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent

from agentic_sample_ad.agent_session_memory_runtime import build_load_session_memory_tool
from agentic_sample_ad.model_settings import resolve_agent_model
from agentic_sample_ad.skill_runtime import build_skill_toolset

from .tool import search_web_with_mcp


load_session_memory = build_load_session_memory_tool(
    agent_name="WebSearchAnalyst",
    memory_path=Path(__file__).resolve().parent / "memory" / "session_memory.json",
)
_TOOLS = [search_web_with_mcp, load_session_memory]
_SKILL_TOOLSET = build_skill_toolset(Path(__file__).resolve().parent / "skills")
if _SKILL_TOOLSET is not None:
    _TOOLS.append(_SKILL_TOOLSET)


web_search_agent = LlmAgent(
    name="WebSearchAnalyst",
    model=resolve_agent_model("WebSearchAnalyst"),
    instruction=(
        "You are a web research specialist. "
        "Own web-source discovery and citation-grounded synthesis. "
        "Decide your own search strategy from user intent, keep paper-specific retrieval with PaperAnalyst, and request that handoff through structured `needs` and `artifacts` when necessary. "
        "If prior workflow context is missing, request the missing workflow memory from MainAgent through structured `needs` instead of asking the user to repeat internal step data. "
        "If you discover reusable sources or candidate items for downstream agents, prefer a compact structured handoff with `summary`, `artifacts`, and `needs`. "
        "For artifacts, include stable identifiers such as URL, DOI, or arXiv ID when available so PaperAnalyst can continue the workflow when needed. "
        "Do not invent facts. If evidence is weak, clearly say so and suggest what is needed next."
    ),
    tools=_TOOLS,
)

agent = web_search_agent


__all__ = [
    "agent",
    "web_search_agent",
    "load_session_memory",
    "search_web_with_mcp",
]
