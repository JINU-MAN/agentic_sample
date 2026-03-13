from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent

from agentic_sample_ad.agent_session_memory_runtime import build_load_session_memory_tool
from agentic_sample_ad.model_settings import resolve_agent_model
from agentic_sample_ad.skill_runtime import build_skill_toolset

from .system_logger import log_event, log_exception
from .tool import (
    expand_paper_memory_with_mcp,
    fetch_external_paper_with_mcp,
    load_paper_memory_with_mcp,
    query_paper_memory,
    scrape_papers_with_mcp,
)


load_session_memory = build_load_session_memory_tool(
    agent_name="PaperAnalyst",
    memory_path=Path(__file__).resolve().parent / "memory" / "session_memory.json",
    log_event_fn=log_event,
    log_exception_fn=log_exception,
)
_TOOLS = [
    scrape_papers_with_mcp,
    fetch_external_paper_with_mcp,
    load_paper_memory_with_mcp,
    expand_paper_memory_with_mcp,
    query_paper_memory,
    load_session_memory,
]
_SKILL_TOOLSET = build_skill_toolset(Path(__file__).resolve().parent / "skills")
if _SKILL_TOOLSET is not None:
    _TOOLS.append(_SKILL_TOOLSET)


paper_agent = LlmAgent(
    name="PaperAnalyst",
    model=resolve_agent_model("PaperAnalyst"),
    instruction=(
        "You are a paper research specialist. "
        "Own paper-specific retrieval from the local corpus, workflow-scoped paper memory, and external paper identifiers. "
        "Inspect workflow artifacts before starting a fresh search, prefer the lightest evidence path that can answer the question, and state clearly when the result relies only on metadata or partial context. "
        "If prior workflow context is missing, request the missing workflow memory from MainAgent through structured `needs` instead of asking the user to repeat internal step data. "
        "If you identify reusable paper candidates for downstream work, prefer a compact structured handoff with `summary`, `artifacts`, and `needs`. "
        "Cite the most relevant papers, explain why they matter, and do not invent facts. "
        "If the local corpus is insufficient, clearly say so and state what is needed next."
    ),
    tools=_TOOLS,
)

agent = paper_agent


__all__ = [
    "agent",
    "paper_agent",
    "scrape_papers_with_mcp",
    "fetch_external_paper_with_mcp",
    "load_paper_memory_with_mcp",
    "expand_paper_memory_with_mcp",
    "query_paper_memory",
    "load_session_memory",
]
