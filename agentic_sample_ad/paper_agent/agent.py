from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent

from agentic_sample_ad.agent_session_memory_runtime import build_load_session_memory_tool
from agentic_sample_ad.handoff_contract_tool import format_handoff_contract
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
    format_handoff_contract,
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
        "Use your private skills, tools, and session memory directly before declaring a blocker, and never ask the coordinator to load your own skill or session memory. "
        "Your direct tools return JSON with `ok`, `tool_name`, `summary`, `content_type`, `items`, `data`, `errors`, and `metadata`; read those fields internally and do not echo raw tool output as the final workflow answer. "
        "If prior workflow context is missing, request the missing workflow memory from MainAgent through structured `needs` instead of asking the user to repeat internal step data. "
        "If you identify reusable paper candidates for downstream work, prefer a compact structured handoff with `summary`, optional `text_response`, `artifacts`, and `needs`. "
        "CRITICAL OUTPUT RULE: Never write JSON manually and never end your response with plain prose. "
        "When your research is complete, call `format_handoff_contract(status, summary, text_response, artifacts_json, needs_json)`. "
        "After the tool returns, write its return value as your next message — copy the returned string exactly as a text message with no additions. "
        "If `format_handoff_contract` returns a line starting with 'format_handoff_contract failed', fix the arguments and call it again before writing anything. "
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
    "format_handoff_contract",
]
