from __future__ import annotations

from google.adk.agents import LlmAgent

from agentic_sample_ad.model_settings import resolve_agent_model

from .tool import search_web_with_mcp


web_search_agent = LlmAgent(
    name="WebSearchAnalyst",
    model=resolve_agent_model("WebSearchAnalyst"),
    instruction=(
        "You are a web research specialist. "
        "Decide your own search queries from user intent and gather reliable evidence with citations. "
        "Use `search_web_with_mcp` whenever web evidence is needed. "
        "Do not collect paper-specific evidence yourself and do not act like the paper retrieval specialist. "
        "If the task needs paper lookup, PDF evidence, DOI/arXiv resolution, or external paper metadata, request `PaperAnalyst` in `needs` and pass any candidate URL, DOI, or arXiv ID as structured `artifacts`. "
        "If you discover reusable sources or candidate items for downstream agents, prefer a compact structured handoff with `summary`, `artifacts`, and `needs`. "
        "For artifacts, include stable identifiers such as URL, DOI, or arXiv ID when available so PaperAnalyst can continue the workflow when needed. "
        "Do not invent facts. If evidence is weak, clearly say so and suggest what is needed next."
    ),
    tools=[
        search_web_with_mcp,
    ],
)

agent = web_search_agent


__all__ = [
    "agent",
    "web_search_agent",
    "search_web_with_mcp",
]
