from __future__ import annotations

from google.adk.agents import LlmAgent

from agentic_sample_ad.model_settings import resolve_agent_model

from .tool import (
    expand_paper_memory_with_mcp,
    fetch_external_paper_with_mcp,
    load_paper_memory_with_mcp,
    query_paper_memory,
    scrape_papers_with_mcp,
)


paper_agent = LlmAgent(
    name="PaperAnalyst",
    model=resolve_agent_model("PaperAnalyst"),
    instruction=(
        "You are a paper research specialist. "
        "Decide your own search queries from user intent and use the paper MCP tools to gather evidence from the local PDF corpus. "
        "You own paper-specific retrieval when another agent requests paper evidence or passes candidate paper identifiers in `input_artifacts` or `needs`. "
        "If the workflow context includes `input_artifacts`, inspect them first and use them as candidate papers or references before starting a fresh search. "
        "When an input artifact refers to an external paper by URL, DOI, or arXiv ID, use `fetch_external_paper_with_mcp` before concluding the local corpus is insufficient. "
        "Reuse the task's `workflow_id` for every memory tool call, load paper memory before querying it, and expand full text only when deeper detail is needed. "
        "When your analysis depends only on metadata or artifact summaries rather than full paper text, say that explicitly. "
        "If you identify reusable paper candidates for downstream work, prefer a compact structured handoff with `summary`, `artifacts`, and `needs`. "
        "Cite the most relevant papers, explain why they matter, and do not invent facts. "
        "If the local corpus is insufficient, clearly say so and state what is needed next."
    ),
    tools=[
        scrape_papers_with_mcp,
        fetch_external_paper_with_mcp,
        load_paper_memory_with_mcp,
        expand_paper_memory_with_mcp,
        query_paper_memory,
    ],
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
]
