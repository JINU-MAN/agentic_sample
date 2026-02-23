from __future__ import annotations

from agentic_sample_ad.agent.paper_agent import (
    expand_paper_memory_with_mcp,
    load_paper_memory_with_mcp,
    query_paper_memory,
    research_agent as paper_agent,
    scrape_papers_with_mcp,
)


agent = paper_agent


__all__ = [
    "agent",
    "paper_agent",
    "scrape_papers_with_mcp",
    "load_paper_memory_with_mcp",
    "expand_paper_memory_with_mcp",
    "query_paper_memory",
]


