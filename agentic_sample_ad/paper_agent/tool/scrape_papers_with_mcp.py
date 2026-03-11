from __future__ import annotations

from ._core import scrape_papers_with_mcp as _scrape_papers_with_mcp_impl


def scrape_papers_with_mcp(query: str) -> str:
    """
    Search PDFs with MCP paper server and return normalized JSON text.
    """
    return _scrape_papers_with_mcp_impl(query=query)


__all__ = ["scrape_papers_with_mcp"]
