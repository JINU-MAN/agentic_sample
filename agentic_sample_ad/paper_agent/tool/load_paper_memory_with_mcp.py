from __future__ import annotations

from ._core import load_paper_memory_with_mcp as _load_paper_memory_with_mcp_impl
from ._core import DEFAULT_WORKFLOW_ID, MAX_MEMORY_CHARS_PER_PAPER


def load_paper_memory_with_mcp(
    query: str,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    max_papers: int = 3,
    max_chars_per_paper: int = MAX_MEMORY_CHARS_PER_PAPER,
    load_mode: str = "overview",
) -> str:
    """
    Load workflow-scoped paper memory.
    """
    return _load_paper_memory_with_mcp_impl(
        query=query,
        workflow_id=workflow_id,
        max_papers=max_papers,
        max_chars_per_paper=max_chars_per_paper,
        load_mode=load_mode,
    )


__all__ = ["load_paper_memory_with_mcp"]
