from __future__ import annotations

from ._core import expand_paper_memory_with_mcp as _expand_paper_memory_with_mcp_impl
from ._core import DEFAULT_WORKFLOW_ID, MAX_MEMORY_CHARS_PER_PAPER


def expand_paper_memory_with_mcp(
    question: str,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    max_papers: int = 2,
    max_chars_per_paper: int = MAX_MEMORY_CHARS_PER_PAPER,
) -> str:
    """
    Lazily expand workflow paper memory with full text for the most relevant papers.
    """
    return _expand_paper_memory_with_mcp_impl(
        question=question,
        workflow_id=workflow_id,
        max_papers=max_papers,
        max_chars_per_paper=max_chars_per_paper,
    )


__all__ = ["expand_paper_memory_with_mcp"]
