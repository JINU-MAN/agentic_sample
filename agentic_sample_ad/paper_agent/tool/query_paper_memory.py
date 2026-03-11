from __future__ import annotations

from ._core import query_paper_memory as _query_paper_memory_impl
from ._core import DEFAULT_WORKFLOW_ID, MAX_MEMORY_SNIPPETS


def query_paper_memory(
    question: str,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    max_snippets: int = MAX_MEMORY_SNIPPETS,
) -> str:
    """
    Query loaded paper memory and return the most relevant snippets quickly.
    """
    return _query_paper_memory_impl(
        question=question,
        workflow_id=workflow_id,
        max_snippets=max_snippets,
    )


__all__ = ["query_paper_memory"]
