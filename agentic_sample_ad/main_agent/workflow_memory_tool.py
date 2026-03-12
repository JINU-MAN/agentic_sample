from __future__ import annotations

from agentic_sample_ad.workflow_memory_runtime import render_active_workflow_memory


def read_workflow_memory(query: str = "", max_items: int = 6) -> str:
    """Read the shared workflow memory for the current run.

    Use this when you need prior step outputs, reusable artifacts, open needs,
    or pending-step context before deciding the next coordinator action.
    Do not use it for fresh external research.
    """

    try:
        normalized_max_items = max(1, min(int(max_items or 6), 12))
    except (TypeError, ValueError):
        normalized_max_items = 6
    return render_active_workflow_memory(query=query, max_items=normalized_max_items)


__all__ = ["read_workflow_memory"]
