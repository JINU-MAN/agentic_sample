from __future__ import annotations

from agentic_sample_ad.tool_output_utils import render_tool_output
from agentic_sample_ad.main_agent.system_logger import log_event, log_exception
from agentic_sample_ad.workflow_memory_runtime import build_active_workflow_memory_view


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
    log_event(
        "tool.read_workflow_memory",
        "call_started",
        {"query": query, "max_items": normalized_max_items},
        direction="outbound",
    )
    try:
        view = build_active_workflow_memory_view(query=query, max_items=normalized_max_items)
        rendered = render_tool_output(
            tool_name="read_workflow_memory",
            ok=bool(view.get("ok")),
            summary=(
                "Loaded active workflow memory."
                if bool(view.get("ok"))
                else str(view.get("message", "No active workflow memory is available."))
            ),
            content_type="memory",
            data=view,
            errors=[] if bool(view.get("ok")) else [str(view.get("message", "workflow_memory_unavailable"))],
            metadata={"query": query, "max_items": normalized_max_items},
        )
        log_event(
            "tool.read_workflow_memory",
            "call_completed",
            {"query": query, "max_items": normalized_max_items},
            direction="inbound",
        )
        return rendered
    except Exception as e:
        log_exception(
            "tool.read_workflow_memory",
            "call_failed",
            e,
            {"query": query, "max_items": normalized_max_items},
        )
        raise


__all__ = ["read_workflow_memory"]
