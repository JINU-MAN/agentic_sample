from __future__ import annotations

from typing import Any, Dict, List

from agentic_sample_ad.event_manager import execute_plan as _legacy_execute_plan

from .system_logger import log_main_event


def execute_plan(
    plan: Dict[str, Any],
    main_agent: Any,
    available_agents: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
) -> Any:
    log_main_event(
        "execute_plan_forwarded",
        {
            "num_available_agents": len(available_agents),
            "has_context": bool(context),
        },
    )
    return _legacy_execute_plan(
        plan=plan,
        main_agent=main_agent,
        available_agents=available_agents,
        context=context,
    )


