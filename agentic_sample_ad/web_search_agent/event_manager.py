from __future__ import annotations

from typing import Any, Dict, List

from agentic_sample_ad.common.local_agent_event_manager import LocalAgentEventManager

from .agent import web_search_agent


_MANAGER = LocalAgentEventManager(
    agent_name="WebSearchAnalyst",
    agent_obj=web_search_agent,
    component="ad.web_search_agent.event_manager",
)


def enqueue_task(command: str, context: Dict[str, Any] | None = None) -> None:
    _MANAGER.enqueue(command, context=context)


def run_next_task() -> Dict[str, Any]:
    return _MANAGER.run_next()


def run_all_tasks() -> List[Dict[str, Any]]:
    return _MANAGER.run_until_empty()


def run_single_task(command: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    enqueue_task(command, context=context)
    return run_next_task()

