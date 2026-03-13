from __future__ import annotations

from agentic_sample_ad.common.a2a_agent_server import run_server
from agentic_sample_ad.web_search_agent.agent import agent as web_search_agent
from agentic_sample_ad.web_search_agent.system_logger import (
    finalize_process_logging,
    initialize_process_logging,
    log_event,
    log_exception,
    start_new_logging_session,
)


def main() -> None:
    run_server(
        module_name=__name__,
        agent_obj=web_search_agent,
        component="a2a.agent_server.web_search",
        default_name="WebSearchAnalyst",
        default_description=(
            "Search the web via the Tavily-backed MCP server and synthesize citation-grounded evidence."
        ),
        default_tags=[
            "worker",
            "web_search",
            "web_evidence_summary",
            "web_research",
        ],
        initialize_logging_fn=initialize_process_logging,
        finalize_logging_fn=finalize_process_logging,
        start_logging_session_fn=start_new_logging_session,
        log_event_fn=log_event,
        log_exception_fn=log_exception,
    )


if __name__ == "__main__":
    main()
