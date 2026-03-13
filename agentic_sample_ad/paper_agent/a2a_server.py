from __future__ import annotations

from agentic_sample_ad.common.a2a_agent_server import run_server
from agentic_sample_ad.paper_agent.agent import agent as paper_agent
from agentic_sample_ad.paper_agent.system_logger import (
    finalize_process_logging,
    initialize_process_logging,
    log_event,
    log_exception,
    start_new_logging_session,
)


def main() -> None:
    run_server(
        module_name=__name__,
        agent_obj=paper_agent,
        component="a2a.agent_server.paper",
        default_name="PaperAnalyst",
        default_description=(
            "Search the local PDF corpus, inspect workflow handoff artifacts, fetch external paper references, "
            "and answer from paper evidence."
        ),
        default_tags=[
            "worker",
            "paper_search",
            "paper_memory",
            "external_paper_fetch",
            "paper_evidence",
        ],
        initialize_logging_fn=initialize_process_logging,
        finalize_logging_fn=finalize_process_logging,
        start_logging_session_fn=start_new_logging_session,
        log_event_fn=log_event,
        log_exception_fn=log_exception,
    )


if __name__ == "__main__":
    main()
