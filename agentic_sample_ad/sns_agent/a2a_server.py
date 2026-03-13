from __future__ import annotations

from agentic_sample_ad.common.a2a_agent_server import run_server
from agentic_sample_ad.sns_agent.agent import agent as sns_agent
from agentic_sample_ad.sns_agent.system_logger import (
    finalize_process_logging,
    initialize_process_logging,
    log_event,
    log_exception,
    start_new_logging_session,
)


def main() -> None:
    run_server(
        module_name=__name__,
        agent_obj=sns_agent,
        component="a2a.agent_server.sns",
        default_name="SocialMediaAnalyst",
        default_description=("Search SNS posts via MCP and summarize relevant social signals."),
        default_tags=[
            "worker",
            "sns_search",
            "sns_summary",
            "social_signal_analysis",
        ],
        initialize_logging_fn=initialize_process_logging,
        finalize_logging_fn=finalize_process_logging,
        start_logging_session_fn=start_new_logging_session,
        log_event_fn=log_event,
        log_exception_fn=log_exception,
    )


if __name__ == "__main__":
    main()
