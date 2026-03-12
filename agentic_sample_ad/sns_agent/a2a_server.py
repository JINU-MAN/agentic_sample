from __future__ import annotations

from agentic_sample_ad.common.a2a_agent_server import run_server
from agentic_sample_ad.sns_agent.agent import agent as sns_agent


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
    )


if __name__ == "__main__":
    main()
