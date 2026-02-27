from __future__ import annotations

from agentic_sample_ad.common.a2a_agent_server import run_server
from agentic_sample_ad.sns_agent.agent import agent as sns_agent


def main() -> None:
    run_server(
        module_name=__name__,
        agent_obj=sns_agent,
        component="a2a.agent_server.sns",
        default_name="SocialMediaAnalyst",
        default_description=(
            "SNS post collection, relevance filtering, and concise social signal summarization."
        ),
        default_tags=[
            "sns_search",
            "sns_summary",
            "agentic",
            "local-agent",
        ],
    )


if __name__ == "__main__":
    main()
