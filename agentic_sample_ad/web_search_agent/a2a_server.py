from __future__ import annotations

from agentic_sample_ad.common.a2a_agent_server import run_server
from agentic_sample_ad.web_search_agent.agent import agent as web_search_agent


def main() -> None:
    run_server(
        module_name=__name__,
        agent_obj=web_search_agent,
        component="a2a.agent_server.web_search",
        default_name="WebSearchAnalyst",
        default_description=(
            "Web search candidate discovery, page fetch validation, and evidence summarization."
        ),
        default_tags=[
            "web_search",
            "web_summary",
            "fact_check",
            "agentic",
            "local-agent",
        ],
    )


if __name__ == "__main__":
    main()
