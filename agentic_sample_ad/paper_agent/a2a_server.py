from __future__ import annotations

from agentic_sample_ad.common.a2a_agent_server import run_server
from agentic_sample_ad.paper_agent.agent import agent as paper_agent


def main() -> None:
    run_server(
        module_name=__name__,
        agent_obj=paper_agent,
        component="a2a.agent_server.paper",
        default_name="PaperAnalyst",
        default_description=(
            "Paper search, full-text memory expansion, and follow-up Q&A from local PDF corpus."
        ),
        default_tags=[
            "paper_search",
            "paper_summary",
            "paper_fulltext_memory",
            "paper_memory_query",
            "agentic",
            "local-agent",
        ],
    )


if __name__ == "__main__":
    main()
