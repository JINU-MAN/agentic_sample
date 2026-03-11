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
            "Search the local PDF corpus, inspect workflow handoff artifacts, fetch external paper references, "
            "and answer from paper evidence."
        ),
        default_tags=[
            "worker",
            "paper_search",
            "paper_memory",
            "external_paper_fetch",
            "scrape_papers_with_mcp",
            "fetch_external_paper_with_mcp",
            "load_paper_memory_with_mcp",
            "expand_paper_memory_with_mcp",
            "query_paper_memory",
        ],
    )


if __name__ == "__main__":
    main()
