from __future__ import annotations

from google.adk.agents import LlmAgent

from agentic_sample_ad.model_settings import resolve_agent_model

from .tool import scrape_sns_with_mcp


sns_agent = LlmAgent(
    name="SocialMediaAnalyst",
    model=resolve_agent_model("SocialMediaAnalyst"),
    instruction=(
        "You are a social media research specialist. "
        "Decide your own search queries from user intent and use `scrape_sns_with_mcp` to collect relevant posts. "
        "If you discover reusable posts, accounts, links, or signals for downstream agents, prefer a compact structured handoff with `summary`, `artifacts`, and `needs`. "
        "Summarize the strongest signals, explain why they matter, and include enough source detail for follow-up. "
        "Do not invent facts. If evidence is weak or missing, clearly say so and state what is needed next."
    ),
    tools=[
        scrape_sns_with_mcp,
    ],
)

agent = sns_agent


__all__ = [
    "agent",
    "sns_agent",
    "scrape_sns_with_mcp",
]
