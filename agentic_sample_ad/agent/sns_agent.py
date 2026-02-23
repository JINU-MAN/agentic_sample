from __future__ import annotations

import json
from pathlib import Path

from google.adk.agents import LlmAgent

from agentic_sample_ad.agent.tool.slack_mcp_tool import slack_post_message
from agentic_sample_ad.mcp_local.client import call_mcp_tool
from agentic_sample_ad.system_logger import log_event, log_exception


BASE_DIR = Path(__file__).parent.parent
SNS_MCP_SERVER = BASE_DIR / "mcp_local" / "sns_server.py"


def scrape_sns_with_mcp(keyword: str) -> str:
    """
    Search SNS JSON posts via MCP server and return normalized JSON text.
    """
    log_event(
        "tool.scrape_sns_with_mcp",
        "call_started",
        {"keyword": keyword, "server_script_path": str(SNS_MCP_SERVER)},
        direction="outbound",
    )
    try:
        result = call_mcp_tool(
            server_script_path=str(SNS_MCP_SERVER),
            tool_name="search_sns_posts",
            arguments={"keyword": keyword},
        )
        log_event(
            "tool.scrape_sns_with_mcp",
            "call_completed",
            {"keyword": keyword, "result": result},
            direction="inbound",
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        log_exception(
            "tool.scrape_sns_with_mcp",
            "call_failed",
            e,
            {"keyword": keyword, "server_script_path": str(SNS_MCP_SERVER)},
        )
        raise


sns_agent = LlmAgent(
    name="SocialMediaAnalyst",
    model="gemini-2.0-flash",
    instruction=(
        "You are a social media analysis specialist.\n"
        "Specialization-first policy: SNS/social signal tasks are your primary responsibility.\n"
        "Decide search angle and semantic scope yourself; do not rely on fixed keyword templates.\n\n"
        "Tool usage guide:\n"
        "1) Start with `scrape_sns_with_mcp` using one or more intent-driven query variants.\n"
        "2) Select and summarize posts by meaning-level relevance to the user request.\n"
        "3) Explain why selected posts are relevant.\n"
        "4) If the user asks for Slack posting, call `slack_post_message(channel, text)`.\n"
        "5) Use indirect delegation only: do not call other agents directly.\n"
        "6) If local SNS evidence is weak or missing, request follow-up work in `Additional Needs:`.\n"
        "   Example: `- [WebSearchAnalyst] Collect external sources that validate this signal.`\n"
        "   Example: `- [MainAgent] Ask user for platform, timeframe, or entity clarification.`\n"
        "7) Always end with one of these:\n"
        "   - `Additional Needs: none`\n"
        "   - `Additional Needs:` followed by bullet lines in format `[TargetAgentName] request`.\n"
        "8) Keep output concise and actionable."
    ),
    tools=[
        scrape_sns_with_mcp,
        slack_post_message,
    ],
)


__all__ = ["scrape_sns_with_mcp", "sns_agent"]

