import json
import os
from typing import Any, Dict

from agentic_sample_ad.mcp_local.client import call_mcp_tool
from agentic_sample_ad.system_logger import log_event, log_exception


def _resolve_slack_server_path() -> str:
    # Resolve at call time so values loaded from `.env` after import are reflected.
    return str(os.getenv("SLACK_MCP_SERVER_PATH", "")).strip()


def slack_post_message(channel: str, text: str) -> str:
    """
    Send a message through a Slack MCP server.
    """
    slack_server_path = _resolve_slack_server_path()
    log_event(
        "tool.slack_post_message",
        "call_started",
        {
            "channel": channel,
            "text": text,
            "has_server_path": bool(slack_server_path),
            "server_script_path": slack_server_path,
        },
        direction="outbound",
    )

    if not slack_server_path:
        message = "SLACK_MCP_SERVER_PATH is not configured."
        log_event(
            "tool.slack_post_message",
            "call_skipped",
            {"reason": "missing_server_path", "message": message},
            level="ERROR",
        )
        return message

    tool_name = "post_message"
    arguments: Dict[str, Any] = {"channel": channel, "text": text}

    try:
        result = call_mcp_tool(
            server_script_path=slack_server_path,
            tool_name=tool_name,
            arguments=arguments,
        )
        log_event(
            "tool.slack_post_message",
            "call_completed",
            {"channel": channel, "result": result},
            direction="inbound",
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        log_exception(
            "tool.slack_post_message",
            "call_failed",
            e,
            {"channel": channel, "has_server_path": bool(slack_server_path)},
        )
        raise


__all__ = ["slack_post_message"]

