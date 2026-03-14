import json
import os
from typing import Any, Dict

from agentic_sample_ad.mcp_local.client import call_mcp_tool
from agentic_sample_ad.main_agent.system_logger import log_event, log_exception
from agentic_sample_ad.tool_output_utils import render_tool_output


def _resolve_slack_server_path() -> str:
    # Resolve at call time so values loaded from `.env` after import are reflected.
    return str(os.getenv("SLACK_MCP_SERVER_PATH", "")).strip()


def slack_post_message(channel: str, text: str) -> str:
    """
    Send a message through a Slack MCP server and return the shared tool output contract.
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
        return render_tool_output(
            tool_name="slack_post_message",
            ok=False,
            summary=message,
            content_type="error",
            data={"channel": channel, "text": text},
            errors=[message],
            metadata={"server_script_path": slack_server_path},
        )

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
        ok = not bool(result.get("isError")) if isinstance(result, dict) else True
        errors = []
        if isinstance(result, dict) and result.get("isError"):
            errors.append("Slack MCP server returned an error.")
        return render_tool_output(
            tool_name="slack_post_message",
            ok=ok,
            summary=(
                f"Posted message to Slack channel '{channel}'."
                if ok
                else f"Failed to post message to Slack channel '{channel}'."
            ),
            content_type="delivery",
            data={"channel": channel, "text": text, "result": result},
            errors=errors,
            metadata={"server_script_path": slack_server_path},
        )
    except Exception as e:
        log_exception(
            "tool.slack_post_message",
            "call_failed",
            e,
            {"channel": channel, "has_server_path": bool(slack_server_path)},
        )
        raise


__all__ = ["slack_post_message"]
