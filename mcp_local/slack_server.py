import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

from mcp.server.fastmcp import FastMCP
from system_logger import log_event


mcp = FastMCP("slack-mcp-server", json_response=True)

SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


def _load_env_file() -> None:
    """
    Lightweight .env loader to avoid requiring python-dotenv.
    Existing process env values are not overwritten.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_slack_token() -> str:
    # Prefer bot token, but allow common fallback names for compatibility.
    return (
        os.getenv("SLACK_BOT_TOKEN")
        or os.getenv("SLACK_API_TOKEN")
        or os.getenv("SLACK_TOKEN")
        or ""
    )


_load_env_file()


@mcp.tool()
def post_message(channel: str, text: str, thread_ts: Optional[str] = None) -> Dict[str, Any]:
    """
    Send a message to Slack using chat.postMessage.
    """
    log_event(
        "mcp_server.slack",
        "tool_called",
        {
            "tool": "post_message",
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        },
        direction="inbound",
    )
    token = _get_slack_token()
    if not token:
        result = {
            "ok": False,
            "error": "missing_slack_token",
            "message": "Set SLACK_BOT_TOKEN (or SLACK_API_TOKEN / SLACK_TOKEN).",
        }
        log_event("mcp_server.slack", "tool_completed", {"result": result}, direction="outbound")
        return result

    if not channel.strip():
        result = {"ok": False, "error": "invalid_channel", "message": "channel is required."}
        log_event("mcp_server.slack", "tool_completed", {"result": result}, direction="outbound")
        return result
    if not text.strip():
        result = {"ok": False, "error": "invalid_text", "message": "text is required."}
        log_event("mcp_server.slack", "tool_completed", {"result": result}, direction="outbound")
        return result

    payload: Dict[str, Any] = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        SLACK_POST_MESSAGE_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )

    try:
        with request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        result = {
            "ok": False,
            "error": "http_error",
            "status_code": e.code,
            "response": raw,
        }
        log_event("mcp_server.slack", "tool_completed", {"result": result}, direction="outbound")
        return result
    except error.URLError as e:
        result = {
            "ok": False,
            "error": "network_error",
            "message": str(e.reason),
        }
        log_event("mcp_server.slack", "tool_completed", {"result": result}, direction="outbound")
        return result
    except Exception as e:  # pragma: no cover
        result = {"ok": False, "error": "unexpected_error", "message": str(e)}
        log_event("mcp_server.slack", "tool_completed", {"result": result}, direction="outbound")
        return result

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        parsed_result = {"ok": False, "error": "invalid_json_response", "response": raw}
        log_event("mcp_server.slack", "tool_completed", {"result": parsed_result}, direction="outbound")
        return parsed_result

    if not result.get("ok"):
        parsed_result = {
            "ok": False,
            "error": result.get("error", "unknown_error"),
            "response": result,
        }
        log_event("mcp_server.slack", "tool_completed", {"result": parsed_result}, direction="outbound")
        return parsed_result

    parsed_result = {
        "ok": True,
        "channel": result.get("channel"),
        "ts": result.get("ts"),
        "message": result.get("message"),
    }
    log_event("mcp_server.slack", "tool_completed", {"result": parsed_result}, direction="outbound")
    return parsed_result


if __name__ == "__main__":
    # Example: python -m mcp.slack_server
    mcp.run(transport="stdio")
