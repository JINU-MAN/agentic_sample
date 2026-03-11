from __future__ import annotations

import json
from pathlib import Path

from agentic_sample_ad.mcp_local.client import call_mcp_tool
from agentic_sample_ad.system_logger import log_event, log_exception


BASE_DIR = Path(__file__).parent.parent.parent
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


__all__ = ["scrape_sns_with_mcp"]

