from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agentic_sample_ad.mcp_local.client import call_mcp_tool
from agentic_sample_ad.sns_agent.system_logger import log_event, log_exception
from agentic_sample_ad.tool_output_utils import render_tool_output


BASE_DIR = Path(__file__).parent.parent.parent
SNS_MCP_SERVER = BASE_DIR / "mcp_local" / "sns_server.py"


def _extract_error_text(raw: Dict[str, Any]) -> str:
    messages: List[str] = []
    for item in raw.get("content", []):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            messages.append(text.strip())
    return "\n".join(messages).strip()


def _extract_sns_items(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    if raw.get("isError"):
        message = _extract_error_text(raw) or "MCP tool returned an error."
        return [{"error": message}]

    structured = raw.get("structuredContent", {})
    if isinstance(structured, dict):
        result = structured.get("result")
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]

    content = raw.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return []
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
    return []


def scrape_sns_with_mcp(keyword: str) -> str:
    """
    Search SNS posts via MCP server and return the shared tool output contract.
    """
    safe_keyword = str(keyword or "").strip()
    log_event(
        "tool.scrape_sns_with_mcp",
        "call_started",
        {"keyword": safe_keyword, "server_script_path": str(SNS_MCP_SERVER)},
        direction="outbound",
    )
    try:
        raw = call_mcp_tool(
            server_script_path=str(SNS_MCP_SERVER),
            tool_name="search_sns_posts",
            arguments={"keyword": safe_keyword},
        )
        normalized = _extract_sns_items(raw)
        error_messages = [
            str(item.get("error", "")).strip()
            for item in normalized
            if isinstance(item, dict) and str(item.get("error", "")).strip()
        ]
        items = [item for item in normalized if not (isinstance(item, dict) and item.get("error"))]
        ok = not error_messages
        rendered = render_tool_output(
            tool_name="scrape_sns_with_mcp",
            ok=ok,
            summary=(
                f"Collected {len(items)} SNS result(s) for keyword '{safe_keyword}'."
                if ok
                else (error_messages[0] if error_messages else "SNS search failed.")
            ),
            content_type="collection",
            items=items,
            data={"keyword": safe_keyword, "result_count": len(items)},
            errors=error_messages,
            metadata={"server_script_path": str(SNS_MCP_SERVER)},
        )
        log_event(
            "tool.scrape_sns_with_mcp",
            "call_completed",
            {"keyword": safe_keyword, "result_count": len(items), "ok": ok},
            direction="inbound",
        )
        return rendered
    except Exception as e:
        log_exception(
            "tool.scrape_sns_with_mcp",
            "call_failed",
            e,
            {"keyword": safe_keyword, "server_script_path": str(SNS_MCP_SERVER)},
        )
        raise


__all__ = ["scrape_sns_with_mcp"]
