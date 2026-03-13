from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agentic_sample_ad.mcp_local.client import call_mcp_tool
from agentic_sample_ad.web_search_agent.system_logger import log_event, log_exception


BASE_DIR = Path(__file__).parent.parent.parent
WEB_SEARCH_MCP_SERVER = BASE_DIR / "mcp_local" / "web_search_server.py"
DEFAULT_RESULT_COUNT = 6


def _extract_error_text(raw: Dict[str, Any]) -> str:
    messages: List[str] = []
    for item in raw.get("content", []):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            messages.append(text.strip())
    return "\n".join(messages).strip()


def _extract_mcp_list_result(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                    if isinstance(parsed, list):
                        return [item for item in parsed if isinstance(item, dict)]
                except json.JSONDecodeError:
                    pass
    return []


def search_web_with_mcp(query: str, max_results: int = DEFAULT_RESULT_COUNT) -> str:
    """
    Query the web-search MCP server and return normalized ranked results.
    """
    safe_query = query.strip()
    safe_max_results = max(1, min(int(max_results), 10))
    log_event(
        "tool.search_web_with_mcp",
        "call_started",
        {
            "query": safe_query,
            "max_results": safe_max_results,
            "server_script_path": str(WEB_SEARCH_MCP_SERVER),
        },
        direction="outbound",
    )
    try:
        raw = call_mcp_tool(
            server_script_path=str(WEB_SEARCH_MCP_SERVER),
            tool_name="search_web",
            arguments={"query": safe_query, "max_results": safe_max_results},
        )
        normalized = _extract_mcp_list_result(raw)
        result_text = json.dumps(normalized, ensure_ascii=False, indent=2)
        log_event(
            "tool.search_web_with_mcp",
            "call_completed",
            {
                "query": safe_query,
                "max_results": safe_max_results,
                "result_count": len(normalized),
            },
            direction="inbound",
        )
        return result_text
    except Exception as e:
        log_exception(
            "tool.search_web_with_mcp",
            "call_failed",
            e,
            {
                "query": safe_query,
                "max_results": safe_max_results,
                "server_script_path": str(WEB_SEARCH_MCP_SERVER),
            },
        )
        raise


__all__ = ["search_web_with_mcp"]
