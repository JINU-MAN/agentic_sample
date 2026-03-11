from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import httpx
from mcp.server.fastmcp import FastMCP

from agentic_sample_ad.system_logger import log_event, log_exception


mcp = FastMCP("web-search-mcp-server", json_response=True)

DEFAULT_MAX_RESULTS = 6
MAX_MAX_RESULTS = 10
REQUEST_TIMEOUT_SECONDS = 20.0
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file() -> None:
    if not ENV_PATH.exists():
        return

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _to_text(value: Any, default: str = "") -> str:
    token = str(value or "").strip()
    return token if token else default


def _extract_results(payload: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
    items = payload.get("results", [])
    if not isinstance(items, list):
        return []

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = _to_text(item.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        title = _to_text(item.get("title"), default="Untitled")
        snippet = _to_text(item.get("content"))
        score = item.get("score")
        published = _to_text(item.get("published_date"))

        row: Dict[str, Any] = {
            "rank": len(rows) + 1,
            "title": title[:200],
            "url": url,
            "snippet": snippet[:500],
            "source": "tavily",
        }
        if isinstance(score, (int, float)):
            row["score"] = float(score)
        if published:
            row["published_date"] = published
        rows.append(row)

        if len(rows) >= max_results:
            break

    return rows


def _tavily_search(*, query: str, max_results: int) -> List[Dict[str, Any]]:
    api_key = _to_text(os.getenv("TAVILY_API_KEY"))
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing. Set it in .env.")

    body = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = client.post(TAVILY_SEARCH_ENDPOINT, json=body)
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        payload = {}
    return _extract_results(payload, max_results=max_results)


@mcp.tool()
def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, Any]]:
    """
    Search the public web via Tavily and return ranked result items.
    """
    _load_env_file()

    normalized_query = query.strip()
    safe_max_results = _clamp(int(max_results), 1, MAX_MAX_RESULTS)
    log_event(
        "mcp_server.web_search",
        "tool_called",
        {
            "tool": "search_web",
            "query": normalized_query,
            "max_results": safe_max_results,
            "provider": "tavily",
        },
        direction="inbound",
    )

    if not normalized_query:
        log_event(
            "mcp_server.web_search",
            "tool_completed",
            {
                "tool": "search_web",
                "query": normalized_query,
                "result_count": 0,
                "reason": "empty_query",
            },
            direction="outbound",
        )
        return []

    try:
        results = _tavily_search(query=normalized_query, max_results=safe_max_results)
        log_event(
            "mcp_server.web_search",
            "tool_completed",
            {
                "tool": "search_web",
                "query": normalized_query,
                "max_results": safe_max_results,
                "provider": "tavily",
                "result_count": len(results),
                "top_urls": [str(item.get("url", "")) for item in results[:5]],
            },
            direction="outbound",
        )
        return results
    except Exception as e:
        log_exception(
            "mcp_server.web_search",
            "tool_failed",
            e,
            {
                "tool": "search_web",
                "query": normalized_query,
                "max_results": safe_max_results,
                "provider": "tavily",
            },
        )
        raise


if __name__ == "__main__":
    mcp.run(transport="stdio")
