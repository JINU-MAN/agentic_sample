from __future__ import annotations

import base64
import html
import re
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import httpx
from mcp.server.fastmcp import FastMCP

from agentic_sample_ad.system_logger import log_event, log_exception


mcp = FastMCP("web-search-mcp-server", json_response=True)

DEFAULT_MAX_RESULTS = 6
MAX_MAX_RESULTS = 10
DEFAULT_FETCH_MAX_CHARS = 6000
MAX_FETCH_MAX_CHARS = 20000
REQUEST_TIMEOUT_SECONDS = 15.0
USER_AGENT = "agentic-sample-web-search/1.0 (+https://example.local)"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html_tags(raw_html: str) -> str:
    without_script = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    without_style = re.sub(r"(?is)<style.*?>.*?</style>", " ", without_script)
    without_tags = re.sub(r"(?is)<[^>]+>", " ", without_style)
    return _normalize_whitespace(html.unescape(without_tags))


def _extract_title(raw_html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
    if not match:
        return ""
    return _normalize_whitespace(html.unescape(match.group(1)))


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _add_instant_answer_item(
    results: List[Dict[str, Any]],
    seen_urls: set[str],
    *,
    text: str,
    url: str,
) -> None:
    cleaned_text = _normalize_whitespace(text)
    cleaned_url = url.strip()
    if not cleaned_text or not cleaned_url:
        return
    if cleaned_url in seen_urls:
        return
    if not _is_http_url(cleaned_url):
        return

    title = cleaned_text
    snippet = ""
    if " - " in cleaned_text:
        left, right = cleaned_text.split(" - ", 1)
        title = left.strip() or cleaned_text
        snippet = right.strip()

    seen_urls.add(cleaned_url)
    results.append(
        {
            "title": title[:200],
            "url": cleaned_url,
            "snippet": snippet[:400],
            "source": "duckduckgo_instant_answer",
        }
    )


def _extract_instant_answer_results(payload: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    abstract_url = str(payload.get("AbstractURL", "")).strip()
    abstract_text = str(payload.get("AbstractText", "")).strip()
    heading = str(payload.get("Heading", "")).strip()
    if abstract_url and abstract_text:
        seen_urls.add(abstract_url)
        results.append(
            {
                "title": heading or abstract_text[:120],
                "url": abstract_url,
                "snippet": abstract_text[:400],
                "source": "duckduckgo_abstract",
            }
        )

    for item in payload.get("Results", []):
        if len(results) >= max_results:
            break
        if not isinstance(item, dict):
            continue
        _add_instant_answer_item(
            results,
            seen_urls,
            text=str(item.get("Text", "")),
            url=str(item.get("FirstURL", "")),
        )

    for item in payload.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if not isinstance(item, dict):
            continue

        # DuckDuckGo may return either direct topic items or grouped Topics arrays.
        topic_items = item.get("Topics")
        if isinstance(topic_items, list):
            for topic in topic_items:
                if len(results) >= max_results:
                    break
                if not isinstance(topic, dict):
                    continue
                _add_instant_answer_item(
                    results,
                    seen_urls,
                    text=str(topic.get("Text", "")),
                    url=str(topic.get("FirstURL", "")),
                )
            continue

        _add_instant_answer_item(
            results,
            seen_urls,
            text=str(item.get("Text", "")),
            url=str(item.get("FirstURL", "")),
        )

    ranked: List[Dict[str, Any]] = []
    for idx, item in enumerate(results[:max_results], start=1):
        enriched = dict(item)
        enriched["rank"] = idx
        ranked.append(enriched)
    return ranked


def _decode_bing_tracking_url(raw_url: str) -> str:
    cleaned = html.unescape(raw_url).strip()
    if not cleaned:
        return ""

    try:
        parsed = urlparse(cleaned)
    except Exception:
        return cleaned

    if parsed.netloc.lower().endswith("bing.com") and parsed.path.startswith("/ck/"):
        params = parse_qs(parsed.query)
        encoded = params.get("u", [""])[0]
        if encoded:
            candidate = encoded[2:] if encoded.startswith("a1") else encoded
            padding = "=" * (-len(candidate) % 4)
            try:
                decoded = base64.urlsafe_b64decode(candidate + padding).decode("utf-8", errors="ignore").strip()
            except Exception:
                decoded = ""
            if _is_http_url(decoded):
                return decoded

    return cleaned


def _extract_bing_results(raw_html: str, max_results: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    link_pattern = re.compile(
        r'(?is)<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    )

    for match in link_pattern.finditer(raw_html):
        if len(results) >= max_results:
            break

        url = _decode_bing_tracking_url(match.group(1))
        if not _is_http_url(url):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = _normalize_whitespace(_strip_html_tags(match.group(2)))
        next_chunk = raw_html[match.end() : match.end() + 1400]
        snippet_match = re.search(r'(?is)<p[^>]*>(.*?)</p>', next_chunk)
        snippet_raw = snippet_match.group(1) if snippet_match else ""
        snippet = _normalize_whitespace(_strip_html_tags(snippet_raw))

        results.append(
            {
                "rank": len(results) + 1,
                "title": title[:200],
                "url": url,
                "snippet": snippet[:400],
                "source": "bing_html",
            }
        )

    return results


@mcp.tool()
def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, Any]]:
    """
    Search the public web using DuckDuckGo Instant Answer API.

    Returns ranked items with title, url, snippet, and source.
    """
    normalized_query = query.strip()
    safe_max_results = _clamp(int(max_results), 1, MAX_MAX_RESULTS)
    log_event(
        "mcp_server.web_search",
        "tool_called",
        {
            "tool": "search_web",
            "query": normalized_query,
            "max_results": safe_max_results,
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
        ddg_params = {
            "q": normalized_query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
            "skip_disambig": "1",
        }
        headers = {"User-Agent": USER_AGENT}
        results: List[Dict[str, Any]] = []
        provider = "duckduckgo_instant_answer"
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True, headers=headers) as client:
            ddg_response = client.get("https://api.duckduckgo.com/", params=ddg_params)
            ddg_response.raise_for_status()
            payload = ddg_response.json()
            if not isinstance(payload, dict):
                payload = {}
            results = _extract_instant_answer_results(payload, safe_max_results)

            if not results:
                provider = "bing_html_fallback"
                bing_response = client.get(
                    "https://www.bing.com/search",
                    params={"q": normalized_query, "setlang": "en-US", "mkt": "en-US"},
                )
                bing_response.raise_for_status()
                results = _extract_bing_results(bing_response.text or "", safe_max_results)

        log_event(
            "mcp_server.web_search",
            "tool_completed",
            {
                "tool": "search_web",
                "query": normalized_query,
                "max_results": safe_max_results,
                "provider": provider,
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
            },
        )
        raise


@mcp.tool()
def fetch_page(url: str, max_chars: int = DEFAULT_FETCH_MAX_CHARS) -> Dict[str, Any]:
    """
    Fetch one page and return extracted plain text.
    """
    normalized_url = url.strip()
    safe_max_chars = _clamp(int(max_chars), 200, MAX_FETCH_MAX_CHARS)
    log_event(
        "mcp_server.web_search",
        "tool_called",
        {
            "tool": "fetch_page",
            "url": normalized_url,
            "max_chars": safe_max_chars,
        },
        direction="inbound",
    )

    if not _is_http_url(normalized_url):
        log_event(
            "mcp_server.web_search",
            "tool_completed",
            {
                "tool": "fetch_page",
                "url": normalized_url,
                "reason": "invalid_url",
            },
            direction="outbound",
            level="ERROR",
        )
        return {
            "ok": False,
            "url": normalized_url,
            "error": "Only http/https URLs are allowed.",
        }

    try:
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True, headers=headers) as client:
            response = client.get(normalized_url)
            raw_html = response.text or ""

        status_code = int(response.status_code)
        if status_code >= 400:
            result = {
                "ok": False,
                "url": normalized_url,
                "final_url": str(response.url),
                "status_code": status_code,
                "error": f"HTTP {status_code}",
                "content": "",
            }
            log_event(
                "mcp_server.web_search",
                "tool_completed",
                {
                    "tool": "fetch_page",
                    "url": normalized_url,
                    "status_code": status_code,
                    "reason": "http_error_status",
                },
                direction="outbound",
                level="ERROR",
            )
            return result

        title = _extract_title(raw_html)
        plain_text = _strip_html_tags(raw_html)
        truncated = len(plain_text) > safe_max_chars
        content = plain_text[:safe_max_chars]

        result = {
            "ok": True,
            "url": normalized_url,
            "final_url": str(response.url),
            "status_code": status_code,
            "title": title,
            "content": content,
            "truncated": truncated,
            "content_length": len(plain_text),
        }
        log_event(
            "mcp_server.web_search",
            "tool_completed",
            {
                "tool": "fetch_page",
                "url": normalized_url,
                "status_code": int(response.status_code),
                "content_length": len(plain_text),
                "truncated": truncated,
            },
            direction="outbound",
        )
        return result
    except Exception as e:
        log_exception(
            "mcp_server.web_search",
            "tool_failed",
            e,
            {
                "tool": "fetch_page",
                "url": normalized_url,
                "max_chars": safe_max_chars,
            },
        )
        raise


if __name__ == "__main__":
    # Example: python -m mcp_local.web_search_server
    mcp.run(transport="stdio")

