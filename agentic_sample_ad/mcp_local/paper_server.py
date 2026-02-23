from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mcp.server.fastmcp import FastMCP

from agentic_sample_ad.system_logger import log_event, log_exception

try:
    # Optional dependency. If unavailable, the server still works with filename matching.
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore


mcp = FastMCP("paper-mcp-server", json_response=True)

DB_ROOT = Path(__file__).parent.parent / "db" / "paper"
MAX_RESULT_COUNT = 20
MAX_SCAN_PAGES = 8
MAX_EXTRACTED_CHARS = 20000
MAX_PREVIEW_CHARS = 300
MAX_HEAD_SCAN_PAGES = 4
MAX_HEAD_CONTENT_CHARS = 40000
MAX_FULL_CONTENT_CHARS = 300000

# key: absolute path, value: (mtime, extracted_text)
_TEXT_CACHE: Dict[str, Tuple[float, str]] = {}
_HEAD_TEXT_CACHE: Dict[str, Tuple[float, str]] = {}
_FULL_TEXT_CACHE: Dict[str, Tuple[float, str]] = {}


def _tokenize_query(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9\uAC00-\uD7A3]{2,}", query.lower())
    deduped: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _extract_pdf_text(path: Path, max_pages: int | None, max_chars: int) -> str:
    if PdfReader is None:
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""

    if max_pages is None:
        pages = reader.pages
    else:
        pages = reader.pages[:max_pages]

    parts: List[str] = []
    total = 0
    for page in pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break

    return " ".join(parts)[:max_chars]


def _get_cached_pdf_text(path: Path) -> str:
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""

    cached = _TEXT_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    text = _extract_pdf_text(path=path, max_pages=MAX_SCAN_PAGES, max_chars=MAX_EXTRACTED_CHARS)
    _TEXT_CACHE[key] = (mtime, text)
    return text


def _get_cached_full_pdf_text(path: Path) -> str:
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""

    cached = _FULL_TEXT_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    text = _extract_pdf_text(path=path, max_pages=None, max_chars=MAX_FULL_CONTENT_CHARS)
    _FULL_TEXT_CACHE[key] = (mtime, text)
    return text


def _get_cached_head_pdf_text(path: Path) -> str:
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""

    cached = _HEAD_TEXT_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    text = _extract_pdf_text(path=path, max_pages=MAX_HEAD_SCAN_PAGES, max_chars=MAX_HEAD_CONTENT_CHARS)
    _HEAD_TEXT_CACHE[key] = (mtime, text)
    return text


def _build_preview(text: str, matched_terms: List[str]) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""

    first_pos = -1
    lower = compact.lower()
    for term in matched_terms:
        idx = lower.find(term)
        if idx >= 0 and (first_pos < 0 or idx < first_pos):
            first_pos = idx

    if first_pos < 0:
        return compact[:MAX_PREVIEW_CHARS]

    start = max(0, first_pos - 120)
    end = min(len(compact), start + MAX_PREVIEW_CHARS)
    preview = compact[start:end]
    if start > 0:
        preview = "..." + preview
    if end < len(compact):
        preview = preview + "..."
    return preview


def _build_reason(match_filename: bool, match_content: bool, has_text: bool) -> str:
    if match_filename and not has_text and PdfReader is None:
        return "Matched in filename (content scan unavailable: pypdf not installed)."
    if match_filename and not has_text:
        return "Matched in filename (content extraction unavailable for this file)."
    if match_filename and match_content:
        return "Matched in both filename and extracted content."
    if match_filename:
        return "Matched in filename."
    if match_content:
        return "Matched in extracted content."
    if not has_text and PdfReader is None:
        return "No content scan (pypdf not installed); filename-only mode."
    if not has_text:
        return "Could not extract text from PDF."
    return "Fallback result."


def _score_pdf(path: Path, terms: List[str]) -> Dict[str, Any]:
    filename = path.name
    filename_lower = filename.lower()
    text = _get_cached_pdf_text(path)
    text_lower = text.lower()

    matched_terms: List[str] = []
    score = 0
    matched_in_filename = False
    matched_in_content = False

    for term in terms:
        in_filename = term in filename_lower
        hit_count = text_lower.count(term) if text_lower else 0
        in_content = hit_count > 0

        if in_filename or in_content:
            matched_terms.append(term)

        if in_filename:
            matched_in_filename = True
            score += 6

        if in_content:
            matched_in_content = True
            score += min(hit_count, 5) * 2

    if not terms:
        # Empty query fallback to listing files with low default score.
        score = 1

    return {
        "filename": filename,
        "path": str(path),
        "score": score,
        "matched_terms": matched_terms,
        "match_in_filename": matched_in_filename,
        "match_in_content": matched_in_content,
        "preview": _build_preview(text, matched_terms),
        "reason": _build_reason(matched_in_filename, matched_in_content, bool(text)),
    }


def _resolve_paper_path(path_value: str) -> Path | None:
    raw = path_value.strip()
    if not raw:
        return None

    path = Path(raw)
    try:
        if not path.is_absolute():
            path = (DB_ROOT / path).resolve()
        else:
            path = path.resolve()
    except Exception:
        return None

    try:
        path.relative_to(DB_ROOT.resolve())
    except Exception:
        return None

    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() != ".pdf":
        return None
    return path


@mcp.tool()
def search_papers(query: str) -> List[Dict[str, Any]]:
    """
    Search PDFs under db/paper using filename + extracted text matching.

    Returns ranked candidates with:
    - filename/path
    - score
    - matched terms
    - short preview from extracted text
    """
    log_event(
        "mcp_server.paper",
        "tool_called",
        {"tool": "search_papers", "query": query, "db_root": str(DB_ROOT)},
        direction="inbound",
    )
    if not DB_ROOT.exists():
        log_event(
            "mcp_server.paper",
            "tool_completed",
            {"tool": "search_papers", "query": query, "result_count": 0, "reason": "db_root_missing"},
            direction="outbound",
        )
        return []

    try:
        terms = _tokenize_query(query)
        results: List[Dict[str, Any]] = []

        for path in DB_ROOT.rglob("*.pdf"):
            item = _score_pdf(path, terms)
            if terms and item["score"] <= 0:
                continue
            results.append(item)

        results.sort(key=lambda x: (int(x.get("score", 0)), x.get("filename", "")), reverse=True)
        top_results = results[:MAX_RESULT_COUNT]
        log_event(
            "mcp_server.paper",
            "tool_completed",
            {
                "tool": "search_papers",
                "query": query,
                "terms": terms,
                "result_count": len(top_results),
                "top_filenames": [str(item.get("filename", "")) for item in top_results[:5]],
            },
            direction="outbound",
        )
        return top_results
    except Exception as e:
        log_exception(
            "mcp_server.paper",
            "tool_failed",
            e,
            {"tool": "search_papers", "query": query},
        )
        raise


@mcp.tool()
def get_paper_content(path: str, max_chars: int = 120000) -> Dict[str, Any]:
    """
    Return extracted paper content for one PDF path.

    - path must point to a file under db/paper.
    - Uses cached full-text extraction for faster follow-up calls.
    """
    safe_max_chars = max(1000, min(int(max_chars), MAX_FULL_CONTENT_CHARS))
    log_event(
        "mcp_server.paper",
        "tool_called",
        {"tool": "get_paper_content", "path": path, "max_chars": safe_max_chars},
        direction="inbound",
    )
    try:
        if not DB_ROOT.exists():
            payload = {
                "ok": False,
                "error": "db_root_missing",
                "path": path,
                "max_chars": safe_max_chars,
            }
            log_event("mcp_server.paper", "tool_completed", payload, direction="outbound")
            return payload

        resolved = _resolve_paper_path(path)
        if resolved is None:
            payload = {
                "ok": False,
                "error": "invalid_path",
                "path": path,
                "max_chars": safe_max_chars,
            }
            log_event("mcp_server.paper", "tool_completed", payload, direction="outbound")
            return payload

        full_text = _get_cached_full_pdf_text(resolved)
        content = full_text[:safe_max_chars]
        payload = {
            "ok": bool(content),
            "path": str(resolved),
            "filename": resolved.name,
            "char_count": len(content),
            "full_char_count": len(full_text),
            "truncated": len(full_text) > len(content),
            "content": content,
        }
        if not content:
            if PdfReader is None:
                payload["error"] = "pypdf_not_installed_or_extract_failed"
            else:
                payload["error"] = "extract_failed_or_empty"

        log_event(
            "mcp_server.paper",
            "tool_completed",
            {
                "tool": "get_paper_content",
                "path": str(resolved),
                "ok": bool(payload.get("ok")),
                "char_count": int(payload.get("char_count", 0)),
                "full_char_count": int(payload.get("full_char_count", 0)),
                "truncated": bool(payload.get("truncated", False)),
            },
            direction="outbound",
        )
        return payload
    except Exception as e:
        log_exception(
            "mcp_server.paper",
            "tool_failed",
            e,
            {"tool": "get_paper_content", "path": path, "max_chars": safe_max_chars},
        )
        raise


@mcp.tool()
def get_paper_head(path: str, max_chars: int = 12000) -> Dict[str, Any]:
    """
    Return head section text (first pages only) for one PDF path.

    - path must point to a file under db/paper.
    - Uses cached head extraction (first MAX_HEAD_SCAN_PAGES pages).
    """
    safe_max_chars = max(1000, min(int(max_chars), MAX_HEAD_CONTENT_CHARS))
    log_event(
        "mcp_server.paper",
        "tool_called",
        {"tool": "get_paper_head", "path": path, "max_chars": safe_max_chars},
        direction="inbound",
    )
    try:
        if not DB_ROOT.exists():
            payload = {
                "ok": False,
                "error": "db_root_missing",
                "path": path,
                "max_chars": safe_max_chars,
            }
            log_event("mcp_server.paper", "tool_completed", payload, direction="outbound")
            return payload

        resolved = _resolve_paper_path(path)
        if resolved is None:
            payload = {
                "ok": False,
                "error": "invalid_path",
                "path": path,
                "max_chars": safe_max_chars,
            }
            log_event("mcp_server.paper", "tool_completed", payload, direction="outbound")
            return payload

        head_text = _get_cached_head_pdf_text(resolved)
        content = head_text[:safe_max_chars]
        payload = {
            "ok": bool(content),
            "path": str(resolved),
            "filename": resolved.name,
            "char_count": len(content),
            "head_full_char_count": len(head_text),
            "head_max_pages": MAX_HEAD_SCAN_PAGES,
            "truncated": len(head_text) > len(content),
            "content": content,
        }
        if not content:
            if PdfReader is None:
                payload["error"] = "pypdf_not_installed_or_extract_failed"
            else:
                payload["error"] = "extract_failed_or_empty"

        log_event(
            "mcp_server.paper",
            "tool_completed",
            {
                "tool": "get_paper_head",
                "path": str(resolved),
                "ok": bool(payload.get("ok")),
                "char_count": int(payload.get("char_count", 0)),
                "head_full_char_count": int(payload.get("head_full_char_count", 0)),
                "truncated": bool(payload.get("truncated", False)),
            },
            direction="outbound",
        )
        return payload
    except Exception as e:
        log_exception(
            "mcp_server.paper",
            "tool_failed",
            e,
            {"tool": "get_paper_head", "path": path, "max_chars": safe_max_chars},
        )
        raise


if __name__ == "__main__":
    mcp.run(transport="stdio")

