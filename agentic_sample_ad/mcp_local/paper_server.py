from __future__ import annotations

import html as html_lib
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import httpx
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
MAX_EXTERNAL_METADATA_CHARS = 40000
EXTERNAL_FETCH_TIMEOUT_SEC = 20.0
HTTP_USER_AGENT = "agentic-sample-paper-agent/1.0"

# key: absolute path, value: (mtime, extracted_text)
_TEXT_CACHE: Dict[str, Tuple[float, str]] = {}
_HEAD_TEXT_CACHE: Dict[str, Tuple[float, str]] = {}
_FULL_TEXT_CACHE: Dict[str, Tuple[float, str]] = {}
ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)", flags=re.IGNORECASE)
DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", flags=re.IGNORECASE)


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


def _clean_text(text: str, max_chars: int | None = None) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars is not None and len(compact) > max_chars:
        return compact[:max_chars]
    return compact


def _strip_markup(text: str, max_chars: int | None = None) -> str:
    unescaped = html_lib.unescape(str(text or ""))
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    return _clean_text(stripped, max_chars=max_chars)


def _extract_url_from_text(text: str) -> str:
    match = re.search(r"https?://[^\s<>\"]+", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(0).rstrip(").,]")


def _extract_doi_from_text(text: str) -> str:
    match = DOI_RE.search(str(text or ""))
    if not match:
        return ""
    return match.group(1).strip().rstrip(").,;")


def _extract_arxiv_id_from_text(text: str) -> str:
    match = ARXIV_ID_RE.search(str(text or ""))
    if not match:
        return ""
    return str(match.group("id") or "").strip()


def _http_client() -> httpx.Client:
    return httpx.Client(
        timeout=EXTERNAL_FETCH_TIMEOUT_SEC,
        follow_redirects=True,
        headers={"User-Agent": HTTP_USER_AGENT},
    )


def _extract_pdf_text_from_bytes(data: bytes, max_pages: int = 4, max_chars: int = MAX_EXTERNAL_METADATA_CHARS) -> str:
    if PdfReader is None or not data:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        return ""
    parts: List[str] = []
    total = 0
    for page in reader.pages[:max_pages]:
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
    return _clean_text(" ".join(parts), max_chars=max_chars)


def _meta_value(html_text: str, *patterns: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _strip_markup(match.group(1), max_chars=MAX_EXTERNAL_METADATA_CHARS)
    return ""


def _extract_title_from_html(html_text: str) -> str:
    title = _meta_value(
        html_text,
        r'<meta[^>]+name=["\']citation_title["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        r"<title>(.*?)</title>",
    )
    return _clean_text(title, max_chars=240)


def _crossref_date(message: Dict[str, Any]) -> str:
    for key in ["published-print", "published-online", "created", "issued"]:
        value = message.get(key)
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts")
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            continue
        date_bits = [str(item) for item in parts[0] if item]
        if date_bits:
            return "-".join(date_bits[:3])
    return ""


def _fetch_crossref_record(doi: str, max_chars: int) -> Dict[str, Any]:
    with _http_client() as client:
        response = client.get(f"https://api.crossref.org/works/{quote(doi, safe='')}")
        response.raise_for_status()
        message = response.json().get("message", {})

    title_values = message.get("title") or []
    title = _clean_text(title_values[0] if title_values else "", max_chars=240)
    authors: List[str] = []
    for item in message.get("author", []) or []:
        if not isinstance(item, dict):
            continue
        name = _clean_text(" ".join([str(item.get("given", "")).strip(), str(item.get("family", "")).strip()]), max_chars=80)
        if name:
            authors.append(name)
        if len(authors) >= 12:
            break
    abstract = _strip_markup(str(message.get("abstract", "") or ""), max_chars=max_chars)
    venue_values = message.get("container-title") or []
    venue = _clean_text(venue_values[0] if venue_values else "", max_chars=180)
    links = [str(item.get("URL", "")).strip() for item in message.get("link", []) if isinstance(item, dict) and str(item.get("URL", "")).strip()]
    return {
        "ok": True,
        "reference_type": "doi",
        "doi": doi,
        "title": title,
        "authors": authors,
        "published": _crossref_date(message),
        "venue": venue,
        "abstract": abstract,
        "resolved_url": _clean_text(str(message.get("URL", "") or ""), max_chars=320),
        "candidate_urls": links[:6],
    }


def _fetch_arxiv_record(arxiv_id: str, max_chars: int) -> Dict[str, Any]:
    api_url = f"http://export.arxiv.org/api/query?id_list={quote(arxiv_id, safe='')}"
    with _http_client() as client:
        response = client.get(api_url)
        response.raise_for_status()
        xml_text = response.text
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return {"ok": False, "error": "arxiv_entry_not_found", "arxiv_id": arxiv_id}

    def _entry_text(path: str, max_len: int) -> str:
        node = entry.find(path, ns)
        return _clean_text(node.text if node is not None else "", max_chars=max_len)

    authors = [
        _clean_text(author.findtext("atom:name", default="", namespaces=ns), max_chars=80)
        for author in entry.findall("atom:author", ns)
        if _clean_text(author.findtext("atom:name", default="", namespaces=ns), max_chars=80)
    ][:12]
    links: List[str] = []
    pdf_url = ""
    for link in entry.findall("atom:link", ns):
        href = _clean_text(str(link.attrib.get("href", "")).strip(), max_chars=320)
        rel = str(link.attrib.get("rel", "")).strip().lower()
        link_type = str(link.attrib.get("type", "")).strip().lower()
        if href:
            links.append(href)
        if (link_type == "application/pdf" or href.endswith(".pdf")) and not pdf_url:
            pdf_url = href
        if rel == "alternate" and not pdf_url and href.endswith(".pdf"):
            pdf_url = href

    return {
        "ok": True,
        "reference_type": "arxiv",
        "arxiv_id": arxiv_id,
        "title": _entry_text("atom:title", 240),
        "authors": authors,
        "published": _entry_text("atom:published", 40),
        "abstract": _entry_text("atom:summary", max_chars),
        "resolved_url": _entry_text("atom:id", 320),
        "pdf_url": pdf_url,
        "candidate_urls": links[:6],
    }


def _fetch_generic_url_record(url: str, max_chars: int) -> Dict[str, Any]:
    with _http_client() as client:
        response = client.get(url)
        response.raise_for_status()
        resolved_url = str(response.url)
        content_type = str(response.headers.get("content-type", "")).lower()
        body = response.content

    if "application/pdf" in content_type or resolved_url.lower().endswith(".pdf"):
        content = _extract_pdf_text_from_bytes(body, max_pages=4, max_chars=max_chars)
        title = ""
        if content:
            first_line = content.split(". ")[0]
            title = _clean_text(first_line, max_chars=240)
        return {
            "ok": bool(content),
            "reference_type": "pdf_url",
            "title": title or Path(resolved_url).name,
            "resolved_url": resolved_url,
            "content": content,
        }

    html_text = body.decode(response.encoding or "utf-8", errors="ignore")
    doi = (
        _meta_value(
            html_text,
            r'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+name=["\']dc\.identifier["\'][^>]+content=["\'](.*?)["\']',
        )
        or _extract_doi_from_text(html_text)
    )
    arxiv_id = _extract_arxiv_id_from_text(resolved_url) or _extract_arxiv_id_from_text(html_text)

    if arxiv_id:
        record = _fetch_arxiv_record(arxiv_id, max_chars=max_chars)
        record["source_url"] = resolved_url
        return record

    if doi:
        record = _fetch_crossref_record(doi, max_chars=max_chars)
        record["source_url"] = resolved_url
        return record

    description = _meta_value(
        html_text,
        r'<meta[^>]+name=["\']citation_abstract["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
    )
    return {
        "ok": True,
        "reference_type": "url",
        "title": _extract_title_from_html(html_text) or Path(resolved_url).name,
        "resolved_url": resolved_url,
        "abstract": _clean_text(description, max_chars=max_chars),
    }


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


@mcp.tool()
def fetch_external_paper(
    reference: str = "",
    url: str = "",
    doi: str = "",
    arxiv_id: str = "",
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """
    Fetch external paper metadata or a compact content preview from URL / DOI / arXiv ID.

    Intended for workflow handoff artifacts discovered by other agents.
    """
    safe_reference = _clean_text(reference, max_chars=600)
    safe_url = _clean_text(url, max_chars=600)
    safe_doi = _clean_text(doi, max_chars=220)
    safe_arxiv_id = _clean_text(arxiv_id, max_chars=80)
    safe_max_chars = max(1000, min(int(max_chars), MAX_EXTERNAL_METADATA_CHARS))
    log_event(
        "mcp_server.paper",
        "tool_called",
        {
            "tool": "fetch_external_paper",
            "reference": safe_reference,
            "url": safe_url,
            "doi": safe_doi,
            "arxiv_id": safe_arxiv_id,
            "max_chars": safe_max_chars,
        },
        direction="inbound",
    )
    try:
        resolved_url = safe_url or _extract_url_from_text(safe_reference)
        resolved_doi = safe_doi or _extract_doi_from_text(safe_reference)
        resolved_arxiv_id = (
            safe_arxiv_id
            or _extract_arxiv_id_from_text(safe_reference)
            or _extract_arxiv_id_from_text(resolved_url)
        )

        if resolved_arxiv_id:
            payload = _fetch_arxiv_record(resolved_arxiv_id, max_chars=safe_max_chars)
        elif resolved_doi:
            payload = _fetch_crossref_record(resolved_doi, max_chars=safe_max_chars)
        elif resolved_url:
            payload = _fetch_generic_url_record(resolved_url, max_chars=safe_max_chars)
        else:
            payload = {
                "ok": False,
                "error": "missing_reference",
            }

        payload["input_reference"] = safe_reference
        if resolved_url and "source_url" not in payload:
            payload["source_url"] = resolved_url
        if resolved_doi and not str(payload.get("doi", "")).strip():
            payload["doi"] = resolved_doi
        if resolved_arxiv_id and not str(payload.get("arxiv_id", "")).strip():
            payload["arxiv_id"] = resolved_arxiv_id

        log_event(
            "mcp_server.paper",
            "tool_completed",
            {
                "tool": "fetch_external_paper",
                "ok": bool(payload.get("ok")),
                "reference_type": str(payload.get("reference_type", "")),
                "title": str(payload.get("title", ""))[:160],
                "doi": str(payload.get("doi", ""))[:120],
                "arxiv_id": str(payload.get("arxiv_id", ""))[:40],
                "resolved_url": str(payload.get("resolved_url", "") or payload.get("source_url", ""))[:220],
            },
            direction="outbound",
        )
        return payload
    except Exception as e:
        log_exception(
            "mcp_server.paper",
            "tool_failed",
            e,
            {
                "tool": "fetch_external_paper",
                "reference": safe_reference,
                "url": safe_url,
                "doi": safe_doi,
                "arxiv_id": safe_arxiv_id,
                "max_chars": safe_max_chars,
            },
        )
        raise


if __name__ == "__main__":
    mcp.run(transport="stdio")

