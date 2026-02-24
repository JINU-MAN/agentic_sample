from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from google.adk.agents import LlmAgent

from agentic_sample_ad.mcp_local.client import call_mcp_tool
from agentic_sample_ad.system_logger import log_event, log_exception


BASE_DIR = Path(__file__).parent.parent
PAPER_MCP_SERVER = BASE_DIR / "mcp_local" / "paper_server.py"
MAX_MEMORY_PAPERS = 5
MAX_MEMORY_CHARS_PER_PAPER = 120000
MAX_MEMORY_SNIPPETS = 8
MAX_OVERVIEW_CHARS = 3500
MAX_HEAD_FETCH_CHARS = 16000
MAX_FULL_EXPAND_PAPERS = 3
DEFAULT_WORKFLOW_ID = "default"

# workflow_id -> memory payload
_PAPER_MEMORY: Dict[str, Dict[str, Any]] = {}


def _normalize_workflow_id(workflow_id: str | None) -> str:
    token = str(workflow_id or "").strip()
    if not token:
        return DEFAULT_WORKFLOW_ID
    return token[:80]


def _extract_error_text(raw: Dict[str, Any]) -> str:
    messages: List[str] = []
    for item in raw.get("content", []):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            messages.append(text.strip())
    return "\n".join(messages).strip()


def _extract_mcp_result(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize MCP response shape into a list of paper dicts.
    """
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


def _extract_mcp_object_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    if raw.get("isError"):
        message = _extract_error_text(raw) or "MCP tool returned an error."
        return {"ok": False, "error": message}

    structured = raw.get("structuredContent", {})
    if isinstance(structured, dict):
        result = structured.get("result")
        if isinstance(result, dict):
            return result

    content = raw.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
    return {}


def _tokenize_text(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9\uAC00-\uD7A3]{2,}", str(text).lower())
    seen: set[str] = set()
    deduped: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _build_snippet(content: str, position: int, width: int = 450) -> str:
    if not content:
        return ""
    start = max(0, position - (width // 2))
    end = min(len(content), start + width)
    snippet = content[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return " ".join(snippet.split())


def _compact_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _fallback_title_from_filename(filename: str) -> str:
    token = Path(str(filename or "").strip()).stem
    token = re.sub(r"[_\-]+", " ", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token[:180] if token else "Untitled paper"


def _extract_title_from_head(head_text: str, fallback_filename: str) -> str:
    for raw_line in str(head_text or "").splitlines()[:30]:
        line = " ".join(raw_line.split()).strip()
        lower = line.lower()
        if len(line) < 10:
            continue
        if len(line) > 220:
            continue
        if lower.startswith(("abstract", "introduction", "keywords", "index terms")):
            continue
        if line.count("@") >= 1:
            continue
        return line
    return _fallback_title_from_filename(fallback_filename)


def _extract_section_by_heading(
    head_text: str,
    heading_patterns: List[str],
    stop_patterns: List[str],
    max_chars: int = 1200,
) -> str:
    text = str(head_text or "")
    if not text.strip():
        return ""

    start_idx = -1
    for pattern in heading_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            start_idx = match.end()
            break
    if start_idx < 0:
        return ""

    tail = text[start_idx:]
    end_idx = len(tail)
    for pattern in stop_patterns:
        match = re.search(pattern, tail, flags=re.IGNORECASE | re.MULTILINE)
        if match and match.start() > 20:
            end_idx = min(end_idx, match.start())

    return _compact_text(tail[:end_idx], max_chars=max_chars)


def _build_overview_text(
    *,
    filename: str,
    head_text: str,
    preview: str,
    reason: str,
    matched_terms: List[str],
) -> str:
    title = _extract_title_from_head(head_text, filename)
    abstract = _extract_section_by_heading(
        head_text=head_text,
        heading_patterns=[
            r"(?im)^\s*(?:\d+[\.\)]\s*)?abstract\b[:\s-]*",
            r"(?i)\babstract\b[:\s-]*",
        ],
        stop_patterns=[
            r"(?im)^\s*(?:\d+[\.\)]\s*)?(keywords?|index terms?)\b",
            r"(?im)^\s*(?:\d+[\.\)]\s*)?introduction\b",
            r"(?im)^\s*(?:\d+[\.\)]\s*)?(background|related work|methods?)\b",
        ],
        max_chars=1400,
    )
    introduction = _extract_section_by_heading(
        head_text=head_text,
        heading_patterns=[
            r"(?im)^\s*(?:\d+[\.\)]\s*)?introduction\b[:\s-]*",
            r"(?i)\bintroduction\b[:\s-]*",
        ],
        stop_patterns=[
            r"(?im)^\s*(?:\d+[\.\)]\s*)?(background|related work|methods?|materials|experiments?|results|conclusion)\b",
        ],
        max_chars=1500,
    )

    if not abstract:
        abstract = _compact_text(preview, 800)
    if not introduction:
        # fallback to early head chunk if explicit introduction heading is missing.
        introduction = _compact_text(head_text, 900)

    matched_text = ", ".join(token for token in matched_terms if token)[:240]
    lines: List[str] = [f"Title: {title}"]
    if abstract:
        lines.append(f"Abstract: {abstract}")
    if introduction:
        lines.append(f"Introduction: {introduction}")
    if reason:
        lines.append(f"Relevance: {_compact_text(reason, 220)}")
    if matched_text:
        lines.append(f"Matched terms: {matched_text}")
    return _compact_text("\n".join(lines), MAX_OVERVIEW_CHARS)


def _memory_source_text(paper: Dict[str, Any]) -> str:
    if bool(paper.get("content_loaded")) and str(paper.get("content", "")).strip():
        return str(paper.get("content", ""))
    overview = str(paper.get("overview", "")).strip()
    if overview:
        return overview
    return str(paper.get("content", ""))


def _is_detail_heavy_question(question: str) -> bool:
    lower = str(question or "").lower()
    detail_terms = [
        "method",
        "methodology",
        "experiment",
        "results",
        "limitation",
        "equation",
        "algorithm",
        "dataset",
        "implementation",
        "details",
        "세부",
        "방법",
        "실험",
        "결과",
        "한계",
        "수식",
    ]
    return len(lower) >= 90 or any(term in lower for term in detail_terms)


def _score_paper_for_question(paper: Dict[str, Any], terms: List[str]) -> int:
    base_score = int(paper.get("score", 0))
    if not terms:
        return base_score

    source = _memory_source_text(paper).lower()
    filename = str(paper.get("filename", "")).lower()
    matched_terms = [str(item).lower() for item in paper.get("matched_terms", []) if str(item).strip()]

    score = base_score
    for term in terms:
        if term in filename:
            score += 6
        if term in matched_terms:
            score += 4
        count = source.count(term)
        if count > 0:
            score += min(count, 8) * 2
    return score


def scrape_papers_with_mcp(query: str) -> str:
    """
    Search PDFs with MCP paper server and return normalized JSON text.
    """
    safe_query = query.strip()
    log_event(
        "tool.scrape_papers_with_mcp",
        "call_started",
        {"query": safe_query, "server_script_path": str(PAPER_MCP_SERVER)},
        direction="outbound",
    )
    try:
        raw = call_mcp_tool(
            server_script_path=str(PAPER_MCP_SERVER),
            tool_name="search_papers",
            arguments={"query": safe_query},
        )
        normalized = _extract_mcp_result(raw)
        result_text = json.dumps(normalized, ensure_ascii=False, indent=2)
        log_event(
            "tool.scrape_papers_with_mcp",
            "call_completed",
            {"query": safe_query, "result_count": len(normalized)},
            direction="inbound",
        )
        return result_text
    except Exception as e:
        log_exception(
            "tool.scrape_papers_with_mcp",
            "call_failed",
            e,
            {"query": safe_query, "server_script_path": str(PAPER_MCP_SERVER)},
        )
        raise


def load_paper_memory_with_mcp(
    query: str,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    max_papers: int = 3,
    max_chars_per_paper: int = MAX_MEMORY_CHARS_PER_PAPER,
    load_mode: str = "overview",
) -> str:
    """
    Load workflow-scoped paper memory.

    - overview mode (default): store compact overview from title/abstract/introduction-like head text.
    - full mode: also store full body text for detailed QA.
    """
    safe_query = query.strip()
    safe_workflow_id = _normalize_workflow_id(workflow_id)
    safe_max_papers = max(1, min(int(max_papers), MAX_MEMORY_PAPERS))
    safe_max_chars = max(2000, min(int(max_chars_per_paper), MAX_MEMORY_CHARS_PER_PAPER))
    safe_load_mode = str(load_mode or "overview").strip().lower()
    if safe_load_mode not in {"overview", "full"}:
        safe_load_mode = "overview"
    safe_head_chars = max(2000, min(safe_max_chars, MAX_HEAD_FETCH_CHARS))
    log_event(
        "tool.load_paper_memory_with_mcp",
        "call_started",
        {
            "query": safe_query,
            "workflow_id": safe_workflow_id,
            "max_papers": safe_max_papers,
            "max_chars_per_paper": safe_max_chars,
            "load_mode": safe_load_mode,
            "head_fetch_chars": safe_head_chars,
            "server_script_path": str(PAPER_MCP_SERVER),
        },
        direction="outbound",
    )

    try:
        raw_search = call_mcp_tool(
            server_script_path=str(PAPER_MCP_SERVER),
            tool_name="search_papers",
            arguments={"query": safe_query},
        )
        candidates = _extract_mcp_result(raw_search)
        if candidates and isinstance(candidates[0], dict) and candidates[0].get("error"):
            payload = {
                "ok": False,
                "workflow_id": safe_workflow_id,
                "query": safe_query,
                "loaded_count": 0,
                "error": str(candidates[0].get("error", "search_error")),
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        ranked = [item for item in candidates if isinstance(item, dict)]
        ranked.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
        selected = ranked[:safe_max_papers]

        loaded_papers: List[Dict[str, Any]] = []
        for item in selected:
            paper_path = str(item.get("path", "")).strip()
            if not paper_path:
                continue
            filename_hint = str(item.get("filename", "")).strip()
            matched_terms = [str(x).strip() for x in item.get("matched_terms", []) if str(x).strip()]
            reason = str(item.get("reason", "")).strip()
            preview = str(item.get("preview", "")).strip()

            content = ""
            content_loaded = False
            head_text = ""
            char_count = 0
            full_char_count = 0
            resolved_path = paper_path

            if safe_load_mode == "full":
                raw_content = call_mcp_tool(
                    server_script_path=str(PAPER_MCP_SERVER),
                    tool_name="get_paper_content",
                    arguments={"path": paper_path, "max_chars": safe_max_chars},
                )
                content_obj = _extract_mcp_object_result(raw_content)
                if not bool(content_obj.get("ok")):
                    continue
                content = str(content_obj.get("content", "") or "")
                if not content.strip():
                    continue
                head_text = content[:safe_head_chars]
                content_loaded = True
                char_count = int(content_obj.get("char_count", len(content)))
                full_char_count = int(content_obj.get("full_char_count", len(content)))
                filename_hint = str(content_obj.get("filename", "") or filename_hint).strip()
                resolved_path = str(content_obj.get("path", "") or paper_path).strip()
            else:
                raw_head = call_mcp_tool(
                    server_script_path=str(PAPER_MCP_SERVER),
                    tool_name="get_paper_head",
                    arguments={"path": paper_path, "max_chars": safe_head_chars},
                )
                head_obj = _extract_mcp_object_result(raw_head)
                if not bool(head_obj.get("ok")) and not preview:
                    continue
                head_text = str(head_obj.get("content", "") or "")
                char_count = len(head_text)
                full_char_count = int(head_obj.get("head_full_char_count", len(head_text)))
                filename_hint = str(head_obj.get("filename", "") or filename_hint).strip()
                resolved_path = str(head_obj.get("path", "") or paper_path).strip()

            overview = _build_overview_text(
                filename=filename_hint,
                head_text=head_text,
                preview=preview,
                reason=reason,
                matched_terms=matched_terms,
            )

            loaded_papers.append(
                {
                    "filename": filename_hint,
                    "path": resolved_path,
                    "score": int(item.get("score", 0)),
                    "matched_terms": matched_terms,
                    "reason": reason,
                    "char_count": int(char_count),
                    "full_char_count": int(full_char_count),
                    "overview": overview,
                    "overview_char_count": len(overview),
                    "head_char_count": len(head_text),
                    "content": content,
                    "content_loaded": content_loaded,
                    "load_mode": "full" if content_loaded else "overview",
                }
            )

        full_loaded_count = sum(1 for item in loaded_papers if bool(item.get("content_loaded")))
        memory_payload = {
            "workflow_id": safe_workflow_id,
            "query": safe_query,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "papers": loaded_papers,
            "loaded_count": len(loaded_papers),
            "full_loaded_count": full_loaded_count,
            "load_mode": safe_load_mode,
            "total_chars": sum(int(item.get("char_count", 0)) for item in loaded_papers),
        }
        _PAPER_MEMORY[safe_workflow_id] = memory_payload

        result_payload = {
            "ok": bool(loaded_papers),
            "workflow_id": safe_workflow_id,
            "query": safe_query,
            "load_mode": safe_load_mode,
            "loaded_count": len(loaded_papers),
            "full_loaded_count": full_loaded_count,
            "papers": [
                {
                    "filename": str(item.get("filename", "")),
                    "path": str(item.get("path", "")),
                    "score": int(item.get("score", 0)),
                    "char_count": int(item.get("char_count", 0)),
                    "full_char_count": int(item.get("full_char_count", 0)),
                    "memory_mode": str(item.get("load_mode", "overview")),
                    "preview": _build_snippet(_memory_source_text(item), 0, width=280),
                }
                for item in loaded_papers
            ],
        }
        result_text = json.dumps(result_payload, ensure_ascii=False, indent=2)
        log_event(
            "tool.load_paper_memory_with_mcp",
            "call_completed",
            {
                "workflow_id": safe_workflow_id,
                "query": safe_query,
                "loaded_count": len(loaded_papers),
                "full_loaded_count": full_loaded_count,
                "load_mode": safe_load_mode,
                "total_chars": int(memory_payload.get("total_chars", 0)),
            },
            direction="inbound",
        )
        return result_text
    except Exception as e:
        log_exception(
            "tool.load_paper_memory_with_mcp",
            "call_failed",
            e,
            {
                "query": safe_query,
                "workflow_id": safe_workflow_id,
                "max_papers": safe_max_papers,
                "max_chars_per_paper": safe_max_chars,
            },
        )
        raise


def expand_paper_memory_with_mcp(
    question: str,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    max_papers: int = 2,
    max_chars_per_paper: int = MAX_MEMORY_CHARS_PER_PAPER,
) -> str:
    """
    Lazily expand workflow paper memory with full text for the most relevant papers.
    """
    safe_question = question.strip()
    safe_workflow_id = _normalize_workflow_id(workflow_id)
    safe_max_papers = max(1, min(int(max_papers), MAX_FULL_EXPAND_PAPERS))
    safe_max_chars = max(2000, min(int(max_chars_per_paper), MAX_MEMORY_CHARS_PER_PAPER))
    log_event(
        "tool.expand_paper_memory_with_mcp",
        "call_started",
        {
            "workflow_id": safe_workflow_id,
            "question": safe_question,
            "max_papers": safe_max_papers,
            "max_chars_per_paper": safe_max_chars,
            "server_script_path": str(PAPER_MCP_SERVER),
        },
        direction="outbound",
    )
    try:
        memory = _PAPER_MEMORY.get(safe_workflow_id)
        if not memory or not isinstance(memory.get("papers"), list):
            payload = {
                "ok": False,
                "workflow_id": safe_workflow_id,
                "error": "memory_empty",
                "message": "No loaded paper memory. Call load_paper_memory_with_mcp first.",
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        papers = [item for item in memory.get("papers", []) if isinstance(item, dict)]
        terms = _tokenize_text(safe_question)
        candidates: List[Dict[str, Any]] = []
        for paper in papers:
            if bool(paper.get("content_loaded")):
                continue
            path = str(paper.get("path", "")).strip()
            if not path:
                continue
            candidates.append(
                {
                    "paper": paper,
                    "score": _score_paper_for_question(paper, terms),
                }
            )

        candidates.sort(key=lambda item: int(item.get("score", 0)), reverse=True)
        selected = candidates[:safe_max_papers]

        expanded: List[Dict[str, Any]] = []
        for candidate in selected:
            paper = candidate.get("paper", {})
            path = str(paper.get("path", "")).strip()
            if not path:
                continue

            raw_content = call_mcp_tool(
                server_script_path=str(PAPER_MCP_SERVER),
                tool_name="get_paper_content",
                arguments={"path": path, "max_chars": safe_max_chars},
            )
            content_obj = _extract_mcp_object_result(raw_content)
            if not bool(content_obj.get("ok")):
                continue

            content = str(content_obj.get("content", "") or "")
            if not content.strip():
                continue

            paper["content"] = content
            paper["content_loaded"] = True
            paper["load_mode"] = "full"
            paper["char_count"] = int(content_obj.get("char_count", len(content)))
            paper["full_char_count"] = int(content_obj.get("full_char_count", len(content)))
            paper["filename"] = str(content_obj.get("filename", "") or paper.get("filename", "")).strip()
            paper["path"] = str(content_obj.get("path", "") or paper.get("path", "")).strip()

            expanded.append(
                {
                    "filename": str(paper.get("filename", "")),
                    "path": str(paper.get("path", "")),
                    "char_count": int(paper.get("char_count", 0)),
                    "full_char_count": int(paper.get("full_char_count", 0)),
                }
            )

        memory["papers"] = papers
        memory["full_loaded_count"] = sum(1 for item in papers if bool(item.get("content_loaded")))
        memory["total_chars"] = sum(int(item.get("char_count", 0)) for item in papers)
        memory["updated_at"] = datetime.now(timezone.utc).isoformat()
        _PAPER_MEMORY[safe_workflow_id] = memory

        payload = {
            "ok": bool(expanded),
            "workflow_id": safe_workflow_id,
            "question": safe_question,
            "expanded_count": len(expanded),
            "full_loaded_count": int(memory.get("full_loaded_count", 0)),
            "expanded_papers": expanded,
        }
        result_text = json.dumps(payload, ensure_ascii=False, indent=2)
        log_event(
            "tool.expand_paper_memory_with_mcp",
            "call_completed",
            {
                "workflow_id": safe_workflow_id,
                "question": safe_question,
                "expanded_count": len(expanded),
                "full_loaded_count": int(memory.get("full_loaded_count", 0)),
            },
            direction="inbound",
        )
        return result_text
    except Exception as e:
        log_exception(
            "tool.expand_paper_memory_with_mcp",
            "call_failed",
            e,
            {
                "workflow_id": safe_workflow_id,
                "question": safe_question,
                "max_papers": safe_max_papers,
            },
        )
        raise


def query_paper_memory(
    question: str,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    max_snippets: int = MAX_MEMORY_SNIPPETS,
) -> str:
    """
    Query loaded paper memory and return the most relevant snippets quickly.
    """
    safe_question = question.strip()
    safe_workflow_id = _normalize_workflow_id(workflow_id)
    safe_max_snippets = max(1, min(int(max_snippets), MAX_MEMORY_SNIPPETS))
    log_event(
        "tool.query_paper_memory",
        "call_started",
        {
            "workflow_id": safe_workflow_id,
            "question": safe_question,
            "max_snippets": safe_max_snippets,
        },
        direction="outbound",
    )
    try:
        memory = _PAPER_MEMORY.get(safe_workflow_id)
        if not memory or not isinstance(memory.get("papers"), list):
            payload = {
                "ok": False,
                "workflow_id": safe_workflow_id,
                "error": "memory_empty",
                "message": "No loaded paper memory. Call load_paper_memory_with_mcp first.",
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        terms = _tokenize_text(safe_question)
        snippet_candidates: List[Dict[str, Any]] = []
        for paper in memory.get("papers", []):
            if not isinstance(paper, dict):
                continue
            filename = str(paper.get("filename", "")).strip()
            content = _memory_source_text(paper)
            if not content:
                continue
            memory_mode = "full" if bool(paper.get("content_loaded")) else "overview"
            lower = content.lower()

            if not terms:
                snippet_candidates.append(
                    {
                        "filename": filename,
                        "path": str(paper.get("path", "")),
                        "score": 1,
                        "memory_mode": memory_mode,
                        "snippet": _build_snippet(content, 0),
                    }
                )
                continue

            found_any = False
            for term in terms:
                idx = lower.find(term)
                if idx < 0:
                    continue
                found_any = True
                score = max(1, lower.count(term))
                snippet_candidates.append(
                    {
                        "filename": filename,
                        "path": str(paper.get("path", "")),
                        "score": score,
                        "term": term,
                        "memory_mode": memory_mode,
                        "snippet": _build_snippet(content, idx, width=450 if memory_mode == "full" else 300),
                    }
                )
            if not found_any:
                snippet_candidates.append(
                    {
                        "filename": filename,
                        "path": str(paper.get("path", "")),
                        "score": 0,
                        "term": "",
                        "memory_mode": memory_mode,
                        "snippet": _build_snippet(content, 0, width=450 if memory_mode == "full" else 300),
                    }
                )

        snippet_candidates.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
        picked: List[Dict[str, Any]] = []
        seen_snippets: set[str] = set()
        for item in snippet_candidates:
            snippet = str(item.get("snippet", "")).strip()
            if not snippet:
                continue
            key = f"{str(item.get('filename', ''))}:{snippet[:120]}"
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            picked.append(
                {
                    "filename": str(item.get("filename", "")),
                    "path": str(item.get("path", "")),
                    "score": int(item.get("score", 0)),
                    "matched_term": str(item.get("term", "")),
                    "memory_mode": str(item.get("memory_mode", "overview")),
                    "snippet": snippet,
                }
            )
            if len(picked) >= safe_max_snippets:
                break

        full_loaded_count = int(memory.get("full_loaded_count", 0))
        loaded_count = int(memory.get("loaded_count", 0))
        overview_only_count = max(0, loaded_count - full_loaded_count)
        recommend_expand = _is_detail_heavy_question(safe_question) and (
            overview_only_count > 0 and any(str(item.get("memory_mode", "")) == "overview" for item in picked)
        )

        payload = {
            "ok": True,
            "workflow_id": safe_workflow_id,
            "question": safe_question,
            "terms": terms,
            "memory_loaded_count": loaded_count,
            "memory_full_loaded_count": full_loaded_count,
            "memory_overview_only_count": overview_only_count,
            "recommend_expand_full_text": recommend_expand,
            "recommended_action": (
                "Call expand_paper_memory_with_mcp(question, workflow_id, ...) and then query_paper_memory again for detail-level answers."
                if recommend_expand
                else ""
            ),
            "snippets": picked,
        }
        result_text = json.dumps(payload, ensure_ascii=False, indent=2)
        log_event(
            "tool.query_paper_memory",
            "call_completed",
            {
                "workflow_id": safe_workflow_id,
                "question": safe_question,
                "snippet_count": len(picked),
                "memory_loaded_count": loaded_count,
                "memory_full_loaded_count": full_loaded_count,
                "recommend_expand_full_text": recommend_expand,
            },
            direction="inbound",
        )
        return result_text
    except Exception as e:
        log_exception(
            "tool.query_paper_memory",
            "call_failed",
            e,
            {"workflow_id": safe_workflow_id, "question": safe_question},
        )
        raise


research_agent = LlmAgent(
    name="PaperAnalyst",
    model="gemini-3-flash-preview",
    instruction=(
        "You are a paper analysis specialist.\n"
        "Specialization-first policy: if the task is paper/research/PDF related, you should lead it.\n"
        "Use MCP tools to find relevant papers and summarize key points clearly.\n"
        "Decide search angle and semantic scope yourself; do not rely on fixed keyword templates.\n\n"
        "Workflow memory rules:\n"
        "1) Read `Workflow ID` from the task input and pass the same workflow_id to memory tools.\n"
        "2) Early in the step, call `load_paper_memory_with_mcp(query, workflow_id, ..., load_mode='overview')` first.\n"
        "   This stores compact title/abstract/introduction-level memory without loading full body text.\n"
        "3) For detail-heavy follow-up (methods/results/limitations/equations), call\n"
        "   `expand_paper_memory_with_mcp(question, workflow_id, ...)` to lazily load only relevant full papers.\n"
        "4) Use `query_paper_memory(question, workflow_id, ...)` for fast memory-based answers.\n\n"
        "Task rules:\n"
        "1) Explain why selected papers are relevant to the user goal.\n"
        "2) Slack posting must be delegated to MainAgent.\n"
        "   If Slack posting is needed, request it as `[MainAgent] Post this result to Slack ...`.\n"
        "3) Use indirect delegation only: do not call other agents directly.\n"
        "4) If local DB coverage is insufficient, request specialist follow-up in `Additional Needs:`.\n"
        "   Example: `- [WebSearchAnalyst] Find trustworthy web sources and recent reports for <topic>.`\n"
        "   Example: `- [MainAgent] Ask user to narrow domain, timeframe, or keywords.`\n"
        "5) Always end with one of these:\n"
        "   - `Additional Needs: none`\n"
        "   - `Additional Needs:` followed by bullet lines in format `[TargetAgentName] request`.\n"
        "6) Keep response concise and actionable."
    ),
    tools=[
        scrape_papers_with_mcp,
        load_paper_memory_with_mcp,
        expand_paper_memory_with_mcp,
        query_paper_memory,
    ],
)


__all__ = [
    "scrape_papers_with_mcp",
    "load_paper_memory_with_mcp",
    "expand_paper_memory_with_mcp",
    "query_paper_memory",
    "research_agent",
]

