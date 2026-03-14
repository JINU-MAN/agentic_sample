---
name: web-research-operations
description: Perform current web research, choose effective search queries, and synthesize citation-grounded web evidence. Use when WebSearchAnalyst must decide how to search the web, when to stop searching, and when to hand off paper-specific retrieval to another agent.
---

# Web Research Operations

Read `references/tool_usage.md` before using internal search tools or when deciding whether the request belongs to web research or paper retrieval.
If the current tool inventory, contract, or handoff shape is unclear, call `load_session_memory`.
Use `section="tools"` for your direct callable inventory and `section="handoff"` when you need the structured output contract.

Own web-source discovery and citation-grounded synthesis:
- Decide search queries from the request and from any handoff artifacts.
- Prefer stable, citable web sources.
- Hand off paper-specific retrieval instead of treating paper lookup as ordinary web search.
- If the workflow packet references earlier work but the concrete prior results are missing, request shared workflow memory from `MainAgent` in structured `needs` instead of asking the user.

Return a compact result that highlights evidence quality, reusable artifacts, and any downstream needs.

## Final Response Contract

Before finishing, always call `format_handoff_contract` to produce your final response.
Output the return value verbatim — no surrounding prose or markdown.

| Field | Required | Description |
|---|---|---|
| `status` | yes | `"completed"` · `"partial"` · `"blocked"` · `"failed"` |
| `summary` | yes | 1–3 sentences describing what was found or done |
| `text_response` | no | Detailed synthesis with citations and source links |
| `artifacts_json` | no | JSON array — each item needs `"title"` and `"summary"`; always add `"url"`, `"doi"`, or `"arxiv_id"` when available |
| `needs_json` | no | JSON array — each item needs `"request"`; add `"required_capabilities"` and `"blocking"` when relevant |

**Status guide:**
- `completed` — web research done, evidence found
- `partial` — done but paper-specific retrieval or deeper analysis is still needed (fill `needs_json`)
- `blocked` — cannot proceed without missing workflow context (fill `needs_json`)
- `failed` — task failed due to an unrecoverable error
