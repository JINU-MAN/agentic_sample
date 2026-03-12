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
