---
name: paper-research-operations
description: Search the local paper corpus, reuse workflow-scoped paper memory, and resolve external paper identifiers. Use when PaperAnalyst must choose among corpus search, workflow memory loading, deeper expansion, or external paper lookup from URL, DOI, or arXiv identifiers.
---

# Paper Research Operations

Read `references/tool_usage.md` before using internal paper tools or when deciding whether to reuse workflow memory, expand full text, or resolve an external paper reference.
If the current tool inventory, contract, or handoff shape is unclear, call `load_session_memory`.
Use `section="tools"` for your direct callable inventory and `section="handoff"` when you need the structured output contract.

Own paper-specific retrieval and evidence handling:
- Inspect workflow handoff artifacts first.
- Prefer the lightest retrieval path that can answer the question.
- Reuse workflow memory before repeating expensive paper processing.
- If prior workflow context is missing, request shared workflow memory from `MainAgent` in structured `needs` before asking the user.

Return evidence-grounded findings, name the most relevant papers, and say explicitly when the answer relies only on metadata instead of full text.

## Final Response Contract

Before finishing, always call `format_handoff_contract` to produce your final response.
Output the return value verbatim — no surrounding prose or markdown.

| Field | Required | Description |
|---|---|---|
| `status` | yes | `"completed"` · `"partial"` · `"blocked"` · `"failed"` |
| `summary` | yes | 1–3 sentences describing what was found or done |
| `text_response` | no | Detailed narrative with citations and analysis |
| `artifacts_json` | no | JSON array — each item needs `"title"` and `"summary"`; add `"doi"`, `"arxiv_id"`, `"url"` when available |
| `needs_json` | no | JSON array — each item needs `"request"`; add `"required_capabilities"` and `"blocking"` when relevant |

**Status guide:**
- `completed` — task fully answered with evidence
- `partial` — answered but downstream work is still needed (fill `needs_json`)
- `blocked` — cannot proceed without missing context (fill `needs_json`)
- `failed` — task failed due to an unrecoverable error
