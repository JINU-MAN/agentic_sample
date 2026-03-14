# Internal Tool Usage

## Available internal actions

- All direct tools return one JSON object with:
  - `ok`, `tool_name`, `summary`, `content_type`, `items`, `data`, `errors`, `metadata`
  - Read `summary` first, then inspect `items` or `data`.
  - Do not copy the raw tool JSON as your final workflow-step answer. Convert it into the workflow handoff contract.

- `load_session_memory(section="", query="", max_items=6)`
  - Use when you need your current tool inventory, ownership rules, or handoff contract.
  - Prefer `section="tools"` for direct callable inventory.
  - Prefer `section="handoff"` when shaping `summary`, `artifacts`, and `needs`.

- `scrape_papers_with_mcp(query)`
  - Use to search the local PDF corpus and identify candidate papers.
- `fetch_external_paper_with_mcp(reference="", url="", doi="", arxiv_id="", max_chars=12000)`
  - Use when handoff artifacts or the request mention a URL, DOI, arXiv ID, or another external paper reference.
- `load_paper_memory_with_mcp(query, workflow_id, max_papers=3, max_chars_per_paper=..., load_mode="overview")`
  - Use to initialize workflow-scoped paper memory for the current topic before asking targeted questions.
- `query_paper_memory(question, workflow_id, max_snippets=...)`
  - Use for fast question answering against already-loaded memory.
- `expand_paper_memory_with_mcp(question, workflow_id, max_papers=2, max_chars_per_paper=...)`
  - Use only when overview memory is insufficient and deeper full-text evidence is needed.

- `format_handoff_contract(status, summary, text_response="", artifacts_json="", needs_json="")`
  - **Call this as the final step before finishing every response.**
  - Pass `status` as one of: `"completed"`, `"partial"`, `"blocked"`, `"failed"`.
  - Pass `summary` as a concise 1–3 sentence description of findings.
  - Pass `text_response` when you have a detailed narrative with citations (optional).
  - Pass `artifacts_json` as a JSON array string when you have reusable items. Each object must have `"title"` and `"summary"`. Add `"doi"`, `"arxiv_id"`, `"url"` when available.
    - Example: `'[{"title": "Attention Is All You Need", "summary": "Introduces transformer architecture.", "arxiv_id": "1706.03762"}]'`
  - Pass `needs_json` as a JSON array string when downstream work is required. Each object must have `"request"`. Add `"required_capabilities"` (list) and `"blocking"` (bool) when relevant.
    - Example: `'[{"request": "Fetch full text for arXiv:1706.03762", "required_capabilities": ["PaperAnalyst"], "blocking": true}]'`
  - Output the return value verbatim as your entire final response. Do not wrap it in prose or code fences.
  - If the tool returns a validation error message (starts with "format_handoff_contract failed"), fix the issues and call again.

## Decision rules

- Reuse the same `workflow_id` across all memory-related actions in the same workflow.
- Inspect `input_artifacts` before starting a fresh search.
- Use `load_session_memory(section="contracts")` if ownership or routing boundaries are unclear.
- Start with metadata or overview memory when possible, then expand only if the answer still lacks support.
- If neither the local corpus nor external reference lookup can support the request, say what evidence is missing.
