# Internal Tool Usage

## Available internal action

- All direct tools return one JSON object with:
  - `ok`, `tool_name`, `summary`, `content_type`, `items`, `data`, `errors`, `metadata`
  - Read `summary` first, then inspect `items` or `data`.
  - Do not copy the raw tool JSON as your final workflow-step answer. Convert it into the workflow handoff contract.

- `load_session_memory(section="", query="", max_items=6)`
  - Use when you need your current tool inventory, ownership rules, or handoff contract.
  - Prefer `section="tools"` for direct callable inventory.
  - Prefer `section="handoff"` when shaping `summary`, `artifacts`, and `needs`.

- `search_web_with_mcp(query, max_results=6)`
  - Use for general web evidence, current-information lookup, and finding citable sources.
  - Prefer a few focused queries over many weak ones.
  - Increase `max_results` only when the first pass is clearly insufficient.

- `format_handoff_contract(status, summary, text_response="", artifacts_json="", needs_json="")`
  - **Call this as the final step before finishing every response.**
  - Pass `status` as one of: `"completed"`, `"partial"`, `"blocked"`, `"failed"`.
  - Pass `summary` as a concise 1–3 sentence description of what was found.
  - Pass `text_response` when you have a detailed synthesis with citations (optional).
  - Pass `artifacts_json` as a JSON array string when you have reusable sources. Each object must have `"title"` and `"summary"`. Always add `"url"`, `"doi"`, or `"arxiv_id"` when available.
    - Example: `'[{"title": "OpenAI Blog – GPT-4", "summary": "Overview of GPT-4 capabilities.", "url": "https://openai.com/research/gpt-4"}]'`
  - Pass `needs_json` as a JSON array string when downstream work is required. Each object must have `"request"`. Add `"required_capabilities"` (list) and `"blocking"` (bool) when relevant.
    - Example: `'[{"request": "Retrieve full paper for arXiv:2303.08774", "required_capabilities": ["PaperAnalyst"], "blocking": false}]'`
  - Output the return value verbatim as your entire final response. Do not wrap it in prose or code fences.
  - If the tool returns a validation error message (starts with "format_handoff_contract failed"), fix the issues and call again.

## Decision rules

- Do not use this tool for PDF parsing, local paper-memory work, DOI/arXiv resolution, or external paper retrieval.
- Use `load_session_memory(section="contracts")` if ownership or routing boundaries are unclear.
- If the task needs paper-specific evidence, return a handoff-ready output and request `PaperAnalyst` in `needs`.
- If a previous workflow step should already have collected the needed web results, request `MainAgent` workflow memory in `needs` before asking the user to resend them.
- Include stable identifiers in `artifacts` when available, especially URL, DOI, and arXiv ID.
- State clearly when the answer is based on weak or incomplete web evidence.
