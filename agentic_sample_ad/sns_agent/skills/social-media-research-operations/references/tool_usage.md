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

- `scrape_sns_with_mcp(keyword)`
  - Use to collect SNS posts relevant to the current keyword or phrase.
  - Prefer targeted keywords and rerun with a refined phrase instead of broad, vague searches.

- `format_handoff_contract(status, summary, text_response="", artifacts_json="", needs_json="")`
  - **Call this as the final step before finishing every response.**
  - Pass `status` as one of: `"completed"`, `"partial"`, `"blocked"`, `"failed"`.
  - Pass `summary` as a concise 1–3 sentence description of the social signals found.
  - Pass `text_response` when you have a detailed narrative of top posts or accounts (optional).
  - Pass `artifacts_json` as a JSON array string when you have reusable posts or accounts. Each object must have `"title"` and `"summary"`. Include account handle, post text summary, and `"url"` when available.
    - Example: `'[{"title": "@researcher – post on LLM alignment", "summary": "Highlights concerns about RLHF limitations.", "url": "https://twitter.com/researcher/status/..."}]'`
  - Pass `needs_json` as a JSON array string when downstream work is required. Each object must have `"request"`. Add `"required_capabilities"` (list) and `"blocking"` (bool) when relevant.
    - Example: `'[{"request": "Find academic papers related to RLHF alignment concerns", "required_capabilities": ["PaperAnalyst"], "blocking": false}]'`
  - Output the return value verbatim as your entire final response. Do not wrap it in prose or code fences.
  - If the tool returns a validation error message (starts with "format_handoff_contract failed"), fix the issues and call again.

## Decision rules

- Use `load_session_memory(section="contracts")` if ownership or routing boundaries are unclear.
- Summarize signals, not just raw post volume.
- Preserve enough source detail for follow-up: account, post text summary, and any relevant links when available.
- If the search result is noisy or weak, say so explicitly and suggest the next useful keyword or evidence source.
