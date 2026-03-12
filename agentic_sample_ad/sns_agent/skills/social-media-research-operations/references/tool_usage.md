# Internal Tool Usage

## Available internal action

- `load_session_memory(section="", query="", max_items=6)`
  - Use when you need your current tool inventory, ownership rules, or handoff contract.
  - Prefer `section="tools"` for direct callable inventory.
  - Prefer `section="handoff"` when shaping `summary`, `artifacts`, and `needs`.

- `scrape_sns_with_mcp(keyword)`
  - Use to collect SNS posts relevant to the current keyword or phrase.
  - Prefer targeted keywords and rerun with a refined phrase instead of broad, vague searches.

## Decision rules

- Use `load_session_memory(section="contracts")` if ownership or routing boundaries are unclear.
- Summarize signals, not just raw post volume.
- Preserve enough source detail for follow-up: account, post text summary, and any relevant links when available.
- If the search result is noisy or weak, say so explicitly and suggest the next useful keyword or evidence source.
