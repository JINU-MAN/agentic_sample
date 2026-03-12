# Internal Tool Usage

## Available internal action

- `load_session_memory(section="", query="", max_items=6)`
  - Use when you need your current tool inventory, ownership rules, or handoff contract.
  - Prefer `section="tools"` for direct callable inventory.
  - Prefer `section="handoff"` when shaping `summary`, `artifacts`, and `needs`.

- `search_web_with_mcp(query, max_results=6)`
  - Use for general web evidence, current-information lookup, and finding citable sources.
  - Prefer a few focused queries over many weak ones.
  - Increase `max_results` only when the first pass is clearly insufficient.

## Decision rules

- Do not use this tool for PDF parsing, local paper-memory work, DOI/arXiv resolution, or external paper retrieval.
- Use `load_session_memory(section="contracts")` if ownership or routing boundaries are unclear.
- If the task needs paper-specific evidence, return a handoff-ready output and request `PaperAnalyst` in `needs`.
- If a previous workflow step should already have collected the needed web results, request `MainAgent` workflow memory in `needs` before asking the user to resend them.
- Include stable identifiers in `artifacts` when available, especially URL, DOI, and arXiv ID.
- State clearly when the answer is based on weak or incomplete web evidence.
