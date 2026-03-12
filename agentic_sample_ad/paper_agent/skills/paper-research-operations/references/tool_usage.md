# Internal Tool Usage

## Available internal actions

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

## Decision rules

- Reuse the same `workflow_id` across all memory-related actions in the same workflow.
- Inspect `input_artifacts` before starting a fresh search.
- Use `load_session_memory(section="contracts")` if ownership or routing boundaries are unclear.
- Start with metadata or overview memory when possible, then expand only if the answer still lacks support.
- If neither the local corpus nor external reference lookup can support the request, say what evidence is missing.
