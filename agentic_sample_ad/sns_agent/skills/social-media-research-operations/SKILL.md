---
name: social-media-research-operations
description: Collect relevant social posts, choose useful search keywords, and summarize social signals. Use when SocialMediaAnalyst must decide how to search SNS content, what signals are worth surfacing, and how to produce a structured handoff for downstream agents.
---

# Social Media Research Operations

Read `references/tool_usage.md` before using internal SNS tools or when deciding whether the current social evidence is strong enough to report.
If the current tool inventory, contract, or handoff shape is unclear, call `load_session_memory`.
Use `section="tools"` for your direct callable inventory and `section="handoff"` when you need the structured output contract.

Own SNS evidence gathering and signal summarization:
- Choose search keywords from user intent and handoff artifacts.
- Focus on posts, accounts, links, and recurring signals that materially change the answer.
- Prefer compact structured handoff output when other agents may continue from the findings.
- If prior workflow context is missing, request shared workflow memory from `MainAgent` in structured `needs` before asking the user.

Report the strongest signals first and state clearly when the social evidence is weak, noisy, or missing.

## Final Response Contract

Before finishing, always call `format_handoff_contract` to produce your final response.
Output the return value verbatim — no surrounding prose or markdown.

| Field | Required | Description |
|---|---|---|
| `status` | yes | `"completed"` · `"partial"` · `"blocked"` · `"failed"` |
| `summary` | yes | 1–3 sentences describing the signals found |
| `text_response` | no | Detailed narrative with top posts, accounts, and recurring themes |
| `artifacts_json` | no | JSON array — each item needs `"title"` and `"summary"`; include account handle, post text summary, and `"url"` when available |
| `needs_json` | no | JSON array — each item needs `"request"`; add `"required_capabilities"` and `"blocking"` when relevant |

**Status guide:**
- `completed` — SNS evidence collected and summarized
- `partial` — signals found but deeper research is still needed (fill `needs_json`)
- `blocked` — cannot proceed without missing context (fill `needs_json`)
- `failed` — task failed due to an unrecoverable error
