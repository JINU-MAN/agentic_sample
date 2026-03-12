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
