---
name: coordinator-operations
description: Coordinate multi-agent execution, decide when work can stay with MainAgent, and handle owned delivery actions such as Slack messaging. Use when MainAgent must choose whether to answer directly, delegate, or send a final delivery through its internal channel tool.
---

# Coordinator Operations

Read `references/tool_usage.md` before using internal delivery tools or when deciding whether MainAgent should keep work instead of delegating it.
If you need the current coordinator tool inventory or the latest known sub-agent contracts, call `load_session_memory`.
Use `section="contracts"` for agent ownership and delegation boundaries, and `section="tools"` only when you truly need tool inventory details.
If another agent asks for prior step context, handoff artifacts, or missing workflow state, use shared workflow memory before asking the user.

Keep MainAgent focused on work it owns:
- Orchestrate, replan, clarify user intent, and package final delivery.
- Keep specialist evidence gathering with worker agents.
- Use direct delivery actions only when the user explicitly needs them or the workflow contract requires them.

Return concise coordinator output. When delegating, make the handoff target and the reason explicit.
