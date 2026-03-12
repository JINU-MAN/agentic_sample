# Internal Tool Usage

## Available internal action

- `load_session_memory(section="", query="", max_items=6)`
  - Use when you need MainAgent's current tool inventory, known sub-agent contracts, or the handoff schema without re-reading long instructions.
  - Prefer `section="contracts"` for ownership and delegation rules.
  - Use `section="tools"` only when you truly need tool inventory details.
  - Do not use it for current workflow state.

- `read_workflow_memory(query="", max_items=6)`
  - Use when MainAgent needs prior step outputs, handoff artifacts, or current workflow state for itself or for a blocked worker.
  - Use before asking the user for internal workflow context that should already exist inside the run.
  - Prefer a focused `query` such as `latest search results`, `current artifacts`, or `open needs`.

- `slack_post_message(channel, text)`
  - Use only when the user explicitly wants Slack delivery or the workflow already established Slack as the delivery channel.
  - Require a concrete channel and a ready-to-send message body.
  - Do not use it for intermediate notes, speculative drafts, or specialist evidence gathering.

## Decision rules

- Keep the task in MainAgent when the remaining work is coordination, plan repair, user clarification, or final delivery formatting.
- Delegate when the next useful step is domain evidence gathering or analysis that a specialist owns.
- Use `load_session_memory` for static contracts and `read_workflow_memory` for run-specific state.
- If a worker is blocked on missing internal workflow context, reply from `read_workflow_memory` instead of asking the user to repeat prior step outputs.
- If delivery is blocked by missing channel or incomplete content, ask for the missing information instead of sending.
