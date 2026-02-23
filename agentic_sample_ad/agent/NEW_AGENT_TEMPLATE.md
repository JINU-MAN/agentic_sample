# New Agent Template

Use this template when adding a new local agent so it behaves like existing agents.

## 1) File placement

- Create a new file under `agent/`, for example: `agent/<domain>_agent.py`.
- Define exactly one exported `LlmAgent` object for the main agent instance in that file.

## 2) Tool wrapper pattern

Use MCP tool wrappers with logging and normalized return text.

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from google.adk.agents import LlmAgent

from agent.tool.slack_mcp_tool import slack_post_message
from mcp_local.client import call_mcp_tool
from system_logger import log_event, log_exception

BASE_DIR = Path(__file__).parent.parent
MY_MCP_SERVER = BASE_DIR / "mcp_local" / "my_server.py"


def _extract_mcp_result(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    structured = raw.get("structuredContent", {})
    if isinstance(structured, dict):
        result = structured.get("result")
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
    return []


def search_my_domain_with_mcp(query: str) -> str:
    log_event(
        "tool.search_my_domain_with_mcp",
        "call_started",
        {"query": query, "server_script_path": str(MY_MCP_SERVER)},
        direction="outbound",
    )
    try:
        raw = call_mcp_tool(
            server_script_path=str(MY_MCP_SERVER),
            tool_name="search_my_domain",
            arguments={"query": query},
        )
        normalized = _extract_mcp_result(raw)
        result_text = json.dumps(normalized, ensure_ascii=False, indent=2)
        log_event(
            "tool.search_my_domain_with_mcp",
            "call_completed",
            {"query": query, "result_count": len(normalized)},
            direction="inbound",
        )
        return result_text
    except Exception as e:
        log_exception(
            "tool.search_my_domain_with_mcp",
            "call_failed",
            e,
            {"query": query, "server_script_path": str(MY_MCP_SERVER)},
        )
        raise
```

## 3) Agent instruction pattern (required)

Your instruction must include:

- semantic search guidance (not fixed keyword templates),
- uncertainty behavior,
- Slack behavior only when user asks,
- mandatory `Additional Needs` protocol for collaboration handoff.

```python
my_domain_agent = LlmAgent(
    name="MyDomainAnalyst",
    model="gemini-2.0-flash",
    instruction=(
        "You are a <domain> analysis specialist.\n"
        "Specialization-first policy: if the task matches your domain, you should lead execution.\n"
        "Decide search angle and semantic scope yourself; do not rely on fixed keyword templates.\n\n"
        "Tool usage guide:\n"
        "1) Use `search_my_domain_with_mcp` with semantic query variants.\n"
        "2) Select evidence by meaning-level relevance and summarize clearly.\n"
        "3) If the user explicitly asks for Slack posting, call `slack_post_message(channel, text)`.\n"
        "4) Use indirect delegation only: do not call other agents directly.\n"
        "5) If local evidence is missing or weak, request follow-up work in `Additional Needs:`.\n"
        "   Example: `- [WebSearchAnalyst] Collect external evidence for <topic>.`\n"
        "   Example: `- [MainAgent] Ask user to clarify constraints.`\n"
        "6) Always end with one of these:\n"
        "   - `Additional Needs: none`\n"
        "   - `Additional Needs:` followed by bullet lines `[TargetAgentName] request`.\n"
        "7) Keep output concise and actionable."
    ),
    tools=[
        search_my_domain_with_mcp,
        slack_post_message,
    ],
)

__all__ = ["search_my_domain_with_mcp", "my_domain_agent"]
```

## 4) Registration and discovery

- Preferred: add metadata in `agent_cards/*.json` with fields `name`, `type=local`, `module`, `attr`, `capabilities`.
- Runtime discovery also scans `agent/*.py` for `LlmAgent` objects, so card is optional but recommended.

## 5) Collaboration contract

The orchestrator passes step context with:

- prior step outputs,
- open additional needs,
- remaining planned steps,
- available agent names for additional-need targeting.
- capability profiles for available agents.

Your agent should:

- complete assigned step,
- prioritize your specialization for matching tasks,
- request specialist follow-up via `Additional Needs` when needed,
- explicitly request missing work in `Additional Needs`,
- target another agent with `[AgentName]` when needed,
- use `[MainAgent]` when user clarification is needed.
