"""
Run standalone tests for paper/sns/web agents with MCP tools.

Usage:
  python scripts/run_agent_test.py sns   "AI"
  python scripts/run_agent_test.py paper "machine learning"
  python scripts/run_agent_test.py web   "latest AI policy update"
"""

import asyncio
import os
import sys
from pathlib import Path

from google.adk.runners import InMemoryRunner
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system_logger import initialize_process_logging


def _load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


def _resolve_agent(agent_type: str):
    agent_type = agent_type.lower()
    if agent_type == "sns":
        from agent.sns_agent import sns_agent

        return sns_agent
    if agent_type == "paper":
        from agent.paper_agent import research_agent

        return research_agent
    if agent_type == "web":
        from agent.web_search_agent import web_search_agent

        return web_search_agent
    raise ValueError("agent_type must be 'sns', 'paper', or 'web'.")


async def _run_agent_test(agent_type: str, query: str) -> None:
    agent = _resolve_agent(agent_type)
    print(f"[{agent_type} agent] query: {query}\n")

    runner = InMemoryRunner(agent=agent, app_name=f"{agent_type}-agent-test")
    try:
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="test-user",
        )
        new_message = types.Content(role="user", parts=[types.Part(text=query)])

        printed = False
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=new_message,
        ):
            if event.content and event.content.parts:
                text = "".join(part.text or "" for part in event.content.parts).strip()
                if text:
                    print(f"[{event.author}] {text}")
                    printed = True

        if not printed:
            print("(No text response emitted.)")
    finally:
        await runner.close()


def main() -> None:
    initialize_process_logging()
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_agent_test.py <sns|paper|web> <query>")
        sys.exit(1)

    agent_type = sys.argv[1].lower()
    query = " ".join(sys.argv[2:]).strip()

    try:
        asyncio.run(_run_agent_test(agent_type=agent_type, query=query))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
