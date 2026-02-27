from __future__ import annotations

import shlex
from uuid import uuid4
from typing import Callable, Tuple

from .agent import run_main_agent
from .session_memory import clear_session
from .system_logger import finalize_main_logging, initialize_main_logging


def _parse_setting_command(user_input: str) -> Tuple[bool, str, str, str]:
    try:
        tokens = shlex.split(str(user_input or "").strip())
    except ValueError as e:
        return False, "", "", f"Invalid command: {e}"

    if not tokens or tokens[0].lower() != "/setting":
        return False, "", "", "Usage: /setting {AgentName} -m {ModelName}"
    if len(tokens) < 4:
        return False, "", "", "Usage: /setting {AgentName} -m {ModelName}"

    agent_name = str(tokens[1]).strip()
    if not agent_name:
        return False, "", "", "Usage: /setting {AgentName} -m {ModelName}"

    model_name = ""
    idx = 2
    while idx < len(tokens):
        flag = str(tokens[idx]).strip().lower()
        if flag in {"-m", "-model"}:
            if idx + 1 >= len(tokens):
                return False, "", "", "Usage: /setting {AgentName} -m {ModelName}"
            model_name = str(tokens[idx + 1]).strip()
            idx += 2
            continue
        return False, "", "", f"Unknown option: {tokens[idx]}"

    if not model_name:
        return False, "", "", "Usage: /setting {AgentName} -m {ModelName}"
    return True, agent_name, model_name, ""


def input_loop(on_model_setting: Callable[[str, str], str] | None = None) -> None:
    initialize_main_logging()
    session_id = uuid4().hex
    print("agentic_sample_ad main agent started. Type 'exit' or 'quit' to stop.")
    print("Type 'reset' or '/reset' to clear the current session memory.")
    print("Type '/setting {AgentName} -m {ModelName}' to update agent model and reboot target agent.")

    while True:
        user_input = input("\nuser> ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Stopping main agent.")
            break

        if not user_input:
            print("Please enter a non-empty message.")
            continue

        if user_input.lower() in {"reset", "/reset"}:
            clear_session(session_id)
            print("Session memory cleared.")
            continue

        if user_input.strip().lower().startswith("/setting"):
            ok, agent_name, model_name, error = _parse_setting_command(user_input)
            if not ok:
                print(error)
                continue

            if on_model_setting is None:
                print("Runtime setting handler is unavailable in this execution mode.")
                continue

            try:
                result = on_model_setting(agent_name.strip(), model_name.strip())
                print(result)
            except Exception as e:
                print(f"Setting command failed: {e}")
            continue

        try:
            response = run_main_agent(user_input, session_id=session_id)
            print("\n[MainAgent]")
            print(response)
        except Exception as e:
            print(f"\nMain agent execution failed: {e}")


if __name__ == "__main__":
    try:
        input_loop()
    finally:
        finalize_main_logging()
