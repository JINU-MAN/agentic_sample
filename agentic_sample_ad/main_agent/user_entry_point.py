from __future__ import annotations

from uuid import uuid4

from .agent import run_main_agent
from .session_memory import clear_session
from .system_logger import finalize_main_logging, initialize_main_logging


def input_loop() -> None:
    initialize_main_logging()
    session_id = uuid4().hex
    print("agentic_sample_ad main agent started. Type 'exit' or 'quit' to stop.")
    print("Type 'reset' or '/reset' to clear the current session memory.")

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
