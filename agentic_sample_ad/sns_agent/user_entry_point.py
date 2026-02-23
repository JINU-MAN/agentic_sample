from __future__ import annotations

from uuid import uuid4

from .event_manager import run_single_task
from .session_memory import clear_session, get_or_create_session


def input_loop() -> None:
    session_id = uuid4().hex
    print("SocialMediaAnalyst standalone mode. Type 'exit' or 'quit' to stop.")
    print("Type 'reset' or '/reset' to clear the local session memory.")

    while True:
        user_input = input("\nsns-user> ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Stopping SocialMediaAnalyst standalone mode.")
            break

        if not user_input:
            print("Please enter a non-empty message.")
            continue

        if user_input.lower() in {"reset", "/reset"}:
            clear_session(session_id)
            print("Session memory cleared.")
            continue

        session = get_or_create_session(session_id)
        session.add_user_turn(user_input)
        result = run_single_task(user_input, context={"session_id": session_id})
        text = result.get("response") if result.get("ok") else result.get("error")
        session.add_assistant_turn(str(text))
        print("\n[SocialMediaAnalyst]")
        print(text)


if __name__ == "__main__":
    input_loop()

