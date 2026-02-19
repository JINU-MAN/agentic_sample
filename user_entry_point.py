from __future__ import annotations

from uuid import uuid4

from agentic import run_main_agent
from session_store import clear_session
from system_logger import initialize_process_logging


def input_loop() -> None:
    """
    Interactive entrypoint for local testing.

    Commands:
    - exit / quit: terminate loop
    - reset / /reset: clear in-memory context for current session
    """
    initialize_process_logging()
    session_id = uuid4().hex
    print("에이전트 시스템에 오신 것을 환영합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("대화 컨텍스트를 초기화하려면 'reset' 또는 '/reset'을 입력하세요.")

    while True:
        user_input = input("\n사용자 입력> ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("에이전트를 종료합니다.")
            break

        if not user_input:
            print("빈 입력입니다. 다시 입력해 주세요.")
            continue

        if user_input.lower() in {"reset", "/reset"}:
            clear_session(session_id)
            print("현재 세션의 대화 컨텍스트를 초기화했습니다.")
            continue

        try:
            response = run_main_agent(user_input, session_id=session_id)
            print("\n[메인 에이전트 응답]")
            print(response)
        except Exception as e:
            print(f"\n에이전트 실행 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    input_loop()
