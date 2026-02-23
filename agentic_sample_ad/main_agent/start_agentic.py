from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Tuple

from agentic_sample_ad.scripts.start_a2a_agents import launch_bridges, stop_bridges

from .card_registry import collect_sub_agent_card_files
from .system_logger import initialize_main_logging, log_main_event, log_main_exception
from .user_entry_point import input_loop


BridgeProcess = Tuple[str, Any, str, int]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
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


def launch_sub_agent_bridges() -> List[BridgeProcess]:
    bridges: List[BridgeProcess] = []
    card_files = collect_sub_agent_card_files()
    log_main_event(
        "bridge_launch_started",
        {"card_files": [str(path) for path in card_files]},
    )
    for card_file in card_files:
        try:
            started = launch_bridges(card_path=card_file)
            bridges.extend(started)
        except Exception as e:
            log_main_exception("bridge_launch_failed", e, {"card_file": str(card_file)})
    log_main_event(
        "bridge_launch_completed",
        {"count": len(bridges), "bridges": [name for name, *_ in bridges]},
    )
    return bridges


def main() -> None:
    initialize_main_logging()
    _load_env_file()
    bridges = launch_sub_agent_bridges()
    if bridges:
        names = ", ".join(name for name, *_ in bridges)
        print(f"A2A bridge servers started: {names}")
    else:
        print("A2A bridge servers were not newly started. Continuing.")

    try:
        input_loop()
    finally:
        if bridges:
            stop_bridges(bridges)
            print("A2A bridge servers stopped.")


if __name__ == "__main__":
    main()


