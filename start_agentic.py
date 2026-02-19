from scripts.start_a2a_agents import launch_bridges, stop_bridges
from system_logger import initialize_process_logging
from user_entry_point import input_loop


if __name__ == "__main__":
    initialize_process_logging()
    bridges = launch_bridges()
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

