from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_PARENT_DIR = ROOT_DIR.parent
BRIDGE_MODULE = "agentic_sample_ad.mcp_local.a2a_bridge_server"

if str(PACKAGE_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT_DIR))

from agentic_sample_ad.system_logger import initialize_process_logging, log_event, log_exception


DEFAULT_CARD_PATH = ROOT_DIR / "agent_cards" / "agent_card.json"
BridgeProcess = Tuple[str, subprocess.Popen[Any], str, int]


def _load_env_file() -> None:
    env_path = ROOT_DIR / ".env"
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


def _load_cards(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _parse_host_port(base_url: str) -> Tuple[str, int]:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base_url: {base_url}")
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        port = int(parsed.port)
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid port in base_url: {base_url}")
    return host, port


def _is_port_open(host: str, port: int, timeout_sec: float = 0.5) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _card_url(base_url: str) -> str:
    return f"{str(base_url).rstrip('/')}/.well-known/agent-card.json"


def _is_agent_card_ready(base_url: str, timeout_sec: float = 1.0) -> bool:
    card_url = _card_url(base_url)
    try:
        response = httpx.get(card_url, timeout=timeout_sec)
        if response.status_code != 200:
            return False
        payload = response.json()
        if not isinstance(payload, dict):
            return False
        return bool(str(payload.get("name", "")).strip())
    except Exception:
        return False


def _wait_for_agent_card_ready(
    base_url: str,
    *,
    ready_timeout_sec: float,
    interval_sec: float = 0.25,
) -> bool:
    deadline = time.time() + max(0.5, ready_timeout_sec)
    while time.time() < deadline:
        if _is_agent_card_ready(base_url):
            return True
        time.sleep(interval_sec)
    return _is_agent_card_ready(base_url)


def _read_ready_timeout_sec() -> float:
    raw = str(os.getenv("A2A_BRIDGE_READY_TIMEOUT_SEC", "")).strip()
    if not raw:
        return 15.0
    try:
        value = float(raw)
    except ValueError:
        return 15.0
    return max(1.0, value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start local A2A bridge servers for agents in agent cards.")
    parser.add_argument(
        "--card-path",
        default=str(DEFAULT_CARD_PATH),
        help="Path to agent card JSON file",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated agent names to run (default: all agents with module/attr/base_url)",
    )
    return parser.parse_args()


def launch_bridges(
    card_path: str | Path = DEFAULT_CARD_PATH,
    only: str = "",
) -> List[BridgeProcess]:
    card_path = Path(card_path).resolve()
    if not card_path.exists():
        raise FileNotFoundError(f"Card file not found: {card_path}")

    cards = _load_cards(card_path)
    only_names = {
        token.strip().lower()
        for token in str(only).split(",")
        if token.strip()
    }

    targets: List[Dict[str, Any]] = []
    for card in cards:
        name = str(card.get("name", "")).strip()
        module_name = str(card.get("module", "")).strip()
        attr_name = str(card.get("attr", "")).strip()
        base_url = str(card.get("base_url", "")).strip()
        if only_names and name.lower() not in only_names:
            continue
        if not name or not module_name or not attr_name or not base_url:
            continue
        targets.append(card)

    if not targets:
        print("No bridge targets found. Check card fields: name/module/attr/base_url.")
        return []

    log_event(
        "a2a.bridge.launcher",
        "launch_started",
        {
            "card_path": str(card_path),
            "target_agents": [str(item.get("name", "")) for item in targets],
        },
    )

    processes: List[BridgeProcess] = []
    ready_timeout_sec = _read_ready_timeout_sec()
    for card in targets:
        name = str(card.get("name", "")).strip()
        module_name = str(card.get("module", "")).strip()
        attr_name = str(card.get("attr", "")).strip()
        base_url = str(card.get("base_url", "")).strip()
        description = str(card.get("description", "")).strip()
        tags = [str(token).strip() for token in card.get("capabilities", []) if str(token).strip()]

        try:
            host, port = _parse_host_port(base_url)
            if _is_port_open(host, port):
                if _wait_for_agent_card_ready(
                    base_url,
                    ready_timeout_sec=min(3.0, ready_timeout_sec),
                ):
                    print(f"[skip] {name} already listening on {host}:{port}")
                    log_event(
                        "a2a.bridge.launcher",
                        "launch_skipped_port_open",
                        {"agent": name, "host": host, "port": port, "base_url": base_url},
                    )
                    continue

                print(f"[failed] {name} port {host}:{port} is open but agent card is unavailable")
                log_event(
                    "a2a.bridge.launcher",
                    "launch_failed_port_occupied_unhealthy",
                    {"agent": name, "host": host, "port": port, "base_url": base_url},
                    level="ERROR",
                )
                continue

            cmd = [
                sys.executable,
                "-m",
                BRIDGE_MODULE,
                "--module",
                module_name,
                "--attr",
                attr_name,
                "--name",
                name,
                "--description",
                description or f"A2A bridge for {name}",
                "--host",
                host,
                "--port",
                str(port),
                "--tags",
                ",".join(tags) if tags else "agentic,local-bridge",
            ]
            process = subprocess.Popen(cmd, cwd=str(PACKAGE_PARENT_DIR))
            print(f"[starting] {name} -> {host}:{port} (pid={process.pid})")

            if not _wait_for_agent_card_ready(base_url, ready_timeout_sec=ready_timeout_sec):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                print(
                    f"[failed] {name} bridge started but card endpoint not ready within {ready_timeout_sec:.1f}s"
                )
                log_event(
                    "a2a.bridge.launcher",
                    "launch_failed_card_not_ready",
                    {
                        "agent": name,
                        "host": host,
                        "port": port,
                        "pid": process.pid,
                        "base_url": base_url,
                        "card_url": _card_url(base_url),
                        "ready_timeout_sec": ready_timeout_sec,
                    },
                    level="ERROR",
                )
                continue

            processes.append((name, process, host, port))
            print(f"[started] {name} -> {host}:{port} (pid={process.pid})")
            log_event(
                "a2a.bridge.launcher",
                "launch_completed",
                {
                    "agent": name,
                    "host": host,
                    "port": port,
                    "pid": process.pid,
                    "command": cmd,
                    "card_url": _card_url(base_url),
                    "ready_timeout_sec": ready_timeout_sec,
                },
            )
        except Exception as e:
            print(f"[failed] {name}: {e}")
            log_exception(
                "a2a.bridge.launcher",
                "launch_failed",
                e,
                {"agent": name, "module": module_name, "attr": attr_name, "base_url": base_url},
            )

    return processes


def stop_bridges(processes: List[BridgeProcess]) -> None:
    for name, process, host, port in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        log_event(
            "a2a.bridge.launcher",
            "bridge_process_stopped",
            {"agent": name, "host": host, "port": port, "pid": process.pid},
        )


def main() -> None:
    initialize_process_logging()
    _load_env_file()
    args = _parse_args()

    processes = launch_bridges(card_path=args.card_path, only=args.only)
    if not processes:
        print("No new bridge process was started.")
        return

    print("A2A bridges are running. Press Ctrl+C to stop all.")
    try:
        while True:
            time.sleep(1.0)
            dead: List[str] = []
            for name, process, host, port in processes:
                if process.poll() is not None:
                    dead.append(f"{name}({host}:{port})")
            if dead:
                print(f"[warning] exited bridge process: {', '.join(dead)}")
                log_event(
                    "a2a.bridge.launcher",
                    "bridge_process_exited",
                    {"agents": dead},
                    level="ERROR",
                )
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_bridges(processes)
        print("All bridge processes stopped.")


if __name__ == "__main__":
    main()

