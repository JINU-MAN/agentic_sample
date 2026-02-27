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

if str(PACKAGE_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT_DIR))

from agentic_sample_ad.system_logger import (
    finalize_process_logging,
    initialize_process_logging,
    log_event,
    log_exception,
)
from agentic_sample_ad.model_settings import normalize_agent_name, read_model_overrides


DEFAULT_CARD_PATH = ROOT_DIR / "agent_cards" / "agent_card.json"
AgentServerProcess = Tuple[str, subprocess.Popen[Any], str, int]
RUNTIME_AGENT_CARDS_ENV_KEY = "AGENTIC_RUNTIME_AGENT_CARDS"
DYNAMIC_PORTS_ENV_KEY = "A2A_DYNAMIC_PORTS"


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


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "disable", "disabled"}


def _pick_dynamic_port(host: str) -> int:
    bind_host = str(host).strip() or "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _build_runtime_card(
    card: Dict[str, Any],
    *,
    runtime_base_url: str,
    runtime_model: str = "",
) -> Dict[str, Any]:
    payload = dict(card)
    payload["base_url"] = runtime_base_url
    payload["runtime_base_url"] = runtime_base_url
    payload["runtime_source"] = "a2a_agent_server_launcher"
    if str(runtime_model).strip():
        payload["runtime_model"] = str(runtime_model).strip()
    return payload


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
    raw = str(os.getenv("A2A_AGENT_SERVER_READY_TIMEOUT_SEC", "")).strip()
    if not raw:
        raw = str(os.getenv("A2A_BRIDGE_READY_TIMEOUT_SEC", "")).strip()
    if not raw:
        return 15.0
    try:
        value = float(raw)
    except ValueError:
        return 15.0
    return max(1.0, value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start local A2A agent servers for agents in agent cards.")
    parser.add_argument(
        "--card-path",
        default=str(DEFAULT_CARD_PATH),
        help="Path to agent card JSON file",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated agent names to run (default: all agents with server_module/base_url)",
    )
    return parser.parse_args()


def launch_agent_servers(
    card_path: str | Path = DEFAULT_CARD_PATH,
    only: str = "",
    model_overrides: Dict[str, str] | None = None,
) -> Tuple[List[AgentServerProcess], List[Dict[str, Any]]]:
    card_path = Path(card_path).resolve()
    if not card_path.exists():
        raise FileNotFoundError(f"Card file not found: {card_path}")

    cards = _load_cards(card_path)
    only_names = {
        token.strip().lower()
        for token in str(only).split(",")
        if token.strip()
    }
    env_overrides = read_model_overrides()
    explicit_overrides = {
        normalize_agent_name(k): str(v).strip()
        for k, v in (model_overrides or {}).items()
        if normalize_agent_name(k) and str(v).strip()
    }
    resolved_model_overrides = dict(env_overrides)
    resolved_model_overrides.update(explicit_overrides)

    targets: List[Dict[str, Any]] = []
    for card in cards:
        name = str(card.get("name", "")).strip()
        server_module = str(card.get("server_module", "")).strip()
        base_url = str(card.get("base_url", "")).strip()
        if only_names and name.lower() not in only_names:
            continue
        if not name or not server_module or not base_url:
            continue
        targets.append(card)

    if not targets:
        print("No launch targets found. Check card fields: name/server_module/base_url.")
        return [], []

    dynamic_ports = _env_flag(DYNAMIC_PORTS_ENV_KEY, True)

    log_event(
        "a2a.agent_server.launcher",
        "launch_started",
        {
            "card_path": str(card_path),
            "target_agents": [str(item.get("name", "")) for item in targets],
            "dynamic_ports": dynamic_ports,
            "model_overrides": resolved_model_overrides,
        },
    )

    processes: List[AgentServerProcess] = []
    runtime_cards: List[Dict[str, Any]] = []
    ready_timeout_sec = _read_ready_timeout_sec()
    for card in targets:
        name = str(card.get("name", "")).strip()
        server_module = str(card.get("server_module", "")).strip()
        base_url = str(card.get("base_url", "")).strip()
        runtime_base_url = base_url
        description = str(card.get("description", "")).strip()
        tags = [str(token).strip() for token in card.get("capabilities", []) if str(token).strip()]
        model_name = (
            resolved_model_overrides.get(normalize_agent_name(name), "")
            or str(card.get("model", "")).strip()
        )

        try:
            host, configured_port = _parse_host_port(base_url)
            port = configured_port
            if dynamic_ports:
                port = _pick_dynamic_port(host)
            runtime_base_url = _base_url(host, port)

            if not dynamic_ports and _is_port_open(host, port):
                if _wait_for_agent_card_ready(
                    runtime_base_url,
                    ready_timeout_sec=min(3.0, ready_timeout_sec),
                ):
                    print(f"[skip] {name} already listening on {host}:{port}")
                    runtime_cards.append(
                        _build_runtime_card(
                            card,
                            runtime_base_url=runtime_base_url,
                            runtime_model=model_name,
                        )
                    )
                    log_event(
                        "a2a.agent_server.launcher",
                        "launch_skipped_port_open",
                        {
                            "agent": name,
                            "host": host,
                            "port": port,
                            "base_url": runtime_base_url,
                            "model": model_name,
                        },
                    )
                    continue

                print(f"[failed] {name} port {host}:{port} is open but agent card is unavailable")
                log_event(
                    "a2a.agent_server.launcher",
                    "launch_failed_port_occupied_unhealthy",
                    {"agent": name, "host": host, "port": port, "base_url": runtime_base_url},
                    level="ERROR",
                )
                continue

            if dynamic_ports and _is_port_open(host, port):
                port = _pick_dynamic_port(host)
                runtime_base_url = _base_url(host, port)

            cmd = [
                sys.executable,
                "-m",
                server_module,
                "--name",
                name,
                "--description",
                description or f"A2A server for {name}",
                "--host",
                host,
                "--port",
                str(port),
                "--tags",
                ",".join(tags) if tags else "agentic,local-agent",
            ]
            if model_name:
                cmd.extend(["--model", model_name])
            process = subprocess.Popen(cmd, cwd=str(PACKAGE_PARENT_DIR))
            print(f"[starting] {name} -> {host}:{port} (pid={process.pid})")

            if not _wait_for_agent_card_ready(runtime_base_url, ready_timeout_sec=ready_timeout_sec):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                print(
                    f"[failed] {name} server started but card endpoint not ready within {ready_timeout_sec:.1f}s"
                )
                log_event(
                    "a2a.agent_server.launcher",
                    "launch_failed_card_not_ready",
                    {
                        "agent": name,
                        "host": host,
                        "port": port,
                        "pid": process.pid,
                        "base_url": runtime_base_url,
                        "card_url": _card_url(runtime_base_url),
                        "ready_timeout_sec": ready_timeout_sec,
                    },
                    level="ERROR",
                )
                continue

            processes.append((name, process, host, port))
            runtime_cards.append(
                _build_runtime_card(
                    card,
                    runtime_base_url=runtime_base_url,
                    runtime_model=model_name,
                )
            )
            print(f"[started] {name} -> {host}:{port} (pid={process.pid})")
            log_event(
                "a2a.agent_server.launcher",
                "launch_completed",
                {
                    "agent": name,
                    "host": host,
                    "port": port,
                    "pid": process.pid,
                    "command": cmd,
                    "card_url": _card_url(runtime_base_url),
                    "base_url": runtime_base_url,
                    "configured_base_url": base_url,
                    "model": model_name,
                    "ready_timeout_sec": ready_timeout_sec,
                },
            )
        except Exception as e:
            print(f"[failed] {name}: {e}")
            log_exception(
                "a2a.agent_server.launcher",
                "launch_failed",
                e,
                {
                    "agent": name,
                    "server_module": server_module,
                    "configured_base_url": base_url,
                    "runtime_base_url": runtime_base_url,
                },
            )

    return processes, runtime_cards


def stop_agent_servers(processes: List[AgentServerProcess]) -> None:
    for name, process, host, port in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        log_event(
            "a2a.agent_server.launcher",
            "agent_server_process_stopped",
            {"agent": name, "host": host, "port": port, "pid": process.pid},
        )


def launch_bridges(
    card_path: str | Path = DEFAULT_CARD_PATH,
    only: str = "",
    model_overrides: Dict[str, str] | None = None,
) -> Tuple[List[AgentServerProcess], List[Dict[str, Any]]]:
    return launch_agent_servers(card_path=card_path, only=only, model_overrides=model_overrides)


def stop_bridges(processes: List[AgentServerProcess]) -> None:
    stop_agent_servers(processes)


def main() -> None:
    try:
        initialize_process_logging()
        _load_env_file()
        args = _parse_args()

        processes, runtime_cards = launch_agent_servers(card_path=args.card_path, only=args.only)
        if runtime_cards:
            os.environ[RUNTIME_AGENT_CARDS_ENV_KEY] = json.dumps(runtime_cards, ensure_ascii=False)
            resolved = ", ".join(
                f"{str(item.get('name', '')).strip()}={str(item.get('base_url', '')).strip()}"
                for item in runtime_cards
                if str(item.get("name", "")).strip() and str(item.get("base_url", "")).strip()
            )
            if resolved:
                print(f"Resolved A2A agent endpoints: {resolved}")
        if not processes:
            print("No new agent server process was started.")
            return

        print("A2A agent servers are running. Press Ctrl+C to stop all.")
        try:
            while True:
                time.sleep(1.0)
                dead: List[str] = []
                for name, process, host, port in processes:
                    if process.poll() is not None:
                        dead.append(f"{name}({host}:{port})")
                if dead:
                    print(f"[warning] exited agent server process: {', '.join(dead)}")
                    log_event(
                        "a2a.agent_server.launcher",
                        "agent_server_process_exited",
                        {"agents": dead},
                        level="ERROR",
                    )
                    break
        except KeyboardInterrupt:
            pass
        finally:
            stop_agent_servers(processes)
            print("All agent server processes stopped.")
    finally:
        finalize_process_logging()


if __name__ == "__main__":
    main()
