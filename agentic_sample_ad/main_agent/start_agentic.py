from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from agentic_sample_ad.scripts.start_a2a_agents import launch_agent_servers, stop_agent_servers
from agentic_sample_ad.model_settings import (
    normalize_agent_name,
    read_default_model,
    read_model_overrides,
    write_model_overrides,
)

from .card_registry import collect_sub_agent_card_files
from .system_logger import (
    finalize_main_logging,
    initialize_main_logging,
    log_main_event,
    log_main_exception,
    start_main_logging_session,
)
from .user_entry_point import input_loop


AgentServerProcess = Tuple[str, Any, str, int]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_AGENT_CARDS_ENV_KEY = "AGENTIC_RUNTIME_AGENT_CARDS"


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


def _publish_runtime_agent_cards(cards: List[Dict[str, Any]]) -> None:
    dedup: Dict[str, Dict[str, Any]] = {}
    for raw in cards:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        base_url = str(raw.get("base_url", "")).strip()
        if not name or not base_url:
            continue
        payload = dict(raw)
        payload["name"] = name
        payload["base_url"] = base_url
        dedup[name.lower()] = payload

    runtime_cards = list(dedup.values())
    if not runtime_cards:
        os.environ.pop(RUNTIME_AGENT_CARDS_ENV_KEY, None)
        log_main_event("runtime_agent_cards_published", {"count": 0})
        return

    os.environ[RUNTIME_AGENT_CARDS_ENV_KEY] = json.dumps(runtime_cards, ensure_ascii=False)
    log_main_event(
        "runtime_agent_cards_published",
        {
            "count": len(runtime_cards),
            "agents": [
                {
                    "name": str(item.get("name", "")),
                    "base_url": str(item.get("base_url", "")),
                }
                for item in runtime_cards
            ],
        },
    )


def launch_sub_agent_servers(
    *,
    only: str = "",
    model_overrides: Dict[str, str] | None = None,
    publish_runtime_cards: bool = True,
) -> Tuple[List[AgentServerProcess], List[Dict[str, Any]]]:
    servers: List[AgentServerProcess] = []
    runtime_cards: List[Dict[str, Any]] = []
    card_files = collect_sub_agent_card_files()
    log_main_event(
        "agent_server_launch_started",
        {
            "card_files": [str(path) for path in card_files],
            "only": str(only or "").strip(),
            "model_overrides": model_overrides or {},
        },
    )
    for card_file in card_files:
        try:
            started, resolved_cards = launch_agent_servers(
                card_path=card_file,
                only=only,
                model_overrides=model_overrides,
            )
            servers.extend(started)
            runtime_cards.extend(resolved_cards)
        except Exception as e:
            log_main_exception("agent_server_launch_failed", e, {"card_file": str(card_file)})

    if publish_runtime_cards:
        _publish_runtime_agent_cards(runtime_cards)
    log_main_event(
        "agent_server_launch_completed",
        {
            "count": len(servers),
            "servers": [name for name, *_ in servers],
            "runtime_cards": [
                {
                    "name": str(item.get("name", "")),
                    "base_url": str(item.get("base_url", "")),
                }
                for item in runtime_cards
            ],
        },
    )
    return servers, runtime_cards


def _collect_configured_sub_agent_names() -> Dict[str, str]:
    names: Dict[str, str] = {}
    card_files = collect_sub_agent_card_files()
    for card_file in card_files:
        try:
            payload = json.loads(card_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        cards: List[Dict[str, Any]]
        if isinstance(payload, list):
            cards = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            cards = [payload]
        else:
            cards = []
        for item in cards:
            raw_name = str(item.get("name", "")).strip()
            key = normalize_agent_name(raw_name)
            if key and raw_name:
                names[key] = raw_name
    return names


def _merge_runtime_cards(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source in [existing, incoming]:
        for item in source:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            key = normalize_agent_name(name)
            if not key:
                continue
            merged[key] = dict(item)
    return list(merged.values())


def _make_model_setting_handler(
    *,
    get_servers: Callable[[], List[AgentServerProcess]],
    set_servers: Callable[[List[AgentServerProcess]], None],
    get_runtime_cards: Callable[[], List[Dict[str, Any]]],
    set_runtime_cards: Callable[[List[Dict[str, Any]]], None],
) -> Callable[[str, str], str]:
    configured_agents = _collect_configured_sub_agent_names()

    def _handler(agent_name: str, model_name: str) -> str:
        raw_agent = str(agent_name or "").strip()
        raw_model = str(model_name or "").strip()
        if not raw_agent or not raw_model:
            return "Usage: /setting {AgentName} -m {ModelName}"

        target_key = normalize_agent_name(raw_agent)
        default_model = read_default_model()
        if normalize_agent_name("MainAgent") == target_key:
            model_overrides = read_model_overrides()
            model_overrides[target_key] = raw_model
            write_model_overrides(model_overrides)
            log_main_event(
                "agent_model_setting_applied",
                {
                    "agent": "MainAgent",
                    "model": raw_model,
                    "default_model": default_model,
                    "reboot": "not_required",
                },
            )
            return (
                f"[setting] MainAgent model updated: {raw_model}\n"
                "MainAgent is recreated on each request, so no server reboot was required."
            )

        configured_name = configured_agents.get(target_key, raw_agent)
        if target_key not in configured_agents:
            return (
                f"[setting] Unknown agent: {raw_agent}\n"
                "Available: MainAgent, PaperAnalyst, SocialMediaAnalyst, WebSearchAnalyst"
            )

        model_overrides = read_model_overrides()
        model_overrides[target_key] = raw_model
        write_model_overrides(model_overrides)

        servers = list(get_servers())
        runtime_cards = list(get_runtime_cards())

        stop_targets = [entry for entry in servers if normalize_agent_name(entry[0]) == target_key]
        if stop_targets:
            stop_agent_servers(stop_targets)
            servers = [entry for entry in servers if normalize_agent_name(entry[0]) != target_key]

        runtime_cards = [
            item
            for item in runtime_cards
            if normalize_agent_name(str(item.get("name", ""))) != target_key
        ]

        started, resolved_cards = launch_sub_agent_servers(
            only=configured_name,
            model_overrides=model_overrides,
            publish_runtime_cards=False,
        )
        servers.extend(started)
        runtime_cards = _merge_runtime_cards(runtime_cards, resolved_cards)
        _publish_runtime_agent_cards(runtime_cards)

        set_servers(servers)
        set_runtime_cards(runtime_cards)

        if not resolved_cards:
            log_main_event(
                "agent_model_setting_failed",
                {"agent": configured_name, "model": raw_model, "reason": "restart_failed"},
            )
            return (
                f"[setting] {configured_name} model override saved: {raw_model}\n"
                "But agent reboot failed. Check logs and retry."
            )

        base_url = str(resolved_cards[0].get("base_url", "")).strip()
        log_main_event(
            "agent_model_setting_applied",
            {
                "agent": configured_name,
                "model": raw_model,
                "base_url": base_url,
                "default_model": default_model,
                "reboot": "completed",
            },
        )
        return (
            f"[setting] {configured_name} model updated: {raw_model}\n"
            f"Agent server rebooted at {base_url}"
        )

    return _handler


def main() -> None:
    # Reset session log once at process start, not during workflow execution.
    start_main_logging_session(reset_files=True)
    initialize_main_logging()
    _load_env_file()
    model_overrides = read_model_overrides()
    servers, runtime_cards = launch_sub_agent_servers(model_overrides=model_overrides)
    if servers:
        endpoints = ", ".join(
            f"{str(item.get('name', '')).strip()}={str(item.get('base_url', '')).strip()}"
            for item in runtime_cards
            if str(item.get("name", "")).strip() and str(item.get("base_url", "")).strip()
        )
        if endpoints:
            print(f"A2A agent servers started: {endpoints}")
        else:
            names = ", ".join(name for name, *_ in servers)
            print(f"A2A agent servers started: {names}")
    else:
        print("A2A agent servers were not newly started. Continuing.")

    def _get_servers() -> List[AgentServerProcess]:
        return list(servers)

    def _set_servers(updated: List[AgentServerProcess]) -> None:
        nonlocal servers
        servers = list(updated)

    def _get_runtime_cards() -> List[Dict[str, Any]]:
        return list(runtime_cards)

    def _set_runtime_cards(updated: List[Dict[str, Any]]) -> None:
        nonlocal runtime_cards
        runtime_cards = list(updated)

    setting_handler = _make_model_setting_handler(
        get_servers=_get_servers,
        set_servers=_set_servers,
        get_runtime_cards=_get_runtime_cards,
        set_runtime_cards=_set_runtime_cards,
    )

    try:
        input_loop(on_model_setting=setting_handler)
    finally:
        try:
            if servers:
                stop_agent_servers(servers)
                print("A2A agent servers stopped.")
        finally:
            finalize_main_logging()


if __name__ == "__main__":
    main()

