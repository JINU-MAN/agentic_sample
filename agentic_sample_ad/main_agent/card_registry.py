from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


AD_ROOT = Path(__file__).resolve().parents[1]
AGENT_CARDS_FILE = AD_ROOT / "agent_cards" / "agent_card.json"
EXTRA_CARD_PATHS_ENV_KEY = "A2A_EXTRA_CARD_PATHS"
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    "agent",
    "agent_cards",
    "common",
    "db",
    "log",
    "main_agent",
    "mcp_local",
    "scripts",
}
RUNTIME_AGENT_CARDS_ENV_KEY = "AGENTIC_RUNTIME_AGENT_CARDS"


def sub_agent_dirs() -> List[Path]:
    dirs: List[Path] = []
    for path in sorted(AD_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        if path.name in EXCLUDED_DIR_NAMES:
            continue
        candidate = path / "well_known" / "agent_card.json"
        if candidate.exists() and candidate.is_file():
            dirs.append(path)
    return dirs


def collect_sub_agent_card_files() -> List[Path]:
    card_files: List[Path] = []
    for directory in sub_agent_dirs():
        candidate = directory / "well_known" / "agent_card.json"
        if candidate.exists() and candidate.is_file():
            card_files.append(candidate)
    return card_files


def _split_card_path_tokens(raw: str) -> List[str]:
    normalized = str(raw or "").strip()
    if not normalized:
        return []

    # Allow both platform separator and comma-separated list.
    parts: List[str] = []
    for token in normalized.replace(",", os.pathsep).split(os.pathsep):
        item = token.strip()
        if item:
            parts.append(item)
    return parts


def _resolve_card_path(token: str) -> Path:
    candidate = Path(token.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (AD_ROOT / candidate).resolve()


def _collect_extra_card_files() -> List[Path]:
    files: List[Path] = []
    seen: set[str] = set()

    if AGENT_CARDS_FILE.exists() and AGENT_CARDS_FILE.is_file():
        resolved = AGENT_CARDS_FILE.resolve()
        key = str(resolved).lower()
        seen.add(key)
        files.append(resolved)

    for token in _split_card_path_tokens(os.getenv(EXTRA_CARD_PATHS_ENV_KEY, "")):
        path = _resolve_card_path(token)
        if not path.exists() or not path.is_file():
            continue
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        files.append(path)
    return files


def _load_cards_from_file(path: Path) -> List[Dict[str, Any]]:
    # `utf-8-sig` allows BOM-prefixed JSON card files created by Windows editors.
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _runtime_cards_from_env() -> Dict[str, Dict[str, Any]]:
    raw = str(os.getenv(RUNTIME_AGENT_CARDS_ENV_KEY, "")).strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except Exception:
        return {}

    runtime_cards: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        runtime_cards = [item for item in parsed if isinstance(item, dict)]
    elif isinstance(parsed, dict):
        for key, value in parsed.items():
            if isinstance(value, dict):
                payload = dict(value)
                payload.setdefault("name", str(key))
                runtime_cards.append(payload)
            elif isinstance(value, str):
                runtime_cards.append({"name": str(key), "base_url": value})

    mapped: Dict[str, Dict[str, Any]] = {}
    for item in runtime_cards:
        name = str(item.get("name", "")).strip()
        base_url = str(item.get("base_url", "")).strip()
        if not name or not base_url:
            continue
        payload = dict(item)
        payload["name"] = name
        payload["base_url"] = base_url
        mapped[name.lower()] = payload
    return mapped


def load_sub_agent_cards() -> List[Dict[str, Any]]:
    loaded: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    runtime_overrides = _runtime_cards_from_env()
    source_files: List[Path] = collect_sub_agent_card_files() + _collect_extra_card_files()

    for card_file in source_files:
        try:
            cards = _load_cards_from_file(card_file)
        except Exception:
            continue
        for card in cards:
            item = dict(card)
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)

            runtime = runtime_overrides.get(key)
            if runtime is not None:
                if str(runtime.get("base_url", "")).strip():
                    item["base_url"] = str(runtime.get("base_url", "")).strip()
                for field in ("type", "module", "attr", "server_module", "description", "capabilities"):
                    if field not in runtime:
                        continue
                    value = runtime.get(field)
                    if value is None:
                        continue
                    if isinstance(value, str) and not value.strip():
                        continue
                    item[field] = value
                item["runtime_endpoint_source"] = "env"

            if not str(item.get("type", "")).strip():
                item["type"] = "a2a"
            item["source_card_path"] = str(card_file)
            loaded.append(item)

    for key, runtime in runtime_overrides.items():
        if key in seen_names:
            continue
        item = dict(runtime)
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if not str(item.get("type", "")).strip():
            item["type"] = "a2a"
        item["source_card_path"] = "<runtime_env>"
        item["runtime_endpoint_source"] = "env_only"
        loaded.append(item)
        seen_names.add(key)

    return loaded
