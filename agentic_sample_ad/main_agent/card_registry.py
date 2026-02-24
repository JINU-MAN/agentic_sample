from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


AD_ROOT = Path(__file__).resolve().parents[1]
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


def _load_cards_from_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def load_sub_agent_cards() -> List[Dict[str, Any]]:
    loaded: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    for card_file in collect_sub_agent_card_files():
        for card in _load_cards_from_file(card_file):
            item = dict(card)
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)

            if not str(item.get("type", "")).strip():
                item["type"] = "a2a"
            item["source_card_path"] = str(card_file)
            loaded.append(item)

    return loaded
