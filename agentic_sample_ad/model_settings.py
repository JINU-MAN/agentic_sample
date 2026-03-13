from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict


AGENT_DEFAULT_MODEL_ENV_KEY = "AGENTIC_DEFAULT_MODEL"
AGENT_MODEL_OVERRIDES_ENV_KEY = "AGENTIC_AGENT_MODEL_OVERRIDES"
DEFAULT_AGENT_MODEL = "gemini-2.5-flash-lite"


def normalize_agent_name(name: str) -> str:
    return " ".join(str(name or "").split()).strip().lower()


def read_default_model() -> str:
    token = str(os.getenv(AGENT_DEFAULT_MODEL_ENV_KEY, "")).strip()
    return token or DEFAULT_AGENT_MODEL


def write_default_model(model_name: str) -> None:
    token = str(model_name or "").strip()
    if token:
        os.environ[AGENT_DEFAULT_MODEL_ENV_KEY] = token
    else:
        os.environ.pop(AGENT_DEFAULT_MODEL_ENV_KEY, None)


def read_model_overrides() -> Dict[str, str]:
    raw = str(os.getenv(AGENT_MODEL_OVERRIDES_ENV_KEY, "")).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}

    overrides: Dict[str, str] = {}
    for key, value in parsed.items():
        agent_key = normalize_agent_name(str(key))
        model_name = str(value or "").strip()
        if not agent_key or not model_name:
            continue
        overrides[agent_key] = model_name
    return overrides


def write_model_overrides(overrides: Dict[str, str]) -> None:
    payload: Dict[str, str] = {}
    for key, value in (overrides or {}).items():
        agent_key = normalize_agent_name(str(key))
        model_name = str(value or "").strip()
        if not agent_key or not model_name:
            continue
        payload[agent_key] = model_name
    if payload:
        os.environ[AGENT_MODEL_OVERRIDES_ENV_KEY] = json.dumps(payload, ensure_ascii=False)
    else:
        os.environ.pop(AGENT_MODEL_OVERRIDES_ENV_KEY, None)


def _env_file_path() -> Path:
    return Path(__file__).resolve().parent / ".env"


def _upsert_env_key(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    updated = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            current_key, _ = line.split("=", 1)
            if current_key.strip() == key:
                updated.append(f"{key}={value}")
                replaced = True
                continue
        updated.append(line)

    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def _remove_env_key(env_path: Path, key: str) -> None:
    if not env_path.exists():
        return
    updated = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            current_key, _ = line.split("=", 1)
            if current_key.strip() == key:
                continue
        updated.append(line)
    env_path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def persist_default_model(model_name: str) -> None:
    token = str(model_name or "").strip()
    if not token:
        raise ValueError("Default model must not be empty.")
    write_default_model(token)
    _upsert_env_key(_env_file_path(), AGENT_DEFAULT_MODEL_ENV_KEY, token)


def persist_model_overrides(overrides: Dict[str, str]) -> None:
    payload: Dict[str, str] = {}
    for key, value in (overrides or {}).items():
        agent_key = normalize_agent_name(str(key))
        model_name = str(value or "").strip()
        if not agent_key or not model_name:
            continue
        payload[agent_key] = model_name

    write_model_overrides(payload)
    env_path = _env_file_path()
    if payload:
        _upsert_env_key(env_path, AGENT_MODEL_OVERRIDES_ENV_KEY, json.dumps(payload, ensure_ascii=False))
    else:
        _remove_env_key(env_path, AGENT_MODEL_OVERRIDES_ENV_KEY)


def resolve_agent_model(agent_name: str, fallback: str = "") -> str:
    overrides = read_model_overrides()
    key = normalize_agent_name(agent_name)
    if key and key in overrides:
        return overrides[key]

    fallback_token = str(fallback or "").strip()
    if fallback_token:
        return fallback_token
    return read_default_model()
