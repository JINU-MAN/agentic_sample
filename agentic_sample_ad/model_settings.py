from __future__ import annotations

import json
import os
from typing import Dict


AGENT_DEFAULT_MODEL_ENV_KEY = "AGENTIC_DEFAULT_MODEL"
AGENT_MODEL_OVERRIDES_ENV_KEY = "AGENTIC_AGENT_MODEL_OVERRIDES"
DEFAULT_AGENT_MODEL = "gemini-2.5-flash-lite"


def normalize_agent_name(name: str) -> str:
    return " ".join(str(name or "").split()).strip().lower()


def read_default_model() -> str:
    token = str(os.getenv(AGENT_DEFAULT_MODEL_ENV_KEY, "")).strip()
    return token or DEFAULT_AGENT_MODEL


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


def resolve_agent_model(agent_name: str, fallback: str = "") -> str:
    overrides = read_model_overrides()
    key = normalize_agent_name(agent_name)
    if key and key in overrides:
        return overrides[key]

    fallback_token = str(fallback or "").strip()
    if fallback_token:
        return fallback_token
    return read_default_model()

