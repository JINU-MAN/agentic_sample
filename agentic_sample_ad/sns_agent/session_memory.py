from __future__ import annotations

from agentic_sample_ad.common.session_memory import SessionMemory, SessionMemoryStore


_STORE = SessionMemoryStore(component="ad.sns_agent.session_memory")


def get_or_create_session(session_id: str) -> SessionMemory:
    return _STORE.get_or_create(session_id)


def clear_session(session_id: str) -> None:
    _STORE.clear(session_id)


def export_session(session_id: str) -> dict:
    return _STORE.export(session_id)

