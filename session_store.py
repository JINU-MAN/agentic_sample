from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from system_logger import log_event


@dataclass
class SessionContext:
    """
    In-memory context for one interactive chat session.
    """

    session_id: str
    turns: List[Dict[str, str]] = field(default_factory=list)
    max_turns: int = 20

    def add_user_turn(self, text: str) -> None:
        self.turns.append({"role": "user", "text": text})
        self._trim()
        log_event(
            "session_store",
            "user_turn_added",
            {"session_id": self.session_id, "text": text, "turn_count": len(self.turns)},
            direction="inbound",
        )

    def add_assistant_turn(self, text: str) -> None:
        self.turns.append({"role": "assistant", "text": text})
        self._trim()
        log_event(
            "session_store",
            "assistant_turn_added",
            {"session_id": self.session_id, "text": text, "turn_count": len(self.turns)},
            direction="outbound",
        )

    def history_as_text(self, limit: int = 8) -> str:
        if not self.turns:
            return ""
        selected = self.turns[-(limit * 2) :]
        lines: List[str] = []
        for item in selected:
            role = item.get("role", "unknown").upper()
            text = item.get("text", "").strip()
            if not text:
                continue
            lines.append(f"{role}: {text}")
        return "\n".join(lines).strip()

    def clear(self) -> None:
        self.turns.clear()

    def _trim(self) -> None:
        overflow = len(self.turns) - self.max_turns
        if overflow > 0:
            del self.turns[0:overflow]


_SESSIONS: Dict[str, SessionContext] = {}


def get_or_create_session(session_id: str) -> SessionContext:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionContext(session_id=session_id)
        log_event("session_store", "session_created", {"session_id": session_id})
    return _SESSIONS[session_id]


def clear_session(session_id: str) -> None:
    session = _SESSIONS.get(session_id)
    if session is not None:
        session.clear()
        log_event("session_store", "session_cleared", {"session_id": session_id})
