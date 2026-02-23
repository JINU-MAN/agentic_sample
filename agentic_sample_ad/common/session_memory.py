from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List

from agentic_sample_ad.system_logger import log_event


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionMemory:
    session_id: str
    turns: List[Dict[str, str]] = field(default_factory=list)
    workflow_contexts: List[Dict[str, Any]] = field(default_factory=list)

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(
            {
                "role": str(role).strip() or "unknown",
                "text": str(text),
                "ts": _utc_now_iso(),
            }
        )

    def add_user_turn(self, text: str) -> None:
        self.add_turn("user", text)

    def add_assistant_turn(self, text: str) -> None:
        self.add_turn("assistant", text)

    def add_workflow_context(self, payload: Dict[str, Any]) -> None:
        self.workflow_contexts.append(
            {
                "ts": _utc_now_iso(),
                "payload": payload,
            }
        )

    def history_as_text(self, max_turns: int = 14) -> str:
        if not self.turns:
            return ""
        turns = self.turns[-max_turns:]
        return "\n".join(f"{item['role']}: {item['text']}" for item in turns).strip()

    def export(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": list(self.turns),
            "workflow_contexts": list(self.workflow_contexts),
        }


class SessionMemoryStore:
    def __init__(self, component: str) -> None:
        self._component = str(component).strip() or "ad.session_memory"
        self._sessions: Dict[str, SessionMemory] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> SessionMemory:
        normalized = str(session_id).strip() or "default"
        with self._lock:
            if normalized not in self._sessions:
                self._sessions[normalized] = SessionMemory(session_id=normalized)
                log_event(self._component, "session_created", {"session_id": normalized})
            return self._sessions[normalized]

    def clear(self, session_id: str) -> None:
        normalized = str(session_id).strip() or "default"
        with self._lock:
            if normalized in self._sessions:
                del self._sessions[normalized]
            log_event(self._component, "session_cleared", {"session_id": normalized})

    def export(self, session_id: str) -> Dict[str, Any]:
        memory = self.get_or_create(session_id)
        return memory.export()


