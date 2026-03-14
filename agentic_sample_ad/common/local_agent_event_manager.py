from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List

from google.adk.runners import InMemoryRunner
from google.genai import types

from agentic_sample_ad.network_retry import collect_response_parts_with_network_retry
from agentic_sample_ad.runtime_utils import finalize_text_response, run_coroutine_sync
from agentic_sample_ad.system_logger import log_event as default_log_event, log_exception as default_log_exception

@dataclass
class AgentCommandEvent:
    command: str
    context: Dict[str, Any] = field(default_factory=dict)


class LocalAgentEventManager:
    def __init__(
        self,
        *,
        agent_name: str,
        agent_obj: Any,
        component: str,
        log_event_fn: Callable[..., None] = default_log_event,
        log_exception_fn: Callable[..., None] = default_log_exception,
    ) -> None:
        self._agent_name = str(agent_name).strip() or "UnknownAgent"
        self._agent_obj = agent_obj
        self._component = str(component).strip() or "ad.local_agent.event_manager"
        self._queue: Deque[AgentCommandEvent] = deque()
        self._lock = threading.Lock()
        self._log_event = log_event_fn
        self._log_exception = log_exception_fn

    def enqueue(self, command: str, context: Dict[str, Any] | None = None) -> None:
        event = AgentCommandEvent(command=str(command), context=dict(context or {}))
        with self._lock:
            self._queue.append(event)
        self._log_event(
            self._component,
            "task_enqueued",
            {"agent": self._agent_name, "command": event.command, "queue_size": len(self._queue)},
        )

    def run_next(self) -> Dict[str, Any]:
        with self._lock:
            if not self._queue:
                return {"ok": False, "agent": self._agent_name, "error": "No queued task."}
            event = self._queue.popleft()
        return run_coroutine_sync(self._async_run_command(event))

    def run_until_empty(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        while True:
            with self._lock:
                if not self._queue:
                    break
            results.append(self.run_next())
        return results

    async def _async_run_command(self, event: AgentCommandEvent) -> Dict[str, Any]:
        runner = InMemoryRunner(agent=self._agent_obj, app_name=f"ad-{self._agent_name}-event-manager")
        self._log_event(
            self._component,
            "task_started",
            {
                "agent": self._agent_name,
                "command": event.command,
                "context": event.context,
            },
            direction="outbound",
        )
        try:
            context_json = json.dumps(event.context, ensure_ascii=False, indent=2).strip() if event.context else "{}"
            prompt = (
                "Task command:\n"
                f"{event.command}\n\n"
                "Context:\n"
                f"{context_json}"
            )
            new_message = types.Content(role="user", parts=[types.Part(text=prompt)])
            collected = await collect_response_parts_with_network_retry(
                runner=runner,
                user_id="ad-local-agent-event-manager",
                new_message=new_message,
                component=self._component,
                operation_name=f"local_agent_event:{self._agent_name}",
                retry_details={"agent": self._agent_name, "command": event.command},
                log_event_fn=self._log_event,
            )
            chunks = list(collected.get("chunks", []))
            response_text = finalize_text_response(chunks)
            result = {
                "ok": True,
                "agent": self._agent_name,
                "command": event.command,
                "response": response_text,
                "normalized_response_parts": list(collected.get("normalized_parts", [])),
                "non_text_part_types": list(collected.get("non_text_part_types", [])),
                "used_non_text_fallback": bool(collected.get("used_non_text_fallback")),
            }
            self._log_event(
                self._component,
                "task_completed",
                {
                    "agent": self._agent_name,
                    "command": event.command,
                    "context": event.context,
                    "response": response_text,
                    "normalized_response_parts": list(collected.get("normalized_parts", [])),
                    "non_text_part_types": list(collected.get("non_text_part_types", [])),
                    "used_non_text_fallback": bool(collected.get("used_non_text_fallback")),
                },
                direction="inbound",
            )
            return result
        except Exception as e:
            self._log_exception(
                self._component,
                "task_failed",
                e,
                {"agent": self._agent_name, "command": event.command},
            )
            return {
                "ok": False,
                "agent": self._agent_name,
                "command": event.command,
                "error": str(e),
            }
        finally:
            await runner.close()


