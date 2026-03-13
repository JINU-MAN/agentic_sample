from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List

from google.adk.runners import InMemoryRunner
from google.genai import types

from agentic_sample_ad.network_retry import collect_text_response_with_network_retry
from agentic_sample_ad.system_logger import log_event as default_log_event, log_exception as default_log_exception


def _run_coroutine_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Any = None
    error: Exception | None = None

    def _target() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except Exception as e:  # pragma: no cover
            error = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result


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
        return _run_coroutine_sync(self._async_run_command(event))

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
            chunks = await collect_text_response_with_network_retry(
                runner=runner,
                user_id="ad-local-agent-event-manager",
                new_message=new_message,
                component=self._component,
                operation_name=f"local_agent_event:{self._agent_name}",
                retry_details={"agent": self._agent_name, "command": event.command},
                log_event_fn=self._log_event,
            )
            response_text = "\n".join(chunks).strip() or "(No text response emitted.)"
            result = {
                "ok": True,
                "agent": self._agent_name,
                "command": event.command,
                "response": response_text,
            }
            self._log_event(
                self._component,
                "task_completed",
                {
                    "agent": self._agent_name,
                    "command": event.command,
                    "context": event.context,
                    "response": response_text,
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


