from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from agentic_sample_ad.system_logger import initialize_process_logging, log_event, log_exception


BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    env_path = BASE_DIR / ".env"
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


def _resolve_local_agent(module_name: str, attr_name: str) -> Tuple[str, LlmAgent]:
    module = importlib.import_module(module_name)
    if not hasattr(module, attr_name):
        raise AttributeError(f"Attribute '{attr_name}' not found in module '{module_name}'.")
    agent_obj = getattr(module, attr_name)
    if not isinstance(agent_obj, LlmAgent):
        raise TypeError(f"Resolved object '{module_name}.{attr_name}' is not LlmAgent.")
    name = str(getattr(agent_obj, "name", "")).strip() or attr_name
    return name, agent_obj


def _extract_user_text(message_payload: Dict[str, Any]) -> str:
    parts = message_payload.get("parts", [])
    if not isinstance(parts, list):
        return ""

    chunks: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if str(part.get("kind", "")).strip().lower() != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks).strip()


async def _run_local_agent(agent_obj: LlmAgent, agent_name: str, user_input: str) -> str:
    runner = InMemoryRunner(agent=agent_obj, app_name=f"a2a-bridge-{agent_name}")
    log_event(
        "a2a.bridge",
        "agent_execution_started",
        {"agent": agent_name, "user_input": user_input},
        direction="outbound",
    )
    try:
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="a2a-bridge-user",
        )
        new_message = types.Content(role="user", parts=[types.Part(text=user_input)])

        chunks: List[str] = []
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=new_message,
        ):
            if event.content and event.content.parts:
                text = "".join(part.text or "" for part in event.content.parts).strip()
                if text:
                    chunks.append(text)

        response_text = "\n".join(chunks).strip() or "(No text response emitted.)"
        log_event(
            "a2a.bridge",
            "agent_execution_completed",
            {"agent": agent_name, "response": response_text},
            direction="inbound",
        )
        return response_text
    finally:
        await runner.close()


def _build_agent_card(
    *,
    agent_name: str,
    description: str,
    host: str,
    port: int,
    tags: List[str],
) -> Dict[str, Any]:
    base_url = f"http://{host}:{port}"
    return {
        "name": agent_name,
        "description": description,
        "url": f"{base_url}/",
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "preferredTransport": "JSONRPC",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "respond",
                "name": "respond",
                "description": "Handle user message and return agent response.",
                "tags": tags,
                "examples": ["Summarize this request and provide actionable result."],
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
            }
        ],
    }


def _jsonrpc_error_response(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def create_app(
    *,
    module_name: str,
    attr_name: str,
    agent_name: str,
    agent_obj: LlmAgent,
    description: str,
    host: str,
    port: int,
    tags: List[str],
) -> FastAPI:
    card_payload = _build_agent_card(
        agent_name=agent_name,
        description=description,
        host=host,
        port=port,
        tags=tags,
    )
    app = FastAPI(title=f"A2A Bridge - {agent_name}")

    @app.get("/.well-known/agent-card.json")
    async def get_agent_card() -> Dict[str, Any]:
        log_event(
            "a2a.bridge",
            "agent_card_requested",
            {"agent": agent_name, "module": module_name, "attr": attr_name},
            direction="inbound",
        )
        return card_payload

    @app.post("/")
    async def handle_jsonrpc(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(_jsonrpc_error_response(None, -32700, "Invalid JSON payload."))

        if not isinstance(payload, dict):
            return JSONResponse(_jsonrpc_error_response(None, -32600, "Invalid JSON-RPC request object."))

        request_id = payload.get("id")
        method = str(payload.get("method", "")).strip()
        if method != "message/send":
            return JSONResponse(
                _jsonrpc_error_response(request_id, -32601, f"Unsupported method: {method or '(empty)'}")
            )

        params = payload.get("params", {})
        if not isinstance(params, dict):
            return JSONResponse(_jsonrpc_error_response(request_id, -32602, "Invalid params object."))

        message_payload = params.get("message", {})
        if not isinstance(message_payload, dict):
            return JSONResponse(_jsonrpc_error_response(request_id, -32602, "Missing or invalid params.message."))

        user_input = _extract_user_text(message_payload)
        if not user_input:
            return JSONResponse(_jsonrpc_error_response(request_id, -32602, "No text part found in message."))

        log_event(
            "a2a.bridge",
            "rpc_request_received",
            {
                "agent": agent_name,
                "request_id": request_id,
                "method": method,
                "input_chars": len(user_input),
            },
            direction="inbound",
        )

        try:
            response_text = await _run_local_agent(agent_obj=agent_obj, agent_name=agent_name, user_input=user_input)
            result_message: Dict[str, Any] = {
                "kind": "message",
                "messageId": uuid4().hex,
                "role": "agent",
                "parts": [{"kind": "text", "text": response_text}],
            }

            context_id = message_payload.get("contextId")
            if isinstance(context_id, str) and context_id.strip():
                result_message["contextId"] = context_id
            task_id = message_payload.get("taskId")
            if isinstance(task_id, str) and task_id.strip():
                result_message["taskId"] = task_id

            response_payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result_message,
            }
            log_event(
                "a2a.bridge",
                "rpc_request_completed",
                {"agent": agent_name, "request_id": request_id},
                direction="outbound",
            )
            return JSONResponse(response_payload)
        except Exception as e:
            log_exception(
                "a2a.bridge",
                "rpc_request_failed",
                e,
                {"agent": agent_name, "request_id": request_id},
            )
            return JSONResponse(_jsonrpc_error_response(request_id, -32000, str(e)))

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local LlmAgent as an A2A-compatible JSON-RPC bridge server.")
    parser.add_argument("--module", required=True, help="Python module path (e.g., agent.web_search_agent)")
    parser.add_argument("--attr", required=True, help="Agent object attribute in module")
    parser.add_argument("--name", default="", help="Override agent display name")
    parser.add_argument("--description", default="", help="Agent card description override")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, required=True, help="Bind port")
    parser.add_argument("--tags", default="local-bridge,agentic", help="Comma-separated tags for agent card skill")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    return parser.parse_args()


def main() -> None:
    initialize_process_logging()
    _load_env_file()
    args = _parse_args()

    resolved_name, agent_obj = _resolve_local_agent(args.module, args.attr)
    agent_name = args.name.strip() or resolved_name
    description = args.description.strip() or f"A2A bridge for local agent '{agent_name}'."
    tags = [token.strip() for token in str(args.tags).split(",") if token.strip()]
    if not tags:
        tags = ["local-bridge", "agentic"]

    app = create_app(
        module_name=args.module,
        attr_name=args.attr,
        agent_name=agent_name,
        agent_obj=agent_obj,
        description=description,
        host=args.host,
        port=args.port,
        tags=tags,
    )
    log_event(
        "a2a.bridge",
        "server_starting",
        {
            "agent": agent_name,
            "module": args.module,
            "attr": args.attr,
            "host": args.host,
            "port": args.port,
        },
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=str(args.log_level).lower())


if __name__ == "__main__":
    main()


