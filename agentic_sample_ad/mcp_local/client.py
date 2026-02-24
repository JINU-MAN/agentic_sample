from __future__ import annotations

import asyncio
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentic_sample_ad.system_logger import log_event, log_exception


def _child_process_env(*, pythonpath_prepend: Path | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    if pythonpath_prepend is not None:
        prepend = str(pythonpath_prepend)
        current = str(env.get("PYTHONPATH", "")).strip()
        env["PYTHONPATH"] = os.pathsep.join([prepend, current]) if current else prepend
    return env


def _server_params(server_script_path: str) -> StdioServerParameters:
    """
    Build stdio server params from an MCP server path.
    """
    server_path = Path(server_script_path).resolve()

    if server_path.parent.name == "mcp_local" and server_path.name.endswith("_server.py"):
        # Standalone-friendly execution:
        # run package module and pin PYTHONPATH to package parent.
        project_root = server_path.parent.parent
        package_name = project_root.name
        module = f"{package_name}.mcp_local.{server_path.stem}"
        child_env = _child_process_env(pythonpath_prepend=project_root.parent)
        log_event(
            "mcp_client",
            "server_params_resolved",
            {
                "server_script_path": str(server_path),
                "execution_mode": "module",
                "module": module,
                "project_root": str(project_root),
                "pythonpath_prepend": str(project_root.parent),
            },
        )
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", module],
            env=child_env,
        )

    log_event(
        "mcp_client",
        "server_params_resolved",
        {
            "server_script_path": str(server_path),
            "execution_mode": "script",
        },
    )
    return StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env=_child_process_env(),
    )


async def _async_call_mcp_tool(
    server_script_path: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute MCP server over stdio and call one tool.
    """
    log_event(
        "mcp_client",
        "tool_call_started",
        {
            "server_script_path": server_script_path,
            "tool_name": tool_name,
            "arguments": arguments,
        },
        direction="outbound",
    )

    async with AsyncExitStack() as stack:
        server_params = _server_params(server_script_path)
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        result = await session.call_tool(tool_name, arguments)
        dumped = result.model_dump(mode="json", exclude_none=True)
        log_event(
            "mcp_client",
            "tool_call_completed",
            {
                "server_script_path": server_script_path,
                "tool_name": tool_name,
                "result": dumped,
            },
            direction="inbound",
        )
        return dumped


def call_mcp_tool(
    server_script_path: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Sync wrapper that can be used from regular functions.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        log_event("mcp_client", "call_mode", {"mode": "sync_asyncio_run", "tool_name": tool_name})
        try:
            return asyncio.run(_async_call_mcp_tool(server_script_path, tool_name, arguments))
        except Exception as e:
            log_exception(
                "mcp_client",
                "tool_call_failed",
                e,
                {
                    "server_script_path": server_script_path,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "mode": "sync_asyncio_run",
                },
            )
            raise

    result: Dict[str, Any] = {}
    error: Exception | None = None
    log_event("mcp_client", "call_mode", {"mode": "threaded_event_loop", "tool_name": tool_name})

    def _run_in_thread() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(_async_call_mcp_tool(server_script_path, tool_name, arguments))
        except Exception as e:  # pragma: no cover
            error = e

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        log_exception(
            "mcp_client",
            "tool_call_failed",
            error,
            {
                "server_script_path": server_script_path,
                "tool_name": tool_name,
                "arguments": arguments,
                "mode": "threaded_event_loop",
            },
        )
        raise error

    log_event(
        "mcp_client",
        "tool_call_returned",
        {
            "server_script_path": server_script_path,
            "tool_name": tool_name,
            "result": result,
        },
        direction="inbound",
    )
    return result


__all__ = ["call_mcp_tool"]
