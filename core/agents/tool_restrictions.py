from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Iterable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command


DEEPAGENTS_FILESYSTEM_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "ls",
        "glob",
        "grep",
        "execute",
    }
)

DEEPAGENTS_HIDDEN_INTERNAL_TOOLS = DEEPAGENTS_FILESYSTEM_TOOLS | {"write_todos"}
"""Built-in DeepAgents tools hidden from VoxelInsight workflows."""


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        name = tool.get("name")
        function = tool.get("function")
        if not name and isinstance(function, dict):
            name = function.get("name")
        return str(name or "")
    return str(getattr(tool, "name", "") or "")


class BlockedToolMiddleware(AgentMiddleware):
    """Hide and reject tools that are outside the VoxelInsight domain contract."""

    def __init__(self, blocked_tools: Iterable[str]) -> None:
        self.blocked_tools = frozenset(str(name) for name in blocked_tools)

    def _request_without_blocked_tools(self, request: ModelRequest) -> ModelRequest:
        tools = [
            tool
            for tool in request.tools
            if _tool_name(tool) not in self.blocked_tools
        ]
        return request.override(tools=tools)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        return handler(self._request_without_blocked_tools(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        return await handler(self._request_without_blocked_tools(request))

    def _blocked_result(self, request: ToolCallRequest) -> ToolMessage:
        tool_name = str(request.tool_call.get("name") or "")
        tool_call_id = str(request.tool_call.get("id") or "")
        message = (
            f"Tool '{tool_name}' is disabled for VoxelInsight domain agents. "
            "Use a registered VoxelInsight tool with artifact_id handoff."
        )
        return ToolMessage(
            content=json.dumps(
                {
                    "ok": False,
                    "status": "error",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "errors": [message],
                    "error": message,
                }
            ),
            name=tool_name,
            tool_call_id=tool_call_id,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if str(request.tool_call.get("name") or "") in self.blocked_tools:
            return self._blocked_result(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> ToolMessage | Command[Any]:
        if str(request.tool_call.get("name") or "") in self.blocked_tools:
            return self._blocked_result(request)
        return await handler(request)


class BlockedSubagentMiddleware(AgentMiddleware):
    """Reject dispatch to disabled DeepAgents subagent targets."""

    def __init__(self, blocked_subagents: Iterable[str]) -> None:
        self.blocked_subagents = frozenset(str(name) for name in blocked_subagents)

    def _blocked_subagent(self, request: ToolCallRequest) -> str | None:
        if str(request.tool_call.get("name") or "") != "task":
            return None
        args = request.tool_call.get("args") or {}
        if not isinstance(args, dict):
            return None
        subagent_type = str(args.get("subagent_type") or "")
        return subagent_type if subagent_type in self.blocked_subagents else None

    def _blocked_result(
        self,
        request: ToolCallRequest,
        subagent_type: str,
    ) -> ToolMessage:
        tool_call_id = str(request.tool_call.get("id") or "")
        message = (
            f"Subagent '{subagent_type}' is disabled for VoxelInsight biomedical "
            "workflows. Route the request to a named domain subagent or answer "
            "directly when all requested work is complete."
        )
        return ToolMessage(
            content=json.dumps(
                {
                    "ok": False,
                    "status": "error",
                    "tool_name": "task",
                    "tool_call_id": tool_call_id,
                    "subagent_type": subagent_type,
                    "errors": [message],
                    "error": message,
                }
            ),
            name="task",
            tool_call_id=tool_call_id,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        subagent_type = self._blocked_subagent(request)
        if subagent_type is not None:
            return self._blocked_result(request, subagent_type)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        subagent_type = self._blocked_subagent(request)
        if subagent_type is not None:
            return self._blocked_result(request, subagent_type)
        return await handler(request)


class IDCViewerIntentMiddleware(AgentMiddleware):
    """Prevent viewer URL generation when the original request is download-only."""

    @staticmethod
    def _without_unsolicited_viewer_urls(request: ToolCallRequest) -> ToolCallRequest:
        if str(request.tool_call.get("name") or "") != "idc_series_search":
            return request
        args = dict(request.tool_call.get("args") or {})
        if not args.get("include_viewer_url"):
            return request
        requested = request.state.get("requested_deliverables") or []
        needs = {
            str(need)
            for deliverable in requested
            if isinstance(deliverable, dict)
            for need in (deliverable.get("needs") or [])
        }
        if "download" not in needs or "viewer_link" in needs:
            return request
        tool_call = dict(request.tool_call)
        args["include_viewer_url"] = False
        tool_call["args"] = args
        return request.override(tool_call=tool_call)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return handler(self._without_unsolicited_viewer_urls(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return await handler(self._without_unsolicited_viewer_urls(request))
