from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from core.agents.visibility import validate_deliverables
from core.agents.external_links import idc_viewer_urls_from_rows


def _records_for_current_run(state: dict[str, Any], key: str) -> list[dict[str, Any]]:
    registry = state.get(key) or {}
    if not isinstance(registry, dict):
        return []
    run_id = str(state.get("current_run_id") or "")
    return [
        record
        for record in registry.values()
        if isinstance(record, dict)
        and (not run_id or str(record.get("run_id") or "") == run_id)
    ]


def _visible_outputs_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for record in _records_for_current_run(state, "data_registry"):
        if record.get("visibility", "user") == "user" and record.get("kind") == "table":
            visible.append(
                {
                    "kind": "dataframe",
                    "source": "tool_outputs",
                    "title": str(record.get("name") or "table"),
                }
            )
            for viewer_url in idc_viewer_urls_from_rows(record.get("rows") or []):
                visible.append(
                    {
                        "kind": "link",
                        "source": "tool_outputs",
                        "title": "Open in IDC Viewer",
                        "url": viewer_url,
                    }
                )

    for record in _records_for_current_run(state, "artifact_registry"):
        if (
            record.get("status") != "verified"
            or record.get("role") == "input"
            or not record.get("downloadable", True)
        ):
            continue
        raw_path = record.get("path")
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        if not path.exists():
            continue
        kind = str(record.get("kind") or "")
        if record.get("source_tool") == "idc_download" and kind != "directory":
            continue
        if kind == "image":
            visible_kind = "image"
        elif kind == "plotly":
            visible_kind = "plotly"
        else:
            visible_kind = "file"
        visible.append(
            {
                "kind": visible_kind,
                "source": "verified_file",
                "title": str(record.get("name") or kind or "artifact"),
                "path": str(path),
            }
        )
    return visible


def _missing_deliverable_names(state: dict[str, Any]) -> list[str]:
    requested = state.get("requested_deliverables") or []
    if not isinstance(requested, list) or not requested:
        return []
    artifacts = _records_for_current_run(state, "artifact_registry")
    results = validate_deliverables(
        requested,
        _visible_outputs_from_state(state),
        artifacts=artifacts,
    )
    return [
        str(result.get("name") or "deliverable")
        for result in results
        if result.get("status") != "satisfied"
    ]


def _has_successful_intermediate_output(state: dict[str, Any]) -> bool:
    run_id = str(state.get("current_run_id") or "")
    events = state.get("tool_events") or []
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            if run_id and str(event.get("run_id") or "") != run_id:
                continue
            if event.get("status") == "ok":
                return True
    return bool(
        _records_for_current_run(state, "data_registry")
        or _records_for_current_run(state, "artifact_registry")
    )


def _response_messages(response: ModelResponse | AIMessage) -> Iterable[BaseMessage]:
    if isinstance(response, AIMessage):
        return [response]
    return response.result


def _response_has_tool_call(response: ModelResponse | AIMessage) -> bool:
    return any(
        isinstance(message, AIMessage) and bool(message.tool_calls)
        for message in _response_messages(response)
    )


class CompletionGuardMiddleware(AgentMiddleware):
    """Give the supervisor one bounded retry when it abandons a ready handoff."""

    @staticmethod
    def _should_retry(
        request: ModelRequest,
        response: ModelResponse | AIMessage,
    ) -> tuple[bool, list[str]]:
        state = dict(request.state or {})
        missing = _missing_deliverable_names(state)
        should_retry = bool(
            missing
            and not _response_has_tool_call(response)
            and _has_successful_intermediate_output(state)
            and request.tools
        )
        return should_retry, missing

    @staticmethod
    def _retry_request(request: ModelRequest, missing: list[str]) -> ModelRequest:
        existing = request.system_message.text if request.system_message is not None else ""
        reminder = (
            "\n\nCompletion guard: successful intermediate outputs exist, but these requested "
            f"deliverables are still missing: {', '.join(missing)}. Do not end with a "
            "progress statement or describe a future action. Make the next concrete `task` "
            "tool call now using the registered outputs. This is the single guarded routing "
            "retry, so choose the next owning subagent directly."
        )
        return request.override(
            system_message=SystemMessage(content=existing + reminder),
            tool_choice="required",
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        response = handler(request)
        should_retry, missing = self._should_retry(request, response)
        if not should_retry:
            return response
        return handler(self._retry_request(request, missing))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        response = await handler(request)
        should_retry, missing = self._should_retry(request, response)
        if not should_retry:
            return response
        return await handler(self._retry_request(request, missing))
