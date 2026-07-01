from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage


_STRUCTURED_RESULT_TOOLS = {"SubagentResult", "IDCSubagentResult"}
_NON_EXECUTED_STATUSES = {"", "not_executed", "not executed", "pending"}


def _response_messages(response: ModelResponse | AIMessage) -> Iterable[BaseMessage]:
    if isinstance(response, AIMessage):
        return [response]
    return response.result


def _structured_calls(
    response: ModelResponse | AIMessage,
) -> list[tuple[AIMessage, int, dict[str, Any]]]:
    calls: list[tuple[AIMessage, int, dict[str, Any]]] = []
    for message in _response_messages(response):
        if not isinstance(message, AIMessage):
            continue
        for index, call in enumerate(message.tool_calls or []):
            if str(call.get("name") or "") in _STRUCTURED_RESULT_TOOLS:
                calls.append((message, index, call))
    return calls


def unsupported_tool_claims(
    state: dict[str, Any],
    response: ModelResponse | AIMessage,
) -> list[str]:
    """Return model-authored subagent claims that lack deterministic state evidence."""

    run_id = str(state.get("current_run_id") or "")
    events = [
        event
        for event in (state.get("tool_events") or [])
        if isinstance(event, dict)
        and (not run_id or str(event.get("run_id") or "") == run_id)
    ]
    artifacts = state.get("artifact_registry") or {}
    data = state.get("data_registry") or {}
    issues: list[str] = []

    for _message, _index, structured_call in _structured_calls(response):
        payload = structured_call.get("args") or {}
        if not isinstance(payload, dict):
            continue
        for claimed in payload.get("tool_calls") or []:
            if not isinstance(claimed, dict):
                continue
            claimed_status = str(claimed.get("status") or "").strip().lower()
            if claimed_status in _NON_EXECUTED_STATUSES:
                continue
            name = str(claimed.get("name") or "").strip()
            call_id = str(claimed.get("call_id") or "").strip()
            matches = [
                event
                for event in events
                if str(event.get("tool_name") or "") == name
                and (
                    not call_id
                    or str(event.get("tool_call_id") or "") == call_id
                )
            ]
            if not matches:
                issues.append(
                    f"claimed tool {name or '[missing name]'}"
                    + (f" ({call_id})" if call_id else "")
                    + " has no matching current-run tool event"
                )
                continue
            if claimed_status in {"ok", "completed", "success"} and not any(
                str(event.get("status") or "") == "ok" for event in matches
            ):
                issues.append(f"claimed successful tool {name} has no successful event")

            if name == "idc_download" and claimed_status in {
                "ok",
                "completed",
                "success",
            }:
                produced_ids = {
                    str(artifact_id)
                    for event in matches
                    for artifact_id in (event.get("artifact_ids") or [])
                }
                produced_dicom = any(
                    isinstance(artifacts.get(artifact_id), dict)
                    and artifacts[artifact_id].get("status") == "verified"
                    and artifacts[artifact_id].get("kind") == "directory"
                    and artifacts[artifact_id].get("source_tool") == "idc_download"
                    for artifact_id in produced_ids
                )
                if not produced_dicom:
                    issues.append(
                        "idc_download success has no verified DICOM directory artifact"
                    )

        for artifact_id in payload.get("artifact_ids") or []:
            if str(artifact_id) not in artifacts:
                issues.append(f"unknown artifact_id {artifact_id}")
        for data_id in payload.get("data_ids") or []:
            if str(data_id) not in data:
                issues.append(f"unknown data_id {data_id}")
    return list(dict.fromkeys(issues))


def _grounding_error_response(
    response: ModelResponse | AIMessage,
    issues: list[str],
) -> ModelResponse | AIMessage:
    error = "grounding_failed: " + "; ".join(issues)

    def replace_message(message: BaseMessage) -> BaseMessage:
        if not isinstance(message, AIMessage):
            return message
        calls = [dict(call) for call in (message.tool_calls or [])]
        changed = False
        for index, call in enumerate(calls):
            if str(call.get("name") or "") not in _STRUCTURED_RESULT_TOOLS:
                continue
            args = dict(call.get("args") or {})
            args.update(
                status="error",
                tool_calls=[],
                artifact_ids=[],
                data_ids=[],
                artifacts=[],
                summary="The requested operation was not executed.",
                errors=[error],
                next_recommended_inputs={},
                visible_outputs=[],
                deliverables=[],
            )
            call["args"] = args
            calls[index] = call
            changed = True
        return message.model_copy(update={"tool_calls": calls}) if changed else message

    if isinstance(response, AIMessage):
        return replace_message(response)  # type: ignore[return-value]
    return response.model_copy(
        update={"result": [replace_message(message) for message in response.result]}
    )


class ToolEvidenceMiddleware(AgentMiddleware):
    """Retry, then reject subagent completion claims unsupported by real tool state."""

    @staticmethod
    def _retry_request(request: ModelRequest, issues: list[str]) -> ModelRequest:
        existing = request.system_message.text if request.system_message is not None else ""
        reminder = (
            "\n\nExecution-grounding guard: your structured result claimed work that has no "
            f"matching deterministic tool evidence: {'; '.join(issues)}. Invoke the real "
            "operational tool now. Do not return a SubagentResult until that tool finishes "
            "and its exact artifact_ids/data_ids appear in the injected state."
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
        issues = unsupported_tool_claims(dict(request.state or {}), response)
        if not issues:
            return response
        retried = handler(self._retry_request(request, issues))
        remaining = unsupported_tool_claims(dict(request.state or {}), retried)
        return _grounding_error_response(retried, remaining) if remaining else retried

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        response = await handler(request)
        issues = unsupported_tool_claims(dict(request.state or {}), response)
        if not issues:
            return response
        retried = await handler(self._retry_request(request, issues))
        remaining = unsupported_tool_claims(dict(request.state or {}), retried)
        return _grounding_error_response(retried, remaining) if remaining else retried
