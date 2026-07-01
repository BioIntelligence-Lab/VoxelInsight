from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from core.agents.artifacts import (
    TOOL_ENVELOPE_SCHEMA_VERSION,
    VoxelAgentState,
    decode_payload,
    registry_delta_from_payload,
    state_context_for_model,
)
from core.agents.durable_registry import load_registry_delta, persist_registry_delta

_EXPLICIT_PATH_ARGUMENTS = {
    "dicom_dir",
    "featureset",
    "files",
    "segmentations",
    "segmentations_map",
    "support_images",
    "support_masks",
    "target_images",
}


class ArtifactResolutionError(ValueError):
    """Raised when a tool receives an unusable artifact reference."""


def _is_path_argument(key: str) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _EXPLICIT_PATH_ARGUMENTS or normalized.endswith(
        ("_path", "_paths", "_file", "_files", "_dir", "_dirs")
    )


def _resolve_path_value(
    value: Any,
    *,
    artifact_registry: dict[str, Any],
    data_registry: dict[str, Any],
    argument_name: str,
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _resolve_path_value(
                nested,
                artifact_registry=artifact_registry,
                data_registry=data_registry,
                argument_name=argument_name,
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_path_value(
                nested,
                artifact_registry=artifact_registry,
                data_registry=data_registry,
                argument_name=argument_name,
            )
            for nested in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _resolve_path_value(
                nested,
                artifact_registry=artifact_registry,
                data_registry=data_registry,
                argument_name=argument_name,
            )
            for nested in value
        )
    if not isinstance(value, str):
        return value

    if value.startswith("data-"):
        data_record = data_registry.get(value)
        if not isinstance(data_record, dict):
            raise ArtifactResolutionError(
                f"Unknown data reference for `{argument_name}`: {value}"
            )
        artifact_id = str(data_record.get("artifact_id") or "")
        if not artifact_id:
            raise ArtifactResolutionError(
                f"Data reference for `{argument_name}` has no registered artifact: {value}"
            )
        return _resolve_path_value(
            artifact_id,
            artifact_registry=artifact_registry,
            data_registry=data_registry,
            argument_name=argument_name,
        )

    if not value.startswith("artifact-"):
        return value

    record = artifact_registry.get(value)
    if not isinstance(record, dict):
        raise ArtifactResolutionError(
            f"Unknown artifact reference for `{argument_name}`: {value}"
        )
    if str(record.get("status") or "verified") != "verified":
        raise ArtifactResolutionError(
            f"Artifact reference for `{argument_name}` is not verified: {value}"
        )
    path_value = record.get("path")
    if not path_value:
        raise ArtifactResolutionError(
            f"Artifact reference for `{argument_name}` has no registered path: {value}"
        )
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        raise ArtifactResolutionError(
            f"Registered path for `{argument_name}` does not exist: {value}"
        )
    return str(path.resolve())


def resolve_artifact_references(
    args: Any,
    artifact_registry: dict[str, Any],
    data_registry: dict[str, Any] | None = None,
) -> Any:
    """Resolve artifact/data IDs only in arguments that represent filesystem paths."""
    if not isinstance(args, dict):
        return args
    return {
        str(key): (
            _resolve_path_value(
                value,
                artifact_registry=artifact_registry,
                data_registry=data_registry or {},
                argument_name=str(key),
            )
            if _is_path_argument(str(key))
            else value
        )
        for key, value in args.items()
    }


class ArtifactRegistryMiddleware(AgentMiddleware):
    """Persist canonical tool artifacts/data and expose exact state to each model."""

    state_schema = VoxelAgentState

    @staticmethod
    def _thread_id(request: ModelRequest | ToolCallRequest) -> str:
        runtime = getattr(request, "runtime", None)
        runtime_config = getattr(runtime, "config", {}) or {}
        configurable = runtime_config.get("configurable", {}) or {}
        return str(configurable.get("thread_id") or "").strip()

    @classmethod
    def _request_with_durable_registry(
        cls,
        request: ModelRequest | ToolCallRequest,
    ) -> ModelRequest | ToolCallRequest:
        """Hydrate sibling-subagent outputs written earlier in the same turn."""

        thread_id = cls._thread_id(request)
        if not thread_id:
            return request
        try:
            durable = load_registry_delta(thread_id)
        except Exception:
            return request
        state = dict(request.state or {})
        artifacts = dict(durable.get("artifact_registry") or {})
        artifacts.update(state.get("artifact_registry") or {})
        data = dict(durable.get("data_registry") or {})
        data.update(state.get("data_registry") or {})
        events_by_id: dict[str, dict[str, Any]] = {}
        for event in [
            *(durable.get("tool_events") or []),
            *(state.get("tool_events") or []),
        ]:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "")
            if event_id:
                events_by_id[event_id] = event
        state.update(
            artifact_registry=artifacts,
            data_registry=data,
            tool_events=list(events_by_id.values()),
        )
        return request.override(state=state)

    @classmethod
    def _request_with_state_context(cls, request: ModelRequest) -> ModelRequest:
        request = cls._request_with_durable_registry(request)  # type: ignore[assignment]
        context = state_context_for_model(dict(request.state))
        existing = request.system_message.text if request.system_message is not None else ""
        system_message = SystemMessage(content=existing + context)
        return request.override(system_message=system_message)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        return handler(self._request_with_state_context(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        return await handler(self._request_with_state_context(request))

    @classmethod
    def _request_with_resolved_artifacts(
        cls,
        request: ToolCallRequest,
    ) -> ToolCallRequest:
        request = cls._request_with_durable_registry(request)  # type: ignore[assignment]
        tool_call = dict(request.tool_call)
        resolved_args = resolve_artifact_references(
            tool_call.get("args"),
            dict(request.state.get("artifact_registry") or {}),
            dict(request.state.get("data_registry") or {}),
        )
        if resolved_args == tool_call.get("args"):
            return request
        tool_call["args"] = resolved_args
        return request.override(tool_call=tool_call)

    @staticmethod
    def _resolution_error_message(
        request: ToolCallRequest,
        error: ArtifactResolutionError,
    ) -> ToolMessage:
        tool_name = str(request.tool_call.get("name") or "")
        tool_call_id = str(request.tool_call.get("id") or "")
        message = str(error)
        return ToolMessage(
            content=json.dumps(
                {
                    "schema_version": TOOL_ENVELOPE_SCHEMA_VERSION,
                    "ok": False,
                    "status": "error",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "outputs": {},
                    "ui": [],
                    "artifacts": [],
                    "data": [],
                    "visible_outputs": [],
                    "memory_delta": {},
                    "logs": [],
                    "errors": [message],
                    "error": message,
                }
            ),
            name=tool_name,
            tool_call_id=tool_call_id,
            status="error",
        )

    @staticmethod
    def _state_command(
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        if not isinstance(result, ToolMessage):
            return result

        payload = decode_payload(result.content)
        if not payload or payload.get("schema_version") != TOOL_ENVELOPE_SCHEMA_VERSION:
            return result

        tool_name = str(request.tool_call.get("name") or payload.get("tool_name") or "")
        tool_call_id = str(request.tool_call.get("id") or result.tool_call_id or "")
        run_id = str(request.state.get("current_run_id") or "")
        payload["tool_name"] = tool_name
        payload["tool_call_id"] = tool_call_id
        payload["provenance"] = {
            "producer": "tool",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "run_id": run_id,
        }
        source_artifacts_by_path: dict[str, str] = {}
        for artifact_id, existing in (request.state.get("artifact_registry") or {}).items():
            if not isinstance(existing, dict) or not existing.get("path"):
                continue
            try:
                source_artifacts_by_path[str(Path(str(existing["path"])).resolve())] = str(artifact_id)
            except Exception:
                continue
        for collection_name in ("artifacts", "data"):
            for record in payload.get(collection_name, []) or []:
                if not isinstance(record, dict):
                    continue
                if not record.get("source_tool"):
                    record["source_tool"] = tool_name
                if not record.get("source_call_id"):
                    record["source_call_id"] = tool_call_id
                if not record.get("run_id"):
                    record["run_id"] = run_id
                if collection_name == "artifacts":
                    metadata = dict(record.get("metadata") or {})
                    source_path = metadata.get("source_path")
                    if source_path and not metadata.get("source_artifact_id"):
                        try:
                            source_id = source_artifacts_by_path.get(
                                str(Path(str(source_path)).resolve())
                            )
                        except Exception:
                            source_id = None
                        if source_id:
                            metadata["source_artifact_id"] = source_id
                            record["metadata"] = metadata
        result.content = (
            payload
            if isinstance(result.content, dict)
            else json.dumps(payload, default=str)
        )
        delta = registry_delta_from_payload(
            payload,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            run_id=run_id,
        )
        if tool_name == "verify_artifacts":
            verified_ids = (payload.get("outputs") or {}).get("artifact_ids") or []
            existing_registry = request.state.get("artifact_registry") or {}
            for artifact_id in verified_ids:
                existing = existing_registry.get(str(artifact_id))
                if not isinstance(existing, dict):
                    continue
                verified = dict(existing)
                metadata = dict(verified.get("metadata") or {})
                metadata.update(
                    {
                        "domain_verified": True,
                        "verified_by_tool": tool_name,
                        "verified_by_call_id": tool_call_id,
                    }
                )
                verified["metadata"] = metadata
                delta["artifact_registry"][str(artifact_id)] = verified
        runtime = getattr(request, "runtime", None)
        runtime_config = getattr(runtime, "config", {}) or {}
        configurable = runtime_config.get("configurable", {}) or {}
        thread_id = str(configurable.get("thread_id") or "").strip()
        if thread_id:
            try:
                persist_registry_delta(thread_id, delta)
            except Exception as exc:
                warnings.warn(
                    f"Could not persist durable artifact registry delta: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if payload.get("status") == "error":
            result.status = "error"
        return Command(
            update={
                **delta,
                "messages": [result],
            }
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        try:
            resolved_request = self._request_with_resolved_artifacts(request)
        except ArtifactResolutionError as error:
            return self._state_command(
                request,
                self._resolution_error_message(request, error),
            )
        return self._state_command(
            resolved_request,
            handler(resolved_request),
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> ToolMessage | Command[Any]:
        try:
            resolved_request = self._request_with_resolved_artifacts(request)
        except ArtifactResolutionError as error:
            return self._state_command(
                request,
                self._resolution_error_message(request, error),
            )
        return self._state_command(
            resolved_request,
            await handler(resolved_request),
        )
