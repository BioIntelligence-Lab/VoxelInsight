from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from core.agents.artifacts import artifact_from_path, dump_model
from core.agents.checkpointing import seed_registry_from_child_checkpoints
from core.agents.deep_voxelinsight import get_voxelinsight_graph
from core.agents.durable_registry import load_registry_delta
from core.agents.visibility import infer_requested_deliverables


StreamObserver = Callable[[Any, str], Optional[Awaitable[None]]]


@dataclass
class TurnResult:
    """Authoritative result of one production graph turn."""

    thread_id: str
    run_id: str
    prompt: str
    final_state: Dict[str, Any]
    registry_payload: Dict[str, Any]
    final_text: str
    stream_event_count: int


def build_initial_agent_state(
    user_message: str,
    uploaded_files: List[str],
) -> Dict[str, Any]:
    """Build graph input while keeping uploads structured and visible to the model."""

    run_id = f"run-{uuid.uuid4().hex}"
    artifact_registry: Dict[str, Dict[str, Any]] = {}
    for uploaded_file in uploaded_files:
        artifact, _error = artifact_from_path(
            uploaded_file,
            key="uploaded_file",
            source_tool="user_upload",
            run_id=run_id,
            role="input",
            kind="upload",
            downloadable=False,
        )
        if artifact:
            artifact_registry[artifact.artifact_id] = dump_model(artifact)

    return {
        "messages": [HumanMessage(content=user_message)],
        "uploaded_files": uploaded_files,
        "artifact_registry": artifact_registry,
        "data_registry": {},
        "tool_events": [],
        "current_run_id": run_id,
        "current_user_request": user_message,
        "requested_deliverables": infer_requested_deliverables(user_message),
        "verification_cycle": {},
    }


def merge_registry_delta_into_state(
    state: Dict[str, Any],
    delta: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge durable tool records into the next root-graph input."""

    merged = dict(state)
    artifacts = dict(delta.get("artifact_registry") or {})
    artifacts.update(merged.get("artifact_registry") or {})
    data = dict(delta.get("data_registry") or {})
    data.update(merged.get("data_registry") or {})
    events_by_id: Dict[str, Dict[str, Any]] = {}
    for event in [*(delta.get("tool_events") or []), *(merged.get("tool_events") or [])]:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id") or "")
        if event_id:
            events_by_id[event_id] = event
    merged["artifact_registry"] = artifacts
    merged["data_registry"] = data
    merged["tool_events"] = list(events_by_id.values())
    return merged


def registry_payload_for_run(state: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """Return the current run's renderable records from authoritative graph state."""

    artifacts = [
        record
        for record in (state.get("artifact_registry") or {}).values()
        if isinstance(record, dict) and str(record.get("run_id") or "") == run_id
    ]
    data = [
        record
        for record in (state.get("data_registry") or {}).values()
        if isinstance(record, dict) and str(record.get("run_id") or "") == run_id
    ]
    ui: List[Dict[str, Any]] = []
    for record in artifacts:
        kind = str(record.get("kind") or "")
        role = str(record.get("role") or "")
        if kind not in {"image", "plotly", "binary"} or role != "visualization":
            continue
        ui.append(
            {
                "kind": (
                    "plotly_json_path"
                    if kind == "plotly"
                    else "image_path"
                    if kind == "image"
                    else "binary_path"
                ),
                "path": record.get("path"),
                "title": record.get("name") or kind,
                "artifact_id": record.get("artifact_id"),
            }
        )
    return {
        "schema_version": "voxelinsight.tool-result.v1",
        "ok": True,
        "status": "ok",
        "tool_name": "voxelinsight",
        "provenance": {
            "producer": "tool",
            "tool_name": "voxelinsight",
            "run_id": run_id,
        },
        "outputs": {},
        "ui": ui,
        "artifacts": artifacts,
        "data": data,
        "errors": [],
    }


def extract_stream_part(part: Any) -> tuple[Any, Any, tuple[Any, ...]]:
    """Normalize LangGraph v2 message stream envelopes."""

    if isinstance(part, dict) and {"type", "data"}.issubset(part.keys()):
        part_type = part.get("type")
        data = part.get("data")
        namespace = tuple(part.get("ns") or ())
        if part_type == "messages" and isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1], namespace
        if part_type == "updates":
            return None, data, namespace
        return None, data, namespace
    if isinstance(part, (tuple, list)) and len(part) == 2:
        return part[0], part[1], ()
    return part, {}, ()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


def final_text_from_state(state: Dict[str, Any]) -> str:
    """Return the latest root assistant response from final graph state."""

    for message in reversed(state.get("messages") or []):
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            continue
        text = _message_text(message.content).strip()
        if text:
            return text
    return ""


async def execute_turn(
    prompt: str,
    *,
    thread_id: str,
    uploaded_files: Optional[Sequence[str]] = None,
    callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
    on_stream_event: Optional[StreamObserver] = None,
    graph: Any = None,
) -> TurnResult:
    """Execute one turn through the same graph lifecycle used by the Chainlit UI."""

    normalized_thread_id = str(thread_id).strip()
    if not normalized_thread_id:
        raise ValueError("thread_id is required")

    files = [str(path) for path in (uploaded_files or [])]
    await seed_registry_from_child_checkpoints(normalized_thread_id)
    initial_state = merge_registry_delta_into_state(
        build_initial_agent_state(prompt, files),
        load_registry_delta(normalized_thread_id),
    )
    run_id = str(initial_state["current_run_id"])
    config = {"configurable": {"thread_id": normalized_thread_id}}
    runnable_config = RunnableConfig(callbacks=list(callbacks or []), **config)
    active_graph = graph or await get_voxelinsight_graph()

    event_count = 0
    async for raw_part in active_graph.astream(
        initial_state,
        stream_mode="messages",
        subgraphs=True,
        version="v2",
        config=runnable_config,
    ):
        event_count += 1
        if on_stream_event is not None:
            observed = on_stream_event(raw_part, run_id)
            if inspect.isawaitable(observed):
                await observed

    state_snapshot = await active_graph.aget_state(config)
    final_state = dict(state_snapshot.values or {})
    return TurnResult(
        thread_id=normalized_thread_id,
        run_id=run_id,
        prompt=prompt,
        final_state=final_state,
        registry_payload=registry_payload_for_run(final_state, run_id),
        final_text=final_text_from_state(final_state),
        stream_event_count=event_count,
    )
