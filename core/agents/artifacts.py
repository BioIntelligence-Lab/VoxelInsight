from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Literal, NotRequired, Optional

from pydantic import BaseModel, Field

from core.agents.external_links import idc_viewer_urls_from_rows
from typing_extensions import TypedDict

try:
    from langchain.agents.middleware.types import AgentState
except Exception:
    AgentState = TypedDict  # type: ignore[assignment,misc]


TOOL_ENVELOPE_SCHEMA_VERSION = "voxelinsight.tool-result.v1"
ARTIFACT_REGISTRY_SCHEMA_VERSION = "voxelinsight.artifact-registry.v1"

ToolStatus = Literal["ok", "partial", "error", "no_action"]
ArtifactStatus = Literal["verified", "missing", "invalid", "unverified"]
ArtifactKind = Literal[
    "file",
    "directory",
    "image",
    "plotly",
    "table",
    "csv",
    "nifti",
    "segmentation",
    "registered_image",
    "transform",
    "binary",
    "upload",
    "other",
]
ArtifactRole = Literal[
    "input",
    "output",
    "visualization",
    "download",
    "segmentation",
    "registered_image",
    "transform",
    "table",
    "other",
]


class ArtifactRecord(BaseModel):
    artifact_id: str
    kind: ArtifactKind = "other"
    role: ArtifactRole = "output"
    name: str = ""
    path: str
    mime_type: str = ""
    source_tool: str = ""
    source_call_id: str = ""
    run_id: str = ""
    status: ArtifactStatus = "unverified"
    size_bytes: Optional[int] = None
    checksum_sha256: str = ""
    downloadable: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataRecord(BaseModel):
    data_id: str
    kind: Literal["table", "json", "text"] = "table"
    visibility: Literal["user", "internal"] = "user"
    name: str = ""
    source_tool: str = ""
    source_call_id: str = ""
    run_id: str = ""
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    nrows: int = 0
    complete: bool = True
    artifact_id: str = ""
    content_fingerprint: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UIRecord(BaseModel):
    kind: Literal["image_path", "plotly_json_path", "binary_path"]
    path: str
    title: str = ""
    artifact_id: str = ""


class ToolEvent(BaseModel):
    event_id: str
    tool_name: str
    tool_call_id: str = ""
    run_id: str = ""
    status: ToolStatus = "ok"
    artifact_ids: List[str] = Field(default_factory=list)
    data_ids: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    producer: Literal["tool"] = "tool"
    tool_name: str = ""
    tool_call_id: str = ""
    run_id: str = ""


class CanonicalToolResult(BaseModel):
    schema_version: Literal["voxelinsight.tool-result.v1"] = TOOL_ENVELOPE_SCHEMA_VERSION
    ok: bool
    status: ToolStatus
    tool_name: str = ""
    tool_call_id: str = ""
    provenance: ProvenanceRecord = Field(default_factory=ProvenanceRecord)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    ui: List[UIRecord] = Field(default_factory=list)
    artifacts: List[ArtifactRecord] = Field(default_factory=list)
    data: List[DataRecord] = Field(default_factory=list)
    visible_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    memory_delta: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    error: str = ""


def _merge_registry(
    left: Optional[Dict[str, Dict[str, Any]]],
    right: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_unique_strings(
    left: Optional[List[str]],
    right: Optional[List[str]],
) -> List[str]:
    seen: set[str] = set()
    merged: List[str] = []
    for value in [*(left or []), *(right or [])]:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _merge_events(
    left: Optional[List[Dict[str, Any]]],
    right: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for event in [*(left or []), *(right or [])]:
        event_id = str(event.get("event_id") or "")
        if not event_id:
            event_id = stable_id("event", json.dumps(event, sort_keys=True, default=str))
        if event_id not in by_id:
            order.append(event_id)
        by_id[event_id] = event
    return [by_id[event_id] for event_id in order]


class VoxelAgentState(AgentState):
    """State shared by the supervisor and every synchronous DeepAgents subagent."""

    uploaded_files: Annotated[NotRequired[List[str]], _merge_unique_strings]
    artifact_registry: Annotated[
        NotRequired[Dict[str, Dict[str, Any]]],
        _merge_registry,
    ]
    data_registry: Annotated[
        NotRequired[Dict[str, Dict[str, Any]]],
        _merge_registry,
    ]
    tool_events: Annotated[NotRequired[List[Dict[str, Any]]], _merge_events]
    current_run_id: NotRequired[str]
    current_user_request: NotRequired[str]
    requested_deliverables: NotRequired[List[Dict[str, Any]]]


def dump_model(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts if part is not None)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def safe_slug(value: str, default: str = "artifact") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-._")
    return slug[:80] or default


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def decode_payload(content: Any) -> Optional[Dict[str, Any]]:
    if isinstance(content, dict):
        return dict(content)
    if not isinstance(content, str):
        return None
    try:
        decoded = json.loads(content)
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


def _file_checksum(path: Path) -> str:
    max_bytes = int(os.getenv("VOXELINSIGHT_CHECKSUM_MAX_BYTES", str(256 * 1024 * 1024)))
    try:
        if path.stat().st_size > max_bytes:
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _guess_kind(path: Path, *, role: str = "", key: str = "") -> ArtifactKind:
    lower_name = path.name.lower()
    lower_key = key.lower()
    lower_role = role.lower()
    if path.is_dir():
        return "directory"
    if lower_name.endswith((".nii", ".nii.gz")):
        if "segment" in lower_key or "mask" in lower_key or "segment" in lower_role:
            return "segmentation"
        if "registered" in lower_key or "registered" in lower_role:
            return "registered_image"
        return "nifti"
    if lower_name.endswith(".csv"):
        return "csv"
    if lower_name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return "image"
    if lower_name.endswith(".json") and ("plot" in lower_key or "plot" in lower_role):
        return "plotly"
    if lower_name.endswith((".tfm", ".h5", ".mat")) or "transform" in lower_key:
        return "transform"
    return "file"


def _role_for_key(key: str, *, source_tool: str) -> ArtifactRole:
    lower = key.lower()
    if source_tool in {"viz_slider", "radiomics"} and lower in {
        "image_path",
        "image_paths",
        "mask_path",
        "mask_paths",
    }:
        return "input"
    if "registered" in lower:
        return "registered_image"
    if "transform" in lower:
        return "transform"
    if "segment" in lower or "mask" in lower:
        return "segmentation"
    if "table" in lower or "csv" in lower:
        return "table"
    if "plot" in lower or "figure" in lower or "image" in lower:
        return "visualization"
    if "download" in lower or lower in {"files", "file", "dicom_dir", "dicom_dirs"}:
        return "download"
    return "output"


def artifact_from_path(
    value: Any,
    *,
    key: str = "path",
    source_tool: str = "",
    source_call_id: str = "",
    run_id: str = "",
    role: Optional[ArtifactRole] = None,
    kind: Optional[ArtifactKind] = None,
    name: str = "",
    downloadable: Optional[bool] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[ArtifactRecord], Optional[str]]:
    if not value:
        return None, None
    path = Path(str(value)).expanduser()
    if not path.exists():
        return None, f"Artifact path does not exist: {path}"

    resolved = str(path.resolve())
    selected_role = role or _role_for_key(key, source_tool=source_tool)
    selected_kind = kind or _guess_kind(path, role=selected_role, key=key)
    is_file = path.is_file()
    mime_type = mimetypes.guess_type(path.name)[0] or ""
    record = ArtifactRecord(
        artifact_id=stable_id("artifact", source_tool, selected_role, resolved),
        kind=selected_kind,
        role=selected_role,
        name=name or path.name,
        path=resolved,
        mime_type=mime_type,
        source_tool=source_tool,
        source_call_id=source_call_id,
        run_id=run_id,
        status="verified",
        size_bytes=path.stat().st_size if is_file else None,
        checksum_sha256=_file_checksum(path) if is_file else "",
        downloadable=(
            downloadable
            if downloadable is not None
            else selected_role not in {"input", "visualization"}
        ),
        metadata=json_safe(metadata or {}),
    )
    return record, None


PATH_KEYS = {
    "path",
    "file",
    "file_path",
    "files",
    "dicom_dir",
    "dicom_dirs",
    "csv_path",
    "nifti_path",
    "nifti_paths",
    "image_path",
    "image_paths",
    "mask_path",
    "mask_paths",
    "segmentations",
    "segmentations_batch",
    "segmentations_map",
    "segmentations_map_batch",
    "registered_image",
    "transform",
    "output_dir",
    "output_root",
    "download_dir",
}


def _iter_keyed_paths(key: str, value: Any) -> Iterable[tuple[str, Any]]:
    if value is None:
        return
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            yield from _iter_keyed_paths(key or str(nested_key), nested_value)
        return
    if isinstance(value, (list, tuple, set)):
        for nested_value in value:
            yield from _iter_keyed_paths(key, nested_value)
        return
    yield key, value


def artifacts_from_mapping(
    mapping: Dict[str, Any],
    *,
    source_tool: str = "",
    source_call_id: str = "",
    run_id: str = "",
) -> tuple[List[ArtifactRecord], List[str]]:
    records: Dict[str, ArtifactRecord] = {}
    errors: List[str] = []
    for key, value in mapping.items():
        if key not in PATH_KEYS:
            continue
        for path_key, path_value in _iter_keyed_paths(key, value):
            record, error = artifact_from_path(
                path_value,
                key=path_key,
                source_tool=source_tool,
                source_call_id=source_call_id,
                run_id=run_id,
            )
            if error:
                errors.append(error)
            if record:
                records[record.artifact_id] = record
    return list(records.values()), errors


def visible_outputs_for_envelope(
    *,
    ui: Iterable[UIRecord],
    artifacts: Iterable[ArtifactRecord],
    data: Iterable[DataRecord],
    outputs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    visible: List[Dict[str, Any]] = []
    for item in ui:
        if item.kind == "plotly_json_path":
            kind = "plotly"
        elif item.kind == "image_path":
            kind = "image"
        else:
            kind = "file"
        visible.append(
            {
                "kind": kind,
                "source": "tool_ui",
                "title": item.title or kind,
                "path": item.path,
                "artifact_id": item.artifact_id or None,
            }
        )
    for record in artifacts:
        if record.role in {"input", "visualization"} or not record.downloadable:
            continue
        if record.kind == "image":
            kind = "image"
        elif record.kind == "plotly":
            kind = "plotly"
        else:
            kind = "file"
        visible.append(
            {
                "kind": kind,
                "source": "verified_file",
                "title": record.name,
                "path": record.path,
                "artifact_id": record.artifact_id,
            }
        )
    for record in data:
        if record.visibility != "user":
            continue
        visible.append(
            {
                "kind": "dataframe",
                "source": "tool_outputs",
                "title": record.name or "table",
                "path": None,
                "data_id": record.data_id,
            }
        )
        for viewer_url in idc_viewer_urls_from_rows(record.rows):
            visible.append(
                {
                    "kind": "link",
                    "source": "tool_outputs",
                    "title": "Open in IDC Viewer",
                    "path": None,
                    "url": viewer_url,
                    "data_id": record.data_id,
                }
            )
    if outputs.get("code"):
        visible.append(
            {
                "kind": "code",
                "source": "tool_outputs",
                "title": "Generated code",
                "path": None,
            }
        )
    if outputs.get("text") or outputs.get("summary"):
        visible.append(
            {
                "kind": "text",
                "source": "tool_outputs",
                "title": "text",
                "path": None,
            }
        )

    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in visible:
        key = (
            item.get("kind"),
            item.get("path"),
            item.get("url"),
            item.get("artifact_id"),
            item.get("data_id"),
            item.get("title"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def registry_delta_from_payload(
    payload: Dict[str, Any],
    *,
    tool_name: str = "",
    tool_call_id: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    artifact_registry: Dict[str, Dict[str, Any]] = {}
    data_registry: Dict[str, Dict[str, Any]] = {}

    for raw in payload.get("artifacts", []) or []:
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        normalized = dict(raw)
        if not normalized.get("source_tool"):
            normalized["source_tool"] = tool_name or payload.get("tool_name") or ""
        if not normalized.get("source_call_id"):
            normalized["source_call_id"] = tool_call_id
        if not normalized.get("run_id"):
            normalized["run_id"] = run_id
        try:
            record = ArtifactRecord.model_validate(normalized)
        except Exception:
            continue
        if record.status == "verified" and Path(record.path).exists():
            artifact_registry[record.artifact_id] = dump_model(record)

    for raw in payload.get("data", []) or []:
        if not isinstance(raw, dict):
            continue
        normalized = dict(raw)
        if not normalized.get("source_tool"):
            normalized["source_tool"] = tool_name or payload.get("tool_name") or ""
        if not normalized.get("source_call_id"):
            normalized["source_call_id"] = tool_call_id
        if not normalized.get("run_id"):
            normalized["run_id"] = run_id
        try:
            record = DataRecord.model_validate(normalized)
        except Exception:
            continue
        data_registry[record.data_id] = dump_model(record)

    raw_status = str(payload.get("status") or ("ok" if payload.get("ok") else "error"))
    status: ToolStatus = (
        raw_status
        if raw_status in {"ok", "partial", "error", "no_action"}
        else "error"
    )  # type: ignore[assignment]
    event = ToolEvent(
        event_id=stable_id(
            "event",
            tool_call_id,
            tool_name or payload.get("tool_name") or "",
            run_id,
        ),
        tool_name=tool_name or str(payload.get("tool_name") or ""),
        tool_call_id=tool_call_id,
        run_id=run_id,
        status=status,
        artifact_ids=list(artifact_registry),
        data_ids=list(data_registry),
        errors=[str(error) for error in payload.get("errors", []) or []],
    )
    return {
        "artifact_registry": artifact_registry,
        "data_registry": data_registry,
        "tool_events": [dump_model(event)],
    }


def state_context_for_model(state: Dict[str, Any], *, max_records: int = 40) -> str:
    artifacts = list((state.get("artifact_registry") or {}).values())[-max_records:]
    data_records = list((state.get("data_registry") or {}).values())[-max_records:]
    context = {
        "current_run_id": state.get("current_run_id", ""),
        "current_user_request": state.get("current_user_request", ""),
        "uploaded_files": state.get("uploaded_files", []),
        "artifacts": artifacts,
        "data": data_records,
    }
    return (
        "\n\n<voxelinsight_state_json>\n"
        + json.dumps(json_safe(context), indent=2)
        + "\n</voxelinsight_state_json>\n"
        "This state block is authoritative and is maintained by code. Pass exact "
        "artifact_id/data_id values from it to tools and subagents; deterministic middleware "
        "resolves registered references. Never pass, copy, reconstruct, shorten, join, or "
        "invent local filesystem paths, even when a registry record contains a path. Never "
        "substitute a PatientID, DICOM UID, filename, UI title, or model-authored alias for an "
        "artifact_id/data_id. Use exact table rows only when a tool explicitly requires scalar "
        "values rather than a data_id. Refer to produced outputs by their registered IDs in "
        "your structured result."
    )
