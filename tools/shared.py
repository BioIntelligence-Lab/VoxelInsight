from __future__ import annotations
import asyncio, time, json, re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, Type
from pydantic import BaseModel, PrivateAttr
try:
    from langchain_core.tools import BaseTool
except Exception:
    class BaseTool(BaseModel):  # type: ignore[no-redef]
        name: str
        description: str
        args_schema: Type[BaseModel]

from core.state import Task, ConversationState
from core.agents.artifacts import (
    CanonicalToolResult,
    DataRecord,
    ProvenanceRecord,
    UIRecord,
    artifact_from_path,
    artifacts_from_mapping,
    dump_model,
    json_safe,
    stable_id,
    visible_outputs_for_envelope,
)
from core.storage import get_run_dir, persist_root

class ToolReturn(TypedDict, total=False):
    schema_version: str
    ok: bool
    status: str
    tool_name: str
    tool_call_id: str
    provenance: Dict[str, Any]
    outputs: Dict[str, Any]
    ui: List[Dict[str, Any]]
    artifacts: List[Dict[str, Any]]
    data: List[Dict[str, Any]]
    visible_outputs: List[Dict[str, Any]]
    memory_delta: Dict[str, Any]
    logs: List[str]
    errors: List[str]
    error: str


UI_OUTPUT_KEYS = {
    "ui.image_path": "image_path",
    "ui.plotly_json_path": "plotly_json_path",
    "ui.binary_path": "binary_path",
}

QUERY_TOOLS = {"idc_query", "midrc_query", "bih_query"}
DATA_PREVIEW_MAX_ROWS = 50
DATA_PREVIEW_MAX_CHARS = 6000
DATA_PREVIEW_MAX_STRING_CHARS = 256


def _is_type(value: Any, *, module_prefix: str, class_name: str) -> bool:
    cls = value.__class__
    return cls.__name__ == class_name and cls.__module__.startswith(module_prefix)


def _is_pandas_dataframe(value: Any) -> bool:
    return _is_type(value, module_prefix="pandas", class_name="DataFrame")


def _is_matplotlib_figure(value: Any) -> bool:
    return _is_type(value, module_prefix="matplotlib", class_name="Figure")


def _is_plotly_figure(value: Any) -> bool:
    return _is_type(value, module_prefix="plotly", class_name="Figure")


def _compact_preview_value(value: Any) -> tuple[Any, bool]:
    safe_value = json_safe(value)
    if isinstance(safe_value, str) and len(safe_value) > DATA_PREVIEW_MAX_STRING_CHARS:
        return safe_value[: DATA_PREVIEW_MAX_STRING_CHARS - 1] + "…", True
    if isinstance(safe_value, list) and len(safe_value) > 10:
        return safe_value[:10], True
    if isinstance(safe_value, dict):
        compacted: Dict[str, Any] = {}
        changed = False
        for key, nested in safe_value.items():
            compacted_value, nested_changed = _compact_preview_value(nested)
            compacted[str(key)] = compacted_value
            changed = changed or nested_changed
        return compacted, changed
    return safe_value, False


def _budgeted_dataframe_rows(
    dataframe: Any,
    *,
    max_rows: int = DATA_PREVIEW_MAX_ROWS,
    max_chars: int = DATA_PREVIEW_MAX_CHARS,
) -> tuple[List[Dict[str, Any]], bool]:
    """Return a bounded row preview and whether it represents the full table exactly."""
    if len(dataframe) == 0:
        return [], True

    preview_rows: List[Dict[str, Any]] = []
    used_chars = 2  # JSON list brackets.
    exact = True
    candidates = dataframe.head(max_rows).to_dict("records")
    for raw_row in candidates:
        safe_row = json_safe(raw_row)
        encoded = json.dumps(safe_row, separators=(",", ":"), default=str)
        row_changed = False
        if used_chars + len(encoded) + 1 > max_chars:
            compacted_row, row_changed = _compact_preview_value(safe_row)
            encoded = json.dumps(compacted_row, separators=(",", ":"), default=str)
            safe_row = compacted_row
        if used_chars + len(encoded) + 1 > max_chars:
            exact = False
            break
        preview_rows.append(safe_row)
        used_chars += len(encoded) + 1
        exact = exact and not row_changed

    complete = exact and len(preview_rows) == len(dataframe)
    return preview_rows, complete


def _write_plotly_json(fig: Any, path: Path) -> None:
    try:
        import plotly

        path.write_text(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))  # type: ignore
    except Exception:
        if hasattr(fig, "to_json"):
            path.write_text(fig.to_json())
        else:
            raise

def _semantic_error(text: str) -> bool:
    normalized = " ".join(str(text).strip().lower().split())
    if not normalized:
        return False
    # Status summaries commonly contain phrases such as ``0 failed`` and
    # ``0 cancelled``.  Remove those explicit zero outcomes before applying
    # the legacy prose fallback so successful batch summaries are not errors.
    normalized = re.sub(
        r"\b0(?:\.0+)?\s+(?:(?:items?|files?|series|patients?|tasks?|jobs?)\s+)?"
        r"(?:failed|cancelled|canceled)\b",
        "",
        normalized,
    )
    markers = (
        "error:",
        " error",
        "timed out",
        "could not",
        "cannot ",
        "can't ",
        " is required",
        "are required",
        "please provide",
        "provide either",
        "provide exactly",
        "path does not exist",
        "file does not exist",
        "invalid ",
        "unsupported ",
        "produced no ",
        "no volume provided",
        "no input file",
    )
    return (
        normalized.startswith(("error", "failed", "cancelled", "canceled", "invalid"))
        or bool(re.search(r"\b(?:failed|cancelled|canceled)\b", normalized))
        or any(marker in normalized for marker in markers)
    )


def _typed_output_error(output: Any) -> tuple[str, bool]:
    """Return a typed failure and whether structured outcome fields exist."""

    if not isinstance(output, dict):
        return "", False

    typed_fields_present = False
    error_value = output.get("error")
    if isinstance(error_value, str) and error_value.strip():
        return error_value.strip(), True

    status_value = output.get("status")
    if isinstance(status_value, str) and status_value.strip():
        typed_fields_present = True
        normalized_status = status_value.strip().lower()
        if normalized_status in {"error", "failed", "failure", "cancelled", "canceled"}:
            return status_value, True

    failure_counts: List[tuple[str, int]] = []
    for key, value in output.items():
        normalized_key = str(key).strip().lower()
        if normalized_key not in {"failed", "cancelled", "canceled"} and not normalized_key.endswith(
            ("_failed", "_cancelled", "_canceled")
        ):
            continue
        typed_fields_present = True
        if isinstance(value, bool):
            if value:
                return f"Structured tool outcome reported {key}=true.", True
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            failure_counts.append((str(key), count))

    if failure_counts:
        detail = ", ".join(f"{key}={count}" for key, count in failure_counts)
        return f"Structured tool outcome reported failures: {detail}.", True
    return "", typed_fields_present


def _persistent_path_from_plain_output(value: Any) -> Optional[Path]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "\n" in text or len(text) > 4096:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute() or not candidate.exists():
        return None
    resolved = candidate.resolve()
    try:
        resolved.relative_to(persist_root().resolve())
    except ValueError:
        return None
    return resolved


def _write_dataframe_artifact(
    dataframe: Any,
    *,
    source_tool: str,
    name: str,
    source_call_id: str,
) -> tuple[DataRecord, Any]:
    table_dir = get_run_dir("tool_data", persist=True)
    table_path = table_dir / f"{stable_id('table', source_tool, name, len(dataframe))}.json"
    dataframe.to_json(table_path, orient="records", date_format="iso")
    artifact, error = artifact_from_path(
        table_path,
        key="table",
        source_tool=source_tool,
        source_call_id=source_call_id,
        role="table",
        kind="table",
        name=f"{name}.json",
        downloadable=False,
        metadata={
            "columns": [str(column) for column in dataframe.columns],
            "nrows": int(len(dataframe)),
        },
    )
    if error or artifact is None:
        raise RuntimeError(error or "Could not register table artifact.")
    rows, complete = _budgeted_dataframe_rows(dataframe)
    data = DataRecord(
        data_id=stable_id("data", source_tool, name, artifact.artifact_id),
        kind="table",
        visibility="internal" if source_tool == "table_chart" else "user",
        name=name,
        source_tool=source_tool,
        source_call_id=source_call_id,
        rows=rows,
        columns=[str(column) for column in dataframe.columns],
        nrows=int(len(dataframe)),
        complete=complete,
        artifact_id=artifact.artifact_id,
        content_fingerprint=(
            f"sha256:{artifact.checksum_sha256}"
            if artifact.checksum_sha256
            else ""
        ),
    )
    return data, artifact


def _data_record_for_artifact(
    dataframe: Any,
    *,
    artifact: Any,
    source_tool: str,
    name: str,
    source_call_id: str,
    visibility: str = "user",
    metadata: Optional[Dict[str, Any]] = None,
) -> DataRecord:
    rows, complete = _budgeted_dataframe_rows(dataframe)
    fingerprint = (
        f"sha256:{artifact.checksum_sha256}"
        if getattr(artifact, "checksum_sha256", "")
        else stable_id(
            "table-content",
            name,
            json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str),
            len(dataframe),
        )
    )
    return DataRecord(
        data_id=stable_id("data", source_tool, name, artifact.artifact_id),
        kind="table",
        visibility="internal" if visibility == "internal" else "user",
        name=name,
        source_tool=source_tool,
        source_call_id=source_call_id,
        rows=rows,
        columns=[str(column) for column in dataframe.columns],
        nrows=int(len(dataframe)),
        complete=complete,
        artifact_id=artifact.artifact_id,
        content_fingerprint=fingerprint,
        metadata=json_safe(metadata or {}),
    )


def _mapping_data_record(
    mapping: Dict[str, Any],
    *,
    source_tool: str,
    source_call_id: str,
) -> Optional[DataRecord]:
    if source_tool not in QUERY_TOOLS:
        return None
    row = {
        str(key): json_safe(value)
        for key, value in mapping.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    if not row:
        return None
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    fingerprint = stable_id("mapping", canonical)
    return DataRecord(
        data_id=stable_id("data", source_tool, fingerprint),
        kind="json",
        visibility="internal",
        name="query_result",
        source_tool=source_tool,
        source_call_id=source_call_id,
        rows=[row],
        columns=list(row),
        nrows=1,
        complete=True,
        content_fingerprint=fingerprint,
    )


def _materialize_figure(
    value: Any,
    *,
    source_tool: str,
    source_call_id: str,
    name: str,
) -> tuple[UIRecord, Any]:
    output_dir = get_run_dir("visualizations", persist=True)
    if _is_matplotlib_figure(value):
        path = output_dir / f"{stable_id('image', source_tool, name)}.png"
        value.savefig(path, bbox_inches="tight")
        kind = "image_path"
        artifact_kind = "image"
    else:
        path = output_dir / f"{stable_id('plotly', source_tool, name)}.json"
        _write_plotly_json(value, path)
        kind = "plotly_json_path"
        artifact_kind = "plotly"
    artifact, error = artifact_from_path(
        path,
        key=name,
        source_tool=source_tool,
        source_call_id=source_call_id,
        role="visualization",
        kind=artifact_kind,
        name=path.name,
        downloadable=False,
    )
    if error or artifact is None:
        raise RuntimeError(error or "Could not register visualization artifact.")
    return (
        UIRecord(
            kind=kind,
            path=artifact.path,
            title=name,
            artifact_id=artifact.artifact_id,
        ),
        artifact,
    )


def _materialize_binary(
    value: Any,
    *,
    source_tool: str,
    source_call_id: str,
    name: str,
) -> tuple[UIRecord, Any]:
    output_dir = get_run_dir("binary_outputs", persist=True)
    path = output_dir / f"{stable_id('binary', source_tool, name)}.bin"
    try:
        value.seek(0)
    except Exception:
        pass
    path.write_bytes(value.read())
    artifact, error = artifact_from_path(
        path,
        key=name,
        source_tool=source_tool,
        source_call_id=source_call_id,
        role="download",
        kind="binary",
        name=path.name,
        downloadable=True,
    )
    if error or artifact is None:
        raise RuntimeError(error or "Could not register binary artifact.")
    return (
        UIRecord(
            kind="binary_path",
            path=artifact.path,
            title=name,
            artifact_id=artifact.artifact_id,
        ),
        artifact,
    )


def normalize_task_result(
    res: Any,
    *,
    tool_name: str = "",
    tool_call_id: str = "",
) -> ToolReturn:
    outputs: Dict[str, Any] = {}
    ui: List[UIRecord] = []
    artifacts_by_id: Dict[str, Any] = {}
    data_by_id: Dict[str, DataRecord] = {}
    mem: Dict[str, Any] = {}
    logs: List[str] = []
    errors: List[str] = [str(error) for error in (getattr(res, "errors", None) or [])]

    if res is None:
        return error_tool_result("Tool returned no result.", tool_name=tool_name)

    arts = getattr(res, "artifacts", {}) or {}
    out  = getattr(res, "output", None)

    for k, value in arts.items():
        if k == "code" or k in {
            "segmentations",
            "segmentations_map",
            "segmentations_batch",
            "segmentations_map_batch",
            "output_dir",
            "output_root",
            "image_path",
            "image_paths",
            "mask_path",
            "mask_paths",
            "files",
            "nifti_path",
            "nifti_paths",
            "csv_path",
            "registered_image",
            "transform",
        }:
            outputs[k] = json_safe(value)
    for k in ("image_path","segmentations","segmentations_map"):
        if k in arts: mem[k] = arts[k]

    artifact_records, artifact_errors = artifacts_from_mapping(
        arts,
        source_tool=tool_name,
        source_call_id=tool_call_id,
    )
    errors.extend(artifact_errors)

    # Preserve cohort lineage after the generic artifact walker flattens nested
    # input -> ROI -> mask mappings. Middleware can then link source_path to the
    # registered input artifact ID before committing these records to state.
    segmentation_lineage: Dict[str, Dict[str, Any]] = {}
    batch_map = arts.get("segmentations_map_batch")
    if isinstance(batch_map, dict):
        for source_path, roi_map in batch_map.items():
            if not isinstance(roi_map, dict):
                continue
            for roi, mask_path in roi_map.items():
                if not isinstance(mask_path, str):
                    continue
                try:
                    resolved_mask = str(Path(mask_path).expanduser().resolve())
                except Exception:
                    resolved_mask = mask_path
                segmentation_lineage[resolved_mask] = {
                    "lineage_type": "segmentation",
                    "case_id": Path(mask_path).parent.name,
                    "source_path": str(source_path),
                    "source_name": Path(str(source_path)).name,
                    "roi": str(roi),
                }
    for artifact in artifact_records:
        lineage = segmentation_lineage.get(str(Path(artifact.path).expanduser().resolve()))
        if lineage:
            artifact.metadata = {**dict(artifact.metadata or {}), **lineage}
        artifacts_by_id[artifact.artifact_id] = artifact

    if _is_pandas_dataframe(out):
        data_record, artifact = _write_dataframe_artifact(
            out,
            source_tool=tool_name,
            source_call_id=tool_call_id,
            name="df",
        )
        data_by_id[data_record.data_id] = data_record
        artifacts_by_id[artifact.artifact_id] = artifact
        outputs["df_preview"] = {
            "rows": data_record.rows,
            "nrows": int(len(out)),
            "data_id": data_record.data_id,
            "complete": data_record.complete,
        }
    elif _is_matplotlib_figure(out) or _is_plotly_figure(out):
        ui_record, artifact = _materialize_figure(
            out,
            source_tool=tool_name,
            source_call_id=tool_call_id,
            name="figure",
        )
        ui.append(ui_record)
        artifacts_by_id[artifact.artifact_id] = artifact
    elif hasattr(out, "read"):
        ui_record, artifact = _materialize_binary(
            out,
            source_tool=tool_name,
            source_call_id=tool_call_id,
            name="binary",
        )
        ui.append(ui_record)
        artifacts_by_id[artifact.artifact_id] = artifact
    elif isinstance(out, dict):
        table_data_specs = out.get("table_data")
        for k, v in out.items():
            if k == "table_data":
                continue
            ui_kind = UI_OUTPUT_KEYS.get(str(k))
            if ui_kind and v:
                artifact, artifact_error = artifact_from_path(
                    v,
                    key=str(k),
                    source_tool=tool_name,
                    source_call_id=tool_call_id,
                    role="visualization" if ui_kind != "binary_path" else "download",
                    kind=(
                        "plotly"
                        if ui_kind == "plotly_json_path"
                        else "image"
                        if ui_kind == "image_path"
                        else "binary"
                    ),
                    downloadable=ui_kind == "binary_path",
                )
                if artifact_error:
                    errors.append(artifact_error)
                if artifact:
                    artifacts_by_id[artifact.artifact_id] = artifact
                    ui.append(
                        UIRecord(
                            kind=ui_kind,
                            path=artifact.path,
                            title=str(k),
                            artifact_id=artifact.artifact_id,
                        )
                    )
                outputs[str(k)] = json_safe(v)
            elif _is_pandas_dataframe(v):
                data_record, artifact = _write_dataframe_artifact(
                    v,
                    source_tool=tool_name,
                    source_call_id=tool_call_id,
                    name=str(k),
                )
                data_by_id[data_record.data_id] = data_record
                artifacts_by_id[artifact.artifact_id] = artifact
                outputs[f"{k}_df_preview"] = {
                    "rows": data_record.rows,
                    "nrows": int(len(v)),
                    "data_id": data_record.data_id,
                    "complete": data_record.complete,
                }
            elif _is_matplotlib_figure(v) or _is_plotly_figure(v):
                ui_record, artifact = _materialize_figure(
                    v,
                    source_tool=tool_name,
                    source_call_id=tool_call_id,
                    name=str(k),
                )
                ui.append(ui_record)
                artifacts_by_id[artifact.artifact_id] = artifact
            elif hasattr(v, "read"):
                ui_record, artifact = _materialize_binary(
                    v,
                    source_tool=tool_name,
                    source_call_id=tool_call_id,
                    name=str(k),
                )
                ui.append(ui_record)
                artifacts_by_id[artifact.artifact_id] = artifact
            elif v is not None:
                outputs[k] = json_safe(v)
        output_artifacts, output_errors = artifacts_from_mapping(
            out,
            source_tool=tool_name,
            source_call_id=tool_call_id,
        )
        errors.extend(output_errors)
        for artifact in output_artifacts:
            artifacts_by_id[artifact.artifact_id] = artifact
        if isinstance(table_data_specs, list):
            table_summaries: List[Dict[str, Any]] = []
            artifacts_by_path = {
                str(Path(artifact.path).resolve()): artifact
                for artifact in artifacts_by_id.values()
                if getattr(artifact, "path", "")
            }
            for index, spec in enumerate(table_data_specs):
                if not isinstance(spec, dict):
                    continue
                dataframe = spec.get("dataframe")
                artifact_path = spec.get("artifact_path")
                if not _is_pandas_dataframe(dataframe) or not artifact_path:
                    errors.append(
                        "table_data entries require a pandas DataFrame and artifact_path."
                    )
                    continue
                resolved_path = str(Path(str(artifact_path)).expanduser().resolve())
                artifact = artifacts_by_path.get(resolved_path)
                if artifact is None:
                    artifact, artifact_error = artifact_from_path(
                        resolved_path,
                        key="csv_path",
                        source_tool=tool_name,
                        source_call_id=tool_call_id,
                        role="table",
                        downloadable=True,
                    )
                    if artifact_error:
                        errors.append(artifact_error)
                    if artifact is None:
                        continue
                    artifacts_by_id[artifact.artifact_id] = artifact
                    artifacts_by_path[resolved_path] = artifact
                name = str(spec.get("name") or f"table_{index + 1}")
                data_record = _data_record_for_artifact(
                    dataframe,
                    artifact=artifact,
                    source_tool=tool_name,
                    source_call_id=tool_call_id,
                    name=name,
                    visibility=str(spec.get("visibility") or "user"),
                    metadata=spec.get("metadata")
                    if isinstance(spec.get("metadata"), dict)
                    else None,
                )
                data_by_id[data_record.data_id] = data_record
                table_summaries.append(
                    {
                        "name": name,
                        "data_id": data_record.data_id,
                        "artifact_id": artifact.artifact_id,
                        "nrows": data_record.nrows,
                        "ncolumns": len(data_record.columns),
                        "preview_rows": len(data_record.rows),
                        "complete": data_record.complete,
                    }
                )
            if table_summaries:
                outputs["table_data"] = table_summaries
        mapping_data = _mapping_data_record(
            out,
            source_tool=tool_name,
            source_call_id=tool_call_id,
        )
        if mapping_data:
            data_by_id[mapping_data.data_id] = mapping_data
    elif out is not None:
        persistent_path = _persistent_path_from_plain_output(out)
        if persistent_path is not None:
            artifact, artifact_error = artifact_from_path(
                persistent_path,
                key="files",
                source_tool=tool_name,
                source_call_id=tool_call_id,
                role="download",
                downloadable=True,
            )
            if artifact_error:
                errors.append(artifact_error)
            if artifact:
                artifacts_by_id[artifact.artifact_id] = artifact
                outputs["files"] = [artifact.path]
        else:
            outputs["text"] = str(out)

    semantic_error_message, has_typed_outcome = _typed_output_error(out)
    if not semantic_error_message and isinstance(out, str) and _semantic_error(out):
        semantic_error_message = out
    elif not semantic_error_message and isinstance(out, dict):
        for key in ("error", "message", "text", "status"):
            if has_typed_outcome and key in {"message", "text"}:
                continue
            value = out.get(key)
            if isinstance(value, str) and _semantic_error(value):
                semantic_error_message = value
                break
    if semantic_error_message and semantic_error_message not in errors:
        errors.append(semantic_error_message)

    explicit_status = str(getattr(res, "status", "") or "")
    has_usable_result = bool(artifacts_by_id or data_by_id or ui)
    if explicit_status in {"error", "no_action"}:
        status = explicit_status
    elif errors:
        # A tool cannot override concrete validation or semantic failures with
        # status="ok". Retain usable artifacts as a partial result.
        status = "partial" if has_usable_result else "error"
    elif explicit_status == "partial":
        status = "partial"
    elif explicit_status == "ok":
        status = "ok"
    elif out is None and not artifacts_by_id:
        status = "no_action"
    else:
        status = "ok"

    artifact_values = list(artifacts_by_id.values())
    data_values = list(data_by_id.values())
    visible_outputs = visible_outputs_for_envelope(
        ui=ui,
        artifacts=artifact_values,
        data=data_values,
        outputs=outputs,
    )
    error = errors[0] if status == "error" and errors else ""
    envelope = CanonicalToolResult(
        ok=status in {"ok", "partial"},
        status=status,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        provenance=ProvenanceRecord(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        ),
        outputs=json_safe(outputs),
        ui=ui,
        artifacts=artifact_values,
        data=data_values,
        visible_outputs=visible_outputs,
        memory_delta=json_safe(mem),
        logs=logs,
        errors=errors,
        error=error,
    )
    return dump_model(envelope)  # type: ignore[return-value]


def error_tool_result(
    message: str,
    *,
    tool_name: str = "",
    logs: Optional[List[str]] = None,
) -> ToolReturn:
    envelope = CanonicalToolResult(
        ok=False,
        status="error",
        tool_name=tool_name,
        provenance=ProvenanceRecord(tool_name=tool_name),
        outputs={},
        ui=[],
        artifacts=[],
        data=[],
        visible_outputs=[],
        memory_delta={},
        logs=logs or [],
        errors=[str(message)],
        error=str(message),
    )
    return dump_model(envelope)  # type: ignore[return-value]

def _cs() -> ConversationState:
    s = ConversationState(); s.memory = {}; return s

TOOL_REGISTRY: List[BaseTool] = []

class AgentTool(BaseTool):
    """Generic wrapper that calls an async runner and normalizes TaskResult."""
    name: str
    description: str
    args_schema: Type[BaseModel]
    timeout_s: int = 300

    _runner = PrivateAttr(default=None)

    async def _arun(self, *args, **kwargs) -> ToolReturn:
        try:
            if args and not kwargs:
                field_names = list(self.args_schema.model_fields.keys()) if hasattr(self.args_schema, "model_fields") else list(self.args_schema.__fields__.keys())  # v2 vs v1
                if len(field_names) == 1:
                    kwargs = {field_names[0]: args[0]}
                else:
                    return error_tool_result(
                        f"{self.name}: positional args not supported for multi-field schema.",
                        tool_name=self.name,
                    )
        except Exception as e:
            return error_tool_result(
                f"{self.name}: arg normalization failed: {e}",
                tool_name=self.name,
            )

        if self._runner is None:
            return error_tool_result(
                f"{self.name} runner not configured.",
                tool_name=self.name,
            )

        start = time.time()
        try:
            res = await asyncio.wait_for(self._runner(**kwargs), timeout=self.timeout_s)
            out = normalize_task_result(res, tool_name=self.name)
            out.setdefault("logs", []).append(f"{self.name}: elapsed={time.time()-start:.2f}s")
            return out
        except asyncio.TimeoutError:
            return error_tool_result(
                f"{self.name} timed out after {self.timeout_s}s",
                tool_name=self.name,
            )
        except Exception as e:
            return error_tool_result(
                f"{self.name} failed. Error: {e}",
                tool_name=self.name,
            )

    def _run(self, *args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(self._arun(*args, **kwargs))

def toolify_agent(*, name: str, description: str, args_schema: Type[BaseModel], timeout_s: int = 300):
    def _wrap(runner_fn: Callable[..., Any]):
        tool = AgentTool(
            name=name,
            description=description,
            args_schema=args_schema,
            timeout_s=timeout_s,
        )
        tool._runner = runner_fn
        TOOL_REGISTRY.append(tool)
        return tool
    return _wrap
