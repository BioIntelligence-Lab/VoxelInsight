import asyncio
import zipfile
import shutil
import json
import sys
import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, List, Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

import chainlit as cl
from chainlit.types import ThreadDict
import pandas as pd

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    AIMessageChunk,
)
from langchain_core.callbacks import BaseCallbackHandler
from core.agents.deep_voxelinsight import DOMAIN_SUBAGENT_TOOL_NAMES
from core.agents.artifacts import (
    registry_delta_from_payload,
    safe_slug,
)
from core.agents.run_turn import execute_turn, extract_stream_part
from core.interactions import (
    ConfirmationRequest,
    InteractionHandler,
    interaction_context,
)
from core.agents.visibility import (
    infer_requested_deliverables,
    validate_deliverables,
)
from core.agents.external_links import idc_viewer_urls_from_rows
from core.storage import get_run_dir, get_temp_dir
from progress_ui import set_progress_queue, update_progress


def _select_nonoverlapping_artifact_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep top-level directories plus files not already contained by them."""

    by_path: Dict[str, Dict[str, Any]] = {}
    for record in records:
        path_value = record.get("path")
        if not path_value:
            continue
        path = Path(str(path_value)).resolve()
        by_path.setdefault(str(path), record)

    directories = sorted(
        (
            (Path(path), record)
            for path, record in by_path.items()
            if Path(path).is_dir()
        ),
        key=lambda item: len(item[0].parts),
    )
    selected_directories: List[tuple[Path, Dict[str, Any]]] = []
    for path, record in directories:
        if any(path.is_relative_to(parent) for parent, _record in selected_directories):
            continue
        selected_directories.append((path, record))

    selected_files = []
    for path_value, record in by_path.items():
        path = Path(path_value)
        if not path.is_file():
            continue
        if any(path.is_relative_to(parent) for parent, _record in selected_directories):
            continue
        selected_files.append(record)
    return selected_files + [record for _path, record in selected_directories]


def table_render_requested(user_message: str) -> bool:
    """Render tabular UI only when the user explicitly requested a table/list."""
    return any(
        item.get("name") == "table"
        for item in infer_requested_deliverables(user_message)
    )


def _data_content_fingerprint(record: Dict[str, Any]) -> str:
    stored = str(record.get("content_fingerprint") or "")
    if stored:
        return stored
    rows = record.get("rows") or []
    if not rows:
        return ""
    canonical = {
        "columns": record.get("columns") or [],
        "rows": rows,
        "nrows": record.get("nrows", len(rows)),
        "complete": bool(record.get("complete", False)),
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"preview-sha256:{hashlib.sha256(encoded).hexdigest()}"

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


async def _zip_paths(
    paths: List[str],
    zip_path: Path,
    *,
    manifest: Optional[Dict[str, Any]] = None,
) -> List[str]:
    def _worker():
        members: List[str] = []
        seen_paths: set[str] = set()
        seen_arcnames: set[str] = set()

        def _write_file(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
            resolved = str(file_path.resolve())
            if resolved in seen_paths:
                return
            candidate = arcname
            counter = 2
            while candidate in seen_arcnames:
                candidate = f"{counter}-{arcname}"
                counter += 1
            zf.write(file_path, arcname=candidate)
            seen_paths.add(resolved)
            seen_arcnames.add(candidate)
            members.append(candidate)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                pth = Path(p)
                if pth.is_dir():
                    base = pth.parent
                    for f in pth.rglob("*"):
                        if f.is_file():
                            _write_file(zf, f, str(f.relative_to(base)))
                elif pth.is_file():
                    _write_file(zf, pth, pth.name)
            if members and manifest:
                zf.writestr(
                    "artifact-manifest.json",
                    json.dumps(manifest, indent=2, sort_keys=True, default=str),
                )
                members.append("artifact-manifest.json")
        return members

    return await asyncio.to_thread(_worker)


def _collect_tool_payloads(messages: List[Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, dict):
                payload = dict(content)
                if not payload.get("tool_name"):
                    payload["tool_name"] = getattr(m, "name", None)
                if not payload.get("tool_call_id"):
                    payload["tool_call_id"] = getattr(m, "tool_call_id", None)
                payloads.append(payload)
            else:
                try:
                    decoded = json.loads(content)
                    if isinstance(decoded, dict):
                        if not decoded.get("tool_name"):
                            decoded["tool_name"] = getattr(m, "name", None)
                        if not decoded.get("tool_call_id"):
                            decoded["tool_call_id"] = getattr(m, "tool_call_id", None)
                        payloads.append(decoded)
                except Exception:
                    pass
    return payloads


def _iter_artifact_dicts(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _iter_map_values(value: Any) -> List[Any]:
    values: List[Any] = []
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, dict):
                values.extend(_iter_map_values(item))
            elif isinstance(item, (list, tuple, set)):
                values.extend(item)
            else:
                values.append(item)
    return values


SAFE_SUBAGENT_UI_KINDS = {"plotly_json_path", "image_path", "binary_path"}
LOCAL_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9:])/(?:Users|private|tmp|var|opt|home|workspace|mnt)"
    r"(?:/[^\s`\"'<>]+)+"
)


def _redact_local_paths(text: Any) -> str:
    def _replacement(match: re.Match[str]) -> str:
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in ".,;:)]}":
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        name = Path(raw).name or "local artifact"
        return f"`{name}`{trailing}"

    return LOCAL_PATH_PATTERN.sub(_replacement, str(text))


def _summarize_tool_error(value: Any, *, max_chars: int = 280) -> str:
    """Return a concise user-facing tool error without dumping schemas/logs."""
    text = " ".join(str(value or "Tool execution failed.").split())
    if "Allowed values:" in text:
        text = text.split("Allowed values:", 1)[0].rstrip(" .")
        text += ". Use a canonical ROI name such as `kidney_left` or `kidney_right`."
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return _redact_local_paths(text)


def _ui_item_key(item: Dict[str, Any]) -> Optional[str]:
    kind = item.get("kind")
    path = item.get("path")
    if not kind or not path:
        return None
    return f"{kind}:{path}"


def _filter_ui_items(
    ui: List[Dict[str, Any]],
    *,
    rendered_ui_keys: Optional[set[str]] = None,
    allowed_ui_kinds: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for item in ui:
        kind = item.get("kind")
        if allowed_ui_kinds is not None and kind not in allowed_ui_kinds:
            continue
        key = _ui_item_key(item)
        if rendered_ui_keys is not None and key is not None:
            if key in rendered_ui_keys:
                continue
        filtered.append(item)
    return filtered


def _code_item_key(code: str, title: str) -> str:
    digest = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()
    return f"{title}:{digest}"


def _extract_code_items(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    outputs = payload.get("outputs", {}) or {}
    if outputs.get("code"):
        items.append({"title": "Generated code", "code": str(outputs["code"])})

    for artifact in _iter_artifact_dicts(payload.get("artifacts")):
        if artifact.get("type") == "code" and artifact.get("content"):
            title = str(artifact.get("name") or "Generated code")
            items.append({"title": title, "code": str(artifact["content"])})

    recommended = payload.get("next_recommended_inputs")
    if isinstance(recommended, dict):
        for key, value in recommended.items():
            if value and "code" in str(key).lower():
                items.append({"title": str(key), "code": str(value)})

    return items


async def _render_code_items(
    payload: Dict[str, Any],
    *,
    rendered_code_keys: Optional[set[str]] = None,
):
    for item in _extract_code_items(payload):
        code = item["code"]
        title = item["title"]
        if rendered_code_keys is not None:
            key = _code_item_key(code, title)
            if key in rendered_code_keys:
                continue
            rendered_code_keys.add(key)
        code_el = cl.CustomElement(
            name="IdcCodeView",
            props={"code": code, "title": title},
            display="inline",
        )
        await cl.Message(content="Generated code (expand to inspect):", elements=[code_el]).send()


async def _render_payload(
    payload: Dict[str, Any],
    *,
    rendered_ui_keys: Optional[set[str]] = None,
    rendered_code_keys: Optional[set[str]] = None,
    rendered_artifact_ids: Optional[set[str]] = None,
    rendered_data_ids: Optional[set[str]] = None,
    rendered_data_fingerprints: Optional[set[str]] = None,
    rendered_file_paths: Optional[set[str]] = None,
    rendered_external_urls: Optional[set[str]] = None,
    allowed_ui_kinds: Optional[set[str]] = None,
    render_outputs: bool = True,
    render_tables: bool = True,
    render_errors: bool = True,
    render_external_links: bool = True,
) -> List[Dict[str, Any]]:
    rendered: List[Dict[str, Any]] = []
    status = str(payload.get("status") or ("ok" if payload.get("ok", True) else "error"))
    if status == "error":
        err = payload.get("error") or "; ".join(payload.get("errors", []) or [])
        err = err or "Tool returned an error."
        if render_errors:
            await cl.Message(content=f"Warning: {_summarize_tool_error(err)}").send()
        return rendered

    outputs = payload.get("outputs", {}) or {}
    ui = _filter_ui_items(
        payload.get("ui", []) or [],
        rendered_ui_keys=rendered_ui_keys,
        allowed_ui_kinds=allowed_ui_kinds,
    )

    await _render_code_items(payload, rendered_code_keys=rendered_code_keys)

    for item in ui:
        kind = item.get("kind")
        if kind == "plotly_json_path":
            path = item.get("path")
            try:
                from plotly.io import from_json
                spec = Path(path).read_text()
                fig = from_json(spec)
                await cl.Message(
                    content="Interactive chart:",
                    elements=[cl.Plotly(name="plot", figure=fig)],
                ).send()
                rendered.append(
                    {
                        "kind": "plotly",
                        "source": "tool_ui",
                        "title": str(item.get("title") or "Interactive chart"),
                        "path": str(path),
                        "artifact_id": item.get("artifact_id"),
                    }
                )
                if rendered_artifact_ids is not None and item.get("artifact_id"):
                    rendered_artifact_ids.add(str(item["artifact_id"]))
                key = _ui_item_key(item)
                if rendered_ui_keys is not None and key:
                    rendered_ui_keys.add(key)
            except Exception:
                await cl.Message(content="(Plotly figure could not be rendered.)").send()
        elif kind == "image_path":
            path = item.get("path")
            if path and Path(path).exists():
                await cl.Message(
                    content="Here is your result:",
                    elements=[cl.Image(name=Path(path).name, path=path)],
                ).send()
                rendered.append(
                    {
                        "kind": "image",
                        "source": "tool_ui",
                        "title": str(item.get("title") or Path(path).name),
                        "path": str(path),
                        "artifact_id": item.get("artifact_id"),
                    }
                )
                if rendered_artifact_ids is not None and item.get("artifact_id"):
                    rendered_artifact_ids.add(str(item["artifact_id"]))
                key = _ui_item_key(item)
                if rendered_ui_keys is not None and key:
                    rendered_ui_keys.add(key)
        elif kind == "binary_path":
            path = item.get("path")
            if path and Path(path).exists():
                await cl.Message(
                    content="Here is your file:",
                    elements=[cl.File(name=Path(path).name, path=path)],
                ).send()
                rendered.append(
                    {
                        "kind": "file",
                        "source": "tool_ui",
                        "title": str(item.get("title") or Path(path).name),
                        "path": str(path),
                        "artifact_id": item.get("artifact_id"),
                    }
                )
                if rendered_artifact_ids is not None and item.get("artifact_id"):
                    rendered_artifact_ids.add(str(item["artifact_id"]))
                if rendered_file_paths is not None:
                    rendered_file_paths.add(str(Path(path).resolve()))
                key = _ui_item_key(item)
                if rendered_ui_keys is not None and key:
                    rendered_ui_keys.add(key)

    payload_data_records = [
        record
        for record in payload.get("data", []) or []
        if isinstance(record, dict) and record.get("kind") == "table"
    ]
    for record in payload_data_records:
        for viewer_url in (
            idc_viewer_urls_from_rows(record.get("rows") or [])
            if render_external_links
            else []
        ):
            if rendered_external_urls is not None and viewer_url in rendered_external_urls:
                continue
            await cl.Message(
                content=f"[Open this series in the IDC Viewer]({viewer_url})"
            ).send()
            rendered.append(
                {
                    "kind": "link",
                    "source": "tool_outputs",
                    "title": "Open in IDC Viewer",
                    "path": None,
                    "url": viewer_url,
                    "data_id": record.get("data_id"),
                }
            )
            if rendered_external_urls is not None:
                rendered_external_urls.add(viewer_url)
    data_records = [
        record
        for record in payload_data_records
        if render_tables and record.get("visibility", "user") == "user"
    ]
    if render_tables and not payload_data_records:
        for key, preview in outputs.items():
            if not (
                (key == "df_preview" or str(key).endswith("_df_preview"))
                and isinstance(preview, dict)
            ):
                continue
            data_records.append(
                {
                    "data_id": str(preview.get("data_id") or f"legacy:{key}"),
                    "name": str(key),
                    "rows": preview.get("rows") or [],
                    "nrows": preview.get("nrows", len(preview.get("rows") or [])),
                    "complete": bool(preview.get("complete", False)),
                }
            )
    for record in data_records:
        data_id = str(record.get("data_id") or "")
        if rendered_data_ids is not None and data_id in rendered_data_ids:
            continue
        fingerprint = _data_content_fingerprint(record)
        if (
            rendered_data_fingerprints is not None
            and fingerprint
            and fingerprint in rendered_data_fingerprints
        ):
            continue
        rows = record.get("rows") or []
        if not rows:
            continue
        dataframe = pd.DataFrame(rows)
        total_rows = int(record.get("nrows") or len(dataframe))
        suffix = "" if record.get("complete") else f"\n\nShowing {len(dataframe)} of {total_rows} rows."
        await cl.Message(
            content=f"{dataframe.to_markdown(index=False)}{suffix}"
        ).send()
        if rendered_data_ids is not None and data_id:
            rendered_data_ids.add(data_id)
        if rendered_data_fingerprints is not None and fingerprint:
            rendered_data_fingerprints.add(fingerprint)
        rendered.append(
            {
                "kind": "dataframe",
                "source": "tool_outputs",
                "title": str(record.get("name") or "table"),
                "path": None,
                "data_id": data_id or None,
            }
        )

    if not render_outputs:
        return rendered

    verified_records: List[Dict[str, Any]] = []
    for record in payload.get("artifacts", []) or []:
        if not isinstance(record, dict):
            continue
        artifact_id = str(record.get("artifact_id") or "")
        path_value = record.get("path")
        if (
            record.get("status") != "verified"
            or record.get("role") in {"input", "visualization"}
            or not record.get("downloadable", True)
            or not path_value
            or not Path(str(path_value)).exists()
        ):
            continue
        if rendered_artifact_ids is not None and artifact_id in rendered_artifact_ids:
            continue
        resolved = str(Path(str(path_value)).resolve())
        if rendered_file_paths is not None and resolved in rendered_file_paths:
            continue
        verified_records.append(record)

    if not verified_records:
        legacy_files = (
            outputs.get("files")
            or outputs.get("nifti_paths")
            or outputs.get("segmentations")
            or []
        )
        if not legacy_files and isinstance(outputs.get("segmentations_map"), dict):
            legacy_files = [
                str(value)
                for value in _iter_map_values(outputs.get("segmentations_map"))
                if value
            ]
        for legacy_path in legacy_files:
            if Path(str(legacy_path)).exists():
                verified_records.append(
                    {
                        "artifact_id": f"legacy:{legacy_path}",
                        "name": Path(str(legacy_path)).name,
                        "path": str(legacy_path),
                        "kind": "file",
                        "role": "output",
                        "status": "verified",
                        "downloadable": True,
                        "source_tool": payload.get("tool_name") or "",
                    }
                )

    if verified_records:
        selected_records = _select_nonoverlapping_artifact_records(verified_records)
        files = [str(record["path"]) for record in selected_records]
        tool = str(payload.get("tool_name") or outputs.get("tool") or "voxelinsight")
        zip_tmpdir = get_temp_dir(prefix="vi_zip")
        zip_name = f"{safe_slug(tool, 'voxelinsight')}-artifacts.zip"
        zip_path = zip_tmpdir / zip_name
        manifest = {
            "schema_version": "voxelinsight.artifact-package.v1",
            "tool": tool,
            "artifacts": [
                {
                    key: record.get(key)
                    for key in (
                        "artifact_id",
                        "name",
                        "kind",
                        "role",
                        "mime_type",
                        "size_bytes",
                        "checksum_sha256",
                        "source_tool",
                        "source_call_id",
                        "run_id",
                    )
                }
                for record in selected_records
            ],
        }
        members = await _zip_paths(files, zip_path, manifest=manifest)
        payload_members = [
            member for member in members if member != "artifact-manifest.json"
        ]
        if not payload_members:
            zip_path.unlink(missing_ok=True)
            return rendered

        if tool in {"dicom2nifti", "dicom2nifti_batch"}:
            output_content = f"**DICOM to NIfTI conversion complete:**\n- Files: {len(payload_members)}\n\nClick to download:"
        elif tool == "tcia_download":
            output_content = f"**TCIA download complete:**\n- Files: {len(payload_members)}\n\nClick to download:"
        elif tool == "midrc_download":
            output_content = f"**MIDRC download complete:**\n- Files: {len(payload_members)}\n\nClick to download:"
        else:
            output_content = f"**Files ready**\n- Files: {len(payload_members)}\n\nClick to download:"

        await cl.Message(
            content=output_content,
            elements=[cl.File(name=zip_path.name, path=str(zip_path))],
        ).send()
        rendered.append(
            {
                "kind": "file",
                "source": "verified_file",
                "title": zip_path.name,
                "path": str(zip_path),
            }
        )
        # A selected parent directory packages all descendant artifacts. Mark
        # every verified source record as rendered, not only the parent record,
        # so the final registry snapshot cannot send the same files again.
        for record in verified_records:
            artifact_id = str(record.get("artifact_id") or "")
            if rendered_artifact_ids is not None and artifact_id:
                rendered_artifact_ids.add(artifact_id)
            if rendered_file_paths is not None:
                rendered_file_paths.add(str(Path(str(record["path"])).resolve()))

    if render_errors and status == "partial" and payload.get("errors"):
        await cl.Message(
            content="Completed with warnings: "
            + _summarize_tool_error("; ".join(payload.get("errors") or []))
        ).send()
    return rendered


FILE_AVAILABILITY_CLAIM = re.compile(
    r"\b(?:"
    r"(?:can|may)\s+(?:also\s+)?download|"
    r"downloadable|"
    r"(?:file|csv|export|artifact)\s+(?:is\s+)?(?:available|attached|ready)|"
    r"(?:attached|saved|exported)\s+(?:file|csv|artifact)|"
    r"provide(?:d)?\s+(?:the\s+)?(?:sorted\s+)?(?:table\s+as\s+)?"
    r"(?:a\s+)?downloadable"
    r")\b",
    re.IGNORECASE,
)

INTERNAL_REGISTRY_ID = re.compile(
    r"\b(?:artifact|data)-[0-9a-f]{12,64}\b",
    re.IGNORECASE,
)


def remove_internal_registry_references(text: str) -> str:
    """Remove model-authored user-facing lines that expose internal registry IDs."""
    if not INTERNAL_REGISTRY_ID.search(text):
        return text
    kept = [
        line
        for line in text.splitlines()
        if not INTERNAL_REGISTRY_ID.search(line)
    ]
    return "\n".join(kept).strip()


def remove_unsupported_file_claims(
    text: str,
    visible_outputs: List[Dict[str, Any]],
) -> str:
    """Remove file/download claims unless a file was actually rendered."""
    if any(item.get("kind") == "file" for item in visible_outputs):
        return text
    if not FILE_AVAILABILITY_CLAIM.search(text):
        return text

    lines: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            lines.append(line)
            continue
        fragments = [
            fragment.strip()
            for fragment in re.split(r"(?<=[.!?])\s+", line)
            if fragment.strip()
        ]
        kept = [
            fragment
            for fragment in fragments
            if not FILE_AVAILABILITY_CLAIM.search(fragment)
        ]
        if kept:
            lines.append(" ".join(kept))

    cleaned = "\n".join(lines).strip()
    if cleaned:
        return cleaned
    if any(item.get("kind") in {"plotly", "image"} for item in visible_outputs):
        return "The requested chart was created and displayed."
    if any(item.get("kind") == "dataframe" for item in visible_outputs):
        return "The requested table was created and displayed."
    return text


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text") or "")
        return "".join(chunks)
    return str(content or "")


def _decode_tool_args(args: Any) -> Any:
    if not isinstance(args, str):
        return args
    try:
        return json.loads(args)
    except Exception:
        return args


def _has_tool_args(args: Any) -> bool:
    return args not in (None, "", {}, [])


def _normalize_tool_name(name: Any) -> Optional[str]:
    if name is None:
        return None
    normalized = str(name).strip()
    if not normalized:
        return None
    return normalized.rsplit(".", 1)[-1]


def _extract_task_subagent_prompt(args: Any) -> Optional[tuple[str, str]]:
    decoded = _decode_tool_args(args)
    if isinstance(decoded, dict):
        subagent = _extract_task_subagent_name(decoded) or "unspecified"
        prompt = (
            decoded.get("prompt")
            or decoded.get("instructions")
            or decoded.get("task")
            or decoded.get("description")
        )
        if prompt:
            return str(subagent), str(prompt)
        return None
    if isinstance(decoded, str) and decoded.strip():
        stripped = decoded.strip()
        if stripped.startswith(("{", "[")):
            return None
        return "unspecified", decoded
    return None


def _extract_task_subagent_name(args: Any) -> Optional[str]:
    decoded = _decode_tool_args(args)
    if not isinstance(decoded, dict):
        return None
    subagent = (
        decoded.get("subagent_type")
        or decoded.get("subagent_name")
        or decoded.get("agent")
        or decoded.get("name")
    )
    if subagent is None:
        return None
    subagent_name = str(subagent).strip()
    return subagent_name or None


def _sanitize_step_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_step_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_step_value(item, key=key) for item in value]
    if isinstance(value, str):
        if any(
            marker in key.lower()
            for marker in (
                "path",
                "file",
                "dir",
                "mask",
                "segment",
                "artifact",
            )
        ):
            path = Path(value)
            if path.is_absolute():
                return f"<local-file:{path.name}>"
        return _redact_local_paths(value)
    return value


def _task_scope_from_prompt(prompt: str) -> str:
    start_tag = "<supervisor_task>"
    end_tag = "</supervisor_task>"
    if start_tag in prompt and end_tag in prompt:
        return prompt.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return prompt[:800]


SUBAGENT_BY_TOOL = {
    tool_name: subagent_name
    for subagent_name, tool_names in DOMAIN_SUBAGENT_TOOL_NAMES.items()
    for tool_name in tool_names
}


def _extract_nested_tool_call_name(call: Any) -> Optional[str]:
    if not isinstance(call, dict):
        return _normalize_tool_name(getattr(call, "name", None))
    name = (
        call.get("name")
        or call.get("tool_name")
        or call.get("recipient_name")
    )
    function = call.get("function")
    if name is None and isinstance(function, dict):
        name = function.get("name")
    return _normalize_tool_name(name)


def _infer_task_subagent_name_from_output(content: Any) -> Optional[str]:
    decoded = _decode_tool_args(content)
    if not isinstance(decoded, dict):
        return None

    direct = _extract_task_subagent_name(decoded)
    if direct:
        return direct

    tool_calls = decoded.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return None

    subagent_names = {
        SUBAGENT_BY_TOOL[tool_name]
        for call in tool_calls
        if (tool_name := _extract_nested_tool_call_name(call)) in SUBAGENT_BY_TOOL
    }
    if len(subagent_names) == 1:
        return next(iter(subagent_names))
    return None


def _format_tool_step_input(tool_name: str, args: Any) -> Any:
    decoded = _decode_tool_args(args)
    if tool_name != "task":
        return _sanitize_step_value(decoded)

    extracted = _extract_task_subagent_prompt(decoded)
    if not extracted:
        return decoded

    subagent_name, prompt = extracted
    formatted: Dict[str, Any] = {
        "subagent": subagent_name,
        "task": _task_scope_from_prompt(prompt),
    }
    if isinstance(decoded, dict):
        consumed_keys = {
            "subagent_type",
            "subagent_name",
            "agent",
            "name",
            "prompt",
            "instructions",
            "task",
            "description",
        }
        extra = {k: v for k, v in decoded.items() if k not in consumed_keys}
        if extra:
            formatted["args"] = _sanitize_step_value(extra)
    return _sanitize_step_value(formatted)


def _extract_tool_call_parts(call: Any) -> tuple[Optional[str], Optional[str], Any, Optional[int]]:
    if not isinstance(call, dict):
        return (
            _normalize_tool_name(getattr(call, "name", None)),
            getattr(call, "id", None),
            getattr(call, "args", None),
            getattr(call, "index", None),
        )

    function = call.get("function")
    function_args = function.get("arguments") if isinstance(function, dict) else None
    function_name = function.get("name") if isinstance(function, dict) else None
    name = _normalize_tool_name(call.get("name") or function_name)
    call_id = call.get("id")
    args = call.get("args") if "args" in call else call.get("arguments", function_args)
    index = call.get("index")
    if index is not None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = None
    return name, call_id, args, index


def _tool_display_label(
    tool_name: str,
    args: Any = None,
    cached_subagent_name: Optional[str] = None,
) -> str:
    tool_name = _normalize_tool_name(tool_name) or tool_name
    if tool_name == "task":
        subagent_name = _extract_task_subagent_name(args) or cached_subagent_name
        if subagent_name:
            return f"VoxelInsight Subagent: {subagent_name}"
    return TOOL_DESCRIPTIONS.get(tool_name, tool_name)


TOOL_DESCRIPTIONS = {
    "task": "VoxelInsight Subagent",
    "idc_query": "IDC Query Tool",
    "bih_query": "BIH Query Tool",
    "imaging": "TotalSegmentator Segmentation - this may take a while",
    "monai": "MONAI Segmentation - this may take a while",
    "nnunet": "nnU-Net Tumor Segmentation - this may take a while",
    "radiomics": "Radiomics Analysis",
    "viz_slider": "Slider Visualization Tool",
    "dicom2nifti": "DICOM to NIfTI Conversion",
    "dicom2nifti_batch": "DICOM to NIfTI Batch Conversion",
    "dicom_to_nifti": "DICOM to NIfTI Conversion",
    "code_gen": "Code Generation",
    "midrc_query": "MIDRC Query Tool",
    "midrc_download": "MIDRC Download Tool",
    "tcia_download": "TCIA Download Tool",
    "idc_download": "IDC Download Tool",
    "clinical_data_download": "Clinical Data Download",
    "image_registration": "Image Registration",
    "merlin_3d": "Merlin 3D Embedding",
    "biomedclip": "BiomedCLIP Biomedical Image Analysis",
    "brainiac": "BrainIAC MRI Analysis - this may take a while",
    "universeg": "Universeg Segmentation",
    "verify_artifacts": "Artifact Verification",
}


class VoxelInsightHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.node_descriptions = {
            "agent": "VoxelInsight",
            "tools": "Tools",
            "final": "VoxelInsight Final",
            "voxelinsight-deepagent": "VoxelInsight",
        }
        self.tool_descriptions = TOOL_DESCRIPTIONS

    async def _rename_root(self, name: str):
        try:
            step = cl.context.current_step
            if step is not None:
                step.name = name
                await step.update()
        except Exception:
            pass

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        await self._rename_root("VoxelInsight")

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        await self._rename_root("VoxelInsight")

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        await self._rename_root("VoxelInsight")

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "tools")
        label = _tool_display_label(tool_name, input_str)
        await self._rename_root(label)

    async def on_tool_end(self, output: str, **kwargs) -> None:
        pass

    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        await self._rename_root("VoxelInsight")


@cl.on_message
async def on_message(message: cl.Message):
    file_elements = [el for el in (message.elements or []) if isinstance(el, cl.File)]
    files: List[str] = []
    upload_dir = (
        get_run_dir("uploads", persist=True, prefix="message")
        if file_elements
        else None
    )
    for index, f in enumerate(file_elements, start=1):
        assert upload_dir is not None
        filename = safe_slug(Path(f.name).name, f"upload-{index}")
        new_path = upload_dir / filename
        if new_path.exists():
            new_path = upload_dir / f"{index}-{filename}"
        shutil.copy2(f.path, new_path)
        files.append(str(new_path))

    session_thread_id = getattr(cl.context.session, "thread_id", None)
    thread_id = (
        cl.user_session.get("voxelinsight_thread_id")
        or session_thread_id
        or cl.context.session.id
    )
    cl.user_session.set("voxelinsight_thread_id", thread_id)
    status_handler = VoxelInsightHandler()
    await status_handler._rename_root("Initializing VoxelInsight...")

    tool_steps_by_key: Dict[str, cl.Step] = {}
    tool_names_by_key: Dict[str, str] = {}
    tool_args_by_key: Dict[str, Any] = {}
    tool_chunk_keys_by_index: Dict[tuple[tuple[str, ...], int], str] = {}
    task_subagent_names_by_key: Dict[str, str] = {}
    rendered_ui_keys: set[str] = set()
    rendered_code_keys: set[str] = set()
    rendered_artifact_ids: set[str] = set()
    rendered_data_ids: set[str] = set()
    rendered_data_fingerprints: set[str] = set()
    rendered_file_paths: set[str] = set()
    rendered_external_urls: set[str] = set()
    visible_outputs: List[Dict[str, Any]] = []
    run_artifacts: Dict[str, Dict[str, Any]] = {}
    pending_main_text = ""

    def _record_rendered_outputs(items: List[Dict[str, Any]]) -> None:
        seen = {
            (
                item.get("kind"),
                item.get("path"),
                item.get("artifact_id"),
                item.get("data_id"),
                item.get("url"),
            )
            for item in visible_outputs
        }
        for item in items:
            key = (
                item.get("kind"),
                item.get("path"),
                item.get("artifact_id"),
                item.get("data_id"),
                item.get("url"),
            )
            if key in seen:
                continue
            seen.add(key)
            visible_outputs.append(item)

    async def _flush_pending_as_progress() -> None:
        nonlocal pending_main_text
        text = pending_main_text.strip()
        pending_main_text = ""
        if text:
            await cl.Message(content=text).send()

    render_tables = table_render_requested(message.content)
    requested_deliverables = infer_requested_deliverables(message.content)
    render_external_links = any(
        "viewer_link" in (deliverable.get("needs") or [])
        for deliverable in requested_deliverables
        if isinstance(deliverable, dict)
    )
    logged_subagent_prompt_keys = set()

    def _decode_tool_output(content: Any) -> Any:
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                return content
        return content

    def _log_subagent_prompt_once(key: str, args: Any) -> None:
        if key in logged_subagent_prompt_keys:
            return
        extracted = _extract_task_subagent_prompt(args)
        if not extracted:
            return
        subagent, prompt = extracted
        logged_subagent_prompt_keys.add(key)
        print("\n=== DeepAgent subagent prompt ===", flush=True)
        print(f"subagent: {subagent}", flush=True)
        print(f"call_id: {key}", flush=True)
        print("prompt:", flush=True)
        print(prompt, flush=True)
        print("=== End DeepAgent subagent prompt ===\n", flush=True)

    async def _record_tool_call(
        tool_name: Optional[str],
        call_id: Optional[str] = None,
        args: Any = None,
    ):
        key = call_id or tool_name
        tool_name = _normalize_tool_name(tool_name) or tool_names_by_key.get(key or "")
        if not tool_name or not key:
            return

        tool_names_by_key[key] = tool_name

        combined_args = tool_args_by_key.get(key)
        if _has_tool_args(args):
            if isinstance(args, str) and isinstance(combined_args, str):
                combined_args = combined_args + args
            elif not _has_tool_args(combined_args):
                combined_args = args
            else:
                combined_args = args
            tool_args_by_key[key] = combined_args

        if tool_name == "task":
            subagent_name = _extract_task_subagent_name(combined_args)
            if subagent_name:
                task_subagent_names_by_key[key] = subagent_name
        if tool_name == "task" and _has_tool_args(combined_args):
            _log_subagent_prompt_once(key, combined_args)
        label = _tool_display_label(
            tool_name,
            combined_args,
            task_subagent_names_by_key.get(key),
        )
        await _flush_pending_as_progress()
        await status_handler._rename_root(label)

        step = tool_steps_by_key.get(key)
        if step is None:
            root_step = cl.context.current_step
            step = cl.Step(
                name=label,
                type="tool",
                parent_id=getattr(root_step, "id", None),
                show_input="json",
                default_open=False,
            )
            if _has_tool_args(combined_args):
                step.input = _format_tool_step_input(tool_name, combined_args)
            await step.send()
            tool_steps_by_key[key] = step
            return

        needs_update = False
        if step.name != label:
            step.name = label
            needs_update = True
        if _has_tool_args(combined_args):
            step.input = _format_tool_step_input(tool_name, combined_args)
            if tool_name == "task":
                _log_subagent_prompt_once(key, combined_args)
                subagent_name = _extract_task_subagent_name(combined_args)
                if subagent_name:
                    task_subagent_names_by_key[key] = subagent_name
                    label = _tool_display_label(tool_name, combined_args, subagent_name)
                    await status_handler._rename_root(label)
                    if step.name != label:
                        step.name = label
            needs_update = True
        if needs_update:
            await step.update()

    async def _complete_tool_call(tool_message: ToolMessage):
        tool_name = _normalize_tool_name(getattr(tool_message, "name", None))
        call_id = getattr(tool_message, "tool_call_id", None)
        if not tool_name and not call_id:
            return
        key = call_id or tool_name
        tool_name = tool_name or tool_names_by_key.get(key or "")
        if not tool_name or not key:
            return

        if key not in tool_steps_by_key:
            await _record_tool_call(tool_name, call_id)

        step = tool_steps_by_key.get(key)
        if step is None:
            return

        if tool_name == "task" and key not in task_subagent_names_by_key:
            inferred_subagent_name = _infer_task_subagent_name_from_output(tool_message.content)
            if inferred_subagent_name:
                task_subagent_names_by_key[key] = inferred_subagent_name

        label = _tool_display_label(
            tool_name,
            tool_args_by_key.get(key),
            task_subagent_names_by_key.get(key),
        )
        if step.name != label:
            step.name = label
        if _has_tool_args(tool_args_by_key.get(key)) and step.input in (None, "", {}, []):
            step.input = _format_tool_step_input(tool_name, tool_args_by_key[key])
        step.output = _sanitize_step_value(_decode_tool_output(tool_message.content))
        await step.update()
        await status_handler._rename_root("VoxelInsight")

    # Only the newest progress state matters to the custom element. Bounding
    # this queue prevents chat/UI backpressure from retaining thousands of
    # subprocess heartbeat events during cohort jobs.
    progress_q = asyncio.Queue(maxsize=1)
    set_progress_queue(progress_q)

    progress_el = None
    progress_msg = None

    async def _drain_progress():
        nonlocal progress_el, progress_msg
        while True:
            event = await progress_q.get()
            if event is None:
                break
            if progress_msg is None:
                progress_el = cl.CustomElement(
                    name="VoxelProgress",
                    props=event,
                    display="inline",
                )
                progress_msg = cl.Message(
                    content="", author="VoxelInsight", elements=[progress_el]
                )
                await progress_msg.send()
            else:
                progress_el.props = event
                progress_el.content = json.dumps(event)
                await progress_el.update()

    drain_task = asyncio.create_task(_drain_progress())

    async def _handle_stream_event(raw_part: Any, active_run_id: str) -> None:
        nonlocal pending_main_text
        event, meta, namespace = extract_stream_part(raw_part)
        is_subagent = any(str(segment).startswith("tools:") for segment in namespace)
        node = meta.get("langgraph_node") if isinstance(meta, dict) else None
        if node and not is_subagent:
            friendly = status_handler.node_descriptions.get(node, node)
            await status_handler._rename_root(friendly)

        if isinstance(event, ToolMessage):
            await _complete_tool_call(event)
            payloads = _collect_tool_payloads([event])
            for payload in payloads:
                if payload.get("tool_name") == "task":
                    continue
                delta = registry_delta_from_payload(
                    payload,
                    tool_name=str(payload.get("tool_name") or ""),
                    tool_call_id=str(payload.get("tool_call_id") or ""),
                    run_id=active_run_id,
                )
                run_artifacts.update(delta.get("artifact_registry") or {})
                rendered = await _render_payload(
                    payload,
                    rendered_ui_keys=rendered_ui_keys,
                    rendered_code_keys=rendered_code_keys,
                    rendered_artifact_ids=rendered_artifact_ids,
                    rendered_data_ids=rendered_data_ids,
                    rendered_data_fingerprints=rendered_data_fingerprints,
                    rendered_file_paths=rendered_file_paths,
                    rendered_external_urls=rendered_external_urls,
                    allowed_ui_kinds=SAFE_SUBAGENT_UI_KINDS,
                    render_outputs=True,
                    render_tables=render_tables,
                    render_errors=not is_subagent,
                    render_external_links=render_external_links,
                )
                _record_rendered_outputs(rendered)

        if isinstance(event, (AIMessage, AIMessageChunk)):
            content = _stringify_content(getattr(event, "content", None))
            tool_calls = getattr(event, "tool_calls", None) or []
            tool_call_chunks = getattr(event, "tool_call_chunks", None) or []
            has_tool_call = bool(tool_calls)
            has_tool_call_chunks = bool(tool_call_chunks)
            if has_tool_call or has_tool_call_chunks or is_subagent:
                if content and not is_subagent:
                    pending_main_text += content
                for call in tool_calls:
                    name, call_id, args, _index = _extract_tool_call_parts(call)
                    await _record_tool_call(name, call_id, args)
                for chunk in tool_call_chunks:
                    name, call_id, args, index = _extract_tool_call_parts(chunk)
                    if index is not None:
                        ns_key = tuple(str(segment) for segment in namespace)
                        index_key = (ns_key, index)
                        if call_id:
                            tool_chunk_keys_by_index[index_key] = call_id
                        else:
                            call_id = tool_chunk_keys_by_index.get(index_key)
                            if call_id is None and name:
                                call_id = f"{'/'.join(ns_key)}:{name}:{index}"
                                tool_chunk_keys_by_index[index_key] = call_id
                    await _record_tool_call(name, call_id, args)
                return
            if content and not is_subagent:
                pending_main_text += content

    async def _confirm_interaction(request: ConfirmationRequest) -> bool:
        response = await cl.AskActionMessage(
            content=request.content,
            actions=[
                cl.Action(
                    name="continue",
                    payload={"value": "continue"},
                    label="✅ Continue",
                ),
                cl.Action(
                    name="cancel",
                    payload={"value": "cancel"},
                    label="❌ Cancel",
                ),
            ],
        ).send()
        return bool(
            response and response.get("payload", {}).get("value") == "continue"
        )

    async def _notify_interaction(content: str) -> None:
        await cl.Message(content=content).send()

    try:
        with interaction_context(
            InteractionHandler(
                confirm=_confirm_interaction,
                notify=_notify_interaction,
            )
        ):
            turn_result = await execute_turn(
                message.content,
                thread_id=str(thread_id),
                uploaded_files=files,
                callbacks=[status_handler],
                on_stream_event=_handle_stream_event,
            )
        registry_payload = turn_result.registry_payload
        run_artifacts = {
            str(record["artifact_id"]): record
            for record in registry_payload["artifacts"]
            if record.get("artifact_id")
        }
        _record_rendered_outputs(
            await _render_payload(
                registry_payload,
                rendered_ui_keys=rendered_ui_keys,
                rendered_code_keys=rendered_code_keys,
                rendered_artifact_ids=rendered_artifact_ids,
                rendered_data_ids=rendered_data_ids,
                rendered_data_fingerprints=rendered_data_fingerprints,
                rendered_file_paths=rendered_file_paths,
                rendered_external_urls=rendered_external_urls,
                allowed_ui_kinds=SAFE_SUBAGENT_UI_KINDS,
                render_outputs=True,
                render_tables=render_tables,
                render_external_links=render_external_links,
            )
        )

        final_text = pending_main_text.strip()
        if not final_text and visible_outputs:
            final_text = "The requested output was completed and attached."
        if final_text:
            supported_final_text = remove_unsupported_file_claims(
                final_text,
                visible_outputs,
            )
            supported_final_text = remove_internal_registry_references(
                supported_final_text
            )
            await cl.Message(
                content=_redact_local_paths(supported_final_text)
            ).send()

    except asyncio.CancelledError:
        await update_progress(100, "Workflow cancelled", status="cancelled")
        raise
    except TypeError as e:
        await update_progress(100, "Workflow failed", status="error")
        if "subgraphs" not in str(e) and "version" not in str(e):
            raise
        await cl.Message(
            content=(
                "This installed LangGraph/Deep Agents stack does not support the current "
                "`astream(..., subgraphs=True, version='v2')` streaming API. "
                f"Error: {e}"
            )
        ).send()
    except httpx.RemoteProtocolError:
        await update_progress(100, "Model stream ended unexpectedly", status="error")
        await cl.Message(content="The model stream ended unexpectedly. Please retry the request.").send()
    except Exception:
        await update_progress(100, "Workflow failed", status="error")
        requested = infer_requested_deliverables(message.content)
        deliverable_results = (
            validate_deliverables(
                requested,
                visible_outputs,
                artifacts=list(run_artifacts.values()),
            )
            if requested
            else []
        )
        if requested and all(item.get("status") == "satisfied" for item in deliverable_results):
            await cl.Message(
                content="The requested visible output was generated and attached."
            ).send()
            return
        import traceback
        traceback.print_exc()
        await cl.Message(
            content="VoxelInsight could not complete this request because an internal error occurred."
        ).send()
    finally:
        await asyncio.sleep(0)
        set_progress_queue(None)
        await progress_q.put(None)
        try:
            await drain_task
        except Exception:
            pass


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    thread_id = thread.get("id") if isinstance(thread, dict) else None
    if thread_id and not cl.user_session.get("voxelinsight_thread_id"):
        cl.user_session.set("voxelinsight_thread_id", str(thread_id))


@cl.action_callback("action_button")
async def on_action(action):
    await action.remove()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="What can you do?",
            message="Give me a quick tour of VoxelInsight features and common workflows.",
            icon="/public/info.svg",
        ),
        cl.Starter(
            label="How many patients are in IDC?",
            message="How many patients are currently in IDC?",
            icon="/public/database.svg",
        ),
        cl.Starter(
            label="Search IDC & Plot Bar Chart",
            message="Plot a bar chart of the number patients for all breast collections in IDC.",
            icon="/public/search.svg",
        ),
        cl.Starter(
            label="Segment uploaded scan",
            message="Segment the liver in my uploaded CT scan and show me the result.",
            icon="/public/chart.svg",
        ),
    ]
