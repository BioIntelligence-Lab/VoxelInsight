from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from core.agents.external_links import idc_viewer_urls_from_rows


PLOT_DELIVERABLES = {"histogram", "plot", "chart"}
DATAFRAME_DELIVERABLES = {"table", "dataframe"}
FILE_DELIVERABLES = {"file", "download", "csv"}
SEGMENTATION_DELIVERABLES = {"segmentation", "mask"}
NIFTI_DELIVERABLES = {"nifti", "conversion"}
REGISTRATION_DELIVERABLES = {"registered_image", "transform"}
LINK_DELIVERABLES = {"viewer_link"}


def visible_output_policy() -> str:
    return """
Visible output model
- `ui.image_path` means the user sees an image.
- `ui.plotly_json_path` means the user sees an interactive chart.
- `outputs.*_df_preview` means tabular data is available for the app to render or summarize.
- `outputs.code` is visible as code, but code is not a requested visualization.
- A validated `viewer_url` from an IDC tool is a visible external link, not a local file.
- `outputs.files` means the app can attach/download existing files if those paths exist.
- `artifacts.path` is not visible unless the app renders it, offers it as a download, or
  the registry records it as verified.
- Local paths should never be claimed to the user unless verified and attached/rendered.
"""


def infer_requested_deliverables(user_message: str) -> List[Dict[str, Any]]:
    text = user_message.lower()
    deliverables: List[Dict[str, Any]] = []
    segmentation_requested = bool(
        re.search(r"\bsegment(?:ation|ed|ing)?\b|\bmasks?\b", text)
    )
    segmentation_visualization_requested = segmentation_requested and bool(
        re.search(r"\b(show|display|view|overlay)\b", text)
    )

    if re.search(r"\bhistogram\b|\bhist\b", text):
        deliverables.append(
            {
                "name": "histogram",
                "needs": ["histogram"],
                "data_domain": _infer_data_domain(text),
                "must_be_visible": True,
            }
        )
    elif (
        re.search(r"\b(plot|chart|graph|visuali[sz]e|figure)\b", text)
        or segmentation_visualization_requested
    ):
        deliverables.append(
            {
                "name": "plot",
                "needs": ["plot"],
                "data_domain": _infer_data_domain(text),
                "must_be_visible": True,
            }
        )

    if re.search(r"\b(csv|spreadsheet|xlsx|excel)\b", text):
        deliverables.append(
            {
                "name": "csv",
                "needs": ["csv"],
                "data_domain": _infer_data_domain(text),
                "must_be_visible": True,
            }
        )

    if re.search(r"\b(table|list|dataframe)\b", text):
        deliverables.append(
            {
                "name": "table",
                "needs": ["table"],
                "data_domain": _infer_data_domain(text),
                "must_be_visible": False,
            }
        )

    if re.search(r"\b(download|save|export)\b", text):
        deliverables.append(
            {
                "name": "download",
                "needs": ["download"],
                "data_domain": _infer_data_domain(text),
                "must_be_visible": True,
            }
        )

    if (
        re.search(r"\b(view|viewer|display|open)\b", text)
        and re.search(r"\b(sequence|series|study|scan|images?|imaging)\b", text)
        and not re.search(r"\b(download|local|inline)\b", text)
    ):
        deliverables.append(
            {
                "name": "viewer_link",
                "needs": ["viewer_link"],
                "data_domain": _infer_data_domain(text),
                "must_be_visible": True,
            }
        )

    if segmentation_requested:
        deliverables.append(
            {
                "name": "segmentation",
                "needs": ["segmentation"],
                "data_domain": "medical image",
                "must_be_visible": True,
            }
        )

    if re.search(r"\bconvert\b.*\bnifti\b|\bdicom\b.*\bnifti\b", text):
        deliverables.append(
            {
                "name": "nifti",
                "needs": ["nifti"],
                "data_domain": "medical image",
                "must_be_visible": True,
            }
        )

    if re.search(r"\bregistration\b|\bregister(?:ed|ing)?\b", text):
        deliverables.extend(
            [
                {
                    "name": "registered_image",
                    "needs": ["registered_image"],
                    "data_domain": "medical image",
                    "must_be_visible": True,
                },
                {
                    "name": "transform",
                    "needs": ["transform"],
                    "data_domain": "medical image",
                    "must_be_visible": True,
                },
            ]
        )

    return _dedupe_deliverables(deliverables)


def visible_outputs_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    visible: List[Dict[str, Optional[str]]] = []

    for item in payload.get("ui", []) or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        path = _existing_path(item.get("path"))
        if kind == "image_path" and path:
            visible.append(
                {"kind": "image", "source": "tool_ui", "title": str(item.get("name") or "Image"), "path": path}
            )
        elif kind == "plotly_json_path" and path:
            visible.append(
                {"kind": "plotly", "source": "tool_ui", "title": str(item.get("name") or "Interactive chart"), "path": path}
            )
        elif kind == "binary_path" and path:
            visible.append(
                {"kind": "file", "source": "tool_ui", "title": str(item.get("name") or "File"), "path": path}
            )

    outputs = payload.get("outputs", {}) or {}
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if key.endswith("_df_preview") or key == "df_preview":
                if isinstance(value, dict) and value.get("rows") is not None:
                    visible.append(
                        {"kind": "dataframe", "source": "tool_outputs", "title": key, "path": None}
                    )
            elif key == "code" and value:
                visible.append(
                    {"kind": "code", "source": "tool_outputs", "title": "Generated code", "path": None}
                )
            elif key in {"text", "summary"} and value:
                visible.append(
                    {"kind": "text", "source": "tool_outputs", "title": key, "path": None}
                )
            elif key in {"files", "nifti_paths", "segmentations"}:
                for path in _iter_paths(value):
                    existing = _existing_path(path)
                    if existing:
                        visible.append(
                            {"kind": "file", "source": "tool_outputs", "title": key, "path": existing}
                        )

    for item in payload.get("data", []) or []:
        if not isinstance(item, dict):
            continue
        if (
            item.get("kind") == "table"
            and item.get("data_id")
            and item.get("visibility", "user") == "user"
        ):
            visible.append(
                {
                    "kind": "dataframe",
                    "source": "tool_outputs",
                    "title": str(item.get("name") or "table"),
                    "path": None,
                    "data_id": str(item["data_id"]),
                }
            )
            for viewer_url in idc_viewer_urls_from_rows(item.get("rows") or []):
                visible.append(
                    {
                        "kind": "link",
                        "source": "tool_outputs",
                        "title": "Open in IDC Viewer",
                        "path": None,
                        "url": viewer_url,
                        "data_id": str(item["data_id"]),
                    }
                )

    for artifact in _iter_dicts(payload.get("artifacts")):
        if (
            artifact.get("status") != "verified"
            or artifact.get("role") == "input"
            or not artifact.get("downloadable", True)
        ):
            continue
        path = _existing_path(artifact.get("path"))
        if not path:
            continue
        artifact_kind = str(artifact.get("kind") or "")
        if artifact.get("source_tool") == "idc_download" and artifact_kind != "directory":
            continue
        if artifact_kind == "image":
            visible_kind = "image"
        elif artifact_kind == "plotly":
            visible_kind = "plotly"
        else:
            visible_kind = "file"
        visible.append(
            {
                "kind": visible_kind,
                "source": "verified_file",
                "title": str(artifact.get("name") or artifact_kind or "Artifact"),
                "path": path,
                "artifact_id": str(artifact.get("artifact_id") or ""),
            }
        )

    for artifact in _verified_artifacts(payload):
        path = _existing_path(artifact.get("path"))
        if path:
            visible.append(
                {"kind": "file", "source": "verified_file", "title": str(artifact.get("name") or "Verified file"), "path": path}
            )

    return _dedupe_visible_outputs(visible)


def validate_deliverables(
    requested: List[Dict[str, Any]],
    visible_outputs: List[Dict[str, Any]],
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    kinds = {str(item.get("kind")) for item in visible_outputs}
    artifact_items = [
        item
        for item in (artifacts or [])
        if isinstance(item, dict)
        and item.get("status") == "verified"
        and item.get("role") != "input"
        and _existing_path(item.get("path"))
    ]
    artifact_kinds = {str(item.get("kind") or "") for item in artifact_items}
    artifact_roles = {str(item.get("role") or "") for item in artifact_items}

    for deliverable in requested:
        name = str(deliverable.get("name") or "")
        needs = set(deliverable.get("needs") or [name])
        if needs & PLOT_DELIVERABLES:
            satisfied = bool(kinds & {"image", "plotly"})
            evidence = _first_evidence(visible_outputs, {"image", "plotly"})
        elif needs & DATAFRAME_DELIVERABLES:
            satisfied = "dataframe" in kinds
            evidence = _first_evidence(visible_outputs, {"dataframe"})
        elif needs & FILE_DELIVERABLES:
            satisfied = "file" in kinds
            evidence = _first_evidence(visible_outputs, {"file"})
        elif needs & LINK_DELIVERABLES:
            satisfied = "link" in kinds
            evidence = _first_evidence(visible_outputs, {"link"})
        elif needs & SEGMENTATION_DELIVERABLES:
            satisfied = bool(
                artifact_kinds & {"segmentation"}
                or artifact_roles & {"segmentation"}
            )
            evidence = _first_artifact_evidence(
                artifact_items,
                {"segmentation"},
                {"segmentation"},
            )
        elif needs & NIFTI_DELIVERABLES:
            satisfied = bool(artifact_kinds & {"nifti"})
            evidence = _first_artifact_evidence(artifact_items, {"nifti"}, set())
        elif needs & REGISTRATION_DELIVERABLES:
            requested_kind = next(iter(needs & REGISTRATION_DELIVERABLES))
            satisfied = bool(
                requested_kind in artifact_kinds
                or requested_kind in artifact_roles
            )
            evidence = _first_artifact_evidence(
                artifact_items,
                {requested_kind},
                {requested_kind},
            )
        else:
            satisfied = True
            evidence = "No machine-checkable visible output required."
        results.append(
            {
                "name": name,
                "status": "satisfied" if satisfied else "missing",
                "evidence": evidence if satisfied else f"Missing visible output for {name}.",
            }
        )
    return results


def deliverables_satisfied(results: Iterable[Dict[str, str]]) -> bool:
    return all(item.get("status") == "satisfied" for item in results)


def invalid_artifact_paths(payload: Dict[str, Any]) -> List[str]:
    invalid: List[str] = []
    for artifact in _iter_dicts(payload.get("artifacts")):
        for key, value in artifact.items():
            if key not in {
                "path",
                "file",
                "file_path",
                "files",
                "csv_path",
                "nifti_path",
                "nifti_paths",
                "image_path",
                "image_paths",
                "mask_path",
                "mask_paths",
                "segmentations",
                "segmentations_map",
                "output_dir",
                "output_root",
                "registered_image",
                "transform",
            }:
                continue
            for path in _iter_paths(value):
                if path and not _existing_path(path):
                    invalid.append(str(path))
    return invalid


def dataframe_rows_from_payloads(payloads: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for payload in payloads:
        outputs = payload.get("outputs", {}) or {}
        if not isinstance(outputs, dict):
            continue
        for key, value in outputs.items():
            if (key.endswith("_df_preview") or key == "df_preview") and isinstance(value, dict):
                rows = value.get("rows")
                if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
                    return rows
    return []


def _infer_data_domain(text: str) -> str:
    parts: List[str] = []
    if "idc" in text or "imaging data commons" in text:
        parts.append("IDC")
    if "breast" in text:
        parts.append("breast collections")
    if "patient" in text:
        parts.append("patient counts")
    return " ".join(parts) or "unspecified"


def _dedupe_deliverables(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        name = item["name"]
        if name in seen:
            continue
        seen.add(name)
        deduped.append(item)
    return deduped


def _dedupe_visible_outputs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = (
            item.get("kind"),
            item.get("source"),
            item.get("title"),
            item.get("path"),
            item.get("url"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _existing_path(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value)
    if text.startswith(("/artifacts/", "/plots/")):
        return None
    try:
        path = Path(text)
        if path.exists():
            return str(path)
    except OSError:
        return None
    return None


def _iter_paths(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        paths: List[str] = []
        for item in value.values():
            paths.extend(_iter_paths(item))
        return paths
    if isinstance(value, (list, tuple, set)):
        paths: List[str] = []
        for item in value:
            paths.extend(_iter_paths(item))
        return paths
    return []


def _iter_dicts(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _verified_artifacts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if payload.get("tool_name") not in {"verify_artifacts", "verification"}:
        return []
    verified: List[Dict[str, Any]] = []
    outputs = payload.get("outputs", {}) or {}
    for value in outputs.values() if isinstance(outputs, dict) else []:
        if isinstance(value, dict):
            verified.extend(_iter_dicts(value.get("artifacts")))
    return verified


def _first_evidence(visible_outputs: List[Dict[str, Any]], kinds: set[str]) -> str:
    for item in visible_outputs:
        if item.get("kind") in kinds:
            source = item.get("source")
            kind = item.get("kind")
            path = item.get("path")
            url = item.get("url")
            return f"{source}:{kind}" + (f":{path or url}" if path or url else "")
    return ""


def _first_artifact_evidence(
    artifacts: List[Dict[str, Any]],
    kinds: set[str],
    roles: set[str],
) -> str:
    for item in artifacts:
        if item.get("kind") in kinds or item.get("role") in roles:
            return (
                f"artifact:{item.get('artifact_id') or item.get('name') or item.get('kind')}"
            )
    return ""
