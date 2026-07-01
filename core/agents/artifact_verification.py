from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.agents.schemas import SubagentResult


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _is_probably_url(value: str) -> bool:
    lower = value.lower()
    return lower.startswith("http://") or lower.startswith("https://")


def _path_from(value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip() or _is_probably_url(value):
        return None
    return Path(value).expanduser()


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def _csv_has_rows(path: Path) -> bool:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return False
        return any(row for row in reader)


def _iter_map_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, dict):
                yield from _iter_map_values(v)
            elif isinstance(v, (list, tuple, set)):
                yield from v
            else:
                yield v


def _artifact_path_values(artifact: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in (
        "path",
        "file_path",
        "csv_path",
        "nifti_path",
        "image_path",
        "image_paths",
        "output_dir",
        "output_root",
        "files",
        "nifti_paths",
        "segmentations",
        "mask_path",
        "mask_paths",
    ):
        for raw in _as_list(artifact.get(key)):
            if isinstance(raw, str) and raw:
                values.append(raw)
    for raw in _iter_map_values(artifact.get("segmentations_map")):
        if isinstance(raw, str) and raw:
            values.append(raw)
    return values


def verify_artifact_payload(
    artifacts: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> SubagentResult:
    errors: List[str] = []
    warnings: List[str] = []
    checked: List[Dict[str, Any]] = []
    nifti_shapes: Dict[str, tuple[int, ...]] = {}
    mask_paths: List[Path] = []
    image_paths: List[Path] = []

    if not artifacts:
        return SubagentResult(
            status="no_action",
            summary="No artifacts were provided for verification.",
        )

    def check_file(path: Path, role: str) -> None:
        if not path.exists():
            errors.append(f"Missing {role}: {path}")
            checked.append({"path": str(path), "role": role, "exists": False})
            return
        checked.append({"path": str(path), "role": role, "exists": True})
        if _is_csv(path):
            try:
                if not _csv_has_rows(path):
                    errors.append(f"CSV artifact has no data rows: {path}")
            except Exception as e:
                errors.append(f"CSV artifact could not be read: {path} ({type(e).__name__}: {e})")

    def check_dir(path: Path, role: str) -> None:
        if not path.exists():
            errors.append(f"Missing {role}: {path}")
            checked.append({"path": str(path), "role": role, "exists": False})
            return
        if not path.is_dir():
            errors.append(f"{role} is not a directory: {path}")
            checked.append({"path": str(path), "role": role, "exists": True, "is_dir": False})
            return
        files = [p for p in path.rglob("*") if p.is_file()]
        checked.append({"path": str(path), "role": role, "exists": True, "file_count": len(files)})
        if not files:
            errors.append(f"{role} is empty: {path}")

    for artifact in artifacts:
        canonical_path = _path_from(artifact.get("path"))
        if canonical_path is not None:
            kind = str(artifact.get("kind") or "")
            role = str(artifact.get("role") or "")
            if kind == "directory":
                check_dir(canonical_path, role or "directory")
            elif kind == "segmentation" or role == "segmentation":
                mask_paths.append(canonical_path)
                check_file(canonical_path, "segmentation")
            elif kind in {"nifti", "registered_image"} or role == "registered_image":
                image_paths.append(canonical_path)
                check_file(canonical_path, role or kind or "image")
            else:
                check_file(canonical_path, role or kind or "path")

        for key in ("file_path", "csv_path", "nifti_path"):
            for raw in _as_list(artifact.get(key)):
                path = _path_from(raw)
                if path is not None:
                    check_file(path, key)

        for raw in _as_list(artifact.get("image_path")) + _as_list(artifact.get("image_paths")):
            path = _path_from(raw)
            if path is not None:
                image_paths.append(path)
                check_file(path, "image")

        for raw in _as_list(artifact.get("output_dir")) + _as_list(artifact.get("output_root")):
            path = _path_from(raw)
            if path is not None:
                check_dir(path, "output_dir")

        for raw in _as_list(artifact.get("files")) + _as_list(artifact.get("nifti_paths")):
            path = _path_from(raw)
            if path is not None:
                check_file(path, "file")

        for raw in _as_list(artifact.get("segmentations")):
            path = _path_from(raw)
            if path is not None:
                mask_paths.append(path)
                check_file(path, "segmentation")

        for raw in _iter_map_values(artifact.get("segmentations_map")):
            path = _path_from(raw)
            if path is not None:
                mask_paths.append(path)
                check_file(path, "segmentation")

        for raw in _as_list(artifact.get("mask_path")) + _as_list(artifact.get("mask_paths")):
            path = _path_from(raw)
            if path is not None:
                mask_paths.append(path)
                check_file(path, "mask")

    try:
        import nibabel as nib
        import numpy as np
    except Exception:
        nib = None
        np = None

    if nib is not None and np is not None:
        for path in image_paths + mask_paths:
            if not path.exists() or not _is_nifti(path):
                continue
            try:
                img = nib.load(str(path))
                shape = tuple(int(x) for x in img.shape[:3])
                nifti_shapes[str(path)] = shape
                if path in mask_paths:
                    data = img.get_fdata()
                    if not bool(np.any(data)):
                        errors.append(f"Segmentation mask is empty: {path}")
            except Exception as e:
                errors.append(f"NIfTI artifact could not be loaded: {path} ({type(e).__name__}: {e})")
    else:
        nifti_candidates = [
            path for path in image_paths + mask_paths if path.exists() and _is_nifti(path)
        ]
        if nifti_candidates:
            warnings.append("Skipped NIfTI load/dimension checks because nibabel or numpy is unavailable.")

    if image_paths and mask_paths and nifti_shapes:
        first_image_shape = next(
            (nifti_shapes.get(str(path)) for path in image_paths if str(path) in nifti_shapes),
            None,
        )
        if first_image_shape is not None:
            for path in mask_paths:
                mask_shape = nifti_shapes.get(str(path))
                if mask_shape is not None and mask_shape != first_image_shape:
                    errors.append(
                        f"Mask dimensions do not match image dimensions: {path} "
                        f"{mask_shape} != {first_image_shape}"
                    )

    valid_checked = [
        item
        for item in checked
        if item.get("exists")
        and not any(
            str(item.get("path") or "") in error
            for error in errors
        )
    ]
    valid_paths = {
        str(item.get("path"))
        for item in valid_checked
        if item.get("path")
    }
    status = (
        "partial"
        if errors and valid_checked
        else "error"
        if errors
        else "partial"
        if warnings
        else "ok"
    )
    summary_parts = [f"Verified {len(checked)} artifact references."]
    if warnings:
        summary_parts.append("Warnings: " + "; ".join(warnings))

    return SubagentResult(
        status=status,
        tool_calls=[{"name": "verify_artifacts", "status": status}],
        artifact_ids=[
            str(artifact.get("artifact_id"))
            for artifact in artifacts
            if artifact.get("artifact_id")
            and valid_paths.intersection(_artifact_path_values(artifact))
        ],
        artifacts=checked,
        summary=" ".join(summary_parts),
        errors=errors,
        next_recommended_inputs={"verified_artifacts": checked},
    )
