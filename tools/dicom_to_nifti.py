from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List

import dicom2nifti
import nibabel as nib
import pydicom
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, ConfigDict, Field

from core.state import TaskResult
from core.storage import get_run_dir
from tools.shared import toolify_agent


def _safe_slug(value: Any, default: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-._")
    return slug[:80] or default


class DicomToNiftiBatchTool:
    """Resolve registered DICOM artifacts and convert each series independently."""

    name = "dicom2nifti_batch"
    max_concurrency = 4

    @staticmethod
    def _read_dicom(path: Path):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            return None
        return dataset if hasattr(dataset, "SOPClassUID") else None

    def _candidate_series_dirs(self, root: Path) -> List[Path]:
        """Return directories containing at least three readable DICOM instances."""

        candidates: List[Path] = []
        directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
        for directory in directories:
            try:
                files = (path for path in directory.iterdir() if path.is_file())
            except OSError:
                continue
            readable = 0
            for path in files:
                if self._read_dicom(path) is not None:
                    readable += 1
                    if readable >= 3:
                        candidates.append(directory.resolve())
                        break
        return candidates

    def _series_metadata(self, series_dir: Path) -> Dict[str, Any]:
        for path in series_dir.iterdir():
            if not path.is_file():
                continue
            dataset = self._read_dicom(path)
            if dataset is None:
                continue
            return {
                "patient_id": str(getattr(dataset, "PatientID", "") or ""),
                "study_instance_uid": str(
                    getattr(dataset, "StudyInstanceUID", "") or ""
                ),
                "series_instance_uid": str(
                    getattr(dataset, "SeriesInstanceUID", "") or ""
                ),
                "series_number": str(getattr(dataset, "SeriesNumber", "") or ""),
                "series_description": str(
                    getattr(dataset, "SeriesDescription", "") or ""
                ),
            }
        raise ValueError("No readable DICOM instance was found in the series directory.")

    @staticmethod
    def _series_key(series_dir: Path, metadata: Dict[str, Any]) -> str:
        series_uid = str(metadata.get("series_instance_uid") or "").strip()
        return f"uid:{series_uid}" if series_uid else f"path:{series_dir.resolve()}"

    @staticmethod
    def _output_path(
        output_root: Path,
        series_dir: Path,
        metadata: Dict[str, Any],
    ) -> Path:
        patient_id = _safe_slug(metadata.get("patient_id"), "unknown-patient")
        series_number = _safe_slug(metadata.get("series_number"), "unknown-series")
        description = _safe_slug(metadata.get("series_description"), "dicom")
        identity = str(metadata.get("series_instance_uid") or series_dir.resolve())
        uid_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        series_output_dir = output_root / f"{patient_id}-{uid_hash}"
        series_output_dir.mkdir(parents=True, exist_ok=True)
        return series_output_dir / (
            f"{patient_id}_series-{series_number}_{description}_{uid_hash}.nii.gz"
        )

    def _convert_series_sync(
        self,
        series_dir: Path,
        output_root: Path,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_path = self._output_path(output_root, series_dir, metadata)
        dicom2nifti.dicom_series_to_nifti(
            str(series_dir),
            str(output_path),
            reorient_nifti=True,
        )
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError("Converter did not produce a non-empty NIfTI file.")

        image = nib.load(str(output_path))
        shape = [int(value) for value in image.shape]
        spacing = [float(value) for value in image.header.get_zooms()]
        if len(shape) < 3 or any(value <= 0 for value in shape[:3]):
            raise RuntimeError(f"Converted NIfTI has an invalid shape: {shape}")

        return {
            "status": "ok",
            "patient_id": metadata.get("patient_id") or None,
            "study_instance_uid": metadata.get("study_instance_uid") or None,
            "series_instance_uid": metadata.get("series_instance_uid") or None,
            "series_number": metadata.get("series_number") or None,
            "series_description": metadata.get("series_description") or None,
            "output_filename": output_path.name,
            "shape": shape,
            "spacing": spacing,
            "_output_path": str(output_path),
        }

    @staticmethod
    def _conversion_error(
        error: Exception,
        *,
        series_dir: Path,
        output_root: Path,
    ) -> str:
        message = str(error) or error.__class__.__name__
        message = message.replace(str(series_dir), "<registered-dicom-artifact>")
        message = message.replace(str(output_root), "<managed-output>")
        return f"{error.__class__.__name__}: {message}"

    @staticmethod
    def _resolve_artifact(
        artifact_id: str,
        artifact_registry: Dict[str, Any],
    ) -> Path:
        if not artifact_id.startswith("artifact-"):
            raise ValueError(
                "dicom2nifti_batch accepts registered artifact IDs only; "
                "local filesystem paths are not accepted."
            )
        record = artifact_registry.get(artifact_id)
        if not isinstance(record, dict):
            raise ValueError(f"Unknown artifact ID: {artifact_id}")
        if str(record.get("status") or "verified") != "verified":
            raise ValueError(f"Artifact is not verified: {artifact_id}")
        path_value = record.get("path")
        if not path_value:
            raise ValueError(f"Artifact has no registered path: {artifact_id}")
        path = Path(str(path_value)).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Registered artifact path does not exist: {artifact_id}")
        if not path.is_dir():
            raise ValueError(f"DICOM artifact is not a directory: {artifact_id}")
        return path

    async def run(
        self,
        *,
        artifact_ids: List[str],
        artifact_registry: Dict[str, Any],
    ) -> TaskResult:
        requested_ids = list(dict.fromkeys(str(value) for value in artifact_ids))
        if not requested_ids:
            message = "dicom2nifti_batch requires at least one registered artifact ID."
            return TaskResult(
                output={"status": "error", "error": message},
                status="error",
                errors=[message],
            )

        input_errors: List[Dict[str, str]] = []
        resolved_inputs: List[tuple[str, Path]] = []
        for artifact_id in requested_ids:
            try:
                resolved_inputs.append(
                    (artifact_id, self._resolve_artifact(artifact_id, artifact_registry))
                )
            except Exception as exc:
                input_errors.append(
                    {"artifact_id": artifact_id, "error": str(exc)}
                )

        series_by_key: Dict[str, Dict[str, Any]] = {}
        for artifact_id, root in resolved_inputs:
            candidate_dirs = self._candidate_series_dirs(root)
            if not candidate_dirs:
                input_errors.append(
                    {
                        "artifact_id": artifact_id,
                        "error": "Registered directory contains no readable DICOM series.",
                    }
                )
                continue
            for series_dir in candidate_dirs:
                try:
                    metadata = self._series_metadata(series_dir)
                except Exception as exc:
                    input_errors.append(
                        {"artifact_id": artifact_id, "error": str(exc)}
                    )
                    continue
                key = self._series_key(series_dir, metadata)
                existing = series_by_key.get(key)
                if existing is None:
                    series_by_key[key] = {
                        "series_dir": series_dir,
                        "metadata": metadata,
                        "source_artifact_ids": [artifact_id],
                    }
                elif artifact_id not in existing["source_artifact_ids"]:
                    existing["source_artifact_ids"].append(artifact_id)

        output_root = get_run_dir(self.name, persist=True)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def convert(entry: Dict[str, Any]) -> Dict[str, Any]:
            metadata = entry["metadata"]
            base_result = {
                "source_artifact_ids": list(entry["source_artifact_ids"]),
                "patient_id": metadata.get("patient_id") or None,
                "study_instance_uid": metadata.get("study_instance_uid") or None,
                "series_instance_uid": metadata.get("series_instance_uid") or None,
                "series_number": metadata.get("series_number") or None,
                "series_description": metadata.get("series_description") or None,
            }
            async with semaphore:
                try:
                    converted = await asyncio.to_thread(
                        self._convert_series_sync,
                        entry["series_dir"],
                        output_root,
                        metadata,
                    )
                except Exception as exc:
                    return {
                        **base_result,
                        "status": "error",
                        "error": self._conversion_error(
                            exc,
                            series_dir=entry["series_dir"],
                            output_root=output_root,
                        ),
                    }
            return {**base_result, **converted}

        results = await asyncio.gather(
            *(convert(entry) for entry in series_by_key.values())
        )
        successful = [result for result in results if result.get("status") == "ok"]
        failed_series = [result for result in results if result.get("status") != "ok"]
        produced = [str(result.pop("_output_path")) for result in successful]
        failed_count = len(failed_series) + len(input_errors)
        if produced and failed_count == 0:
            status = "ok"
        elif produced:
            status = "partial"
        else:
            status = "error"

        errors = [item["error"] for item in input_errors]
        errors.extend(
            str(item.get("error") or "Series conversion failed.")
            for item in failed_series
        )
        output: Dict[str, Any] = {
            "status": status,
            "requested_artifacts": len(requested_ids),
            "resolved_artifacts": len(resolved_inputs),
            "discovered_series": len(series_by_key),
            "succeeded": len(successful),
            "failed": failed_count,
            "results": results,
            "input_errors": input_errors,
        }
        if produced:
            output.update({"action": "download", "files": produced})
        elif errors:
            output["error"] = errors[0]

        return TaskResult(
            output=output,
            artifacts={"nifti_paths": produced} if produced else {},
            status=status,
            errors=errors,
        )


_BATCH_TOOL = DicomToNiftiBatchTool()


class DicomToNiftiBatchArgs(BaseModel):
    """Model-visible arguments for the artifact-backed batch converter."""

    model_config = ConfigDict(extra="forbid")

    artifact_ids: List[
        Annotated[str, Field(pattern=r"^artifact-[0-9a-f]{20}$")]
    ] = Field(
        ...,
        min_length=1,
        description=(
            "Registered DICOM directory artifact IDs to convert. Every value must be an "
            "exact `artifact-...` ID from voxelinsight_state_json. Never pass a local "
            "filesystem path, DICOM UID, filename, UI label, or reconstructed directory."
        ),
    )
    artifact_registry: Annotated[
        Dict[str, Any],
        InjectedState("artifact_registry"),
    ]


@toolify_agent(
    name="dicom2nifti_batch",
    description=(
        "Convert one or more registered DICOM directory artifacts to NIfTI as one "
        "deterministic batch. The only model-provided input is `artifact_ids`, containing "
        "exact `artifact-...` IDs from voxelinsight_state_json. This tool does not accept "
        "filesystem paths, DICOM UIDs, filenames, or UI labels. It resolves artifacts from "
        "injected registry state, deduplicates overlapping parent/series directories, writes "
        "each DICOM series to a collision-safe output, validates every NIfTI, and returns "
        "per-series success or failure records. Call it exactly once for the full requested batch."
    ),
    args_schema=DicomToNiftiBatchArgs,
    timeout_s=900,
)
async def dicom2nifti_batch_runner(
    artifact_ids: List[str],
    artifact_registry: Dict[str, Any],
):
    return await _BATCH_TOOL.run(
        artifact_ids=artifact_ids,
        artifact_registry=artifact_registry,
    )


# Import compatibility for older application entry points. This is an alias to the
# artifact-ID-only batch tool, not the former path-based interface.
dicom2nifti_runner = dicom2nifti_batch_runner
