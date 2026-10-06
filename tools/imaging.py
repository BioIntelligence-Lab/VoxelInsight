import os
import pathlib
import subprocess
import asyncio
import csv
import difflib
import hashlib
import json
import re
from typing import List, Dict, Optional
from tools.shared import toolify_agent
from tools.file_validation import validate_existing_input_files
from progress_ui import update_progress

from core.state import Task, TaskResult, ConversationState
from core.storage import get_run_dir

_PCT_RE = re.compile(r"(?<!\d)(100|\d{1,2})(?:\.\d+)?%")

_TS_PHASES = (
    ("preparing", "Preparing input", 0, 5, ("prepar", "checking input", "initializ")),
    ("loading", "Loading model", 5, 12, ("download", "loading model", "weights", "checkpoint")),
    ("preprocessing", "Preprocessing and resampling", 12, 28, ("resampl", "preprocess", "crop")),
    ("inference", "Running inference", 28, 84, ("predict", "inference", "forward pass")),
    ("postprocessing", "Postprocessing masks", 84, 94, ("postprocess", "restoring", "uncrop")),
    ("saving", "Saving segmentations", 94, 99, ("saving", "writing", "export")),
)


class _TSProgressTracker:
    """Convert TotalSegmentator's phase-local output into monotonic case progress."""

    def __init__(self) -> None:
        self.phase_index = 0
        self.pct = 0

    def feed(self, text: str) -> tuple[int, str]:
        lowered = text.lower()
        detected = self.phase_index
        for index, (_key, _label, _start, _end, keywords) in enumerate(_TS_PHASES):
            if any(keyword in lowered for keyword in keywords):
                detected = max(detected, index)
        self.phase_index = detected

        _key, label, start, end, _keywords = _TS_PHASES[self.phase_index]
        candidate = start
        matches = _PCT_RE.findall(text)
        if matches:
            raw_pct = max(0, min(100, int(float(matches[-1]))))
            candidate = int(start + (raw_pct / 100.0) * (end - start))
        self.pct = max(self.pct, candidate)
        return self.pct, label


async def _run_ts_stream(cmd, on_progress, progress_start: int = 0, progress_end: int = 100):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    buf = []
    tail = ""
    last_emitted_local_pct: Optional[int] = None
    last_emitted_label: Optional[str] = None
    tracker = _TSProgressTracker()

    async def _emit(local_pct: int, label: str):
        nonlocal last_emitted_local_pct, last_emitted_label
        local_pct = max(0, min(100, int(local_pct)))
        mapped_pct = int(progress_start + (local_pct / 100.0) * (progress_end - progress_start))
        mapped_pct = max(0, min(100, mapped_pct))
        if (
            last_emitted_local_pct is None
            or local_pct > last_emitted_local_pct
            or label != last_emitted_label
        ):
            last_emitted_local_pct = local_pct
            last_emitted_label = label
            await on_progress(mapped_pct, label)

    try:
        while True:
            chunk = await proc.stdout.read(256)
            if not chunk:
                break
            s = chunk.decode(errors="ignore")
            buf.append(s)
            scan_text = tail + s
            tail = scan_text[-256:]
            local_pct, label = tracker.feed(scan_text)
            await _emit(max(1, local_pct), label)
        rc = await proc.wait()
    except BaseException:
        # A cohort-level timeout or cancellation must not leave an expensive
        # TotalSegmentator subprocess running after the tool call has ended.
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        raise
    out = "\n".join(buf)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, " ".join(cmd), output=out, stderr=out)
    await _emit(99, "Finalizing outputs")
    return out

class ImagingAgent:
    name = "imaging"
    model = None 

    def __init__(self, ct_mappings: str):
        self.ct_mappings = ct_mappings
        self.canonical_roi_names: Dict[str, str] = {}
        self.allowed_rois_by_task = self._load_allowed_rois()

    def _load_allowed_rois(self) -> Dict[str, set[str]]:
        mapping_files = (
            pathlib.Path("Data/TotalSegmentatorMappingsCT.tsv"),
            pathlib.Path("Data/TotalSegmentatorMappingsMRI.tsv"),
        )
        allowed: Dict[str, set[str]] = {}
        for mapping_file in mapping_files:
            if not mapping_file.exists():
                continue
            with mapping_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    task = str(row.get("task_name", "")).strip().lower()
                    original_roi = str(row.get("roi_subset", "")).strip()
                    roi = original_roi.lower()
                    if not task or not roi:
                        continue
                    allowed.setdefault(task, set()).add(roi)
                    # TotalSegmentator label names are case-sensitive (e.g. vertebrae_C1).
                    self.canonical_roi_names[roi] = original_roi
        return allowed

    @staticmethod
    def _normalize_roi_name(value: str) -> str:
        return value.strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _empty_mask_names(paths: List[str]) -> List[str]:
        """Names of masks with no foreground voxels (structure outside the field of view)."""
        import nibabel as nib
        import numpy as np

        empty: List[str] = []
        for path in paths:
            try:
                if not np.asanyarray(nib.load(path).dataobj).any():
                    name = pathlib.Path(path).name
                    empty.append(name[:-7] if name.endswith(".nii.gz") else pathlib.Path(name).stem)
            except Exception:
                continue
        return empty

    @staticmethod
    def _mask_name(path: str) -> str:
        name = pathlib.Path(path).name.lower()
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return pathlib.Path(name).stem

    def _expand_roi_aliases(self, task_name: str, rois: List[str]) -> List[str]:
        """Resolve a small set of unambiguous anatomical aliases."""
        task = task_name.strip().lower()
        allowed = self.allowed_rois_by_task.get(task, set())
        aliases = {
            "kidney": ("kidney_left", "kidney_right"),
            "kidneys": ("kidney_left", "kidney_right"),
        }
        # Group names that map to label families in the task's own mapping table.
        group_prefixes = {
            "lung": "lung_", "lungs": "lung_",
            "rib": "rib_", "ribs": "rib_",
            "vertebra": "vertebrae_", "vertebrae": "vertebrae_", "spine": "vertebrae_",
            "cervical_vertebrae": "vertebrae_c", "thoracic_vertebrae": "vertebrae_t",
            "lumbar_vertebrae": "vertebrae_l",
        }
        expanded: List[str] = []
        for roi in rois:
            targets = aliases.get(roi)
            if roi in allowed:
                expanded.append(roi)
            elif targets and all(target in allowed for target in targets):
                expanded.extend(targets)
            else:
                # "lungs" -> lung_left/lung_right, "adrenal_glands" -> adrenal_gland_left/right
                stems = [roi, roi[:-1]] if roi.endswith("s") else [roi]
                bilateral = next(
                    ([f"{stem}_left", f"{stem}_right"] for stem in stems
                     if f"{stem}_left" in allowed and f"{stem}_right" in allowed),
                    None,
                )
                prefix = group_prefixes.get(roi)
                members = sorted(name for name in allowed if prefix and name.startswith(prefix))
                if bilateral:
                    expanded.extend(bilateral)
                elif members:
                    expanded.extend(members)
                else:
                    expanded.append(roi)
        return expanded

    def _validate_mapping_inputs(
        self,
        task_name: str,
        requested_rois: List[str],
        fast: bool,
        all_structures: bool,
    ) -> Optional[str]:
        task = task_name.strip().lower()
        normalized_rois = [self._normalize_roi_name(r) for r in requested_rois]

        if all_structures and task not in {"total", "total_mr"}:
            return "all_structures=true is supported only for task 'total' or 'total_mr'."
        if task in {"total", "total_mr"}:
            if normalized_rois and all_structures:
                return "Specify ROI subsets or all_structures=true, not both."
            if not normalized_rois and not all_structures:
                return (
                    f"Task '{task}' requires roi_subset/roi_subsets unless the user "
                    "explicitly requested every structure and all_structures=true."
                )

        if task != "liver_vessels" and task in self.allowed_rois_by_task and not normalized_rois:
            return None

        if task != "liver_vessels" and task not in self.allowed_rois_by_task and normalized_rois:
            return (
                f"Task '{task_name}' does not accept ROI subsets in the configured mapping. "
                "Omit roi_subset or choose a mapped task."
            )

        if task == "liver_vessels" and fast:
            return "Task 'liver_vessels' cannot be run with fast=true."

        if normalized_rois and task in self.allowed_rois_by_task:
            allowed = self.allowed_rois_by_task[task]
            invalid = [r for r in normalized_rois if r not in allowed]
            if invalid:
                suggestions: List[str] = []
                for roi in invalid:
                    suggestions.extend(difflib.get_close_matches(roi, sorted(allowed), n=2))
                suggestion_text = (
                    f" Close matches: {list(dict.fromkeys(suggestions))}."
                    if suggestions
                    else ""
                )
                return f"Invalid ROI for task '{task_name}': {invalid}.{suggestion_text}"
        return None

    @staticmethod
    def _case_dir_name(index: int, path: pathlib.Path) -> str:
        """Return a stable, collision-safe directory name for one input."""
        name = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "volume"
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
        return f"{index:05d}_{slug[:64]}_{digest}"

    @staticmethod
    def _write_batch_status(path: pathlib.Path, payload: Dict[str, object]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        os.replace(temp_path, path)

    @staticmethod
    def _validate_nifti_inputs(paths: List[pathlib.Path]) -> None:
        unsupported = [
            path.name
            for path in paths
            if not (
                path.name.lower().endswith(".nii")
                or path.name.lower().endswith(".nii.gz")
            )
        ]
        if unsupported:
            raise ValueError(
                "imaging accepts NIfTI inputs only (.nii or .nii.gz); unsupported: "
                + ", ".join(unsupported)
            )

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        validate_existing_input_files(task.files, tool_name="imaging")
        self._validate_nifti_inputs([pathlib.Path(path) for path in task.files])

        if len(task.files) == 1:
            await update_progress(0, "Starting")

            raw_path = pathlib.Path(task.files[0])

            state.memory["image_path"] = str(raw_path)

            fixed_path = raw_path

            out_dir = str(get_run_dir(self.name, persist=True))
            task_name = str(task.kwargs.get("task_name", "total"))

            roi_subset = task.kwargs.get("roi_subset")
            roi_subsets = task.kwargs.get("roi_subsets")
            requested_rois: List[str] = []

            if isinstance(roi_subsets, (list, tuple)):
                requested_rois.extend([str(x) for x in roi_subsets if x])
            if roi_subset:
                if isinstance(roi_subset, (list, tuple)):
                    requested_rois.extend([str(x) for x in roi_subset if x])
                else:
                    requested_rois.append(str(roi_subset))

            requested_rois = [self._normalize_roi_name(roi) for roi in requested_rois]
            requested_rois = self._expand_roi_aliases(task_name, requested_rois)
            seen = set()
            requested_rois = [r for r in requested_rois if not (r in seen or seen.add(r))]
            fast = bool(task.kwargs.get("fast", True))
            all_structures = bool(task.kwargs.get("all_structures", False))

            validation_error = self._validate_mapping_inputs(
                task_name=task_name,
                requested_rois=requested_rois,
                fast=fast,
                all_structures=all_structures,
            )
            if validation_error:
                await update_progress(100, "Invalid segmentation request", status="error")
                return TaskResult(output=validation_error, status="error", errors=[validation_error])

            cmd = [
                "TotalSegmentator",
                "-i", str(fixed_path),
                "-o", out_dir,
                "--task", task_name,
            ]
            if requested_rois:
                cmd += ["--roi_subset"] + [self.canonical_roi_names.get(r, r) for r in requested_rois]

            if fast:
                cmd += ["--fast"]

            await update_progress(20, "Running TotalSegmentator")
            try:
                await _run_ts_stream(cmd, update_progress, progress_start=20, progress_end=98)
            except subprocess.CalledProcessError as e:
                await update_progress(100, "TotalSegmentator failed", status="error")
                msg = (
                    "TotalSegmentator failed.\n\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"STDOUT/STDERR:\n{e.stderr or e.output}"
                )
                return TaskResult(output=msg, status="error", errors=[msg])

            seg_paths = [
                os.path.join(out_dir, f)
                for f in os.listdir(out_dir)
                if f.endswith(".nii") or f.endswith(".nii.gz")
            ]
            seg_paths.sort()

            if not seg_paths:
                message = "TotalSegmentator completed but produced no NIfTI masks."
                await update_progress(100, message, status="error")
                return TaskResult(output=message, status="error", errors=[message])

            seg_map: Dict[str, str] = {}
            if requested_rois:
                lower_files = {self._mask_name(path): path for path in seg_paths}
                for roi in requested_rois:
                    match = lower_files.get(roi)
                    if match:
                        seg_map[roi] = match
                missing_rois = [roi for roi in requested_rois if roi not in seg_map]
                if missing_rois:
                    message = f"Requested masks were not produced: {missing_rois}"
                    await update_progress(100, message, status="error")
                    return TaskResult(output=message, status="error", errors=[message])

            if not requested_rois and seg_paths:
                for p in seg_paths:
                    seg_map[os.path.splitext(os.path.basename(p))[0]] = p

            state.memory["segmentations"] = seg_paths
            state.memory["segmentations_map"] = seg_map
            empty_masks = self._empty_mask_names(seg_paths)

            summary = {
                "agent": "imaging",
                "action": "inference",
                "task": task_name,
                "requested_rois": requested_rois,
                "output_dir": out_dir,
                "num_masks": len(seg_paths),
                "num_nonempty": len(seg_paths) - len(empty_masks),
                "empty_masks": empty_masks,
                "matched": seg_map if len(seg_map) <= 20 else {"count": len(seg_map), "first": sorted(seg_map)[:10]},
            }
            await update_progress(100, "TotalSegmentator complete", status="completed")
            return TaskResult(
                output=summary,
                artifacts={"segmentations": seg_paths, "segmentations_map": seg_map, "output_dir": out_dir},
                status="ok",
            )
        else:
            input_paths = [pathlib.Path(p) for p in task.files]
            state.memory["image_paths"] = [str(p) for p in input_paths]

            total = len(input_paths)
            max_concurrency = max(1, min(8, int(task.kwargs.get("max_concurrency", 1))))
            await update_progress(
                2,
                f"Preparing {total} volumes",
                title="Batch segmentation",
                status="running",
                phase="preparing",
                total=total,
                completed=0,
                failed=0,
                detail=f"Up to {max_concurrency} case(s) at a time",
            )

            fixed_paths = list(input_paths)

            task_name = str(task.kwargs.get("task_name", "total"))
            roi_subset = task.kwargs.get("roi_subset")
            roi_subsets = task.kwargs.get("roi_subsets")
            requested_rois: List[str] = []

            if isinstance(roi_subsets, (list, tuple)):
                requested_rois.extend([str(x) for x in roi_subsets if x])
            if roi_subset:
                if isinstance(roi_subset, (list, tuple)):
                    requested_rois.extend([str(x) for x in roi_subset if x])
                else:
                    requested_rois.append(str(roi_subset))

            requested_rois = [self._normalize_roi_name(roi) for roi in requested_rois]
            requested_rois = self._expand_roi_aliases(task_name, requested_rois)
            seen = set()
            requested_rois = [r for r in requested_rois if not (r in seen or seen.add(r))]
            fast = bool(task.kwargs.get("fast", True))
            all_structures = bool(task.kwargs.get("all_structures", False))

            validation_error = self._validate_mapping_inputs(
                task_name=task_name,
                requested_rois=requested_rois,
                fast=fast,
                all_structures=all_structures,
            )
            if validation_error:
                await update_progress(
                    100,
                    "Invalid segmentation request",
                    title="Batch segmentation",
                    status="error",
                    total=total,
                    completed=0,
                    failed=total,
                    detail=validation_error,
                )
                return TaskResult(output=validation_error, status="error", errors=[validation_error])

            out_root = str(get_run_dir(self.name, persist=True))
            status_path = pathlib.Path(out_root) / "batch_status.json"
            per_input: List[Optional[Dict[str, object]]] = [None] * total
            seg_map_batch: Dict[str, Dict[str, str]] = {}
            seg_paths_batch: Dict[str, List[str]] = {}
            case_progress = [0] * total
            progress_lock = asyncio.Lock()

            def _snapshot() -> Dict[str, object]:
                completed_rows = [row for row in per_input if row is not None]
                succeeded = sum(row.get("status") == "ok" for row in completed_rows)
                failed = sum(row.get("status") == "error" for row in completed_rows)
                return {
                    "status": "running" if len(completed_rows) < total else (
                        "ok" if failed == 0 else "partial" if succeeded else "error"
                    ),
                    "task": task_name,
                    "requested_rois": requested_rois,
                    "total": total,
                    "completed": succeeded,
                    "failed": failed,
                    "max_concurrency": max_concurrency,
                    "output_root": out_root,
                    "cases": completed_rows,
                }

            self._write_batch_status(status_path, _snapshot())

            async def _emit_case_progress(
                index: int,
                local_pct: int,
                label: str,
            ) -> None:
                async with progress_lock:
                    case_progress[index] = max(case_progress[index], max(0, min(100, int(local_pct))))
                    snapshot = _snapshot()
                    overall = 2 + int(96 * sum(case_progress) / max(1, 100 * total))
                    await update_progress(
                        overall,
                        label,
                        title="Batch segmentation",
                        status="running",
                        phase="segmenting",
                        total=total,
                        completed=snapshot["completed"],
                        failed=snapshot["failed"],
                        current=index + 1,
                        current_item=input_paths[index].name,
                        current_pct=case_progress[index],
                    )

            async def _run_case(index: int, fp: pathlib.Path, orig: pathlib.Path) -> Dict[str, object]:
                await _emit_case_progress(
                    index,
                    1,
                    f"Starting case {index + 1}/{total}",
                )
                case_dir = os.path.join(out_root, self._case_dir_name(index + 1, orig))
                os.makedirs(case_dir, exist_ok=True)
                cmd = [
                    "TotalSegmentator",
                    "-i", str(fp),
                    "-o", case_dir,
                    "--task", task_name,
                ]
                if requested_rois:
                    cmd += ["--roi_subset"] + [self.canonical_roi_names.get(r, r) for r in requested_rois]
                if fast:
                    cmd += ["--fast"]

                async def _case_progress(local_pct: int, label: str, **extras):
                    del extras
                    await _emit_case_progress(
                        index,
                        local_pct,
                        f"Case {index + 1}/{total}: {label}",
                    )

                try:
                    await _run_ts_stream(cmd, _case_progress)
                except subprocess.CalledProcessError as e:
                    return {
                        "case_id": pathlib.Path(case_dir).name,
                        "input": str(orig),
                        "input_name": orig.name,
                        "status": "error",
                        "output_dir": case_dir,
                        "num_masks": 0,
                        "error": f"TotalSegmentator exited with code {e.returncode}: {e.stderr or e.output}",
                    }
                except Exception as e:
                    return {
                        "case_id": pathlib.Path(case_dir).name,
                        "input": str(orig),
                        "input_name": orig.name,
                        "status": "error",
                        "output_dir": case_dir,
                        "num_masks": 0,
                        "error": f"{type(e).__name__}: {e}",
                    }

                seg_paths = [
                    os.path.join(case_dir, f)
                    for f in os.listdir(case_dir)
                    if f.endswith(".nii") or f.endswith(".nii.gz")
                ]
                seg_paths.sort()
                if not seg_paths:
                    return {
                        "case_id": pathlib.Path(case_dir).name,
                        "input": str(orig),
                        "input_name": orig.name,
                        "status": "error",
                        "output_dir": case_dir,
                        "num_masks": 0,
                        "error": "TotalSegmentator completed but produced no NIfTI masks.",
                    }

                seg_map: Dict[str, str] = {}
                if requested_rois:
                    lower_files = {self._mask_name(path): path for path in seg_paths}
                    for roi in requested_rois:
                        match = lower_files.get(roi)
                        if match:
                            seg_map[roi] = match
                    missing_rois = [roi for roi in requested_rois if roi not in seg_map]
                    if missing_rois:
                        return {
                            "case_id": pathlib.Path(case_dir).name,
                            "input": str(orig),
                            "input_name": orig.name,
                            "status": "error",
                            "output_dir": case_dir,
                            "num_masks": len(seg_paths),
                            "matched": seg_map,
                            "segmentations": list(seg_map.values()),
                            "error": f"Requested masks were not produced: {missing_rois}",
                        }
                empty_masks = self._empty_mask_names(seg_paths)
                if not requested_rois and seg_paths:
                    for pth in seg_paths:
                        seg_map[os.path.splitext(os.path.basename(pth))[0]] = pth
                return {
                    "case_id": pathlib.Path(case_dir).name,
                    "input": str(orig),
                    "input_name": orig.name,
                    "status": "ok",
                    "output_dir": case_dir,
                    "num_masks": len(seg_paths),
                    "num_nonempty": len(seg_paths) - len(empty_masks),
                    "empty_masks": empty_masks,
                    "matched": seg_map,
                    "segmentations": seg_paths,
                }

            queue: asyncio.Queue = asyncio.Queue()
            for index, pair in enumerate(zip(fixed_paths, input_paths)):
                queue.put_nowait((index, *pair))

            async def _worker() -> None:
                while True:
                    try:
                        index, fp, orig = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        row = await _run_case(index, fp, orig)
                        async with progress_lock:
                            per_input[index] = row
                            case_progress[index] = 100
                            if row.get("status") == "ok":
                                seg_paths_batch[str(orig)] = list(row.get("segmentations") or [])
                                seg_map_batch[str(orig)] = dict(row.get("matched") or {})
                            snapshot = _snapshot()
                            self._write_batch_status(status_path, snapshot)
                            done = int(snapshot["completed"]) + int(snapshot["failed"])
                            await update_progress(
                                2 + int(96 * sum(case_progress) / max(1, 100 * total)),
                                f"Processed {done}/{total} volumes",
                                title="Batch segmentation",
                                status="running",
                                phase="segmenting",
                                total=total,
                                completed=snapshot["completed"],
                                failed=snapshot["failed"],
                                current=index + 1,
                                current_item=orig.name,
                            )
                    finally:
                        queue.task_done()

            workers = [asyncio.create_task(_worker()) for _ in range(min(max_concurrency, total))]
            await asyncio.gather(*workers)

            state.memory["segmentations_batch"] = seg_paths_batch
            state.memory["segmentations_map_batch"] = seg_map_batch

            final_snapshot = _snapshot()
            self._write_batch_status(status_path, final_snapshot)
            status = str(final_snapshot["status"])
            completed = int(final_snapshot["completed"])
            failed = int(final_snapshot["failed"])
            final_ui_status = "completed" if status == "ok" else status
            await update_progress(
                100,
                f"Batch complete: {completed}/{total} succeeded, {failed} failed",
                title="Batch segmentation",
                status=final_ui_status,
                phase="complete",
                total=total,
                completed=completed,
                failed=failed,
            )

            summary: Dict[str, object] = {
                "status": status,
                "agent": "imaging",
                "action": "inference",
                "task": task_name,
                "requested_rois": requested_rois,
                "output_dir": out_root,
                "status_path": str(status_path),
                "total": total,
                "completed": completed,
                "failed": failed,
                "max_concurrency": max_concurrency,
                "masks_per_case": {
                    str(row.get("input_name")): int(row.get("num_masks") or 0)
                    for row in per_input
                    if row is not None
                },
                "per_input": [
                    {
                        key: row[key]
                        for key in ("case_id", "input_name", "status", "num_masks", "num_nonempty", "empty_masks", "error")
                        if key in row
                    }
                    | {"masks_matched": len(row.get("matched") or {})}
                    for row in per_input
                    if row is not None
                ],
            }
            errors = [
                f"{row.get('input_name')}: {row.get('error')}"
                for row in per_input
                if row is not None and row.get("status") == "error"
            ]
            return TaskResult(
                output=summary,
                artifacts={
                    "segmentations_batch": seg_paths_batch,
                    "segmentations_map_batch": seg_map_batch,
                    "output_root": out_root,
                    "files": [str(status_path)],
                },
                status=status,
                errors=errors,
            )

from pydantic import BaseModel, Field
from typing import Optional, List, Union
from tools.shared import toolify_agent, _cs
from core.state import Task

_IM: Optional[ImagingAgent] = None


def configure_imaging_tool(*, ct_mappings: str = ""):
    global _IM
    _IM = ImagingAgent(ct_mappings=ct_mappings)


class ImagingArgs(BaseModel):
    file_path: Optional[str] = Field(
        default=None,
        description="Exact path to input CT/MRI volume (.nii/.nii.gz). Required for one file.",
    )
    file_paths: Optional[List[str]] = Field(
        default=None,
        description="List of input CT volumes (.nii/.nii.gz). Use this for multiple files",
    )
    task_name: str = Field(
        default="total",
        description="TotalSegmentator task to run (e.g., 'total', 'lung_vessels', 'cardiac').",
    )
    roi_subset: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="ROI(s) to extract (string or list). Case-insensitive substring matching.",
    )
    roi_subsets: Optional[List[str]] = Field(
        default=None,
        description="Additional ROIs to include. Merged with roi_subset.",
    )
    all_structures: bool = Field(
        default=False,
        description=(
            "Set true only when the user explicitly requests every structure from "
            "task_name='total' or 'total_mr'."
        ),
    )
    fast: bool = Field(
        default=True,
        description="Use '--fast' mode for quicker but less accurate results.",
    )
    max_concurrency: int = Field(
        default=1,
        ge=1,
        le=8,
        description=(
            "Maximum simultaneous TotalSegmentator processes for a multi-file batch. "
            "Keep at 1 on a single GPU; increase only when hardware capacity is known."
        ),
    )


@toolify_agent(
    name="imaging",
    description=(
        "Performs segmentation on CT scans and MRI scans using TotalSegmentator." 
        "Accepts single or multiple NIfTI files as input."
        "Takes a NIfTI file, runs the chosen task (default: 'total'), and outputs NIfTI masks. "
        "Supports filtering results to requested ROIs and fast/accurate modes. "
        "Saves segmentation files to a temp directory and returns paths."
        "Outputs include: summary dict, segmentation files, and ROI→file mapping."
        "Task-specific rules:"
        "- For task=total or task=total_mr, specify ROI subsets or explicitly set all_structures=true; never omit both."
        "- Set all_structures=true only when the user requests every structure. For all other tasks, never use all_structures."
        "- Incorrect use of `roi_subset` will cause errors."
        "- Special rule: For liver_tumor segmentation, use `task=liver_vessels` with no `roi_subset`. Also you cannot use --fast for task=liver_vessels."  
        "- TotalSegmentator only accepts certain task names and roi subsets. You are provided with these."
    ),
    args_schema=ImagingArgs,
    # Cohort segmentation can legitimately run for hours. The subprocess
    # cleanup above still guarantees that cancellation terminates active jobs.
    timeout_s=86400,
)
async def imaging_runner(
    file_path: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    task_name: str = "total",
    roi_subset: Optional[Union[str, List[str]]] = None,
    roi_subsets: Optional[List[str]] = None,
    all_structures: bool = False,
    fast: bool = True,
    max_concurrency: int = 1,
):
    if _IM is None:
        raise RuntimeError("Imaging tool not configured. Call configure_imaging_tool(...) first.")

    files: List[str] = []
    if file_paths:
        files.extend(file_paths)
    if file_path:
        files.append(file_path)
    # One tool call represents one cohort job. Preserve order while preventing
    # duplicate work when the same artifact appears in both arguments.
    files = list(dict.fromkeys(files))
    validate_existing_input_files(files, tool_name="imaging")
    kwargs = {
        "task_name": task_name,
        "roi_subset": roi_subset,
        "roi_subsets": roi_subsets,
        "all_structures": all_structures,
        "fast": fast,
        "max_concurrency": max_concurrency,
    }

    task = Task(user_msg=f"Run TotalSegmentator task='{task_name}'", files=files, kwargs=kwargs)
    return await _IM.run(task, _cs())
