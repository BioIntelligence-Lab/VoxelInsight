import os
import shutil
import pathlib
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional

from core.state import Task, TaskResult, ConversationState


class ImagingAgent:
    name = "imaging"
    model = None 

    def __init__(self, ct_mappings: str):
        self.ct_mappings = ct_mappings

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        if not task.files:
            return TaskResult(output="No file provided for imaging.")

        raw_path = pathlib.Path(task.files[0])

        state.memory["image_path"] = str(raw_path)

        if raw_path.suffix not in (".nii", ".gz"):
            tmp_dir = pathlib.Path(tempfile.mkdtemp())
            fixed_path = tmp_dir / (raw_path.name + ".nii.gz")
            shutil.copy(raw_path, fixed_path)
        else:
            fixed_path = raw_path

        out_dir = tempfile.mkdtemp(prefix="ts_")
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

        seen = set()
        requested_rois = [r for r in requested_rois if not (r in seen or seen.add(r))]

        cmd = [
            "TotalSegmentator",
            "-i", str(fixed_path),
            "-o", out_dir,
            "--task", task_name,
        ]
        for r in requested_rois:
            cmd += ["--roi_subset"] + requested_rois

        if task.kwargs.get("fast", True):
            cmd += ["--fast"]

        try:
            proc = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            msg = (
                "TotalSegmentator failed.\n\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            )
            return TaskResult(output=msg)

        seg_paths = [
            os.path.join(out_dir, f)
            for f in os.listdir(out_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]
        seg_paths.sort()

        # Build a mapping roi_subset -> file 
        seg_map: Dict[str, str] = {}
        if requested_rois:
            lower_files = {f.lower(): f for f in seg_paths}
            for roi in requested_rois:
                roi_l = roi.lower()
                match = next((lower_files[k] for k in lower_files if roi_l in k), None)
                if match:
                    seg_map[roi] = match

        if not requested_rois and seg_paths:
            for p in seg_paths:
                seg_map[os.path.splitext(os.path.basename(p))[0]] = p

        state.memory["segmentations"] = seg_paths
        state.memory["segmentations_map"] = seg_map

        summary = {
            "agent": "imaging",
            "action": "inference",
            "task": task_name,
            "requested_rois": requested_rois,
            "output_dir": out_dir,
            "num_masks": len(seg_paths),
            "matched": seg_map,
        }
        return TaskResult(
            output=summary,
            artifacts={"segmentations": seg_paths, "segmentations_map": seg_map, "output_dir": out_dir},
        )
