import os
import shutil
import pathlib
import tempfile
import subprocess
import asyncio
from typing import List, Dict, Tuple, Optional
from tools.shared import toolify_agent, normalize_task_result
from progress_ui import update_progress

from core.state import Task, TaskResult, ConversationState

async def _run_ts_stream(cmd, on_progress):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    buf = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        s = line.decode(errors="ignore").strip()
        buf.append(s)
        ls = s.lower()
        #if "inference" in ls or "predict" in ls:
            #await on_progress(50, "Running inference")
    rc = await proc.wait()
    out = "\n".join(buf)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, " ".join(cmd), output=out, stderr=out)
    return out

class ImagingAgent:
    name = "imaging"
    model = None 

    def __init__(self, ct_mappings: str):
        self.ct_mappings = ct_mappings

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        if not task.files:
            return TaskResult(output="No file provided for imaging.")

        if len(task.files) == 1:
            #await update_progress(0, "Starting")

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
            if requested_rois:
                cmd += ["--roi_subset"] + requested_rois

            if task.kwargs.get("fast", True):
                cmd += ["--fast"]

            #await update_progress(20, "Running TotalSegmentator")
            try:
                await _run_ts_stream(cmd, update_progress)
            except subprocess.CalledProcessError as e:
                #await update_progress(100, "Failed")
                msg = (
                    "TotalSegmentator failed.\n\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"STDOUT/STDERR:\n{e.stderr or e.output}"
                )
                return TaskResult(output=msg)

            #await update_progress(100, "TotalSegmentator Done")

            seg_paths = [
                os.path.join(out_dir, f)
                for f in os.listdir(out_dir)
                if f.endswith(".nii") or f.endswith(".nii.gz")
            ]
            seg_paths.sort()

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
        else:
            input_paths = [pathlib.Path(p) for p in task.files]
            state.memory["image_paths"] = [str(p) for p in input_paths]

            #await update_progress(2, "Preparing batch")

            fixed_paths: List[pathlib.Path] = []
            for p in input_paths:
                if p.suffix not in (".nii", ".gz"):
                    tmp_dir = pathlib.Path(tempfile.mkdtemp())
                    fixed_path = tmp_dir / (p.name + ".nii.gz")
                    shutil.copy(p, fixed_path)
                    fixed_paths.append(fixed_path)
                else:
                    fixed_paths.append(p)

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

            out_root = tempfile.mkdtemp(prefix="ts_batch_")
            per_input = []
            seg_map_batch: Dict[str, Dict[str, str]] = {}
            seg_paths_batch: Dict[str, List[str]] = {}

            total = max(1, len(fixed_paths))
            BAR_START = 2
            BAR_END = 98
            SLICE = (BAR_END - BAR_START) / total

            for idx, (fp, orig) in enumerate(zip(fixed_paths, input_paths), start=1):
                slice_start = BAR_START + SLICE * (idx - 1)
                slice_end = BAR_START + SLICE * idx

                #await update_progress(int(slice_start), f"Starting {idx}/{total}")

                case_dir = os.path.join(out_root, pathlib.Path(orig).stem)
                os.makedirs(case_dir, exist_ok=True)
                cmd = [
                    "TotalSegmentator",
                    "-i", str(fp),
                    "-o", case_dir,
                    "--task", task_name,
                ]
                if requested_rois:
                    cmd += ["--roi_subset"] + requested_rois
                if task.kwargs.get("fast", True):
                    cmd += ["--fast"]

                async def _case_progress(local_pct: int, label: str, **extras):
                    local = max(0, min(100, int(local_pct)))
                    global_pct = int(slice_start + (local / 100.0) * (slice_end - slice_start))
                    #await update_progress(global_pct, label)

                try:
                    await _run_ts_stream(cmd, _case_progress)
                except subprocess.CalledProcessError as e:
                    #await update_progress(100, "Failed")
                    return TaskResult(output=("TotalSegmentator failed.\n\n"
                                            f"Command: {' '.join(cmd)}\n"
                                            f"STDOUT/STDERR:\n{e.stderr or e.output}"))

                seg_paths = [
                    os.path.join(case_dir, f)
                    for f in os.listdir(case_dir)
                    if f.endswith(".nii") or f.endswith(".nii.gz")
                ]
                seg_paths.sort()
                seg_paths_batch[str(orig)] = seg_paths

                seg_map: Dict[str, str] = {}
                if requested_rois:
                    lower_files = {f.lower(): f for f in seg_paths}
                    for roi in requested_rois:
                        roi_l = roi.lower()
                        match = next((lower_files[k] for k in lower_files if roi_l in k), None)
                        if match:
                            seg_map[roi] = match
                if not requested_rois and seg_paths:
                    for pth in seg_paths:
                        seg_map[os.path.splitext(os.path.basename(pth))[0]] = pth
                seg_map_batch[str(orig)] = seg_map

                per_input.append({
                    "input": str(orig),
                    "output_dir": case_dir,
                    "num_masks": len(seg_paths),
                    "matched": seg_map,
                })

                #await update_progress(int(slice_end), f"Finished {idx}/{total}")

            state.memory["segmentations_batch"] = seg_paths_batch
            state.memory["segmentations_map_batch"] = seg_map_batch

            #await update_progress(100, "Done")

            summary = {
                "agent": "imaging",
                "action": "inference",
                "task": task_name,
                "requested_rois": requested_rois,
                "output_dir": out_root,
                "per_input": per_input,
            }
            return TaskResult(
                output=summary,
                artifacts={"segmentations_batch": seg_paths_batch, "segmentations_map_batch": seg_map_batch, "output_root": out_root},
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
        description="Path to input CT volume (.nii/.nii.gz). If omitted, uses attached file.",
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
    fast: bool = Field(
        default=True,
        description="Use '--fast' mode for quicker but less accurate results.",
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
        "- If using `task=total` or `task=total_mr`: you may specify `roi_subset` values for specific organs/tissues. For all other tasks: never specify `roi_subset`."
        "- Incorrect use of `roi_subset` will cause errors."
        "- Special rule: For liver_tumor segmentation, use `task=liver_vessels` with no `roi_subset`. Also you cannot use --fast for task=liver_vessels."  
        "- TotalSegmentator only accepts certain task names and roi subsets. You are provided with these."
    ),
    args_schema=ImagingArgs,
    timeout_s=1200,
)
async def imaging_runner(
    file_path: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    task_name: str = "total",
    roi_subset: Optional[Union[str, List[str]]] = None,
    roi_subsets: Optional[List[str]] = None,
    fast: bool = True,
):
    if _IM is None:
        raise RuntimeError("Imaging tool not configured. Call configure_imaging_tool(...) first.")

    files: List[str] = []
    if file_paths:
        files.extend(file_paths)
    if file_path:
        files.append(file_path)
    kwargs = {"task_name": task_name, "roi_subset": roi_subset, "roi_subsets": roi_subsets, "fast": fast}

    task = Task(user_msg=f"Run TotalSegmentator task='{task_name}'", files=files, kwargs=kwargs)
    return await _IM.run(task, _cs())
