from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import nibabel as nib
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from core.state import ConversationState, Task, TaskResult
from core.storage import get_run_dir, get_temp_dir
from progress_ui import update_progress
from tools.file_validation import validate_existing_input_files
from tools.shared import _cs, toolify_agent


BREAST_TUMOR_MODEL_PATH = (
    "/Users/adhrith/Downloads/Trained_Models/Dataset910_breastSeg/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres"
)
BRAIN_TUMOR_MODEL_PATH = (
    "/Users/adhrith/Downloads/Trained_Models/Dataset501_BraTS2021/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres"
)

NNUNET_PREDICT_EXECUTABLE = "nnUNetv2_predict_from_modelfolder"


ModelName = Literal["breast_tumor", "brain_tumor"]
DeviceName = Literal["auto", "cuda", "cpu", "mps"]

_CHANNEL_SUFFIX_RE = re.compile(r"^(?P<case>.+)_(?P<channel>\d{4})$")
_BRATS_MODALITY_SUFFIX_RE = re.compile(
    r"^(?P<case>.+)[_-](?P<modality>t2f|t1n|t1c|t2w)$",
    re.IGNORECASE,
)
_BRATS_MODALITY_TO_CHANNEL = {
    "t2f": 0,
    "t1n": 1,
    "t1c": 2,
    "t2w": 3,
}


@dataclass
class PreparedCase:
    source_case_id: str
    staged_case_id: str
    channels: Dict[int, Path]


@dataclass
class CommandResult:
    returncode: int
    tail: List[str]


class NNUNetCommandError(RuntimeError):
    def __init__(self, returncode: int, tail: List[str]):
        super().__init__(f"nnU-Net exited with code {returncode}.")
        self.returncode = int(returncode)
        self.tail = list(tail)


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def _safe_case_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return safe[:100] or "case"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:10]


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def _stop_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=8)
        return
    except asyncio.TimeoutError:
        pass

    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    await proc.wait()


async def _run_nnunet_command(
    command: List[str],
    *,
    environment: Dict[str, str],
    log_path: Path,
) -> CommandResult:
    """Run nnU-Net while preserving logs and cleaning up on cancellation."""

    tail: deque[str] = deque(maxlen=80)
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            start_new_session=os.name != "nt",
        )
        try:
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.readline()
                if not chunk:
                    break
                text = chunk.decode(errors="replace")
                log_handle.write(text)
                log_handle.flush()
                tail.append(text.rstrip())
            returncode = await proc.wait()
        except BaseException:
            await _stop_process(proc)
            raise

    result = CommandResult(returncode=returncode, tail=list(tail))
    if returncode != 0:
        raise NNUNetCommandError(returncode, result.tail)
    return result


class NNUNetPredictAgent:
    name = "nnunet"
    model = None

    def __init__(
        self,
        *,
        breast_model_path: str,
        brain_model_path: str,
        predict_executable: str,
    ) -> None:
        self.breast_model_path = Path(breast_model_path).expanduser()
        self.brain_model_path = Path(brain_model_path).expanduser()
        self.predict_executable = str(predict_executable).strip()

    def _model_path(self, model_name: ModelName) -> Path:
        if model_name == "breast_tumor":
            return self.breast_model_path
        return self.brain_model_path

    @staticmethod
    def _expected_channels(model_name: ModelName) -> Dict[int, str]:
        if model_name == "breast_tumor":
            return {0: "T1"}
        return {0: "FLAIR", 1: "T1", 2: "T1CE", 3: "T2"}

    @staticmethod
    def _resolve_device(requested: DeviceName) -> Literal["cuda", "cpu", "mps"]:
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available in this environment.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available in this environment.")
        return requested

    def _validate_model(
        self,
        model_name: ModelName,
    ) -> tuple[Path, Dict[int, str], Dict[str, int], List[str]]:
        model_path = self._model_path(model_name).resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Configured {model_name} nnU-Net model directory does not exist."
            )

        dataset_json_path = model_path / "dataset.json"
        plans_json_path = model_path / "plans.json"
        if not dataset_json_path.is_file() or not plans_json_path.is_file():
            raise ValueError(
                f"Configured {model_name} path must contain dataset.json and plans.json."
            )

        try:
            dataset = json.loads(dataset_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not read {model_name} dataset.json: {exc}") from exc

        file_ending = str(dataset.get("file_ending") or "")
        if file_ending.lower() != ".nii.gz":
            raise ValueError(
                f"The configured {model_name} model expects {file_ending or 'an unknown format'}; "
                "this tool currently accepts .nii.gz inputs only."
            )

        configured_channels = {
            int(index): str(label)
            for index, label in (dataset.get("channel_names") or {}).items()
        }
        expected_channels = self._expected_channels(model_name)
        if set(configured_channels) != set(expected_channels):
            raise ValueError(
                f"Configured {model_name} model channels are {configured_channels}; "
                f"expected channel indexes {sorted(expected_channels)}."
            )

        labels = {
            str(name): int(value)
            for name, value in (dataset.get("labels") or {}).items()
            if isinstance(value, (int, float))
        }

        available_folds: List[str] = []
        for fold_dir in model_path.glob("fold_*"):
            if not fold_dir.is_dir() or not (fold_dir / "checkpoint_final.pth").is_file():
                continue
            fold_name = fold_dir.name.removeprefix("fold_")
            if fold_name == "all" or fold_name.isdigit():
                available_folds.append(fold_name)
        available_folds.sort(
            key=lambda value: (value == "all", int(value) if value.isdigit() else 0)
        )
        if not available_folds:
            raise ValueError(
                f"Configured {model_name} model has no fold_* directory with checkpoint_final.pth."
            )

        return model_path, configured_channels, labels, available_folds

    @staticmethod
    def _validate_input_geometry(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
        try:
            image = nib.load(str(path))
        except Exception as exc:
            raise ValueError(f"Input {path.name} is not a readable NIfTI image: {exc}") from exc
        shape = tuple(int(value) for value in image.shape[:3])
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError(f"Input {path.name} has invalid spatial shape {shape}.")
        return shape, np.asarray(image.affine)

    def _prepare_cases(
        self,
        files: List[str],
        *,
        model_name: ModelName,
        input_dir: Path,
        channels: Dict[int, str],
    ) -> List[PreparedCase]:
        paths = [Path(value).expanduser().resolve() for value in files]
        unsupported = [path.name for path in paths if not path.name.lower().endswith(".nii.gz")]
        if unsupported:
            raise ValueError(
                "nnunet accepts .nii.gz inputs only; unsupported: "
                + ", ".join(unsupported)
            )

        grouped: Dict[str, Dict[int, Path]] = {}
        case_order: List[str] = []
        for path in paths:
            stem = _nifti_stem(path)
            channel_match = _CHANNEL_SUFFIX_RE.match(stem)
            brats_match = (
                _BRATS_MODALITY_SUFFIX_RE.match(stem)
                if model_name == "brain_tumor"
                else None
            )
            if channel_match:
                case_id = channel_match.group("case")
                channel = int(channel_match.group("channel"))
            elif brats_match:
                case_id = brats_match.group("case")
                modality = brats_match.group("modality").lower()
                channel = _BRATS_MODALITY_TO_CHANNEL[modality]
            else:
                if len(channels) != 1:
                    expected = ", ".join(
                        f"_{index:04d}={name}" for index, name in sorted(channels.items())
                    )
                    raise ValueError(
                        f"The {model_name} model requires either nnU-Net channel suffixes "
                        f"({expected}) or standard BraTS modality suffixes "
                        "(-t2f=FLAIR, -t1n=T1, -t1c=T1CE, -t2w=T2). "
                        f"Could not identify the modality for {path.name}."
                    )
                case_id = stem
                channel = next(iter(channels))

            if channel not in channels:
                raise ValueError(
                    f"Input {path.name} uses channel _{channel:04d}, but {model_name} "
                    f"expects {sorted(channels)}."
                )

            if case_id in grouped and channel in grouped[case_id]:
                if len(channels) == 1 and channel_match is None:
                    case_id = f"{case_id}_{_short_hash(str(path))}"
                else:
                    raise ValueError(
                        f"Duplicate channel _{channel:04d} for case {case_id}."
                    )
            if case_id not in grouped:
                grouped[case_id] = {}
                case_order.append(case_id)
            grouped[case_id][channel] = path

        expected_indexes = set(channels)
        used_staged_ids: Dict[str, str] = {}
        prepared: List[PreparedCase] = []
        for case_id in case_order:
            case_channels = grouped[case_id]
            actual_indexes = set(case_channels)
            if actual_indexes != expected_indexes:
                missing = sorted(expected_indexes - actual_indexes)
                extra = sorted(actual_indexes - expected_indexes)
                raise ValueError(
                    f"Case {case_id} has channels {sorted(actual_indexes)}; "
                    f"missing={missing}, unexpected={extra}."
                )

            staged_case_id = _safe_case_id(case_id)
            existing = used_staged_ids.get(staged_case_id)
            if existing is not None and existing != case_id:
                staged_case_id = f"{staged_case_id}_{_short_hash(case_id)}"
            used_staged_ids[staged_case_id] = case_id

            reference_shape: Optional[tuple[int, ...]] = None
            reference_affine: Optional[np.ndarray] = None
            for channel, source_path in sorted(case_channels.items()):
                shape, affine = self._validate_input_geometry(source_path)
                if reference_shape is None:
                    reference_shape = shape
                    reference_affine = affine
                elif shape != reference_shape or not np.allclose(
                    affine,
                    reference_affine,
                    rtol=1e-5,
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"Channels for case {case_id} are not geometrically aligned; "
                        f"{source_path.name} does not match the first channel."
                    )
                destination = input_dir / f"{staged_case_id}_{channel:04d}.nii.gz"
                shutil.copy2(source_path, destination)

            prepared.append(
                PreparedCase(
                    source_case_id=case_id,
                    staged_case_id=staged_case_id,
                    channels=dict(case_channels),
                )
            )
        return prepared

    @staticmethod
    def _prediction_path(output_dir: Path, staged_case_id: str) -> Optional[Path]:
        for suffix in (".nii.gz", ".nii"):
            candidate = output_dir / f"{staged_case_id}{suffix}"
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate.resolve()
        return None

    @staticmethod
    def _validate_prediction(
        prediction: Path,
        reference: Path,
    ) -> Dict[str, object]:
        try:
            predicted_image = nib.load(str(prediction))
            reference_image = nib.load(str(reference))
            predicted_shape = tuple(int(value) for value in predicted_image.shape[:3])
            reference_shape = tuple(int(value) for value in reference_image.shape[:3])
            if predicted_shape != reference_shape:
                raise ValueError(
                    f"prediction shape {predicted_shape} does not match input shape {reference_shape}"
                )
            if not np.allclose(
                predicted_image.affine,
                reference_image.affine,
                rtol=1e-5,
                atol=1e-4,
            ):
                raise ValueError("prediction affine does not match the input affine")
            nonzero_voxels = int(np.count_nonzero(np.asanyarray(predicted_image.dataobj)))
        except Exception as exc:
            raise ValueError(f"Invalid prediction {prediction.name}: {exc}") from exc
        return {
            "shape": list(predicted_shape),
            "nonzero_voxels": nonzero_voxels,
        }

    async def run(
        self,
        task: Task,
        state: ConversationState,
    ) -> TaskResult:
        del state
        files = list(dict.fromkeys(str(value) for value in task.files if value))
        requested_device: DeviceName = task.kwargs.get("device") or "auto"
        try:
            validate_existing_input_files(files, tool_name="nnunet")
            model_name: ModelName = task.kwargs["model_name"]
            model_path, channels, labels, folds = self._validate_model(model_name)
            device = self._resolve_device(requested_device)
        except Exception as exc:
            message = str(exc)
            return TaskResult(
                output={"status": "error", "error": message},
                status="error",
                errors=[message],
            )

        input_dir = get_temp_dir(prefix="nnunet_inputs")
        try:
            prepared_cases = self._prepare_cases(
                files,
                model_name=model_name,
                input_dir=input_dir,
                channels=channels,
            )
        except Exception as exc:
            shutil.rmtree(input_dir, ignore_errors=True)
            message = str(exc)
            return TaskResult(
                output={"status": "error", "error": message},
                status="error",
                errors=[message],
            )

        output_dir = get_run_dir(self.name, persist=True)
        log_path = output_dir / "nnunet.log"
        status_path = output_dir / "nnunet_status.json"

        tta = bool(task.kwargs.get("tta", True))
        save_probabilities = bool(task.kwargs.get("save_probabilities", False))
        command = [
            self.predict_executable,
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-m",
            str(model_path),
            "-f",
            *folds,
            "-device",
            device,
            "--disable_progress_bar",
        ]
        if not tta:
            command.append("--disable_tta")
        if save_probabilities:
            command.append("--save_probabilities")

        command_error = ""
        await update_progress(
            5,
            f"Preparing {len(prepared_cases)} nnU-Net case(s)",
            title="nnU-Net segmentation",
            status="running",
            total=len(prepared_cases),
            completed=0,
            failed=0,
        )
        try:
            await update_progress(
                15,
                f"Running {model_name.replace('_', ' ')} model",
                title="nnU-Net segmentation",
                status="running",
                total=len(prepared_cases),
                completed=0,
                failed=0,
            )
            await _run_nnunet_command(
                command,
                environment=os.environ.copy(),
                log_path=log_path,
            )
        except FileNotFoundError:
            command_error = (
                f"{self.predict_executable} was not found. Install nnunetv2 in the active "
                "environment and ensure its console scripts are on PATH."
            )
        except NNUNetCommandError as exc:
            command_error = f"nnU-Net prediction exited with code {exc.returncode}."
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)

        case_results: List[Dict[str, object]] = []
        segmentation_paths: List[str] = []
        segmentation_map: Dict[str, str] = {}
        errors: List[str] = []

        for case in prepared_cases:
            prediction = self._prediction_path(output_dir, case.staged_case_id)
            input_names = {
                f"{channel:04d}": path.name
                for channel, path in sorted(case.channels.items())
            }
            if prediction is None:
                message = f"Case {case.source_case_id} produced no segmentation file."
                errors.append(message)
                case_results.append(
                    {
                        "case_id": case.source_case_id,
                        "status": "error",
                        "input_files": input_names,
                        "error": message,
                    }
                )
                continue

            try:
                validation = self._validate_prediction(
                    prediction,
                    next(iter(case.channels.values())),
                )
            except Exception as exc:
                message = str(exc)
                errors.append(message)
                case_results.append(
                    {
                        "case_id": case.source_case_id,
                        "status": "error",
                        "input_files": input_names,
                        "prediction": prediction.name,
                        "error": message,
                    }
                )
                continue

            prediction_value = str(prediction)
            segmentation_paths.append(prediction_value)
            segmentation_map[case.source_case_id] = prediction_value
            case_results.append(
                {
                    "case_id": case.source_case_id,
                    "status": "ok",
                    "input_files": input_names,
                    "prediction": prediction.name,
                    **validation,
                }
            )

        if command_error:
            errors.insert(0, command_error)

        completed = len(segmentation_paths)
        failed = len(prepared_cases) - completed
        if completed == len(prepared_cases) and not command_error:
            status = "ok"
        elif completed:
            status = "partial"
        else:
            status = "error"

        summary: Dict[str, object] = {
            "status": status,
            "agent": self.name,
            "action": "inference",
            "model_name": model_name,
            "channel_names": {str(index): name for index, name in channels.items()},
            "labels": labels,
            "folds": folds,
            "requested_device": requested_device,
            "device": device,
            "tta": tta,
            "save_probabilities": save_probabilities,
            "total": len(prepared_cases),
            "completed": completed,
            "failed": failed,
            "cases": case_results,
            "errors": errors,
        }
        _write_json(status_path, summary)

        await update_progress(
            100,
            f"nnU-Net complete: {completed}/{len(prepared_cases)} succeeded",
            title="nnU-Net segmentation",
            status="completed" if status == "ok" else status,
            total=len(prepared_cases),
            completed=completed,
            failed=failed,
        )

        artifact_files = [str(status_path)]
        if log_path.is_file():
            artifact_files.append(str(log_path))
        artifacts: Dict[str, object] = {
            "segmentations": segmentation_paths,
            "segmentations_map": segmentation_map,
            "output_dir": str(output_dir),
            "files": artifact_files,
        }

        return TaskResult(
            output=summary,
            artifacts=artifacts,
            status=status,  # type: ignore[arg-type]
            errors=errors,
        )


class NNUNetArgs(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    file_path: Optional[str] = Field(
        default=None,
        description=(
            "Exact artifact_id for one .nii.gz input. For brain_tumor, normally use "
            "file_paths because four channel files are required."
        ),
    )
    file_paths: Optional[List[str]] = Field(
        default=None,
        description=(
            "Exact artifact_ids for all .nii.gz inputs in one nnU-Net batch. Brain tumor "
            "cases accept either matching _0000=FLAIR, _0001=T1, _0002=T1CE, and "
            "_0003=T2 files or standard BraTS -t2f, -t1n, -t1c, and -t2w filenames. "
            "Breast tumor inputs are independent single-channel T1 cases."
        ),
    )
    model_name: ModelName = Field(
        ...,
        description=(
            "Installed nnU-Net model: breast_tumor for Dataset910 breast tumor "
            "segmentation, or brain_tumor for Dataset501 BraTS tumor segmentation."
        ),
    )
    device: DeviceName = Field(
        default="auto",
        description=(
            "Inference device: auto selects CUDA, then MPS, then CPU; an explicit cuda, cpu, "
            "or mps value overrides detection. Use CUDA_VISIBLE_DEVICES outside the tool to "
            "select a particular GPU."
        ),
    )
    tta: bool = Field(
        default=True,
        description="Use nnU-Net test-time mirroring augmentation. Recommended for accuracy.",
    )
    save_probabilities: bool = Field(
        default=False,
        description=(
            "Also save probability arrays. Leave false unless probability outputs are "
            "explicitly required because they consume substantial disk space."
        ),
    )


_NNUNET: Optional[NNUNetPredictAgent] = None


def configure_nnunet_tool(
    *,
    breast_model_path: Optional[str] = None,
    brain_model_path: Optional[str] = None,
    predict_executable: Optional[str] = None,
) -> None:
    """Configure local nnU-Net model folders without loading model weights."""

    global _NNUNET
    _NNUNET = NNUNetPredictAgent(
        breast_model_path=breast_model_path or BREAST_TUMOR_MODEL_PATH,
        brain_model_path=brain_model_path or BRAIN_TUMOR_MODEL_PATH,
        predict_executable=predict_executable or NNUNET_PREDICT_EXECUTABLE,
    )


@toolify_agent(
    name="nnunet",
    description=(
        "Run inference with one of the two configured nnU-Net v2 tumor-segmentation models. "
        "Use model_name=breast_tumor for Dataset910 single-channel T1 breast tumor "
        "segmentation. Use model_name=brain_tumor for Dataset501 BraTS brain tumor "
        "segmentation; every brain case requires four geometrically aligned files. Accept "
        "either nnU-Net suffixes (_0000=FLAIR, _0001=T1, _0002=T1CE, _0003=T2) or "
        "standard BraTS suffixes (-t2f, -t1n, -t1c, -t2w), which are mapped automatically. "
        "Pass exact "
        "registered artifact IDs through file_path or file_paths. The tool batches all cases "
        "into one inference command and returns verified NIfTI segmentation artifacts."
    ),
    args_schema=NNUNetArgs,
    timeout_s=86400,
)
async def nnunet_runner(
    model_name: ModelName,
    file_path: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    device: DeviceName = "auto",
    tta: bool = True,
    save_probabilities: bool = False,
):
    if _NNUNET is None:
        raise RuntimeError(
            "nnunet tool is not configured. Call configure_nnunet_tool(...) first."
        )

    files: List[str] = []
    if file_paths:
        files.extend(file_paths)
    if file_path:
        files.append(file_path)
    files = list(dict.fromkeys(files))

    task = Task(
        user_msg=f"Run nnU-Net model={model_name}",
        files=files,
        kwargs={
            "model_name": model_name,
            "device": device,
            "tta": tta,
            "save_probabilities": save_probabilities,
        },
    )
    return await _NNUNET.run(task, _cs())
