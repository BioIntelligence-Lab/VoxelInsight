from __future__ import annotations

import asyncio
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator

from core.agents.artifacts import safe_slug
from core.state import TaskResult
from core.storage import get_run_dir, get_temp_dir
from tools.shared import toolify_agent


BRAINIAC_MODEL_SIZE = (96, 96, 96)
BRAINIAC_PATCH_SIZE = (16, 16, 16)
BRAINIAC_HIDDEN_SIZE = 768
BRAINIAC_MLP_DIM = 3072
BRAINIAC_NUM_LAYERS = 12
BRAINIAC_NUM_HEADS = 12
BRAINIAC_CHECKPOINT_PATH = "/Users/adhrith/Downloads/BrainIAC.ckpt"
BRAINIAC_ATLAS_PATH = "/Users/adhrith/Downloads/temp_head.nii.gz"
BRAINIAC_DEVICE = "auto"
BRAINIAC_CHECKPOINT_SHA256 = (
    "f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693"
)

BrainIACOperation = Literal["features", "saliency", "both"]


@dataclass
class _PreparedCase:
    index: int
    source_path: Path
    case_name: str
    inference_path: Path
    preprocessed_path: Optional[Path]
    tensor: torch.Tensor
    affine: np.ndarray
    header: nib.Nifti1Header


_BRAINIAC_CONTEXT: Optional[Dict[str, Any]] = None
_MODEL_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_HD_BET_LOCK = threading.Lock()


def configure_brainiac_tool(
    *,
    checkpoint_path: Optional[str] = BRAINIAC_CHECKPOINT_PATH,
    atlas_path: Optional[str] = BRAINIAC_ATLAS_PATH,
    device: Optional[str] = BRAINIAC_DEVICE,
    expected_checkpoint_sha256: Optional[str] = BRAINIAC_CHECKPOINT_SHA256,
) -> None:
    global _BRAINIAC_CONTEXT

    selected_device = (device or BRAINIAC_DEVICE).strip().lower()
    if not selected_device or selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device not in {"cpu", "cuda"}:
        raise ValueError("BrainIAC device must be 'cpu', 'cuda', or 'auto'.")
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BrainIAC was configured for CUDA, but CUDA is unavailable.")

    checkpoint = Path(
        checkpoint_path or BRAINIAC_CHECKPOINT_PATH
    ).expanduser().resolve()
    atlas = Path(atlas_path or BRAINIAC_ATLAS_PATH).expanduser().resolve()
    expected_sha256 = (
        expected_checkpoint_sha256 or BRAINIAC_CHECKPOINT_SHA256
    ).strip().lower()
    if expected_sha256 and (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("Expected BrainIAC checkpoint SHA-256 must be 64 hexadecimal characters.")

    _BRAINIAC_CONTEXT = {
        "checkpoint_path": checkpoint,
        "atlas_path": atlas,
        "device": selected_device,
        "expected_checkpoint_sha256": expected_sha256,
        "model": None,
        "checkpoint_sha256": "",
    }


def _context() -> Dict[str, Any]:
    if _BRAINIAC_CONTEXT is None:
        raise RuntimeError(
            "BrainIAC tool is not configured. Call configure_brainiac_tool(...) first."
        )
    return _BRAINIAC_CONTEXT


def _require_file(path: Optional[Path], *, label: str) -> Path:
    if path is None:
        raise RuntimeError(f"BrainIAC {label} is not configured.")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"BrainIAC {label} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_monai_vit() -> Any:
    try:
        from monai.networks.nets import ViT
    except Exception as exc:
        raise RuntimeError(
            "BrainIAC requires MONAI. Install the project's declared `monai` dependency."
        ) from exc
    return ViT


def _build_brainiac_model(checkpoint_path: Path) -> torch.nn.Module:
    ViT = _load_monai_vit()

    class BrainIACBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = ViT(
                in_channels=1,
                img_size=BRAINIAC_MODEL_SIZE,
                patch_size=BRAINIAC_PATCH_SIZE,
                hidden_size=BRAINIAC_HIDDEN_SIZE,
                mlp_dim=BRAINIAC_MLP_DIM,
                num_layers=BRAINIAC_NUM_LAYERS,
                num_heads=BRAINIAC_NUM_HEADS,
                save_attn=True,
            )

            for block in self.backbone.blocks:
                if not bool(getattr(block, "with_cross_attention", False)):
                    if hasattr(block, "norm_cross_attn"):
                        block.norm_cross_attn = torch.nn.Identity()
                    if hasattr(block, "cross_attn"):
                        block.cross_attn = torch.nn.Identity()

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            output = self.backbone(image)
            tokens = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise RuntimeError(
                    "BrainIAC backbone returned an unexpected output shape; expected "
                    "[batch, tokens, 768]."
                )
            return tokens[:, 0]

    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("BrainIAC checkpoint must contain a state dictionary.")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("BrainIAC checkpoint has no usable state_dict mapping.")

    backbone_state = {
        str(key)[len("backbone.") :]: value
        for key, value in state_dict.items()
        if str(key).startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError("BrainIAC checkpoint contains no `backbone.*` weights.")

    model = BrainIACBackbone()
    model.backbone.load_state_dict(backbone_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _ensure_model_loaded() -> torch.nn.Module:
    context = _context()
    if context.get("model") is not None:
        return context["model"]

    with _MODEL_LOAD_LOCK:
        if context.get("model") is not None:
            return context["model"]

        checkpoint_path = _require_file(
            context.get("checkpoint_path"),
            label="checkpoint",
        )
        expected_sha256 = str(context.get("expected_checkpoint_sha256") or "")
        if expected_sha256:
            actual_sha256 = _sha256(checkpoint_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "BrainIAC checkpoint checksum mismatch: "
                    f"expected {expected_sha256}, observed {actual_sha256}."
                )
            context["checkpoint_sha256"] = actual_sha256

        model = _build_brainiac_model(checkpoint_path)
        model.to(torch.device(str(context["device"])))
        context["model"] = model
        return model


def _is_nifti(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.endswith(".nii") or lowered.endswith(".nii.gz")


def _nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def _validate_source_nifti(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"BrainIAC input does not exist: {path}")
    if not _is_nifti(path):
        raise ValueError(
            f"BrainIAC accepts NIfTI files only; convert DICOM before inference: {path.name}"
        )
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise ValueError(f"BrainIAC could not read NIfTI input {path.name}: {exc}") from exc
    if len(image.shape) != 3:
        raise ValueError(
            f"BrainIAC requires a single 3D MRI volume; {path.name} has shape {image.shape}."
        )
    if any(int(size) < 2 for size in image.shape):
        raise ValueError(f"BrainIAC input has an invalid spatial shape: {image.shape}.")
    return image


def _resized_affine(
    affine: np.ndarray,
    old_shape: Sequence[int],
    new_shape: Sequence[int],
) -> np.ndarray:

    transform = np.eye(4, dtype=np.float64)
    for axis in range(3):
        scale = float(old_shape[axis]) / float(new_shape[axis])
        transform[axis, axis] = scale
        transform[axis, 3] = (scale - 1.0) / 2.0
    return np.asarray(affine, dtype=np.float64) @ transform


def _model_input_from_nifti(
    path: Path,
) -> tuple[torch.Tensor, np.ndarray, nib.Nifti1Header]:
    image = _validate_source_nifti(path)
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"BrainIAC input contains non-finite voxel values: {path.name}")

    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=BRAINIAC_MODEL_SIZE,
        mode="trilinear",
        align_corners=False,
    )
    nonzero = tensor != 0
    if not bool(nonzero.any()):
        raise ValueError(f"BrainIAC input contains no nonzero voxels: {path.name}")
    values = tensor[nonzero]
    mean = values.mean()
    std = values.std(unbiased=False)
    if not bool(torch.isfinite(std)) or float(std) <= 1e-8:
        raise ValueError(
            "BrainIAC input has insufficient nonzero intensity variation: "
            f"{path.name}"
        )
    tensor = torch.where(nonzero, (tensor - mean) / std, torch.zeros_like(tensor))

    affine = _resized_affine(image.affine, image.shape, BRAINIAC_MODEL_SIZE)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(BRAINIAC_MODEL_SIZE)
    return tensor, affine, header


def _save_nifti(
    data: np.ndarray,
    *,
    path: Path,
    affine: np.ndarray,
    header: nib.Nifti1Header,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_header = header.copy()
    output_header.set_data_dtype(np.float32)
    output_header.set_data_shape(data.shape)
    image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32),
        np.asarray(affine, dtype=np.float64),
        header=output_header,
    )
    qform_code = int(header.get("qform_code", 0)) or 1
    sform_code = int(header.get("sform_code", 0)) or 1
    image.set_qform(affine, code=qform_code)
    image.set_sform(affine, code=sform_code)
    nib.save(image, str(path))


def _load_simpleitk() -> Any:
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise RuntimeError(
            "BrainIAC preprocessing requires SimpleITK. Install the project's "
            "declared `SimpleITK>=2.2` dependency."
        ) from exc
    return sitk


def _load_hd_bet() -> tuple[str, tuple[Any, ...]]:
    try:
        from HD_BET.checkpoint_download import maybe_download_parameters
        from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

        return (
            "v2",
            (maybe_download_parameters, get_hdbet_predictor, hdbet_predict),
        )
    except Exception:
        try:
            from HD_BET.run import run_hd_bet

            return "legacy", (run_hd_bet,)
        except Exception as legacy_exc:
            raise RuntimeError(
                "BrainIAC raw-image preprocessing requires HD-BET. Install the "
                "`HD_BET` package, or set preprocess=false for an image that has "
                "already been registered and skull stripped."
            ) from legacy_exc


def _resample_template_to_one_millimeter(template: Any, sitk: Any) -> Any:
    old_size = template.GetSize()
    old_spacing = template.GetSpacing()
    new_spacing = (1.0, 1.0, 1.0)
    new_size = [
        max(1, int(round(old_size[index] * old_spacing[index] / new_spacing[index])))
        for index in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(template.GetOrigin())
    resampler.SetOutputDirection(template.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(template)


def _register_to_brainiac_template(
    source_path: Path,
    registered_path: Path,
    atlas_path: Path,
) -> None:
    sitk = _load_simpleitk()
    fixed = sitk.ReadImage(str(atlas_path), sitk.sitkFloat32)
    if fixed.GetDimension() != 3:
        raise ValueError("BrainIAC atlas must be a 3D image.")
    fixed = _resample_template_to_one_millimeter(fixed, sitk)

    moving = sitk.ReadImage(str(source_path), sitk.sitkFloat32)
    if moving.GetDimension() != 3:
        raise ValueError(f"BrainIAC input must be 3D: {source_path.name}")
    moving = sitk.N4BiasFieldCorrection(moving)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    if hasattr(registration, "SetMetricSamplingSeed"):
        registration.SetMetricSamplingSeed(42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration.Execute(fixed, moving)

    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )
    registered_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(registered, str(registered_path))


def _brain_extract(
    registered_path: Path,
    *,
    destination: Path,
) -> None:
    backend, functions = _load_hd_bet()
    destination.parent.mkdir(parents=True, exist_ok=True)
    configured_device = str(_context()["device"])

    with _HD_BET_LOCK:
        if backend == "v2":
            maybe_download_parameters, get_hdbet_predictor, hdbet_predict = functions
            maybe_download_parameters()
            predictor = get_hdbet_predictor(
                use_tta=False,
                device=torch.device(configured_device),
                verbose=False,
            )
            hdbet_predict(
                str(registered_path),
                str(destination),
                predictor,
                keep_brain_mask=False,
                compute_brain_extracted_image=True,
            )
        else:
            (run_hd_bet,) = functions
            legacy_device: Any = 0 if configured_device == "cuda" else "cpu"
            run_hd_bet(
                str(registered_path),
                str(destination),
                mode="fast",
                device=legacy_device,
                postprocess=False,
                do_tta=False,
                keep_mask=False,
                overwrite=True,
            )

    if not destination.is_file():
        raise RuntimeError("HD-BET completed without producing a skull-stripped NIfTI.")


def _preprocess_case(
    source_path: Path,
    *,
    case_name: str,
    output_dir: Path,
) -> Path:
    context = _context()
    atlas_path = _require_file(context.get("atlas_path"), label="atlas")
    work_dir = get_temp_dir(prefix=f"brainiac-{case_name}")
    registered_path = work_dir / "registered_0000.nii.gz"
    processed_path = output_dir / "preprocessed" / f"{case_name}_brainiac.nii.gz"
    _register_to_brainiac_template(source_path, registered_path, atlas_path)
    _brain_extract(
        registered_path,
        destination=processed_path,
    )
    return processed_path


def _prepare_case(
    index: int,
    source_path: Path,
    *,
    case_name: str,
    preprocess: bool,
    output_dir: Path,
) -> _PreparedCase:
    _validate_source_nifti(source_path)
    preprocessed_path = (
        _preprocess_case(
            source_path,
            case_name=case_name,
            output_dir=output_dir,
        )
        if preprocess
        else None
    )
    inference_path = preprocessed_path or source_path
    tensor, affine, header = _model_input_from_nifti(inference_path)

    if preprocess:
        model_input_path = (
            output_dir / "preprocessed" / f"{case_name}_brainiac_model_input.nii.gz"
        )
        _save_nifti(
            tensor.squeeze(0).squeeze(0).numpy(),
            path=model_input_path,
            affine=affine,
            header=header,
        )
        preprocessed_path = model_input_path
        inference_path = model_input_path

    return _PreparedCase(
        index=index,
        source_path=source_path,
        case_name=case_name,
        inference_path=inference_path,
        preprocessed_path=preprocessed_path,
        tensor=tensor,
        affine=affine,
        header=header,
    )


def _tokens_from_backbone_output(output: Any) -> torch.Tensor:
    tokens = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise RuntimeError(
            "BrainIAC backbone returned an unexpected output; expected token tensor "
            "with shape [batch, tokens, 768]."
        )
    if tokens.shape[-1] != BRAINIAC_HIDDEN_SIZE:
        raise RuntimeError(
            f"BrainIAC backbone returned hidden size {tokens.shape[-1]}, expected 768."
        )
    return tokens


def _forward_case(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    operation: BrainIACOperation,
    saliency_layer: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    device = torch.device(str(_context()["device"]))
    image = tensor.to(device)
    need_features = operation in {"features", "both"}
    need_saliency = operation in {"saliency", "both"}
    captured: Dict[str, torch.Tensor] = {}
    hook = None

    if need_saliency:
        try:
            attention_module = model.backbone.blocks[saliency_layer].attn
        except Exception as exc:
            raise RuntimeError(
                f"BrainIAC transformer layer {saliency_layer} is unavailable."
            ) from exc

        def capture_attention(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("BrainIAC attention hook received no token tensor.")
            tokens = inputs[0]
            qkv = module.qkv(tokens)
            batch_size, token_count, _ = qkv.shape
            qkv = qkv.reshape(
                batch_size,
                token_count,
                3,
                module.num_heads,
                -1,
            ).permute(2, 0, 3, 1, 4)
            query, key = qkv[0], qkv[1]
            captured["attention"] = (
                (query @ key.transpose(-2, -1)) * module.scale
            ).softmax(dim=-1)

        hook = attention_module.register_forward_pre_hook(capture_attention)

    try:
        with _INFERENCE_LOCK, torch.inference_mode():
            output = model.backbone(image)
    finally:
        if hook is not None:
            hook.remove()

    tokens = _tokens_from_backbone_output(output)
    embedding = (
        tokens[:, 0].squeeze(0).detach().cpu().numpy().astype(np.float32)
        if need_features
        else None
    )

    saliency = None
    if need_saliency:
        attention = captured.get("attention")
        if attention is None:
            raise RuntimeError("BrainIAC could not capture transformer attention weights.")
        mean_attention = attention[0].mean(dim=0)
        patch_attention = mean_attention[0, 1:]
        patch_count = int(np.prod([size // patch for size, patch in zip(
            BRAINIAC_MODEL_SIZE,
            BRAINIAC_PATCH_SIZE,
        )]))
        if patch_attention.numel() < patch_count:
            padded = torch.zeros(
                patch_count,
                dtype=patch_attention.dtype,
                device=patch_attention.device,
            )
            padded[: patch_attention.numel()] = patch_attention
            patch_attention = padded
        elif patch_attention.numel() > patch_count:
            patch_attention = patch_attention[:patch_count]

        patch_grid = tuple(
            size // patch
            for size, patch in zip(BRAINIAC_MODEL_SIZE, BRAINIAC_PATCH_SIZE)
        )
        volume = patch_attention.reshape(*patch_grid).unsqueeze(0).unsqueeze(0)
        volume = F.interpolate(
            volume,
            size=BRAINIAC_MODEL_SIZE,
            mode="trilinear",
            align_corners=False,
        ).squeeze()
        minimum = volume.min()
        maximum = volume.max()
        if float(maximum - minimum) > 1e-12:
            volume = (volume - minimum) / (maximum - minimum)
        else:
            volume = torch.zeros_like(volume)
        saliency = volume.detach().cpu().numpy().astype(np.float32)

    return embedding, saliency


def _write_saliency_preview(
    model_input: np.ndarray,
    saliency: np.ndarray,
    *,
    path: Path,
    title: str,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    slice_scores = saliency.reshape(-1, saliency.shape[2]).max(axis=0)
    slice_index = int(np.argmax(slice_scores))
    image_slice = model_input[:, :, slice_index]
    saliency_slice = saliency[:, :, slice_index]

    nonzero = image_slice[image_slice != 0]
    if nonzero.size:
        lower, upper = np.percentile(nonzero, (1, 99))
    else:
        lower, upper = float(image_slice.min()), float(image_slice.max())
    if upper <= lower:
        upper = lower + 1.0
    normalized_image = np.clip((image_slice - lower) / (upper - lower), 0.0, 1.0)

    figure = Figure(figsize=(10, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    image_axis, overlay_axis = figure.subplots(1, 2)
    image_axis.imshow(normalized_image.T, cmap="gray", origin="lower")
    image_axis.set_title("BrainIAC model input")
    image_axis.axis("off")
    overlay_axis.imshow(normalized_image.T, cmap="gray", origin="lower")
    overlay_axis.imshow(
        saliency_slice.T,
        cmap="magma",
        origin="lower",
        alpha=np.clip(saliency_slice.T, 0.0, 1.0) * 0.75,
        vmin=0.0,
        vmax=1.0,
    )
    overlay_axis.set_title(f"Attention overlay (slice {slice_index})")
    overlay_axis.axis("off")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    figure.clear()


class BrainIACArgs(BaseModel):
    image_path: Optional[str] = Field(
        default=None,
        description=(
            "Exact artifact_id for one 3D structural brain MRI NIfTI file. "
            "ArtifactRegistryMiddleware resolves it to a verified host path."
        ),
    )
    image_paths: Optional[List[str]] = Field(
        default=None,
        description=(
            "Ordered exact artifact_ids for a batch of 3D structural brain MRI "
            "NIfTI files. Do not pass DICOM directories."
        ),
    )
    operation: BrainIACOperation = Field(
        default="features",
        description=(
            "`features` writes one 768-dimensional embedding per scan; `saliency` "
            "writes 3D attention maps and previews; `both` produces both outputs."
        ),
    )
    preprocess: bool = Field(
        default=True,
        description=(
            "Run BrainIAC-specific N4 correction, template registration, skull stripping, "
            "96^3 resizing, and nonzero intensity normalization. Set false only when the "
            "input has already been registered and skull stripped for BrainIAC."
        ),
    )
    saliency_layer: int = Field(
        default=-1,
        ge=-BRAINIAC_NUM_LAYERS,
        lt=BRAINIAC_NUM_LAYERS,
        description="Transformer layer used for attention saliency; -1 selects the final layer.",
    )
    normalize_features: bool = Field(
        default=False,
        description="L2-normalize each 768-dimensional feature vector before writing CSV.",
    )
    max_concurrency: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "Maximum parallel preprocessing cases. Model inference remains serialized "
            "to prevent duplicate GPU memory pressure."
        ),
    )

    @model_validator(mode="after")
    def _require_exactly_one_input_shape(self) -> "BrainIACArgs":
        has_single = bool(self.image_path)
        has_batch = bool(self.image_paths)
        if has_single == has_batch:
            raise ValueError("Provide exactly one of image_path or image_paths.")
        return self


class BrainIACTool:
    name = "brainiac"
    model = "BrainIAC ViT-B"

    @staticmethod
    def _case_names(paths: Sequence[Path]) -> List[str]:
        counts: Dict[str, int] = {}
        names: List[str] = []
        for path in paths:
            base = safe_slug(_nifti_stem(path), "brain-mri")
            counts[base] = counts.get(base, 0) + 1
            names.append(base if counts[base] == 1 else f"{base}-{counts[base]}")
        return names

    @staticmethod
    def _normalize_saliency_layer(layer: int) -> int:
        normalized = layer + BRAINIAC_NUM_LAYERS if layer < 0 else layer
        if normalized < 0 or normalized >= BRAINIAC_NUM_LAYERS:
            raise ValueError(
                f"BrainIAC saliency_layer must select one of {BRAINIAC_NUM_LAYERS} layers."
            )
        return normalized

    def _prepare_cases(
        self,
        paths: Sequence[Path],
        *,
        preprocess: bool,
        output_dir: Path,
        max_concurrency: int,
    ) -> tuple[List[_PreparedCase], List[Dict[str, Any]], List[str]]:
        names = self._case_names(paths)
        prepared: List[_PreparedCase] = []
        failures: List[Dict[str, Any]] = []
        errors: List[str] = []

        def prepare(index: int) -> _PreparedCase:
            return _prepare_case(
                index,
                paths[index],
                case_name=names[index],
                preprocess=preprocess,
                output_dir=output_dir,
            )

        if max_concurrency == 1 or len(paths) == 1:
            futures = [(index, None) for index in range(len(paths))]
            for index, _unused in futures:
                try:
                    prepared.append(prepare(index))
                except Exception as exc:
                    message = f"{names[index]}: preprocessing failed: {exc}"
                    errors.append(message)
                    failures.append(
                        {
                            "case": names[index],
                            "source_name": paths[index].name,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
        else:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                future_by_index = {
                    executor.submit(prepare, index): index
                    for index in range(len(paths))
                }
                for future in as_completed(future_by_index):
                    index = future_by_index[future]
                    try:
                        prepared.append(future.result())
                    except Exception as exc:
                        message = f"{names[index]}: preprocessing failed: {exc}"
                        errors.append(message)
                        failures.append(
                            {
                                "case": names[index],
                                "source_name": paths[index].name,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
        prepared.sort(key=lambda item: item.index)
        failures.sort(key=lambda item: names.index(str(item["case"])))
        return prepared, failures, errors

    def _run_sync(
        self,
        paths: Sequence[Path],
        *,
        operation: BrainIACOperation,
        preprocess: bool,
        saliency_layer: int,
        normalize_features: bool,
        max_concurrency: int,
    ) -> TaskResult:
        output_dir = get_run_dir(self.name, persist=True)
        layer = self._normalize_saliency_layer(saliency_layer)
        model = _ensure_model_loaded()
        prepared, case_results, errors = self._prepare_cases(
            paths,
            preprocess=preprocess,
            output_dir=output_dir,
            max_concurrency=max_concurrency,
        )

        feature_rows: List[Dict[str, Any]] = []
        saliency_paths: List[str] = []
        preview_paths: List[str] = []
        preprocessed_paths: List[str] = []

        for case in prepared:
            try:
                embedding, saliency = _forward_case(
                    model,
                    case.tensor,
                    operation=operation,
                    saliency_layer=layer,
                )
                result: Dict[str, Any] = {
                    "case": case.case_name,
                    "source_name": case.source_path.name,
                    "status": "ok",
                    "preprocessed": preprocess,
                }
                if case.preprocessed_path is not None:
                    preprocessed_paths.append(str(case.preprocessed_path))

                if embedding is not None:
                    if normalize_features:
                        norm = float(np.linalg.norm(embedding))
                        if norm <= 1e-12:
                            raise RuntimeError("BrainIAC produced a zero-norm feature vector.")
                        embedding = embedding / norm
                    row: Dict[str, Any] = {
                        "case": case.case_name,
                        "source_name": case.source_path.name,
                    }
                    row.update(
                        {
                            f"brainiac_{index}": float(value)
                            for index, value in enumerate(embedding)
                        }
                    )
                    feature_rows.append(row)
                    result["feature_dimensions"] = int(embedding.shape[0])

                if saliency is not None:
                    saliency_path = (
                        output_dir
                        / "saliency"
                        / f"{case.case_name}_attention_layer_{layer}.nii.gz"
                    )
                    preview_path = (
                        output_dir
                        / "previews"
                        / f"{case.case_name}_attention_layer_{layer}.png"
                    )
                    _save_nifti(
                        saliency,
                        path=saliency_path,
                        affine=case.affine,
                        header=case.header,
                    )
                    _write_saliency_preview(
                        case.tensor.squeeze(0).squeeze(0).numpy(),
                        saliency,
                        path=preview_path,
                        title=f"BrainIAC attention: {case.case_name}",
                    )
                    saliency_paths.append(str(saliency_path))
                    preview_paths.append(str(preview_path))
                    result.update(
                        saliency_layer=layer,
                        saliency_min=float(saliency.min()),
                        saliency_max=float(saliency.max()),
                    )
                case_results.append(result)
            except Exception as exc:
                message = f"{case.case_name}: inference failed: {exc}"
                errors.append(message)
                case_results.append(
                    {
                        "case": case.case_name,
                        "source_name": case.source_path.name,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        case_order = {name: index for index, name in enumerate(self._case_names(paths))}
        case_results.sort(key=lambda item: case_order.get(str(item.get("case")), len(case_order)))
        successful = sum(result.get("status") == "ok" for result in case_results)
        failed = len(case_results) - successful

        artifacts: Dict[str, Any] = {}
        outputs: Dict[str, Any] = {
            "text": (
                f"BrainIAC {operation} completed: {successful} succeeded, {failed} failed."
            ),
            "operation": operation,
            "model": "BrainIAC ViT-B",
            "model_input_shape": list(BRAINIAC_MODEL_SIZE),
            "feature_dimensions": BRAINIAC_HIDDEN_SIZE if feature_rows else 0,
            "saliency_layer": layer if operation in {"saliency", "both"} else None,
            "preprocess": preprocess,
            "requested": len(paths),
            "succeeded": successful,
            "failed": failed,
            "cases": case_results,
            "research_use_notice": (
                "BrainIAC embeddings and attention maps are research outputs, not "
                "clinical diagnoses."
            ),
        }

        if feature_rows:
            features = pd.DataFrame(feature_rows)
            feature_path = output_dir / "brainiac_features.csv"
            features.to_csv(feature_path, index=False)
            artifacts["files"] = [str(feature_path)]
            outputs["table_data"] = [
                {
                    "name": "brainiac_features",
                    "dataframe": features,
                    "artifact_path": str(feature_path),
                    "visibility": "user",
                    "metadata": {
                        "model": "BrainIAC ViT-B",
                        "feature_dimensions": BRAINIAC_HIDDEN_SIZE,
                        "normalized": normalize_features,
                        "preprocessed": preprocess,
                    },
                }
            ]
        if successful:
            artifacts["output_dir"] = str(output_dir)
        if saliency_paths:
            artifacts["nifti_paths"] = [*preprocessed_paths, *saliency_paths]
        elif preprocessed_paths:
            artifacts["nifti_paths"] = preprocessed_paths
        if preview_paths:
            artifacts["image_paths"] = preview_paths
            outputs["ui.image_path"] = preview_paths[0]

        status: Literal["ok", "partial", "error"]
        if successful == len(paths):
            status = "ok"
        elif successful:
            status = "partial"
        else:
            status = "error"
        return TaskResult(
            output=outputs,
            artifacts=artifacts,
            status=status,
            errors=errors,
        )

    async def run(
        self,
        *,
        paths: Sequence[str],
        operation: BrainIACOperation,
        preprocess: bool,
        saliency_layer: int,
        normalize_features: bool,
        max_concurrency: int,
    ) -> TaskResult:
        resolved = [Path(path).expanduser().resolve() for path in paths]
        if not resolved:
            return TaskResult(
                output="BrainIAC requires at least one NIfTI input.",
                status="error",
                errors=["No BrainIAC input files were provided."],
            )
        return await asyncio.to_thread(
            self._run_sync,
            resolved,
            operation=operation,
            preprocess=preprocess,
            saliency_layer=saliency_layer,
            normalize_features=normalize_features,
            max_concurrency=max_concurrency,
        )


_BRAINIAC_TOOL = BrainIACTool()


@toolify_agent(
    name="brainiac",
    description=(
        "Analyze registered 3D structural brain MRI NIfTI files with the BrainIAC "
        "(Brain Imaging Adaptive Core) ViT-B foundation model. Use `features` to produce "
        "one 768-dimensional embedding per scan, `saliency` to produce normalized 3D "
        "transformer-attention NIfTI maps and overlay previews, or `both` for both outputs. "
        "BrainIAC supports structural T1w, T2w, FLAIR, and T1CE MRI; it does not accept "
        "DICOM directories, CT, diffusion MRI, or fMRI. Set preprocess=true for raw NIfTI "
        "so the tool performs N4 correction, template registration, skull stripping, "
        "96^3 resizing, and nonzero intensity normalization. Attention maps are research "
        "visualizations, not segmentations or clinical diagnoses."
    ),
    args_schema=BrainIACArgs,
    timeout_s=86400,
)
async def brainiac_runner(
    image_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    operation: BrainIACOperation = "features",
    preprocess: bool = True,
    saliency_layer: int = -1,
    normalize_features: bool = False,
    max_concurrency: int = 1,
):
    paths: List[str] = []
    if image_paths:
        paths.extend(image_paths)
    if image_path:
        paths.append(image_path)
    paths = list(dict.fromkeys(paths))
    return await _BRAINIAC_TOOL.run(
        paths=paths,
        operation=operation,
        preprocess=preprocess,
        saliency_layer=saliency_layer,
        normalize_features=normalize_features,
        max_concurrency=max_concurrency,
    )
