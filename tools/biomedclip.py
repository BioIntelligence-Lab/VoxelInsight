from __future__ import annotations

import asyncio
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.state import TaskResult
from core.storage import get_run_dir
from tools.shared import toolify_agent


DEFAULT_MODEL_REF = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
MODEL_CONTEXT_LENGTH = 256
RASTER_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
CT_WINDOWS: Dict[str, tuple[float, float]] = {
    "soft_tissue": (40.0, 400.0),
    "lung": (-160.0, 1500.0),
    "brain": (40.0, 80.0),
    "bone": (400.0, 1800.0),
}


_BIOMEDCLIP_CTX: Optional[Dict[str, Any]] = None
_MODEL_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def configure_biomedclip_tool(
    *,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    model_ref: str = DEFAULT_MODEL_REF,
) -> None:

    global _BIOMEDCLIP_CTX
    normalized_device = (device or "auto").strip().lower()
    if normalized_device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    normalized_ref = str(model_ref).strip()
    if not normalized_ref:
        raise ValueError("model_ref cannot be empty")
    _BIOMEDCLIP_CTX = {
        "device_request": normalized_device,
        "cache_dir": str(Path(cache_dir).expanduser()) if cache_dir else None,
        "model_ref": normalized_ref,
        "model": None,
        "preprocess": None,
        "tokenizer": None,
        "torch": None,
        "device": None,
    }


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BiomedCLIP was configured for CUDA, but CUDA is unavailable.")
        return "cuda"
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("BiomedCLIP was configured for MPS, but MPS is unavailable.")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _ensure_model_loaded() -> Dict[str, Any]:
    global _BIOMEDCLIP_CTX
    if _BIOMEDCLIP_CTX is None:
        raise RuntimeError(
            "BiomedCLIP tool not configured. Call configure_biomedclip_tool(...) first."
        )
    if _BIOMEDCLIP_CTX.get("model") is not None:
        return _BIOMEDCLIP_CTX

    with _MODEL_LOAD_LOCK:
        if _BIOMEDCLIP_CTX.get("model") is not None:
            return _BIOMEDCLIP_CTX
        try:
            import open_clip
            import torch
        except Exception as exc:
            raise RuntimeError(
                "BiomedCLIP requires open_clip_torch and transformers. "
                f"Original import error: {type(exc).__name__}: {exc}"
            ) from exc

        device = _resolve_device(torch, str(_BIOMEDCLIP_CTX["device_request"]))
        model_ref = str(_BIOMEDCLIP_CTX["model_ref"])
        cache_dir = _BIOMEDCLIP_CTX.get("cache_dir")
        load_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        try:
            model, preprocess = open_clip.create_model_from_pretrained(
                model_ref,
                **load_kwargs,
            )
        except TypeError:
            # Older OpenCLIP versions may not expose cache_dir on this helper.
            model, preprocess = open_clip.create_model_from_pretrained(model_ref)
        tokenizer = open_clip.get_tokenizer(model_ref)
        model.eval()
        model.to(device)
        _BIOMEDCLIP_CTX.update(
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            open_clip_version=getattr(open_clip, "__version__", "unknown"),
            torch_version=getattr(torch, "__version__", "unknown"),
        )
    return _BIOMEDCLIP_CTX


@dataclass
class PreparedSource:
    source_index: int
    source_name: str
    source_kind: str
    images: List[Any]
    view_indices: List[Optional[int]]
    preprocessing: Dict[str, Any]


def _is_nifti(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _plane_axis(plane: str) -> int:
    return {"sagittal": 0, "coronal": 1, "axial": 2}[plane]


def _extract_plane_slice(data: Any, *, plane: str, index: int) -> np.ndarray:
    if plane == "sagittal":
        array = np.asanyarray(data[index, :, :])
    elif plane == "coronal":
        array = np.asanyarray(data[:, index, :])
    else:
        array = np.asanyarray(data[:, :, index])
    return np.rot90(np.asarray(array, dtype=np.float32))


def _evenly_spaced(indices: Sequence[int], count: int) -> List[int]:
    values = list(indices)
    if not values:
        return []
    if count >= len(values):
        return values
    positions = np.linspace(0, len(values) - 1, count)
    return list(dict.fromkeys(values[int(round(position))] for position in positions))


def _select_slice_indices(
    *,
    axis_size: int,
    selection: str,
    num_slices: int,
    explicit_indices: Optional[List[int]],
    mask_data: Any = None,
    plane: str = "axial",
) -> List[int]:
    if axis_size <= 0:
        raise ValueError("Volume has no selectable slices.")
    if explicit_indices:
        selected = list(dict.fromkeys(int(value) for value in explicit_indices))
        invalid = [value for value in selected if value < 0 or value >= axis_size]
        if invalid:
            raise ValueError(
                f"slice_indices outside valid range 0..{axis_size - 1}: {invalid}"
            )
        return selected

    if selection == "mask":
        if mask_data is None:
            raise ValueError("slice_selection='mask' requires mask_path or mask_paths.")
        candidates = [
            index
            for index in range(axis_size)
            if np.any(_extract_plane_slice(mask_data, plane=plane, index=index) > 0)
        ]
        if not candidates:
            raise ValueError("The supplied mask contains no nonzero slices in this plane.")
        return _evenly_spaced(candidates, num_slices)

    all_indices = list(range(axis_size))
    if selection == "center":
        count = min(num_slices, axis_size)
        start = max(0, (axis_size - count) // 2)
        return list(range(start, start + count))
    return _evenly_spaced(all_indices, num_slices)


def _crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    padding_fraction: float,
) -> tuple[np.ndarray, bool]:
    nonzero = np.argwhere(mask > 0)
    if nonzero.size == 0:
        return image, False
    y0, x0 = nonzero.min(axis=0)
    y1, x1 = nonzero.max(axis=0) + 1
    pad_y = int(math.ceil((y1 - y0) * padding_fraction))
    pad_x = int(math.ceil((x1 - x0) * padding_fraction))
    y0 = max(0, int(y0) - pad_y)
    x0 = max(0, int(x0) - pad_x)
    y1 = min(image.shape[0], int(y1) + pad_y)
    x1 = min(image.shape[1], int(x1) + pad_x)
    return image[y0:y1, x0:x1], True


def _window_parameters(
    *,
    modality: str,
    intensity_mode: str,
    window_preset: str,
    window_center: Optional[float],
    window_width: Optional[float],
) -> Optional[tuple[float, float, str]]:
    if (window_center is None) != (window_width is None):
        raise ValueError("window_center and window_width must be supplied together.")
    if window_center is not None and window_width is not None:
        if window_width <= 0:
            raise ValueError("window_width must be positive.")
        return float(window_center), float(window_width), "custom"
    if intensity_mode == "percentile":
        return None
    should_window = intensity_mode == "ct_window" or (
        intensity_mode == "auto" and modality == "ct" and window_preset != "none"
    )
    if not should_window:
        return None
    selected_preset = "soft_tissue" if window_preset == "auto" else window_preset
    if selected_preset == "none":
        if intensity_mode == "ct_window":
            selected_preset = "soft_tissue"
        else:
            return None
    center, width = CT_WINDOWS[selected_preset]
    return center, width, selected_preset


def _grayscale_to_pil(
    slices: Sequence[np.ndarray],
    *,
    modality: str,
    intensity_mode: str,
    window_preset: str,
    window_center: Optional[float],
    window_width: Optional[float],
    percentile_low: float,
    percentile_high: float,
    invert: bool = False,
) -> tuple[List[Any], Dict[str, Any]]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("BiomedCLIP image preprocessing requires Pillow.") from exc

    if not slices:
        raise ValueError("No image slices were prepared.")
    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError(
            "percentile_low and percentile_high must satisfy 0 <= low < high <= 100."
        )
    window = _window_parameters(
        modality=modality,
        intensity_mode=intensity_mode,
        window_preset=window_preset,
        window_center=window_center,
        window_width=window_width,
    )
    finite_values = [array[np.isfinite(array)] for array in slices]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        raise ValueError("Selected slices contain no finite pixel values.")

    if window is not None:
        center, width, selected_preset = window
        low = center - width / 2.0
        high = center + width / 2.0
        method = "ct_window"
    else:
        concatenated = np.concatenate([values.reshape(-1) for values in finite_values])
        low, high = np.percentile(
            concatenated,
            [percentile_low, percentile_high],
        ).astype(float)
        selected_preset = "none"
        method = "percentile"
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = min(float(np.min(values)) for values in finite_values)
        high = max(float(np.max(values)) for values in finite_values)
    if high <= low:
        high = low + 1.0

    images: List[Any] = []
    for array in slices:
        clean = np.nan_to_num(array, nan=low, posinf=high, neginf=low)
        scaled = np.clip((clean - low) / (high - low), 0.0, 1.0)
        pixels = np.rint(scaled * 255.0).astype(np.uint8)
        if invert:
            pixels = 255 - pixels
        images.append(Image.fromarray(pixels, mode="L").convert("RGB"))
    return images, {
        "intensity_method": method,
        "window_preset": selected_preset,
        "intensity_low": float(low),
        "intensity_high": float(high),
        "inverted": bool(invert),
    }


def _load_mask_nifti(
    mask_path: Optional[str],
    expected_shape: Sequence[int],
    expected_affine: np.ndarray,
) -> Any:
    if not mask_path:
        return None
    path = Path(mask_path).expanduser()
    if not path.is_file() or not _is_nifti(path):
        raise ValueError("Masks for volumetric BiomedCLIP input must be NIfTI files.")
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("NIfTI processing requires nibabel.") from exc
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3:
        raise ValueError("Mask NIfTI must be three-dimensional.")
    if tuple(image.shape) != tuple(expected_shape):
        raise ValueError(
            f"Mask shape {tuple(image.shape)} does not match image shape {tuple(expected_shape)}."
        )
    if not np.allclose(image.affine, expected_affine, rtol=1e-4, atol=1e-3):
        raise ValueError(
            "Mask affine does not match the image affine after canonical reorientation."
        )
    return image.dataobj


def _prepare_nifti(
    path: Path,
    *,
    source_index: int,
    mask_path: Optional[str],
    plane: str,
    slice_selection: str,
    slice_indices: Optional[List[int]],
    num_slices: int,
    crop_to_mask: bool,
    mask_padding: float,
    modality: str,
    intensity_mode: str,
    window_preset: str,
    window_center: Optional[float],
    window_width: Optional[float],
    percentile_low: float,
    percentile_high: float,
) -> PreparedSource:
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("NIfTI processing requires nibabel.") from exc
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3:
        raise ValueError(
            f"BiomedCLIP expects a 3D NIfTI volume, received shape {tuple(image.shape)}."
        )
    mask_data = _load_mask_nifti(mask_path, image.shape, image.affine)
    axis = _plane_axis(plane)
    selected = _select_slice_indices(
        axis_size=int(image.shape[axis]),
        selection=slice_selection,
        num_slices=num_slices,
        explicit_indices=slice_indices,
        mask_data=mask_data,
        plane=plane,
    )
    arrays: List[np.ndarray] = []
    cropped_slices: List[int] = []
    empty_mask_slices: List[int] = []
    for index in selected:
        array = _extract_plane_slice(image.dataobj, plane=plane, index=index)
        if crop_to_mask:
            if mask_data is None:
                raise ValueError("crop_to_mask=true requires mask_path or mask_paths.")
            mask_slice = _extract_plane_slice(mask_data, plane=plane, index=index)
            array, cropped = _crop_to_mask(
                array,
                mask_slice,
                padding_fraction=mask_padding,
            )
            if cropped:
                cropped_slices.append(index)
            else:
                empty_mask_slices.append(index)
        arrays.append(array)
    effective_modality = "other" if modality == "auto" else modality
    images, intensity_metadata = _grayscale_to_pil(
        arrays,
        modality=effective_modality,
        intensity_mode=intensity_mode,
        window_preset=window_preset,
        window_center=window_center,
        window_width=window_width,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    return PreparedSource(
        source_index=source_index,
        source_name=path.name,
        source_kind="nifti_volume",
        images=images,
        view_indices=[int(index) for index in selected],
        preprocessing={
            "canonical_orientation": "RAS",
            "plane": plane,
            "slice_selection": "explicit" if slice_indices else slice_selection,
            "slice_indices": selected,
            "volume_shape": [int(value) for value in image.shape],
            "modality": effective_modality,
            "crop_to_mask": crop_to_mask,
            "cropped_slice_indices": cropped_slices,
            "empty_mask_slice_indices": empty_mask_slices,
            "mask_padding": mask_padding if crop_to_mask else 0.0,
            **intensity_metadata,
        },
    )


def _dicom_sort_key(dataset: Any, path: Path) -> tuple[Any, ...]:
    position = getattr(dataset, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        try:
            coordinates = np.asarray(position[:3], dtype=float)
            orientation = getattr(dataset, "ImageOrientationPatient", None)
            if orientation is not None and len(orientation) >= 6:
                row_direction = np.asarray(orientation[:3], dtype=float)
                column_direction = np.asarray(orientation[3:6], dtype=float)
                normal = np.cross(row_direction, column_direction)
                return (0, float(np.dot(coordinates, normal)), path.name)
            return (0, float(coordinates[2]), path.name)
        except (TypeError, ValueError):
            pass
    for priority, attribute in ((1, "SliceLocation"), (2, "InstanceNumber")):
        value = getattr(dataset, attribute, None)
        if value is not None:
            try:
                return (priority, float(value), path.name)
            except (TypeError, ValueError):
                pass
    return (3, 0.0, path.name)


def _dicom_pixels(dataset: Any) -> np.ndarray:
    array = np.asarray(dataset.pixel_array, dtype=np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    return array * slope + intercept


def _prepare_dicom_directory(
    path: Path,
    *,
    source_index: int,
    plane: str,
    slice_selection: str,
    slice_indices: Optional[List[int]],
    num_slices: int,
    modality: str,
    intensity_mode: str,
    window_preset: str,
    window_center: Optional[float],
    window_width: Optional[float],
    percentile_low: float,
    percentile_high: float,
) -> PreparedSource:
    if plane != "axial":
        raise ValueError(
            "DICOM directory inputs currently support axial source slices only. "
            "Convert to NIfTI for coronal or sagittal reslicing."
        )
    if slice_selection == "mask":
        raise ValueError("Mask-based slice selection currently requires a NIfTI input.")
    try:
        import pydicom
    except Exception as exc:
        raise RuntimeError("DICOM processing requires pydicom.") from exc

    candidates: List[tuple[Any, Path]] = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        try:
            header = pydicom.dcmread(str(candidate), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(header, "Rows", None) and getattr(header, "Columns", None):
            candidates.append((header, candidate))
    if not candidates:
        raise ValueError(f"No readable DICOM image instances found in directory: {path}")
    candidates.sort(key=lambda item: _dicom_sort_key(item[0], item[1]))
    selected = _select_slice_indices(
        axis_size=len(candidates),
        selection=slice_selection,
        num_slices=num_slices,
        explicit_indices=slice_indices,
    )
    arrays: List[np.ndarray] = []
    invert = False
    detected_modality = ""
    for index in selected:
        dataset = pydicom.dcmread(str(candidates[index][1]), force=True)
        pixels = _dicom_pixels(dataset)
        if pixels.ndim != 2:
            raise ValueError(
                "DICOM directory contains a non-2D instance; use a single multi-frame "
                "DICOM path for that object."
            )
        arrays.append(pixels)
        detected_modality = detected_modality or str(
            getattr(dataset, "Modality", "") or ""
        ).lower()
        invert = invert or str(
            getattr(dataset, "PhotometricInterpretation", "") or ""
        ).upper() == "MONOCHROME1"
    effective_modality = modality
    if effective_modality == "auto":
        effective_modality = "ct" if detected_modality == "ct" else (
            "mr" if detected_modality == "mr" else "other"
        )
    images, intensity_metadata = _grayscale_to_pil(
        arrays,
        modality=effective_modality,
        intensity_mode=intensity_mode,
        window_preset=window_preset,
        window_center=window_center,
        window_width=window_width,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        invert=invert,
    )
    return PreparedSource(
        source_index=source_index,
        source_name=path.name,
        source_kind="dicom_series",
        images=images,
        view_indices=[int(index) for index in selected],
        preprocessing={
            "plane": "axial",
            "slice_selection": "explicit" if slice_indices else slice_selection,
            "slice_indices": selected,
            "series_instance_count": len(candidates),
            "modality": effective_modality,
            "crop_to_mask": False,
            **intensity_metadata,
        },
    )


def _prepare_dicom_file(
    path: Path,
    *,
    source_index: int,
    plane: str,
    slice_selection: str,
    slice_indices: Optional[List[int]],
    num_slices: int,
    modality: str,
    intensity_mode: str,
    window_preset: str,
    window_center: Optional[float],
    window_width: Optional[float],
    percentile_low: float,
    percentile_high: float,
) -> PreparedSource:
    if plane != "axial":
        raise ValueError(
            "Multi-frame DICOM inputs currently support frame sampling only. "
            "Convert to NIfTI for coronal or sagittal reslicing."
        )
    if slice_selection == "mask":
        raise ValueError("Mask-based slice selection currently requires a NIfTI input.")
    try:
        import pydicom
    except Exception as exc:
        raise RuntimeError("DICOM processing requires pydicom.") from exc
    dataset = pydicom.dcmread(str(path), force=True)
    pixels = _dicom_pixels(dataset)
    if pixels.ndim == 2:
        frames = [pixels]
    elif pixels.ndim == 3:
        frames = [pixels[index] for index in range(pixels.shape[0])]
    else:
        raise ValueError(f"Unsupported DICOM pixel shape: {tuple(pixels.shape)}")
    selected = _select_slice_indices(
        axis_size=len(frames),
        selection=slice_selection,
        num_slices=num_slices,
        explicit_indices=slice_indices,
    )
    detected = str(getattr(dataset, "Modality", "") or "").lower()
    effective_modality = modality
    if effective_modality == "auto":
        effective_modality = "ct" if detected == "ct" else (
            "mr" if detected == "mr" else "other"
        )
    invert = str(
        getattr(dataset, "PhotometricInterpretation", "") or ""
    ).upper() == "MONOCHROME1"
    images, intensity_metadata = _grayscale_to_pil(
        [frames[index] for index in selected],
        modality=effective_modality,
        intensity_mode=intensity_mode,
        window_preset=window_preset,
        window_center=window_center,
        window_width=window_width,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        invert=invert,
    )
    return PreparedSource(
        source_index=source_index,
        source_name=path.name,
        source_kind="dicom_image" if len(frames) == 1 else "dicom_multiframe",
        images=images,
        view_indices=[int(index) for index in selected],
        preprocessing={
            "plane": "axial",
            "slice_selection": "explicit" if slice_indices else slice_selection,
            "slice_indices": selected,
            "frame_count": len(frames),
            "modality": effective_modality,
            "crop_to_mask": False,
            **intensity_metadata,
        },
    )


def _prepare_raster(path: Path, *, source_index: int) -> PreparedSource:
    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise RuntimeError("Raster image processing requires Pillow.") from exc
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB").copy()
        original_size = [int(value) for value in opened.size]
    return PreparedSource(
        source_index=source_index,
        source_name=path.name,
        source_kind="raster_image",
        images=[image],
        view_indices=[None],
        preprocessing={
            "original_size": original_size,
            "color_mode": "RGB",
            "model_preprocess": "official BiomedCLIP validation transform",
        },
    )


def _prepare_source(
    path_value: str,
    *,
    source_index: int,
    mask_path: Optional[str],
    plane: str,
    slice_selection: str,
    slice_indices: Optional[List[int]],
    num_slices: int,
    crop_to_mask: bool,
    mask_padding: float,
    modality: str,
    intensity_mode: str,
    window_preset: str,
    window_center: Optional[float],
    window_width: Optional[float],
    percentile_low: float,
    percentile_high: float,
) -> PreparedSource:
    path = Path(path_value).expanduser()
    if not path.exists():
        raise ValueError(f"BiomedCLIP input does not exist: {path}")
    if path.is_dir():
        if mask_path or crop_to_mask:
            raise ValueError("Masks are currently supported only for NIfTI volume inputs.")
        return _prepare_dicom_directory(
            path,
            source_index=source_index,
            plane=plane,
            slice_selection=slice_selection,
            slice_indices=slice_indices,
            num_slices=num_slices,
            modality=modality,
            intensity_mode=intensity_mode,
            window_preset=window_preset,
            window_center=window_center,
            window_width=window_width,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
    if _is_nifti(path):
        return _prepare_nifti(
            path,
            source_index=source_index,
            mask_path=mask_path,
            plane=plane,
            slice_selection=slice_selection,
            slice_indices=slice_indices,
            num_slices=num_slices,
            crop_to_mask=crop_to_mask,
            mask_padding=mask_padding,
            modality=modality,
            intensity_mode=intensity_mode,
            window_preset=window_preset,
            window_center=window_center,
            window_width=window_width,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
    if path.suffix.lower() in RASTER_SUFFIXES:
        if mask_path or crop_to_mask:
            raise ValueError("Masks are currently supported only for NIfTI volume inputs.")
        return _prepare_raster(path, source_index=source_index)
    if mask_path or crop_to_mask:
        raise ValueError("Masks are currently supported only for NIfTI volume inputs.")
    return _prepare_dicom_file(
        path,
        source_index=source_index,
        plane=plane,
        slice_selection=slice_selection,
        slice_indices=slice_indices,
        num_slices=num_slices,
        modality=modality,
        intensity_mode=intensity_mode,
        window_preset=window_preset,
        window_center=window_center,
        window_width=window_width,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )


def _l2_normalize(features: Any, torch: Any) -> Any:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _encode_images(images: Sequence[Any], *, batch_size: int) -> np.ndarray:
    context = _ensure_model_loaded()
    torch = context["torch"]
    model = context["model"]
    preprocess = context["preprocess"]
    device = context["device"]
    encoded: List[np.ndarray] = []
    with _INFERENCE_LOCK, torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            tensor = torch.stack([preprocess(image) for image in batch_images]).to(device)
            features = _l2_normalize(model.encode_image(tensor), torch)
            encoded.append(features.detach().float().cpu().numpy())
    return np.concatenate(encoded, axis=0).astype(np.float32)


def _format_prompts(prompts: Sequence[str], template: str) -> List[str]:
    if not prompts:
        raise ValueError("text_prompts is required for action='score_text'.")
    if len(prompts) > 256:
        raise ValueError("At most 256 text prompts may be scored in one call.")
    clean = [str(prompt).strip() for prompt in prompts]
    if any(not prompt for prompt in clean):
        raise ValueError("text_prompts cannot contain empty values.")
    if not template:
        return clean
    if "{label}" not in template and "{}" not in template:
        raise ValueError("text_template must contain either '{label}' or '{}'.")
    try:
        return [
            template.format(label=prompt)
            if "{label}" in template
            else template.format(prompt)
            for prompt in clean
        ]
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Invalid text_template: {exc}") from exc


def _encode_text(prompts: Sequence[str]) -> tuple[np.ndarray, float]:
    context = _ensure_model_loaded()
    torch = context["torch"]
    model = context["model"]
    tokenizer = context["tokenizer"]
    device = context["device"]
    try:
        tokens = tokenizer(list(prompts), context_length=MODEL_CONTEXT_LENGTH)
    except TypeError:
        tokens = tokenizer(list(prompts))
    if hasattr(tokens, "to"):
        tokens = tokens.to(device)
    elif isinstance(tokens, dict):
        tokens = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in tokens.items()
        }
    with _INFERENCE_LOCK, torch.inference_mode():
        try:
            features = model.encode_text(tokens)
        except TypeError:
            features = model.encode_text(**tokens)
        features = _l2_normalize(features, torch)
        logit_scale_value = getattr(model, "logit_scale", None)
        logit_scale = (
            float(logit_scale_value.exp().detach().float().cpu())
            if logit_scale_value is not None
            else 100.0
        )
    return features.detach().float().cpu().numpy().astype(np.float32), logit_scale


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / max(float(np.sum(exponentials)), 1e-12)


def _source_feature_blocks(
    sources: Sequence[PreparedSource],
    slice_features: np.ndarray,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    blocks: List[np.ndarray] = []
    aggregate: List[np.ndarray] = []
    offset = 0
    for source in sources:
        block = slice_features[offset : offset + len(source.images)]
        offset += len(source.images)
        blocks.append(block)
        aggregate.append(_normalize_vector(np.mean(block, axis=0)))
    return blocks, aggregate


def _score_source_against_text(
    *,
    block: np.ndarray,
    aggregate: np.ndarray,
    text_features: np.ndarray,
    aggregation: str,
    aggregation_top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    per_view = block @ text_features.T
    best_positions = np.argmax(per_view, axis=0)
    if aggregation == "max_similarity":
        scores = np.max(per_view, axis=0)
    elif aggregation == "topk_mean_similarity":
        count = min(aggregation_top_k, per_view.shape[0])
        scores = np.mean(np.sort(per_view, axis=0)[-count:, :], axis=0)
    else:
        scores = aggregate @ text_features.T
    return np.asarray(scores, dtype=np.float32), best_positions


class BiomedCLIPArgs(BaseModel):
    action: Literal["embed", "score_text", "retrieve_similar"] = Field(
        ...,
        description=(
            "Operation to perform: extract embeddings, score explicit biomedical text "
            "prompts, or rank the supplied images/volumes by image-image similarity."
        ),
    )
    image_path: Optional[str] = Field(
        default=None,
        description=(
            "One registered raster image, NIfTI volume, DICOM object, or DICOM series "
            "directory. Pass its exact artifact_id; middleware resolves it to a host path."
        ),
    )
    image_paths: Optional[List[str]] = Field(
        default=None,
        description=(
            "Multiple registered raster images, NIfTI volumes, DICOM objects, or DICOM "
            "series directories. Pass exact artifact_ids."
        ),
    )
    text_prompts: Optional[List[str]] = Field(
        default=None,
        description=(
            "Explicit candidate labels or biomedical descriptions for score_text. "
            "Scores are relative similarities, not calibrated diagnostic confidence."
        ),
    )
    text_template: str = Field(
        default="this is a photo of {label}",
        description="Prompt template containing '{label}' or '{}'; empty means use prompts verbatim.",
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum ranked results per query.")
    batch_size: int = Field(default=16, ge=1, le=128, description="Image inference batch size.")
    plane: Literal["axial", "coronal", "sagittal"] = Field(
        default="axial",
        description="Plane used when sampling a NIfTI volume.",
    )
    slice_selection: Literal["uniform", "center", "mask"] = Field(
        default="uniform",
        description="How slices are selected when explicit slice_indices are not supplied.",
    )
    slice_indices: Optional[List[int]] = Field(
        default=None,
        description="Optional exact zero-based slice/frame indices applied to every volume input.",
    )
    num_slices: int = Field(
        default=9,
        ge=1,
        le=64,
        description="Number of slices sampled from each volume when indices are not explicit.",
    )
    mask_path: Optional[str] = Field(
        default=None,
        description="One registered NIfTI mask for a single NIfTI image input.",
    )
    mask_paths: Optional[List[Optional[str]]] = Field(
        default=None,
        description="NIfTI masks aligned one-to-one with image_paths; use null-equivalent omission for unmasked inputs.",
    )
    crop_to_mask: bool = Field(
        default=False,
        description="Crop each selected NIfTI slice to its nonzero mask bounding box.",
    )
    mask_padding: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fractional padding added around a mask crop.",
    )
    modality: Literal["auto", "ct", "mr", "other"] = Field(
        default="auto",
        description=(
            "Modality used for intensity preprocessing. DICOM can infer CT/MR; NIfTI "
            "defaults to 'other' unless specified."
        ),
    )
    intensity_mode: Literal["auto", "ct_window", "percentile"] = Field(
        default="auto",
        description="Use CT windowing when appropriate or percentile normalization.",
    )
    window_preset: Literal["auto", "none", "soft_tissue", "lung", "brain", "bone"] = Field(
        default="auto",
        description="CT display window preset. Auto uses soft tissue for known CT inputs.",
    )
    window_center: Optional[float] = Field(default=None, description="Custom CT window center.")
    window_width: Optional[float] = Field(default=None, gt=0, description="Custom CT window width.")
    percentile_low: float = Field(default=0.5, ge=0.0, le=100.0)
    percentile_high: float = Field(default=99.5, ge=0.0, le=100.0)
    volume_aggregation: Literal[
        "mean_embedding", "max_similarity", "topk_mean_similarity"
    ] = Field(
        default="mean_embedding",
        description=(
            "How slice-level text similarities are combined. Non-mean strategies apply "
            "only to score_text; embeddings and image-image retrieval use normalized means."
        ),
    )
    aggregation_top_k: int = Field(
        default=3,
        ge=1,
        le=64,
        description="Number of highest-scoring slices averaged by topk_mean_similarity.",
    )
    include_slice_embeddings: bool = Field(
        default=True,
        description="Write a slice-level embedding CSV in addition to aggregate embeddings.",
    )


class BiomedCLIPAgent:
    name = "biomedclip"
    model = None

    def run(
        self,
        *,
        action: str,
        image_path: Optional[str],
        image_paths: Optional[List[str]],
        text_prompts: Optional[List[str]],
        text_template: str,
        top_k: int,
        batch_size: int,
        plane: str,
        slice_selection: str,
        slice_indices: Optional[List[int]],
        num_slices: int,
        mask_path: Optional[str],
        mask_paths: Optional[List[Optional[str]]],
        crop_to_mask: bool,
        mask_padding: float,
        modality: str,
        intensity_mode: str,
        window_preset: str,
        window_center: Optional[float],
        window_width: Optional[float],
        percentile_low: float,
        percentile_high: float,
        volume_aggregation: str,
        aggregation_top_k: int,
        include_slice_embeddings: bool,
    ) -> TaskResult:
        inputs = [str(value) for value in (image_paths or []) if str(value).strip()]
        if image_path:
            inputs.append(str(image_path))
        inputs = list(dict.fromkeys(inputs))
        if not inputs:
            raise ValueError("Provide image_path or image_paths.")
        if percentile_low >= percentile_high:
            raise ValueError("percentile_low must be less than percentile_high.")
        if action != "score_text" and volume_aggregation != "mean_embedding":
            raise ValueError(
                "max_similarity and topk_mean_similarity are valid only for score_text."
            )
        if mask_path and mask_paths:
            raise ValueError("Provide mask_path or mask_paths, not both.")
        if mask_path:
            if len(inputs) != 1:
                raise ValueError("mask_path can be used only with one image input.")
            aligned_masks: List[Optional[str]] = [mask_path]
        elif mask_paths is not None:
            if len(mask_paths) != len(inputs):
                raise ValueError("mask_paths must align one-to-one with image inputs.")
            aligned_masks = [str(value) if value else None for value in mask_paths]
        else:
            aligned_masks = [None] * len(inputs)

        sources: List[PreparedSource] = []
        preparation_errors: List[str] = []
        for source_index, (input_path, aligned_mask) in enumerate(
            zip(inputs, aligned_masks)
        ):
            try:
                source = _prepare_source(
                    input_path,
                    source_index=source_index,
                    mask_path=aligned_mask,
                    plane=plane,
                    slice_selection=slice_selection,
                    slice_indices=slice_indices,
                    num_slices=num_slices,
                    crop_to_mask=crop_to_mask,
                    mask_padding=mask_padding,
                    modality=modality,
                    intensity_mode=intensity_mode,
                    window_preset=window_preset,
                    window_center=window_center,
                    window_width=window_width,
                    percentile_low=percentile_low,
                    percentile_high=percentile_high,
                )
                sources.append(source)
            except Exception as exc:
                preparation_errors.append(
                    f"Input {source_index + 1} ({Path(input_path).name}): "
                    f"{type(exc).__name__}: {exc}"
                )
        if not sources:
            raise RuntimeError("No inputs could be prepared. " + "; ".join(preparation_errors))
        if action == "retrieve_similar" and len(sources) < 2:
            raise ValueError("retrieve_similar requires at least two successfully prepared inputs.")

        all_images = [image for source in sources for image in source.images]
        slice_features = _encode_images(all_images, batch_size=batch_size)
        feature_blocks, aggregate_features = _source_feature_blocks(
            sources,
            slice_features,
        )
        embedding_dim = int(slice_features.shape[1])
        result_rows: List[Dict[str, Any]] = []
        formatted_prompts: List[str] = []
        logit_scale: Optional[float] = None

        if action == "embed":
            for source, aggregate in zip(sources, aggregate_features):
                result_rows.append(
                    {
                        "source_index": source.source_index,
                        "source_name": source.source_name,
                        "source_kind": source.source_kind,
                        "views_encoded": len(source.images),
                        "embedding_dimensions": embedding_dim,
                        "embedding_l2_norm": float(np.linalg.norm(aggregate)),
                    }
                )
        elif action == "score_text":
            raw_prompts = list(text_prompts or [])
            formatted_prompts = _format_prompts(raw_prompts, text_template)
            text_features, logit_scale = _encode_text(formatted_prompts)
            for source, block, aggregate in zip(
                sources,
                feature_blocks,
                aggregate_features,
            ):
                scores, best_positions = _score_source_against_text(
                    block=block,
                    aggregate=aggregate,
                    text_features=text_features,
                    aggregation=volume_aggregation,
                    aggregation_top_k=aggregation_top_k,
                )
                probabilities = _softmax(float(logit_scale) * scores)
                order = np.argsort(-scores)[: min(top_k, len(raw_prompts))]
                for rank, prompt_index in enumerate(order, start=1):
                    best_position = int(best_positions[prompt_index])
                    result_rows.append(
                        {
                            "source_index": source.source_index,
                            "source_name": source.source_name,
                            "source_kind": source.source_kind,
                            "rank": rank,
                            "prompt": raw_prompts[prompt_index],
                            "formatted_prompt": formatted_prompts[prompt_index],
                            "cosine_similarity": float(scores[prompt_index]),
                            "relative_probability": float(probabilities[prompt_index]),
                            "aggregation": volume_aggregation,
                            "best_view_index": source.view_indices[best_position],
                        }
                    )
        else:
            matrix = np.stack(aggregate_features)
            similarities = matrix @ matrix.T
            for query_position, source in enumerate(sources):
                candidate_positions = [
                    index for index in range(len(sources)) if index != query_position
                ]
                candidate_positions.sort(
                    key=lambda index: float(similarities[query_position, index]),
                    reverse=True,
                )
                for rank, candidate_position in enumerate(
                    candidate_positions[: min(top_k, len(candidate_positions))],
                    start=1,
                ):
                    candidate = sources[candidate_position]
                    result_rows.append(
                        {
                            "query_source_index": source.source_index,
                            "query_source_name": source.source_name,
                            "rank": rank,
                            "match_source_index": candidate.source_index,
                            "match_source_name": candidate.source_name,
                            "cosine_similarity": float(
                                similarities[query_position, candidate_position]
                            ),
                            "aggregation": "mean_embedding",
                        }
                    )

        results = pd.DataFrame(result_rows)
        aggregate_rows: List[Dict[str, Any]] = []
        for source, feature in zip(sources, aggregate_features):
            row: Dict[str, Any] = {
                "source_index": source.source_index,
                "source_name": source.source_name,
                "source_kind": source.source_kind,
                "views_encoded": len(source.images),
            }
            row.update(
                {
                    f"biomedclip_{index:04d}": float(value)
                    for index, value in enumerate(feature)
                }
            )
            aggregate_rows.append(row)
        aggregate_frame = pd.DataFrame(aggregate_rows)

        slice_rows: List[Dict[str, Any]] = []
        if include_slice_embeddings:
            offset = 0
            for source in sources:
                for view_position, view_index in enumerate(source.view_indices):
                    feature = slice_features[offset]
                    offset += 1
                    row = {
                        "source_index": source.source_index,
                        "source_name": source.source_name,
                        "source_kind": source.source_kind,
                        "view_position": view_position,
                        "view_index": view_index,
                    }
                    row.update(
                        {
                            f"biomedclip_{index:04d}": float(value)
                            for index, value in enumerate(feature)
                        }
                    )
                    slice_rows.append(row)

        output_root = get_run_dir(self.name, persist=True)
        results_path = output_root / "biomedclip_results.csv"
        aggregate_path = output_root / "biomedclip_embeddings.csv"
        manifest_path = output_root / "biomedclip_manifest.json"
        results.to_csv(results_path, index=False)
        aggregate_frame.to_csv(aggregate_path, index=False)
        output_files = [str(results_path), str(aggregate_path)]
        if include_slice_embeddings:
            slice_path = output_root / "biomedclip_slice_embeddings.csv"
            pd.DataFrame(slice_rows).to_csv(slice_path, index=False)
            output_files.append(str(slice_path))

        context = _ensure_model_loaded()
        manifest = {
            "schema_version": "voxelinsight.biomedclip.v1",
            "action": action,
            "model": {
                "model_ref": context["model_ref"],
                "device": context["device"],
                "embedding_dimensions": embedding_dim,
                "context_length": MODEL_CONTEXT_LENGTH,
                "open_clip_version": context.get("open_clip_version", "unknown"),
                "torch_version": context.get("torch_version", "unknown"),
            },
            "sources": [
                {
                    "source_index": source.source_index,
                    "source_name": source.source_name,
                    "source_kind": source.source_kind,
                    "views_encoded": len(source.images),
                    "preprocessing": source.preprocessing,
                }
                for source in sources
            ],
            "text": {
                "raw_prompts": list(text_prompts or []),
                "formatted_prompts": formatted_prompts,
                "template": text_template,
                "logit_scale": logit_scale,
            },
            "volume_aggregation": volume_aggregation,
            "aggregation_top_k": aggregation_top_k,
            "warnings": [
                "BiomedCLIP similarities are research outputs, not calibrated diagnostic confidence.",
                "Volume results are VoxelInsight slice aggregations; BiomedCLIP is a 2D model.",
            ],
            "preparation_errors": preparation_errors,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        output_files.append(str(manifest_path))

        status = "partial" if preparation_errors else "ok"
        summary = (
            f"BiomedCLIP {action} completed for {len(sources)} source(s) using "
            f"{len(all_images)} rendered view(s)."
        )
        return TaskResult(
            output={
                "text": summary,
                "action": action,
                "model_ref": context["model_ref"],
                "device": context["device"],
                "results": results,
                "embedding_dimensions": embedding_dim,
                "source_count": len(sources),
                "view_count": len(all_images),
                "warnings": manifest["warnings"],
            },
            artifacts={
                "files": output_files,
                "output_dir": str(output_root),
            },
            status=status,
            errors=preparation_errors,
        )


@toolify_agent(
    name="biomedclip",
    description=(
        "Run research-oriented BiomedCLIP image-text analysis on registered 2D biomedical "
        "images, NIfTI volumes, DICOM objects, or DICOM series directories. Supports "
        "512-dimensional embeddings, explicit text-prompt similarity ranking, and "
        "image-image retrieval. For volumes, deterministically samples axial/coronal/sagittal "
        "NIfTI slices or axial DICOM frames, supports CT windowing, percentile normalization, "
        "and optional NIfTI mask-based slice selection/cropping. Scores are relative semantic "
        "similarities for research, not calibrated diagnostic confidence. This tool does not "
        "segment images, generate reports, perform open-ended VQA, or natively model 3D volumes."
    ),
    args_schema=BiomedCLIPArgs,
    timeout_s=1800,
)
async def biomedclip_runner(
    action: str,
    image_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    text_prompts: Optional[List[str]] = None,
    text_template: str = "this is a photo of {label}",
    top_k: int = 5,
    batch_size: int = 16,
    plane: str = "axial",
    slice_selection: str = "uniform",
    slice_indices: Optional[List[int]] = None,
    num_slices: int = 9,
    mask_path: Optional[str] = None,
    mask_paths: Optional[List[Optional[str]]] = None,
    crop_to_mask: bool = False,
    mask_padding: float = 0.10,
    modality: str = "auto",
    intensity_mode: str = "auto",
    window_preset: str = "auto",
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
    volume_aggregation: str = "mean_embedding",
    aggregation_top_k: int = 3,
    include_slice_embeddings: bool = True,
) -> TaskResult:
    if _BIOMEDCLIP_CTX is None:
        raise RuntimeError(
            "BiomedCLIP tool not configured. Call configure_biomedclip_tool(...) first."
        )
    agent = BiomedCLIPAgent()
    return await asyncio.to_thread(
        agent.run,
        action=action,
        image_path=image_path,
        image_paths=image_paths,
        text_prompts=text_prompts,
        text_template=text_template,
        top_k=top_k,
        batch_size=batch_size,
        plane=plane,
        slice_selection=slice_selection,
        slice_indices=slice_indices,
        num_slices=num_slices,
        mask_path=mask_path,
        mask_paths=mask_paths,
        crop_to_mask=crop_to_mask,
        mask_padding=mask_padding,
        modality=modality,
        intensity_mode=intensity_mode,
        window_preset=window_preset,
        window_center=window_center,
        window_width=window_width,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        volume_aggregation=volume_aggregation,
        aggregation_top_k=aggregation_top_k,
        include_slice_embeddings=include_slice_embeddings,
    )
