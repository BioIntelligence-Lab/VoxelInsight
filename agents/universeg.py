# agents/universeg.py
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from core.state import Task, TaskResult, ConversationState

ALLOWED_IMG_EXTS = (".nii", ".nii.gz") 
DEFAULT_SIZE = (128, 128)              

def _is_nifti(path: Path) -> bool:
    s = path.name.lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")

def _expand_paths(maybe_paths) -> List[str]:
    """Accept str or list[str] and expands directories recursively to NIfTI files."""
    if not maybe_paths:
        return []
    if isinstance(maybe_paths, str):
        maybe_paths = [maybe_paths]
    out, seen = [], set()
    for raw in maybe_paths:
        p = Path(raw)
        if p.is_dir():
            for f in p.rglob("*"):
                if _is_nifti(f):
                    s = str(f)
                    if s not in seen:
                        seen.add(s); out.append(s)
        elif p.exists() and _is_nifti(p):
            s = str(p)
            if s not in seen:
                seen.add(s); out.append(s)
    return out

def _stem_key(p: Path) -> str:
    """Normalize stem for pairing by dropping extension and common mask suffixes."""
    name = p.name
    base = name[:-7] if name.lower().endswith(".nii.gz") else name[: name.rfind(".")] if "." in name else name
    for suf in ["_mask", "-mask", "_seg", "-seg", "_label", "-label"]:
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
            break
    return base.lower()

def _pair_support(images: List[str], masks: List[str]) -> List[Tuple[str, str]]:
    imap = {_stem_key(Path(i)): i for i in images}
    mmap = {_stem_key(Path(m)): m for m in masks}
    keys = sorted(set(imap) & set(mmap))
    return [(imap[k], mmap[k]) for k in keys]

def _load_nii(path: str) -> nib.Nifti1Image:
    return nib.load(path)

def _vol_to_slices(vol: np.ndarray) -> List[np.ndarray]:
    """Return list of axial slices (H,W) from a (H,W,Z) volume."""
    assert vol.ndim == 3, f"Expected 3D volume, got {vol.shape}"
    return [vol[:, :, z] for z in range(vol.shape[2])]

def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite([mn, mx]).all() or mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn + 1e-6)

def _resize_2d(arr: np.ndarray, size=(128, 128), mode="bilinear") -> np.ndarray:
    """Resize a single (H,W) slice to size using torch"""
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    out = F.interpolate(t, size=size, mode=mode, align_corners=False if mode == "bilinear" else None)
    return out.squeeze(0).squeeze(0).numpy()

def _save_mask_like(ref_img: nib.Nifti1Image, mask_vol: np.ndarray, out_path: str):
    """Save (H,W,Z) uint8 mask with the same affine/header as ref."""
    mask_vol = np.asarray(mask_vol)
    if mask_vol.dtype != np.uint8:
        mask_vol = (mask_vol > 0.5).astype(np.uint8)
    nii = nib.Nifti1Image(mask_vol, affine=ref_img.affine, header=ref_img.header)
    nib.save(nii, out_path)

def _pick_support_slices(img_path: str, msk_path: str, max_slices: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pick up to max_slices informative 2D slices from a support volume (mask > 0)."""
    I = _load_nii(img_path)
    M = _load_nii(msk_path)
    volI = np.asanyarray(I.dataobj)
    volM = np.asanyarray(M.dataobj)
    assert volI.shape == volM.shape, f"Support image/mask shape mismatch: {volI.shape} vs {volM.shape}"

    slices = []
    for z in range(volI.shape[2]):
        m = volM[:, :, z]
        if np.any(m > 0):
            slices.append((volI[:, :, z], (m > 0).astype(np.float32)))
        if len(slices) >= max_slices:
            break
    if not slices:
        mid = volI.shape[2] // 2
        slices = [(volI[:, :, mid], (volM[:, :, mid] > 0).astype(np.float32))]
    return slices

class UniversegAgent:
    """
    kwargs:
      - support_images: str or list[str]    # dirs or files (NIfTI)
      - support_masks:  str or list[str]    # dirs or files (NIfTI)
      - target_images:  str or list[str]    # dirs or files (NIfTI). If omitted, uses uploaded files.
      - max_support_slices: int = 10
      - threshold: float = 0.5
      - resize_to: tuple = (128,128)    
      - output_dir: str (default: ~/vi_universeg/run_<pid>)
    Artifacts:
      - 'segmentations': list[str]
      - 'segmentations_map': dict[target_path -> mask_path]
      - 'output_dir': str
    """
    name = "universeg"
    model = None  

    def __init__(self):
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from universeg import universeg  
            self._model = universeg(pretrained=True)

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        kw = task.kwargs or {}

        support_images = kw.get("support_images") 
        support_masks  = kw.get("support_masks") 
        target_images  = kw.get("target_images")  

        if not target_images and task.files:
            target_images = task.files

        support_images = _expand_paths(support_images)
        support_masks  = _expand_paths(support_masks)
        target_images  = _expand_paths(target_images)

        if not support_images or not support_masks:
            return TaskResult("Universeg: please provide support_images and support_masks (dirs or files).")

        if not target_images:
            return TaskResult("Universeg: please provide target_images (dirs or files), or upload NIfTI(s).")

        pairs = _pair_support(support_images, support_masks)
        if not pairs:
            return TaskResult(
                output="Universeg: could not pair any support images with masks. Ensure names match (e.g., case01.nii.gz ↔ case01_mask.nii.gz).",
                artifacts={"support_images": support_images, "support_masks": support_masks},
            )

        max_support_slices = int(kw.get("max_support_slices", 10))
        threshold = float(kw.get("threshold", 0.5))
        resize_to = tuple(kw.get("resize_to", DEFAULT_SIZE))

        # Output dir
        out_root = kw.get("output_dir")
        if not out_root:
            base = Path(tempfile.mkdtemp(prefix="vi_universeg_"))
            out_root = str(base / f"run_{os.getpid()}")
        Path(out_root).mkdir(parents=True, exist_ok=True)

        # build support tensors (B=1, S, 1, 128, 128)
        support_imgs_2d: List[np.ndarray] = []
        support_msks_2d: List[np.ndarray] = []
        for img_p, msk_p in pairs:
            chosen = _pick_support_slices(img_p, msk_p, max_slices=max_support_slices)
            for si, sm in chosen:
                si_r = _resize_2d(_normalize01(si), resize_to, mode="bilinear")
                sm_r = _resize_2d((sm > 0).astype(np.float32), resize_to, mode="nearest")
                support_imgs_2d.append(si_r)
                support_msks_2d.append(sm_r)

        if not support_imgs_2d:
            return TaskResult(output="Universeg: no informative support slices found (masks empty).")

        support_images_t = torch.from_numpy(
            np.stack(support_imgs_2d)[None, :, None, ...].astype(np.float32)
        )  # (1,S,1,H,W)
        support_labels_t = torch.from_numpy(
            np.stack(support_msks_2d)[None, :, None, ...].astype(np.float32)
        )  

        # Load model
        try:
            self._ensure_model()
        except Exception as e:
            return TaskResult(output=f"Universeg: failed to load model: {e}")

        model = self._model
        model.eval()

        seg_paths: List[str] = []
        seg_map: Dict[str, str] = {}

        for tpath in target_images:
            try:
                nii = _load_nii(tpath)
                vol = np.asanyarray(nii.dataobj)  # (H,W,Z)
                assert vol.ndim == 3, f"Target must be 3D NIfTI; got {vol.shape}"

                H, W, Z = vol.shape
                pred_stack = np.zeros((H, W, Z), dtype=np.float32)

                for z in range(Z):
                    sl = _normalize01(vol[:, :, z])                              # (H,W)
                    sl_r = _resize_2d(sl, resize_to, mode="bilinear")            # (128,128)
                    target_t = torch.from_numpy(sl_r[None, None, ...].astype(np.float32))  # (1,1,128,128)

                    with torch.no_grad():
                        pred = model(target_t, support_images_t, support_labels_t)  # torch (1,1,128,128)

                    # Resize back to original (H,W)
                    pred_up = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
                    pred_np = pred_up.squeeze(0).squeeze(0).cpu().numpy()       
                    pred_stack[:, :, z] = pred_np

                # Threshold to binary
                mask_bin = (pred_stack >= threshold).astype(np.uint8)

                # Save NIfTI aligned to target
                out_name = Path(tpath).stem
                if out_name.endswith(".nii"):
                    out_name = out_name[:-4]
                out_path = str(Path(out_root) / f"{out_name}_seg.nii.gz")
                _save_mask_like(nii, mask_bin, out_path)

                if os.path.exists(out_path):
                    seg_paths.append(out_path)
                    seg_map[tpath] = out_path
                else:
                    seg_map[tpath] = "ERROR: output not written"
            except Exception as e:
                seg_map[tpath] = f"ERROR: {e}"

        if not seg_paths:
            return TaskResult(
                output="Universeg: inference completed but produced no mask files.",
                artifacts={"output_dir": out_root, "segmentations_map": seg_map},
            )

        # Store in state and return
        state.memory["segmentations"] = seg_paths
        state.memory["segmentations_map"] = seg_map

        return TaskResult(
            output={
                "action": "inference",
                "model_name": "universeg(pretrained=True)",
                "output_dir": out_root,
                "count": len(seg_paths),
            },
            artifacts={
                "output_dir": out_root,
                "segmentations": seg_paths,
                "segmentations_map": seg_map,
                "support_pairs": pairs,
            },
        )
