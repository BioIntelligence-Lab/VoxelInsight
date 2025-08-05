# agents/viz_slider.py
import numpy as np
from pathlib import Path
import nibabel as nib
import plotly.graph_objects as go
from matplotlib import cm as mcm

from core.state import Task, TaskResult, ConversationState

class VizSliderAgent:
    """
    Slider-only visualization agent.

    Expected kwargs (all optional except image):
      - image_path: str (NIfTI path) OR image_array: np.ndarray (H,W,Z) or (H,W,Z,C)
      - mask_paths: list[str] OR mask_arrays: list[np.ndarray] (each mask H,W,Z or labelmap)
      - alpha: float in [0,1] (default 0.4)
      - cmaps: list[str] (matplotlib cmap names; optional, wraps if fewer than masks)
      - window_center: float (WL/WW windowing)
      - window_width: float
      - per_slice_minmax: bool (default True if no WL/WW provided)
      - interactive: bool (default True => slider; False => static mid slice)
      - max_dim: int (downsample longest side <= max_dim; default 512)

    It returns a Plotly Figure in TaskResult.output.
    """
    name  = "viz_slider"
    model = None  

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        try:
            kw = task.kwargs or {}
            image_path       = kw.get("image_path") or state.memory.get("image_path")
            image_array      = kw.get("image_array") or state.memory.get("image_array")
            mask_paths       = kw.get("mask_paths") or state.memory.get("mask_paths") \
                               or state.memory.get("segmentations") or []
            mask_arrays      = kw.get("mask_arrays") or state.memory.get("mask_arrays") or []
            alpha            = float(kw.get("alpha", 0.4))
            cmaps            = kw.get("cmaps")  # optional list[str]
            window_center    = kw.get("window_center", None)
            window_width     = kw.get("window_width", None)
            per_slice_minmax = bool(kw.get("per_slice_minmax", (window_center is None or window_width is None)))
            interactive      = bool(kw.get("interactive", True))
            max_dim          = int(kw.get("max_dim", 512))

            # Defaults if cmaps not provided
            DEFAULT_CMAPS = [
                "jet", "spring", "summer", "autumn", "winter",
                "cool", "hot", "viridis", "plasma", "magma", "cividis"
            ]
            if not cmaps:
                cmaps = DEFAULT_CMAPS

            # Load image volume 
            if image_array is not None:
                vol = self._ensure_3d(self._nan_clean(np.asarray(image_array)))
            elif image_path:
                vol = self._load_nifti(image_path)
            else:
                return TaskResult(output="VizSlider: Provide either 'image_path' or 'image_array'.")

            H, W, Z = vol.shape
            if Z < 1:
                return TaskResult(output=f"VizSlider: Volume has no slices (shape={vol.shape}).")

            # Prepare masks 
            mask_vols = self._prepare_masks(mask_paths, mask_arrays)  
            # Validate shape compatibility
            for mv in mask_vols:
                if mv.shape != vol.shape:
                    return TaskResult(
                        output=f"VizSlider: mask shape {mv.shape} != image shape {vol.shape}. "
                               "Resampling required."
                    )

            #  Map image to uint8 slices 
            if (window_center is not None) and (window_width is not None):
                base_slices = [
                    self._downsample2d(
                        self._window_to_uint8(vol[:, :, z], window_center, window_width), max_dim
                    )
                    for z in range(Z)
                ]
            else:
                base_slices = [
                    self._downsample2d(sl, max_dim)
                    for sl in self._per_slice_minmax_uint8(vol)
                ]

            # Downsample masks using the same stride approach
            if mask_vols:
                mask_slices_per_z = []
                for z in range(Z):
                    mslices = []
                    for mv in mask_vols:
                        sl = mv[:, :, z].astype(np.uint8)
                        sl = self._downsample2d(sl, max_dim)
                        mslices.append(sl)
                    mask_slices_per_z.append(mslices)
            else:
                mask_slices_per_z = [[] for _ in range(Z)]

            # Build RGBA frames 
            nz  = len(base_slices)
            mid = nz // 2

            frames = []
            for z in range(nz):
                rgba = self._rgba_stack_with_cmaps(base_slices[z], mask_slices_per_z[z], alpha, cmaps)
                frames.append(go.Frame(data=[go.Image(z=rgba)], name=str(z)))

            fig = go.Figure(
                data=[go.Image(z=self._rgba_stack_with_cmaps(base_slices[mid],
                                                             mask_slices_per_z[mid],
                                                             alpha, cmaps))],
                frames=frames
            )

            if interactive and nz > 1:
                fig.update_layout(
                    sliders=[{
                        "active": mid,
                        "currentvalue": {"prefix": "Slice "},
                        "steps": [
                            {"label": str(k), "method": "animate",
                             "args": [[str(k)], {"mode": "immediate",
                                                 "frame": {"duration": 0, "redraw": True},
                                                 "transition": {"duration": 0}}]}
                            for k in range(nz)
                        ]
                    }],
                    margin=dict(l=0, r=0, t=0, b=0)
                )
            else:
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

            arts = {
                "image_path": str(image_path) if image_path else None,
                "mask_paths": [str(p) for p in (mask_paths or [])],
                "num_slices": Z
            }
            return TaskResult(output=fig, artifacts=arts)

        except Exception as e:
            return TaskResult(output=f"VizSlider error: {e}")

    # Helpers
    def _ensure_3d(self, vol: np.ndarray) -> np.ndarray:
        """Accept (H,W,Z) or (H,W,Z,C) and return (H,W,Z)."""
        if vol.ndim == 4 and vol.shape[-1] > 1:
            return vol[..., 0]
        if vol.ndim == 4 and vol.shape[-1] == 1:
            return vol[..., 0]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume (H,W,Z); got {vol.shape}")
        return vol

    def _nan_clean(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _downsample2d(self, arr: np.ndarray, max_dim: int) -> np.ndarray:
        """Stride downsample to keep largest side <= max_dim."""
        h, w = arr.shape[:2]
        scale = max(h, w) / float(max_dim)
        if scale <= 1:
            return arr
        step = int(np.ceil(scale))
        return arr[::step, ::step] if arr.ndim == 2 else arr[::step, ::step, ...]

    def _window_to_uint8(self, arr: np.ndarray, center: float, width: float) -> np.ndarray:
        lo, hi = center - width/2.0, center + width/2.0
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-6)
        return (arr * 255).astype(np.uint8)

    def _per_slice_minmax_uint8(self, vol: np.ndarray):
        out = []
        for z in range(vol.shape[2]):
            sl = vol[:, :, z]
            mn, mx = np.nanmin(sl), np.nanmax(sl)
            if not np.isfinite([mn, mx]).all() or mx <= mn:
                out.append(np.zeros_like(sl, dtype=np.uint8))
            else:
                sl = (sl - mn) / (mx - mn + 1e-6)
                out.append((sl * 255).astype(np.uint8))
        return out

    def _load_nifti(self, path: str) -> np.ndarray:
        x = nib.load(str(Path(path))).get_fdata()
        return self._ensure_3d(self._nan_clean(x))

    def _to_mask_list(self, arr: np.ndarray):
        """
        Convert a 3D mask (binary or labelmap) to a list of binary masks.
        - If binary {0,1}, returns [mask].
        - If labelmap (ints), splits into one mask per label (excluding 0).
        """
        arr = np.asarray(arr)
        uniq = np.unique(arr[~np.isnan(arr)]) if np.isnan(arr).any() else np.unique(arr)
        if arr.dtype.kind in "iu":
            labels = [int(u) for u in uniq if u != 0]
            if len(labels) <= 1:
                return [(arr > 0).astype(np.uint8)]
            masks = []
            for lab in labels:
                masks.append((arr == lab).astype(np.uint8))
            return masks
        else:
            return [(arr > 0).astype(np.uint8)]

    def _prepare_masks(self, mask_paths, mask_arrays):
        """Load/normalize all masks into a list of binary mask volumes (H,W,Z)."""
        vols = []
        for p in (mask_paths or []):
            vols.append(self._load_nifti(p))
        for a in (mask_arrays or []):
            vols.append(self._ensure_3d(self._nan_clean(np.asarray(a))))
        out = []
        for v in vols:
            out.extend(self._to_mask_list(v))
        return out

    def _rgba_stack_with_cmaps(self, base_uint8: np.ndarray, mask_slices, alpha: float, cmap_names):
        """
        base_uint8: (H,W) uint8
        mask_slices: list of 2D binary masks aligned to base
        alpha: float 0..1
        cmap_names: list[str]
        Returns (H,W,4) uint8 RGBA composite
        """
        h, w = base_uint8.shape
        rgba = np.stack([base_uint8, base_uint8, base_uint8], axis=-1).astype(np.float32)

        if not mask_slices:
            a = np.full((h, w, 1), 255, dtype=np.uint8)
            return np.concatenate([rgba.astype(np.uint8), a], axis=-1)

        for i, m in enumerate(mask_slices):
            if m is None:
                continue
            sl = (m > 0).astype(np.uint8)
            if sl.max() == 0:
                continue
            cmap = mcm.get_cmap(cmap_names[i % len(cmap_names)])
            col = (np.array(cmap(1.0)) * 255).astype(np.uint8)  # solid RGBA color
            mask_idx = sl.astype(bool)
            rgba[mask_idx, 0] = (1 - alpha) * rgba[mask_idx, 0] + alpha * col[0]
            rgba[mask_idx, 1] = (1 - alpha) * rgba[mask_idx, 1] + alpha * col[1]
            rgba[mask_idx, 2] = (1 - alpha) * rgba[mask_idx, 2] + alpha * col[2]

        a = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([rgba.astype(np.uint8), a], axis=-1)
