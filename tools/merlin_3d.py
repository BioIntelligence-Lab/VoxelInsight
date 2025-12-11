# tools/merlin_3d.py

import os
import pathlib
import tempfile
import shutil
from typing import List, Dict, Optional

import numpy as np
import torch
import pandas as pd
from pydantic import BaseModel, Field

from merlin import Merlin
from merlin.data import DataLoader

from core.state import Task, TaskResult, ConversationState
from tools.shared import toolify_agent, _cs

# -------------------------------------------------------------------
# Global MERLIN context (model, device, cache_root)
# -------------------------------------------------------------------

_MERLIN_CTX: Optional[Dict] = None


def configure_merlin_tool(
    *,
    device: Optional[str] = None,
    cache_root: Optional[str] = None,
    merlin_kwargs: Optional[dict] = None,
):
    """
    Configure the global MERLIN model for 3D image embeddings.

    - device: "cuda" or "cpu". If None, auto-select based on torch.cuda.is_available().
    - cache_root: parent directory for MERLIN's internal DataLoader cache.
                  If None, a temp directory under each run's output will be used.
    - merlin_kwargs: passed directly into Merlin(...).
                     By default, uses ImageEmbedding=True as in the official demo.
    """
    global _MERLIN_CTX

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    merlin_kwargs = dict(merlin_kwargs or {})
    # Demo pattern: Merlin(ImageEmbedding=True) for embeddings
    merlin_kwargs.setdefault("ImageEmbedding", True)

    model = Merlin(**merlin_kwargs)
    model.eval()
    model.to(device)

    _MERLIN_CTX = {
        "model": model,
        "device": device,
        "cache_root": cache_root,
    }


# -------------------------------------------------------------------
# Tool argument schema
# -------------------------------------------------------------------

class Merlin3DArgs(BaseModel):
    volume_path: Optional[str] = Field(
        default=None,
        description="Path to a 3D CT volume (.nii or .nii.gz). If omitted, uses attached file."
    )
    volume_paths: Optional[List[str]] = Field(
        default=None,
        description="List of 3D CT volumes (.nii/.nii.gz) for batch embeddings."
    )
    normalize: bool = Field(
        default=True,
        description="If true, L2-normalize embedding vectors before writing CSV."
    )


# -------------------------------------------------------------------
# Core Agent
# -------------------------------------------------------------------

class MerlinEmbeddingAgent:
    """
    Runs MERLIN in 3D ImageEmbedding mode on NIfTI volumes to produce volume-level embeddings.

    - Input: NIfTI volumes (.nii/.nii.gz), one or many.
    - Uses merlin.data.DataLoader with a small datalist per run (mirrors the official demo).
    - Output: a single CSV file with one row per volume:
              filename, merlin_0, merlin_1, ..., merlin_D
    """

    name = "merlin_3d"
    model = None  # metadata only; actual MERLIN model is in _MERLIN_CTX

    def __init__(self):
        if _MERLIN_CTX is None:
            raise RuntimeError(
                "MERLIN 3D tool not configured. Call configure_merlin_tool(...) at app startup."
            )
        self.model = _MERLIN_CTX["model"]
        self.device = _MERLIN_CTX["device"]
        self.cache_root = _MERLIN_CTX["cache_root"]

    # ---------- Helpers ----------

    def _ensure_nifti(self, p: pathlib.Path) -> pathlib.Path:
        """
        Mirror your ImagingAgent pattern:

        - If the file isn't .nii or .nii.gz, copy it to a temp dir with .nii.gz suffix.
        - Merlin's DataLoader expects NIfTI; we let Merlin handle loading/preprocessing.
        """
        if p.suffix not in (".nii", ".gz"):
            tmp_dir = pathlib.Path(tempfile.mkdtemp())
            fixed = tmp_dir / (p.name + ".nii.gz")
            shutil.copy(p, fixed)
            return fixed
        return p

    def _get_cache_dir_for_volume(self, out_root: pathlib.Path, vol_path: pathlib.Path) -> str:
        """
        Choose a cache directory for MERLIN's DataLoader.

        If a global cache_root was provided in configure_merlin_tool, use that as parent;
        otherwise, use the current output root.
        """
        if self.cache_root:
            base = pathlib.Path(self.cache_root)
        else:
            base = out_root
        cache_dir = base / f"merlin_cache_{vol_path.stem}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)

    def _compute_embedding_single(
        self,
        vol_path: pathlib.Path,
        out_root: pathlib.Path,
        normalize: bool,
    ) -> np.ndarray:
        """
        Build a small DataLoader for a single NIfTI, run MERLIN(ImageEmbedding=True),
        and return the resulting embedding as a 1D numpy array.
        """
        # Prepare MERLIN datalist (same pattern as demo: datalist={"image": "<path>"})
        datalist = [{"image": str(vol_path)}]
        cache_dir = self._get_cache_dir_for_volume(out_root, vol_path)

        dataloader = DataLoader(
            datalist=datalist,
            cache_dir=cache_dir,
            batchsize=1,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()
        emb_vec = None

        for batch in dataloader:
            imgs = batch["image"].to(self.device)
            outputs = self.model(imgs)
            # Demo: Image embeddings shape (for downstream tasks) is outputs[0]
            emb = outputs[0]  # tensor (B, D) with B=1 here
            emb = emb.squeeze(0).detach().cpu().numpy()  # (D,)
            emb_vec = emb
            break  # single batch only

        if emb_vec is None:
            raise RuntimeError(f"MERLIN did not produce an embedding for volume: {vol_path}")

        if normalize:
            norm = np.linalg.norm(emb_vec) + 1e-12
            emb_vec = emb_vec / norm

        return emb_vec.astype(np.float32)

    # ---------- Main entrypoint ----------

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        # No explicit volume paths and no attached files → nothing to do
        if not task.files and not (task.kwargs.get("volume_path") or task.kwargs.get("volume_paths")):
            return TaskResult(output="No volume provided for MERLIN 3D embedding.")

        normalize = bool(task.kwargs.get("normalize", True))

        # Collect volume paths from kwargs and/or attached files
        volume_paths: List[str] = []
        if task.kwargs.get("volume_paths"):
            volume_paths.extend(task.kwargs["volume_paths"])
        if task.kwargs.get("volume_path"):
            volume_paths.append(task.kwargs["volume_path"])
        if not volume_paths and task.files:
            volume_paths.extend(task.files)

        # Ensure NIfTI and store fixed paths
        fixed_paths: List[pathlib.Path] = []
        for p_str in volume_paths:
            p = pathlib.Path(p_str)
            fixed_paths.append(self._ensure_nifti(p))

        # Track in state.memory
        state.memory["merlin_input_volumes"] = [str(p) for p in fixed_paths]

        # Output root for this run
        out_root = pathlib.Path(tempfile.mkdtemp(prefix="merlin_3d_"))
        rows: List[Dict[str, float]] = []

        # Compute embeddings and build rows for CSV
        for p in fixed_paths:
            emb_vec = self._compute_embedding_single(p, out_root, normalize=normalize)
            row: Dict[str, float] = {"filename": str(p)}
            for i, val in enumerate(emb_vec):
                row[f"merlin_{i}"] = float(val)
            rows.append(row)

        if not rows:
            return TaskResult(output="MERLIN did not produce any embeddings. No CSV created.")

        # Create DataFrame and CSV
        df = pd.DataFrame(rows)
        csv_path = out_root / "merlin_embeddings.csv"
        df.to_csv(csv_path, index=False)

        # Store in memory for later chaining (e.g., code_gen → clustering)
        state.memory["merlin_embeddings_csv"] = str(csv_path)

        summary = {
            "agent": "merlin_3d",
            "action": "image_embedding",
            "model": "Merlin(ImageEmbedding=True)",
            "num_volumes": len(fixed_paths),
            "csv_path": str(csv_path),
        }

        # IMPORTANT: include files + output_dir so Chainlit can zip & offer download
        return TaskResult(
            output=summary,
            artifacts={
                "files": [str(csv_path)],
                "output_dir": str(out_root),
            },
        )


# -------------------------------------------------------------------
# Tool wrapper
# -------------------------------------------------------------------

@toolify_agent(
    name="merlin_3d",
    description=(
        "Computes 3D CT embeddings using the Merlin vision-language foundation model in ImageEmbedding mode. "
        "Accepts single or multiple NIfTI volumes (.nii/.nii.gz). "
        "Outputs a CSV file with one row per input volume: filename and merlin_* feature columns."
    ),
    args_schema=Merlin3DArgs,
    timeout_s=1200,
)
async def merlin_3d_runner(
    volume_path: Optional[str] = None,
    volume_paths: Optional[List[str]] = None,
    normalize: bool = True,
):
    if _MERLIN_CTX is None:
        raise RuntimeError("MERLIN 3D tool not configured. Call configure_merlin_tool(...) first.")

    files: List[str] = []
    if volume_paths:
        files.extend(volume_paths)
    if volume_path:
        files.append(volume_path)

    kwargs = {
        "volume_path": volume_path,
        "volume_paths": volume_paths,
        "normalize": normalize,
    }

    agent = MerlinEmbeddingAgent()
    task = Task(
        user_msg="Run MERLIN 3D image embedding",
        files=files,
        kwargs=kwargs,
    )
    return await agent.run(task, _cs())
