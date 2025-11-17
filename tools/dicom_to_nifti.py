import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import pydicom
import dicom2nifti
from pydantic import BaseModel, Field

from core.state import TaskResult
from tools.shared import toolify_agent 

class Dicom2NiftiTool:
    name = "dicom2nifti"

    def _clean_path(self, p: str) -> Path:
        return Path(os.path.expanduser((p or "").replace("\\ ", " ").strip())).resolve()

    def _is_dicom(self, f: Path) -> bool:
        if not f.is_file():
            return False
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            return hasattr(ds, "SOPClassUID")
        except Exception:
            return False

    def _candidate_series_dirs(self, root: Path) -> List[Path]:
        # find dirs with >=3 readable dicoms
        out: List[Path] = []
        for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
            try:
                files = list(d.iterdir())
            except Exception:
                continue
            dicoms = [f for f in files if self._is_dicom(f)]
            if len(dicoms) >= 3:
                out.append(d)
        # de-dupe
        seen, uniq = set(), []
        for d in out:
            s = d.as_posix()
            if s not in seen:
                uniq.append(d); seen.add(s)
        return uniq

    async def run(self, *, dicom_dir: str, out_dir: Optional[str] = None) -> TaskResult:
        if not dicom_dir:
            return TaskResult(output="dicom2nifti_py: no dicom_dir provided.", artifacts={})

        root = self._clean_path(dicom_dir)
        if not root.exists():
            return TaskResult(output=f"dicom2nifti_py: path does not exist: {root}", artifacts={})
        if not root.is_dir():
            return TaskResult(output=f"dicom2nifti_py: path is not a directory: {root}", artifacts={})

        out_base = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="vi_dcm2nii_"))

        # Try bulk directory conversion first
        try:
            dicom2nifti.convert_directory(
                str(root), str(out_base),
                compression=True,
                reorient=True
            )
        except Exception:
            pass

        produced = [str(p) for p in out_base.rglob("*.nii*")]

        # If nothing produced, try per-series conversion
        if not produced:
            series_dirs = self._candidate_series_dirs(root)
            errs = []
            for sd in series_dirs:
                out_name = sd.name + ".nii.gz"
                out_path = out_base / out_name
                try:
                    dicom2nifti.dicom_series_to_nifti(
                        str(sd), str(out_path), reorient=True
                    )
                except Exception as e:
                    errs.append(f"{sd}: {e}")

            produced = [str(p) for p in out_base.rglob("*.nii*")]

            if not produced:
                diag: Dict[str, Any] = {
                    "root": str(root),
                    "checked_series_dirs": [str(d) for d in series_dirs[:5]],
                    "errors_tail": errs[-3:],
                }
                return TaskResult(
                    output=f"dicom2nifti_py: conversion produced no NIfTI files.\n{diag}",
                    artifacts={"output_dir": str(out_base)}
                )

        return TaskResult(
            output={
                "action": "download",
                "tool": self.name,
                "download_dir": str(out_base),
                "files": produced
            },
            artifacts={
                "nifti_paths": produced,
                "output_dir": str(out_base),
                "image_path": produced[0] if produced else None,
            },
        )

_TOOL = Dicom2NiftiTool()

class Dicom2NiftiArgs(BaseModel):
    dicom_dir: str = Field(..., description="Directory containing DICOM files or nested DICOM series.")
    out_dir: Optional[str] = Field(None, description="Output directory for NIfTI files (optional).")

@toolify_agent(
    name="dicom2nifti",
    description=(
        "IDC studies follow the DICOM hierarchy: Patient → Study → Series → Instances."
        "A series contains multiple DICOM slices."
        "Use this tool to convert a DICOM directory (or nested series) into NIfTI (.nii.gz). Returns a downloadable file list automatically displayed in UI for users to download."
    ),
    args_schema=Dicom2NiftiArgs,
    timeout_s=900, 
)
async def dicom2nifti_runner(dicom_dir: str, out_dir: Optional[str] = None):
    return await _TOOL.run(dicom_dir=dicom_dir, out_dir=out_dir)