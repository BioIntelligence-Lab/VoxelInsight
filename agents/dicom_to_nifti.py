# agents/dicom2nifti_py.py
import os
import tempfile
from pathlib import Path

import pydicom
import dicom2nifti

from core.state import Task, TaskResult, ConversationState


class Dicom2NiftiPyAgent:
    name = "dicom2nifti"
    model = None

    def _clean_path(self, p: str) -> Path:
        # handle "\ " artifacts and ~
        return Path(os.path.expanduser((p or "").replace("\\ ", " ").strip())).resolve()

    def _is_dicom(self, f: Path) -> bool:
        if not f.is_file():
            return False
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            return hasattr(ds, "SOPClassUID")
        except Exception:
            return False

    def _candidate_series_dirs(self, root: Path) -> list[Path]:
        # find dirs with >=3 readable dicoms
        out = []
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

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        dicom_dir = task.kwargs.get("dicom_dir")
        if not dicom_dir and task.files and os.path.isdir(task.files[0]):
            dicom_dir = task.files[0]

        if not dicom_dir:
            return TaskResult(output="dicom2nifti_py: no dicom_dir provided.")

        root = self._clean_path(dicom_dir)
        if not root.exists():
            return TaskResult(output=f"dicom2nifti_py: path does not exist: {root}")
        if not root.is_dir():
            return TaskResult(output=f"dicom2nifti_py: path is not a directory: {root}")

        out_dir = Path(tempfile.mkdtemp(prefix="vi_dcm2nii_"))

        try:
            dicom2nifti.convert_directory(
                str(root), str(out_dir),
                compression=True,
                reorient=True
            )
        except Exception as e:
            pass

        produced = [str(p) for p in out_dir.rglob("*.nii*")]  # <— RECURSIVE
        if not produced:
            series_dirs = self._candidate_series_dirs(root)
            errs = []
            for sd in series_dirs:
                out_name = sd.name + ".nii.gz"
                out_path = out_dir / out_name
                try:
                    dicom2nifti.dicom_series_to_nifti(
                        str(sd), str(out_path), reorient=True
                    )
                except Exception as e:
                    errs.append(f"{sd}: {e}")

            produced = [str(p) for p in out_dir.rglob("*.nii*")]

            if not produced:
                diag = {
                    "root": str(root),
                    "checked_series_dirs": [str(d) for d in series_dirs[:5]],
                    "errors_tail": errs[-3:],
                }
                return TaskResult(output=f"dicom2nifti_py: conversion produced no NIfTI files.\n{diag}")

        # Seed memory for downstream agents 
        state.memory.setdefault("files", []).extend(produced)
        state.memory["image_path"] = produced[0]

        return TaskResult(
            output={
                "action": "download",
                "download_dir": str(out_dir),
                "files": produced
            },
            artifacts={
                "nifti_paths": produced,
                "output_dir": str(out_dir),
                "image_path": produced[0],
            },
        )
