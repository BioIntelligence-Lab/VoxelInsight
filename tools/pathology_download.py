from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from dicomweb_client.api import DICOMwebClient
import wsidicom
from pydantic import BaseModel, Field

from core.state import Task, TaskResult, ConversationState
from tools.shared import toolify_agent, _cs

DEFAULT_IDC_STORE = "https://proxy.imaging.datacommons.cancer.gov/current/viewer-only-no-downloads-see-tinyurl-dot-com-slash-3j3d9jyp/dicomWeb"
OUTDIR = "pathology_downloads"


class PathologyDownloadArgs(BaseModel):
    dicom_store_url: str = Field(DEFAULT_IDC_STORE, description="DICOMweb base URL for the store.")
    study_instance_uid: str = Field(..., description="StudyInstanceUID for the slide.")
    series_instance_uid: str = Field(..., description="SeriesInstanceUID for the slide.")
    level: int = Field(0, ge=0, description="Resolution level index (0 is highest resolution).")
    x: int = Field(0, ge=0, description="Top-left x pixel coordinate.")
    y: int = Field(0, ge=0, description="Top-left y pixel coordinate.")
    width: int = Field(512, ge=1, description="Region width in pixels.")
    height: int = Field(512, ge=1, description="Region height in pixels.")
    image_type: str = Field("PNG", description="Image format to save (PNG or JPEG).")


class PathologyDownloadAgent:
    name = "pathology_download"
    model = None

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        kw = task.kwargs or {}
        dicom_store_url: str = kw.get("dicom_store_url", DEFAULT_IDC_STORE)
        study_instance_uid: str = kw.get("study_instance_uid")
        series_instance_uid: str = kw.get("series_instance_uid")
        level: int = int(kw.get("level", 0))
        x: int = int(kw.get("x", 0))
        y: int = int(kw.get("y", 0))
        width: int = int(kw.get("width", 512))
        height: int = int(kw.get("height", 512))
        image_type: str = kw.get("image_type", "PNG") or "PNG"

        if not (study_instance_uid and series_instance_uid):
            return TaskResult(output="Missing required UIDs: study_instance_uid and series_instance_uid.", artifacts={})
        if width < 1 or height < 1:
            return TaskResult(output="Width and height must be positive.", artifacts={})
        if level < 0:
            return TaskResult(output="Level must be non-negative.", artifacts={})

        try:
            img_bytes, fmt = await asyncio.to_thread(
                self._fetch_region,
                dicom_store_url,
                study_instance_uid,
                series_instance_uid,
                level,
                x,
                y,
                width,
                height,
                image_type,
            )
        except Exception as e:
            return TaskResult(output=f"PathologyDownload error: {e}", artifacts={})

        out_root = Path(OUTDIR).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        fname = (
            f"study_{study_instance_uid}_"
            f"series_{series_instance_uid}_"
            f"level{level}_x{x}_y{y}_{width}x{height}.{fmt.lower()}"
        )
        out_path = out_root / fname
        out_path.write_bytes(img_bytes)

        summary = (
            f"Downloaded pathology region from DICOMweb store.\n"
            f"- Store: {dicom_store_url}\n"
            f"- StudyInstanceUID: {study_instance_uid}\n"
            f"- SeriesInstanceUID: {series_instance_uid}\n"
            f"- Level: {level}\n"
            f"- Region: ({x}, {y}) size {width}x{height}\n"
            f"- Saved to: {out_path}"
        )

        return TaskResult(
            output={
                "text": summary,
                "tool": self.name,
                "study_instance_uid": study_instance_uid,
                "series_instance_uid": series_instance_uid,
                "level": level,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "image_type": fmt,
                "files": [str(out_path)],
            },
            artifacts={"files": [str(out_path)], "output_dir": str(out_root)},
        )

    def _fetch_region(
        self,
        dicom_store_url: str,
        study_instance_uid: str,
        series_instance_uid: str,
        level: int,
        x: int,
        y: int,
        width: int,
        height: int,
        image_type: str,
    ):
        dw_client = DICOMwebClient(url=dicom_store_url)
        wsidc_client = wsidicom.WsiDicomWebClient(dw_client)
        slide = wsidicom.WsiDicom.open_web(wsidc_client, study_uid=study_instance_uid, series_uids=series_instance_uid)
        levels = list(slide.levels)
        level_obj = None
        level_value = level
        for lv in levels:
            lv_id = getattr(lv, "level", getattr(lv, "level_index", None))
            if lv_id == level:
                level_obj = lv
                level_value = lv_id
                break
        if level_obj is None and 0 <= level < len(levels):
            level_obj = levels[level]
            level_value = getattr(level_obj, "level", getattr(level_obj, "level_index", level))
        if level_obj is None:
            avail = [getattr(lv, "level", getattr(lv, "level_index", idx)) for idx, lv in enumerate(levels)]
            raise ValueError(f"Requested level {level} not available; available levels: {avail}")
        region = slide.read_region(location=(x, y), level=level_value, size=(width, height))
        fmt = image_type.upper()
        if fmt not in {"PNG", "JPEG", "JPG"}:
            fmt = "PNG"
        if fmt == "JPG":
            fmt = "JPEG"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt.lower()}") as tmp:
            tmp_path = Path(tmp.name)
            region.save(tmp_path, format=fmt)
        data = tmp_path.read_bytes()
        tmp_path.unlink(missing_ok=True)
        return data, fmt


_PD: Optional[PathologyDownloadAgent] = None


def configure_pathology_download_tool():
    global _PD
    _PD = PathologyDownloadAgent()


@toolify_agent(
    name="pathology_download",
    description=(
        "Download a region from a histopathology whole-slide image via DICOMweb. "
        "Provide study/series UIDs plus level and region coordinates. "
        "Defaults to the IDC proxied DICOM store."
    ),
    args_schema=PathologyDownloadArgs,
    timeout_s=180,
)
async def pathology_download_runner(
    dicom_store_url: str = DEFAULT_IDC_STORE,
    study_instance_uid: str = "",
    series_instance_uid: str = "",
    level: int = 0,
    x: int = 0,
    y: int = 0,
    width: int = 512,
    height: int = 512,
    image_type: str = "PNG",
):
    if _PD is None:
        raise RuntimeError("Pathology download tool not configured. Call configure_pathology_download_tool() first.")

    kwargs = {
        "dicom_store_url": dicom_store_url,
        "study_instance_uid": study_instance_uid,
        "series_instance_uid": series_instance_uid,
        "level": level,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "image_type": image_type,
    }
    task = Task(user_msg="Download pathology region via DICOMweb", files=[], kwargs=kwargs)
    return await _PD.run(task, _cs())
