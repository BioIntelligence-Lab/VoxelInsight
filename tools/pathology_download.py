from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from core.state import Task, TaskResult, ConversationState
from tools.shared import toolify_agent, _cs

# Default IDC WADO-RS endpoint
DEFAULT_ENDPOINT = "https://portal.imaging.datacommons.cancer.gov/dcm4chee-arc/aets/DCM4CHEE/rs"
OUTDIR = "pathology_downloads"


# ============================================================
# Args schema (for toolify_agent + LLM)
# ============================================================

class PathologyDownloadArgs(BaseModel):
    dicomweb_endpoint: str = Field(
        DEFAULT_ENDPOINT,
        description="DICOMweb endpoint (WADO-RS). Defaults to IDC public endpoint.",
    )
    study_instance_uid: str = Field(..., description="StudyInstanceUID for the slide.")
    series_instance_uid: str = Field(..., description="SeriesInstanceUID for the slide.")
    sop_instance_uid: str = Field(..., description="SOPInstanceUID for the desired instance/frame.")
    frame_number: int = Field(1, description="Frame number to retrieve.")
    size: int = Field(
        512,
        description="Tile size (pixels). Default 512. If user specifies another size, use that.",
    )
    image_type: str = Field("jpeg", description="Image format to request (e.g., jpeg).")


# ============================================================
# Agent implementation (mirrors IDCDownloadAgent style)
# ============================================================

class PathologyDownloadAgent:
    name = "pathology_download"
    model = None

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        kw = task.kwargs or {}

        dicomweb_endpoint: str = kw.get("dicomweb_endpoint", DEFAULT_ENDPOINT)
        study_instance_uid: str = kw.get("study_instance_uid")
        series_instance_uid: str = kw.get("series_instance_uid")
        sop_instance_uid: str = kw.get("sop_instance_uid")
        frame_number: int = int(kw.get("frame_number", 1))
        size: int = int(kw.get("size", 512))
        image_type: str = kw.get("image_type", "jpeg")

        if not (study_instance_uid and series_instance_uid and sop_instance_uid):
            return TaskResult(
                output="Missing required UIDs: study_instance_uid, series_instance_uid, sop_instance_uid.",
                artifacts={},
            )

        base = dicomweb_endpoint.rstrip("/")
        url = (
            f"{base}/studies/{study_instance_uid}"
            f"/series/{series_instance_uid}"
            f"/instances/{sop_instance_uid}"
            f"/frames/{frame_number}"
        )
        params = {
            "contentType": f"image/{image_type}",
            "size": size,
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                content = resp.content
                content_type = resp.headers.get("content-type", "")
        except Exception as e:
            return TaskResult(
                output=f"PathologyDownload error while fetching from WADO-RS: {e}",
                artifacts={},
            )

        # Determine file extension
        if "jpeg" in content_type:
            ext = "jpg"
        elif "png" in content_type:
            ext = "png"
        else:
            ext = image_type

        # Output directory (persistent, similar to idc_downloads)
        out_root = Path(OUTDIR).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        fname = (
            f"study_{study_instance_uid}_"
            f"series_{series_instance_uid}_"
            f"frame{frame_number}_{size}px.{ext}"
        )
        out_path = out_root / fname
        out_path.write_bytes(content)

        summary = (
            f"Downloaded histopathology tile from WADO-RS.\n"
            f"- StudyInstanceUID: {study_instance_uid}\n"
            f"- SeriesInstanceUID: {series_instance_uid}\n"
            f"- SOPInstanceUID: {sop_instance_uid}\n"
            f"- Frame: {frame_number}\n"
            f"- Size: {size}px\n"
            f"- Saved to: {out_path}"
        )

        return TaskResult(
            output={
                "text": summary,
                "tool": self.name,
                "study_instance_uid": study_instance_uid,
                "series_instance_uid": series_instance_uid,
                "sop_instance_uid": sop_instance_uid,
                "frame_number": frame_number,
                "size": size,
                "image_type": image_type,
                "files": [str(out_path)],
            },
            artifacts={"files": [str(out_path)], "output_dir": str(out_root)},
        )


# ============================================================
# Config + tool wrapper (mirrors idc_download style)
# ============================================================

_PD: Optional[PathologyDownloadAgent] = None


def configure_pathology_download_tool():
    """Call this once at startup to configure the pathology download tool."""
    global _PD
    _PD = PathologyDownloadAgent()


@toolify_agent(
    name="pathology_download",
    description=(
        "Download histopathology imagery via DICOMweb WADO-RS. "
        "Provide study/series/instance UIDs and desired frame/size. "
        "Defaults to the IDC public endpoint and 512x512 tiles."
    ),
    args_schema=PathologyDownloadArgs,
    timeout_s=120,
)
async def pathology_download_runner(
    dicomweb_endpoint: str = DEFAULT_ENDPOINT,
    study_instance_uid: str = "",
    series_instance_uid: str = "",
    sop_instance_uid: str = "",
    frame_number: int = 1,
    size: int = 512,
    image_type: str = "jpeg",
):
    if _PD is None:
        raise RuntimeError(
            "Pathology download tool not configured. "
            "Call configure_pathology_download_tool() first."
        )

    kwargs = {
        "dicomweb_endpoint": dicomweb_endpoint,
        "study_instance_uid": study_instance_uid,
        "series_instance_uid": series_instance_uid,
        "sop_instance_uid": sop_instance_uid,
        "frame_number": frame_number,
        "size": size,
        "image_type": image_type,
    }
    task = Task(user_msg="Download pathology tile via WADO-RS", files=[], kwargs=kwargs)
    return await _PD.run(task, _cs())
