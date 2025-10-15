import os, asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import contextlib
import chainlit as cl

from core.state import Task, TaskResult, ConversationState

outdir = "idc_downloads"

class IDCDownloadAgent:
    name  = "idc_download"
    model = None

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        kw = task.kwargs or {}
        series_uid   = kw.get("series_uid")          
        series_uids  = kw.get("series_uids") or []   
        timeout_s    = int(kw.get("timeout_s", 3600))

        res = await cl.AskActionMessage(
            content="Would you like to download series from IDC now?",
            actions=[
                cl.Action(name="continue", payload={"value": "continue"}, label="✅ Continue"),
                cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
            ],
        ).send()

        if not (res and res.get("payload", {}).get("value") == "continue"):
            await cl.Message(content="IDC download cancelled.").send()
            return TaskResult(
                output="IDC download was cancelled by the user via the UI. DO NOT REPEAT without checking with user again if needed.",
                artifacts={}
            )

        await cl.Message(content="Starting IDC download...").send()

        try:
            uids: List[str] = []
            if series_uid:
                uids.append(str(series_uid))
            if series_uids:
                uids.extend([str(u) for u in series_uids if u])

            if not uids:
                return TaskResult(output="Provide series_uid or series_uids.", artifacts={})

            out_root = Path(outdir).expanduser().resolve()
            out_root.mkdir(parents=True, exist_ok=True)

            before = self._snapshot_files(out_root)
            logs: List[str] = []

            rc, so, se = await self._download_series(uids, out_root, timeout_s)
            logs.append(f"download: rc={rc}")
            if se:
                logs.append(f"stderr: {se.strip()[:500]}")

            if rc != 0:
                return TaskResult(
                    output=f"IDC download failed (rc={rc}). {se or ''}".strip(),
                    artifacts={"output_dir": str(out_root), "logs": logs}
                )

            after = self._snapshot_files(out_root)
            new_files = sorted(after - before)

            series_dirs = self._find_series_dirs(out_root, uids)
            if not series_dirs:
                series_dirs = [str(out_root)]

            summary = f"Downloaded series: {len(uids)} | New files: {len(new_files)} → {out_root}"
            return TaskResult(
                output={
                    "text": summary,
                    "series_uids": uids,
                    "files": series_dirs,
                    "output_dir": str(out_root),
                    "logs": logs,
                    "tool": self.name,
                },
                artifacts={"files": series_dirs, "output_dir": str(out_root)}
            )

        except Exception as e:
            return TaskResult(output=f"IDCDownload error: {e}")

    # helpers
    def _snapshot_files(self, dirp: Path) -> Set[str]:
        return {str(p) for p in dirp.rglob("*") if p.is_file()} if dirp.exists() else set()

    def _find_series_dirs(self, root: Path, uids: List[str]) -> List[str]:
        out: List[str] = []
        for u in uids:
            matches = [p for p in root.rglob("*") if p.is_dir() and u in str(p)]
            if matches:
                out.append(str(matches[0]))
        return out

    async def _download_series(self, uids: List[str], out_root: Path, timeout_s: int):
        async def _runner():
            try:
                from idc_index import index
                client = index.IDCClient()
                client.download_from_selection(seriesInstanceUID=uids, downloadDir=str(out_root))
                return 0, "ok", ""
            except Exception as e:
                return 1, "", str(e)

        try:
            return await asyncio.wait_for(asyncio.to_thread(lambda: asyncio.run(_runner())), timeout=timeout_s)
        except asyncio.TimeoutError:
            return 124, "", "Timeout while downloading from IDC"

from pydantic import BaseModel, Field
from tools.shared import toolify_agent, _cs

_DL: Optional[IDCDownloadAgent] = None

def configure_idc_download_tool():
    global _DL
    _DL = IDCDownloadAgent()

class IDCDownloadArgs(BaseModel):
    series_uid: Optional[str] = Field(None, description="Single SeriesInstanceUID.")
    series_uids: Optional[List[str]] = Field(None, description="Multiple SeriesInstanceUIDs.")
    timeout_s: int = Field(default=3600, ge=30, description="Overall timeout seconds.")

@toolify_agent(
    name="idc_download",
    description="Download IDC series via idc_index given one or many SeriesInstanceUIDs. Use this tool for all IDC downloads.",
    args_schema=IDCDownloadArgs,
    timeout_s=3600,
)
async def idc_download_runner(
    series_uid: Optional[str] = None,
    series_uids: Optional[List[str]] = None,
    timeout_s: int = 3600,
):
    if _DL is None:
        raise RuntimeError("IDC download tool not configured. Call configure_idc_download_tool() first.")

    kwargs = {
        "series_uid": series_uid,
        "series_uids": series_uids,
        "timeout_s": timeout_s,
    }
    task = Task(user_msg="Download IDC files", files=[], kwargs=kwargs)
    return await _DL.run(task, _cs())
