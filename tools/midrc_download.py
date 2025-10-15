import os, asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
import contextlib
import tempfile
import chainlit as cl

from core.state import Task, TaskResult, ConversationState

CRED_PATH = os.getenv("MIDRC_CRED", "~/midrc_credentials.json")
ENDPOINT  = "data.midrc.org"                                   
DEFAULT_OUTDIR = "midrc_downloads"                             


class MIDRCDownloadAgent:
    name  = "midrc_download"
    model = None

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        kw = task.kwargs or {}
        object_id   = kw.get("object_id")
        object_ids  = kw.get("object_ids")
        output_dir  = kw.get("output_dir") or DEFAULT_OUTDIR
        timeout_s   = int(kw.get("timeout_s", 3600))
        parallel    = int(kw.get("parallel", 2))

        res = await cl.AskActionMessage(
            content="Would you like to download files from MIDRC now?",
            actions=[
                cl.Action(name="continue", payload={"value": "continue"}, label="✅ Continue"),
                cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
            ],
        ).send()

        if res and res.get("payload").get("value") == "continue":
            await cl.Message(
                content="Starting MIDRC download...",
            ).send()

            try:
                # validate cred path
                credp = Path(CRED_PATH).expanduser().resolve()
                if not credp.exists():
                    return TaskResult(output=f"Cred file not found at {credp}")

                # ensure output dir
                out_root = Path(output_dir).expanduser().resolve()
                out_root.mkdir(parents=True, exist_ok=True)

                # gather ids
                ids: List[str] = []
                if object_id:
                    ids.append(object_id)
                if object_ids:
                    ids.extend([u for u in object_ids if u])

                if not ids:
                    return TaskResult(output="Provide object_id or object_ids.", artifacts={"output_dir": str(out_root)})

                before = self._snapshot_files(out_root)
                logs: List[str] = []

                # download
                sem = asyncio.Semaphore(max(1, min(16, parallel)))

                async def _one(uid: str):
                    async with sem:
                        return await self._pull_object(str(credp), ENDPOINT, uid, out_root, timeout_s)

                results = await asyncio.gather(*[_one(u) for u in ids], return_exceptions=True)

                errs = []
                for r in results:
                    if isinstance(r, Exception):
                        errs.append(str(r)); continue
                    logs.append(f"{r.get('uid')}: rc={r.get('rc')}")
                    if r.get("rc") != 0:
                        errs.append(f"{r.get('uid')}: {r.get('stderr') or r.get('stdout')}")

                if errs:
                    return TaskResult(
                        output="Some downloads failed:\n" + "\n".join(errs[:20]),
                        artifacts={"output_dir": str(out_root), "logs": logs}
                    )

                after = self._snapshot_files(out_root)
                new_files = sorted(after - before)
                summary = f"Downloaded {len(new_files)} file(s) to {out_root}."
                return TaskResult(
                    output={"text": summary, "files": new_files, "output_dir": str(out_root), "logs": logs, "tool": self.name},
                    artifacts={"files": new_files, "output_dir": str(out_root)}
                )

            except Exception as e:
                return TaskResult(output=f"MIDRCDownload error: {e}")

        else:
            await cl.Message(content="MIDRC download cancelled.",).send()
            return TaskResult(output="MIDRC download was cancelled by the user via the UI. DO NOT REPEAT without checking with user again if needed.", artifacts={})

    # helpers 
    def _snapshot_files(self, dirp: Path) -> set[str]:
        return {str(p) for p in dirp.rglob("*") if p.is_file()} if dirp.exists() else set()

    async def _run_cmd(self, cmd: List[str], timeout: Optional[int] = None):
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            return 124, "", "Timeout"
        return proc.returncode, out.decode(errors="ignore"), err.decode(errors="ignore")

    async def _pull_object(self, cred: str, endpoint: str, obj: str, out_dir: Path, timeout: int) -> Dict[str, Any]:
        cmd = [
            "gen3",
            "--auth", cred,
            "--endpoint", endpoint,
            "drs-pull",
            "object", obj,
            "--output-dir", str(out_dir),
        ]
        rc, so, se = await self._run_cmd(cmd, timeout=timeout)
        return {"uid": obj, "rc": rc, "stdout": so, "stderr": se}

from pydantic import BaseModel, Field
from tools.shared import toolify_agent, _cs

_DL: Optional[MIDRCDownloadAgent] = None

def configure_midrc_download_tool():
    global _DL
    _DL = MIDRCDownloadAgent()

class MIDRCDownloadArgs(BaseModel):
    object_id: Optional[str] = Field(None, description="Single DRS object GUID.")
    object_ids: Optional[List[str]] = Field(None, description="Multiple DRS object GUIDs.")
    output_dir: Optional[str] = Field(None, description=f"Download directory. Defaults to '{DEFAULT_OUTDIR}'.")
    timeout_s: int = Field(default=3600, ge=30, description="Overall timeout seconds.")
    parallel: int = Field(default=2, ge=1, le=16, description="Max concurrent pulls when object_ids is used.")

@toolify_agent(
    name="midrc_download",
    description="Download MIDRC files via Gen3 CLI (drs-pull) given one or many DRS GUIDs.",
    args_schema=MIDRCDownloadArgs,
    timeout_s=3600,
)
async def midrc_download_runner(
    object_id: Optional[str] = None,
    object_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    timeout_s: int = 3600,
    parallel: int = 2,
):
    if _DL is None:
        raise RuntimeError("MIDRC download tool not configured. Call configure_midrc_download_tool() first.")

    kwargs = {
        "object_id": object_id,
        "object_ids": object_ids,
        "output_dir": output_dir,
        "timeout_s": timeout_s,
        "parallel": parallel,
    }
    task = Task(user_msg="Download MIDRC files", files=[], kwargs=kwargs)
    return await _DL.run(task, _cs())
