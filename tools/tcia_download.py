# tools/tcia_download.py
import os, json, pathlib, zipfile, tempfile, time, requests, asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

from core.state import Task, TaskResult, ConversationState
from pydantic import BaseModel, Field
from tools.shared import toolify_agent, _cs

BASE_V1 = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
FALLBACK_V3 = "https://services.cancerimagingarchive.net/services/v3/TCIA/query/getImage"

def _ensure_outdir(root: Optional[str], make_subdir: bool, label: str = "tcia") -> Path:
    base = Path(root) if root else Path(f"vi_{label}_{next(tempfile._get_candidate_names())}")
    base = Path.cwd() / "tcia_downloads" / base
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{label}_downloads" if make_subdir else base

def _snapshot(dirp: Path) -> set[str]:
    return {str(p) for p in dirp.rglob("*") if p.is_file()} if dirp.exists() else set()

def _get_series_from_collection(collection: str) -> str:
    r = requests.get(f"{BASE_V1}/getSeries", params={"Collection": collection, "format": "json"}, timeout=60)
    r.raise_for_status()
    for s in r.json():
        try:
            if int(s.get("ImageCount", 0)) > 0:
                return s["SeriesInstanceUID"]
        except Exception:
            continue
    raise ValueError(f"No valid series were found in collection {collection}")

def _get_series_size(series_uid: str) -> Dict[str, Any]:
    r = requests.get(f"{BASE_V1}/getSeriesSize", params={"SeriesInstanceUID": series_uid, "format": "json"}, timeout=60)
    r.raise_for_status()
    return r.json()

def _download_series_zip(series_uid: str, out_zip: Path, use_v3_fallback: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    def _download(url: str):
        with requests.get(url, params={"SeriesInstanceUID": series_uid}, stream=True, timeout=600) as r:
            r.raise_for_status()
            out_zip.parent.mkdir(parents=True, exist_ok=True)
            with out_zip.open("wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

    if out_zip.exists() and not overwrite:
        return {"uid": series_uid, "zip": str(out_zip), "size": out_zip.stat().st_size}

    try:
        _download(f"{BASE_V1}/getImage")
    except Exception:
        if not use_v3_fallback:
            raise
        _download(FALLBACK_V3)

    if not zipfile.is_zipfile(out_zip):
        first_bytes = out_zip.read_bytes()[:256]
        raise RuntimeError(f"Not a ZIP ({out_zip}). First 256 bytes: {first_bytes!r}")

    return {"uid": series_uid, "zip": str(out_zip), "size": out_zip.stat().st_size}

def _extract_zip(zp: Path, to_dir: Path) -> List[str]:
    files: List[str] = []
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(to_dir)
        for n in zf.namelist():
            p = to_dir / n
            if p.exists() and p.is_file():
                files.append(str(p.resolve()))
    return files

class TCIADownloadAgent:
    name  = "tcia_download"
    model = None

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        kw = task.kwargs or {}
        series_uid   = kw.get("series_uid")
        series_uids  = kw.get("series_uids")
        collection   = kw.get("collection")
        output_dir   = kw.get("output_dir")
        make_subdir  = bool(kw.get("make_subdir", True))
        extract_zip  = bool(kw.get("extract_zip", True))
        v3_fallback  = bool(kw.get("v3_fallback", True))
        parallel     = int(kw.get("parallel", 3))
        overwrite    = bool(kw.get("overwrite", False))
        throttle_s   = float(kw.get("throttle_s", 0.0))

        try:
            out_root = _ensure_outdir(output_dir, make_subdir, label="tcia")
            before   = _snapshot(out_root)

            modes = sum(bool(x) for x in [series_uid, series_uids, collection])
            if modes != 1:
                return TaskResult(output="Provide exactly one of: series_uid, series_uids, collection.",
                                  artifacts={"output_dir": str(out_root)})

            if collection:
                uid = await asyncio.to_thread(_get_series_from_collection, collection)
                series_list = [uid]
            elif series_uid:
                series_list = [series_uid]
            else:
                series_list = [u for u in (series_uids or []) if u]

            remaining_series = series_list[5:]
            series_list = series_list[:5] 

            if not series_list:
                return TaskResult(output="No SeriesInstanceUIDs provided.",
                                  artifacts={"output_dir": str(out_root)})

            sem = asyncio.Semaphore(max(1, min(16, parallel)))
            results: List[Dict[str, Any]] = []
            errors: List[str] = []
            zips: List[str] = []
            extracted: List[str] = []

            async def _one(uid: str):
                async with sem:
                    try:
                        dest_zip = out_root / f"{uid}.zip"
                        info = await asyncio.to_thread(_download_series_zip, uid, dest_zip, v3_fallback, overwrite)
                        if throttle_s > 0:
                            await asyncio.sleep(throttle_s)
                        if extract_zip:
                            extract_dir = out_root / uid
                            files = await asyncio.to_thread(_extract_zip, Path(info["zip"]), extract_dir)
                            return {"ok": True, "uid": uid, "zip": info["zip"], "files": files}
                        else:
                            return {"ok": True, "uid": uid, "zip": info["zip"], "files": []}
                    except Exception as e:
                        return {"ok": False, "uid": uid, "error": f"{type(e).__name__}: {e}"}

            tasks = [_one(uid) for uid in series_list]
            for r in await asyncio.gather(*tasks):
                if r["ok"]:
                    zips.append(r["zip"])
                    extracted.extend(r["files"])
                else:
                    errors.append(f"{r['uid']}: {r['error']}")

            after     = _snapshot(out_root)
            new_files = sorted(after - before)
            files_out = new_files if new_files else (zips if zips else [])

            summary = f"Downloaded {len(series_list)} series; files written: {len(files_out)}; output: {out_root}"
            if errors:
                summary += f"\nSome failures ({len(errors)}):\n" + "\n".join(errors[:10])

            return TaskResult(
                output={"text": summary, "files": files_out, "output_dir": str(out_root), "tool": self.name},
                artifacts={"files": files_out, "output_dir": str(out_root)}
            )
        except Exception as e:
            return TaskResult(output=f"TCIADownload error: {e}")

_TCIA: Optional[TCIADownloadAgent] = None

def configure_tcia_download_tool():
    global _TCIA
    _TCIA = TCIADownloadAgent()

class TCIADownloadArgs(BaseModel):
    series_uid: Optional[str] = Field(None, description="Single SeriesInstanceUID")
    series_uids: Optional[List[str]] = Field(None, description="Multiple SeriesInstanceUIDs")
    collection: Optional[str] = Field(None, description="TCIA collection name (pick first non-empty series)")

    output_dir: Optional[str] = Field(None, description="Output directory")
    make_subdir: bool = Field(default=True, description="Create subfolder in output_dir")
    extract_zip: bool = Field(default=True, description="Extract ZIPs to series folder")
    v3_fallback: bool = Field(default=True, description="Fallback to v3 getImage if v1 fails")
    parallel: int = Field(default=3, ge=1, le=16, description="Concurrent series downloads")
    overwrite: bool = Field(default=False, description="Re-download even if ZIP exists")
    throttle_s: float = Field(default=0.0, ge=0.0, description="Sleep between series downloads")

@toolify_agent(
    name="tcia_download",
    description="Download TCIA DICOMs by SeriesInstanceUID(s) or by picking a series from a collection (no API key).",
    args_schema=TCIADownloadArgs,
    timeout_s=36000,
)
async def tcia_download_runner(
    series_uid: Optional[str] = None,
    series_uids: Optional[List[str]] = None,
    collection: Optional[str] = None,
    output_dir: Optional[str] = None,
    make_subdir: bool = True,
    extract_zip: bool = True,
    v3_fallback: bool = True,
    parallel: int = 3,
    overwrite: bool = False,
    throttle_s: float = 0.0,
):
    if _TCIA is None:
        raise RuntimeError("TCIA download tool not configured. Call configure_tcia_download_tool() first.")
    kwargs = {
        "series_uid": series_uid,
        "series_uids": series_uids,
        "collection": collection,
        "output_dir": output_dir,
        "make_subdir": make_subdir,
        "extract_zip": extract_zip,
        "v3_fallback": v3_fallback,
        "parallel": parallel,
        "overwrite": overwrite,
        "throttle_s": throttle_s,
    }
    task = Task(user_msg="Download TCIA series", files=[], kwargs=kwargs)
    return await _TCIA.run(task, _cs())
