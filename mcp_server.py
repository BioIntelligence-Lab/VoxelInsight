from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from core.state import TaskResult
from core.storage import get_run_dir
from tools.shared import normalize_task_result

from idc_index import index

import tools.idc_query as dq_mod
import tools.idc_download as idc_dl_mod
import tools.pathology_download as path_mod
import tools.clinical_data as clin_mod
import tools.idc_web_qa as webqa_mod
import tools.idc_code_qa as code_qa_mod
import tools.dicom_to_nifti as d2n_mod


_CONFIGURED = False


def _ensure_configured() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    load_dotenv(override=True)

    idc_client = index.IDCClient()
    df_IDC = idc_client.index
    try:
        df_BIH = pd.read_csv("Data/BIH_Cases_table.csv", low_memory=False)
    except Exception as e:
        print(f"Warning: could not load BIH data ({e})")
        df_BIH = pd.DataFrame()

    dq_mod.configure_idc_query_tool(
        df_IDC=df_IDC,
        df_BIH=df_BIH,
        system_prompt=Path("prompts/agent_systems/idc_query.txt").read_text(),
    )
    idc_dl_mod.configure_idc_download_tool()
    clin_mod.configure_clinical_data_tool()
    webqa_mod.configure_idc_web_qa_tool(
        system_prompt=Path("prompts/agent_systems/idc_web_qa.txt").read_text(),
    )
    path_mod.configure_pathology_download_tool()
    code_qa_mod.configure_idc_code_qa_tool(
        system_prompt=Path("prompts/agent_systems/idc_code_qa.txt").read_text(),
    )

    _CONFIGURED = True


async def _idc_download_headless(
    *,
    series_uid: Optional[str] = None,
    series_uids: Optional[List[str]] = None,
    timeout_s: int = 3600,
) -> TaskResult:
    agent = idc_dl_mod.IDCDownloadAgent()

    uids: List[str] = []
    if series_uid:
        uids.append(str(series_uid))
    if series_uids:
        uids.extend([str(u) for u in series_uids if u])
    if not uids:
        return TaskResult(output="Provide series_uid or series_uids.", artifacts={})

    out_root = get_run_dir(agent.name, persist=True)

    before = agent._snapshot_files(out_root)
    logs: List[str] = []

    rc, so, se = await agent._download_series(uids, out_root, timeout_s)
    logs.append(f"download: rc={rc}")
    if se:
        logs.append(f"stderr: {se.strip()[:500]}")

    if rc != 0:
        return TaskResult(
            output=f"IDC download failed (rc={rc}). {se or ''}".strip(),
            artifacts={"output_dir": str(out_root), "logs": logs},
        )

    after = agent._snapshot_files(out_root)
    new_files = sorted(after - before)

    series_dirs = agent._find_series_dirs(out_root, uids)
    if not series_dirs:
        series_dirs = [str(out_root)]

    summary = f"Downloaded series: {len(uids)} | New files: {len(new_files)} -> {out_root}"
    return TaskResult(
        output={
            "text": summary,
            "series_uids": uids,
            "files": series_dirs,
            "output_dir": str(out_root),
            "logs": logs,
            "tool": agent.name,
        },
        artifacts={"files": series_dirs, "output_dir": str(out_root)},
    )


mcp = FastMCP("VoxelInsight")


@mcp.tool()
async def idc_query(instructions: str, reasoning_effort: str = "medium") -> dict:
    _ensure_configured()
    res = await dq_mod.idc_query_runner(instructions=instructions, reasoning_effort=reasoning_effort)
    return normalize_task_result(res)


@mcp.tool()
async def idc_download(
    series_uid: Optional[str] = None,
    series_uids: Optional[List[str]] = None,
    timeout_s: int = 3600,
    skip_confirm: bool = True,
) -> dict:
    _ensure_configured()
    if not skip_confirm:
        res = await idc_dl_mod.idc_download_runner(
            series_uid=series_uid,
            series_uids=series_uids,
            timeout_s=timeout_s,
        )
        return normalize_task_result(res)

    res = await _idc_download_headless(
        series_uid=series_uid,
        series_uids=series_uids,
        timeout_s=timeout_s,
    )
    return normalize_task_result(res)


@mcp.tool()
async def pathology_download(
    dicom_store_url: str = path_mod.DEFAULT_IDC_STORE,
    study_instance_uid: str = "",
    series_instance_uid: str = "",
    level: int = 0,
    x: int = 0,
    y: int = 0,
    width: int = 512,
    height: int = 512,
    image_type: str = "PNG",
) -> dict:
    _ensure_configured()
    res = await path_mod.pathology_download_runner(
        dicom_store_url=dicom_store_url,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        level=level,
        x=x,
        y=y,
        width=width,
        height=height,
        image_type=image_type,
    )
    return normalize_task_result(res)


@mcp.tool()
async def clinical_data_download(
    collection_id: str,
    fields: Optional[List[str]] = None,
    filter_field: Optional[str] = None,
    filter_value: Optional[str] = None,
    limit_rows: int = 5000,
) -> dict:
    _ensure_configured()
    res = await clin_mod.clinical_data_download_runner(
        collection_id=collection_id,
        fields=fields,
        filter_field=filter_field,
        filter_value=filter_value,
        limit_rows=limit_rows,
    )
    return normalize_task_result(res)


@mcp.tool()
async def idc_web_qa(
    question: str,
    top_k: int = 5,
    max_chars: int = 8000,
    synthesize: bool = True,
    reasoning_effort: str = "low",
) -> dict:
    _ensure_configured()
    res = await webqa_mod.idc_web_qa_runner(
        question=question,
        top_k=top_k,
        max_chars=max_chars,
        synthesize=synthesize,
        reasoning_effort=reasoning_effort,
    )
    return normalize_task_result(res)


@mcp.tool()
async def idc_code_qa(
    question: str,
    top_k: int = 5,
    max_chars: int = 8000,
    synthesize: bool = True,
    reasoning_effort: str = "low",
) -> dict:
    _ensure_configured()
    res = await code_qa_mod.idc_code_qa_runner(
        question=question,
        top_k=top_k,
        max_chars=max_chars,
        synthesize=synthesize,
        reasoning_effort=reasoning_effort,
    )
    return normalize_task_result(res)


@mcp.tool()
async def dicom2nifti(dicom_dir: str, out_dir: Optional[str] = None) -> dict:
    _ensure_configured()
    res = await d2n_mod.dicom2nifti_runner(dicom_dir=dicom_dir, out_dir=out_dir)
    return normalize_task_result(res)


if __name__ == "__main__":
    _ensure_configured()
    mcp.run()