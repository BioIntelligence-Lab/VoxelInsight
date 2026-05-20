import os
import asyncio
import zipfile
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, List, Any

import httpx
import pandas as pd

import chainlit as cl
from chainlit.types import ThreadDict
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    AIMessageChunk,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import BaseTool

try:
    from deepagents import create_deep_agent
except Exception as e:
    create_deep_agent = None
    _DEEPAGENTS_IMPORT_ERROR = e
else:
    _DEEPAGENTS_IMPORT_ERROR = None

from core.supervisor_llm import build_supervisor_llm
from core.storage import get_temp_dir
from idc_index import index

import tools.idc_query as dq_mod
import tools.imaging as img_mod
import tools.viz_slider as vz_mod
import tools.radiomics as rad_mod
import tools.monai_infer as monai_mod
import tools.dicom_to_nifti as d2n_mod
import tools.code_gen as code_mod
import tools.universeg as ug_mod
import tools.midrc_query as midrc_mod
import tools.midrc_download as midrc_dl_mod
import tools.bih_query as bih_mod
import tools.tcia_download as tcia_dl_mod
import tools.idc_download as idc_dl_mod
import tools.clinical_data as clin_mod
import tools.image_registration as ir_mod
import tools.merlin_3d as merlin3d_mod
from tools.shared import TOOL_REGISTRY


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


load_dotenv()


IDC_Client = index.IDCClient()
df_IDC = IDC_Client.index
try:
    df_BIH = pd.read_csv("Data/BIH_Cases_table.csv", low_memory=False)
except Exception as e:
    print(f"Warning: could not load BIH data ({e})")
    df_BIH = pd.DataFrame()
try:
    df_MIDRC = pd.read_parquet("midrc_mirror/nodes/midrc_files_wide.parquet")
except Exception as e:
    print(f"Warning: could not load MIDRC data ({e})")
    df_MIDRC = pd.DataFrame()


def _read_text(path: str) -> str:
    p = Path(path)
    return p.read_text() if p.exists() else ""


merlin3d_mod.configure_merlin_tool(
    device=None,
    cache_root=None,
    merlin_kwargs=None,
)
dq_mod.configure_idc_query_tool(
    df_IDC=df_IDC,
    df_BIH=df_BIH,
    system_prompt=_read_text("prompts/agent_systems/idc_query.txt"),
)
img_mod.configure_imaging_tool(ct_mappings="")
vz_mod.configure_viz_slider_tool()
rad_mod.configure_radiomics_tool(system_prompt=_read_text("prompts/agent_systems/radiomics.txt"))
monai_mod.configure_monai_tool(
    system_prompt=_read_text("prompts/agent_systems/monai.txt"),
    additional_context=_read_text("Data/monai_bundles_instructions.txt"),
)
code_mod.configure_code_gen_tool(
    system_prompt=_read_text("prompts/agent_systems/code_gen.txt"),
    df_IDC=df_IDC,
)
midrc_mod.configure_midrc_query_tool(
    df_MIDRC=df_MIDRC,
    system_prompt=_read_text("prompts/agent_systems/midrc_query.txt"),
)
bih_mod.configure_bih_query_tool(
    df_BIH=df_BIH,
    system_prompt=_read_text("prompts/agent_systems/bih_query.txt"),
)
midrc_dl_mod.configure_midrc_download_tool()
tcia_dl_mod.configure_tcia_download_tool()
idc_dl_mod.configure_idc_download_tool()
clin_mod.configure_clinical_data_tool()
ir_mod.configure_image_registration_tool()

_ = dq_mod.idc_query_runner
_ = img_mod.imaging_runner
_ = vz_mod.viz_slider_runner
_ = rad_mod.radiomics_runner
_ = monai_mod.monai_runner
_ = code_mod.code_gen_runner
_ = midrc_mod.midrc_query_runner
_ = bih_mod.bih_query_runner
_ = midrc_dl_mod.midrc_download_runner
_ = tcia_dl_mod.tcia_download_runner
_ = idc_dl_mod.idc_download_runner
_ = clin_mod.clinical_data_download_runner
_ = ir_mod.image_registration_runner
_ = merlin3d_mod.merlin_3d_runner
_ = ug_mod.universeg_runner


ALL_TOOLS: tuple[BaseTool, ...] = tuple(TOOL_REGISTRY)
TOOL_NAMES = {tool.name: tool for tool in ALL_TOOLS}
SEGMENTATION_TOOL_NAMES = {"imaging", "monai"}
HIDDEN_TOP_LEVEL_TOOLS = SEGMENTATION_TOOL_NAMES | {"segmentation_orchestrator"}


def resolve_tool_subset(raw: str | None) -> List[BaseTool]:
    available_tools = [tool for tool in ALL_TOOLS if tool.name not in HIDDEN_TOP_LEVEL_TOOLS]
    if not raw:
        return available_tools
    requested = {name.strip() for name in raw.split(",") if name.strip()}
    subset = [
        TOOL_NAMES[name]
        for name in requested
        if name in TOOL_NAMES and name not in HIDDEN_TOP_LEVEL_TOOLS
    ]
    return subset or available_tools


USED_TOOLS = resolve_tool_subset(os.getenv("VOXELINSIGHT_TOOLS"))
SEGMENTATION_TOOLS = [TOOL_NAMES[name] for name in SEGMENTATION_TOOL_NAMES if name in TOOL_NAMES]


def _load_ts_mappings_for_prompt() -> str:
    lines: List[str] = []
    for mapping_file in ("Data/TotalSegmentatorMappingsCT.tsv", "Data/TotalSegmentatorMappingsMRI.tsv"):
        path = Path(mapping_file)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            continue
        if {"task_name", "roi_subset"}.issubset(df.columns):
            grouped = (
                df[["task_name", "roi_subset"]]
                .dropna()
                .astype(str)
                .groupby("task_name")["roi_subset"]
                .apply(lambda s: ", ".join(sorted(set(s))))
            )
            lines.extend(f"- {task}: {rois}" for task, rois in grouped.items())
    return "\n".join(lines) if lines else "(No TotalSegmentator mappings available.)"


def _main_policy() -> str:
    return f"""
You are VoxelInsight, an AI radiology assistant/agent.
You have access to VoxelInsight tools plus a Deep Agents `task` tool for delegation.

Behavior
- Do not ask follow-up questions unless absolutely necessary.
- Do not do more than the user requests.
- Do not suggest next steps unless the user asks for them.
- Before every tool call, tell the user what you are doing in concise, understandable language.
- Assume tools and subagents cannot see each other's detailed outputs unless you pass the needed information explicitly.
- Retry a failing tool a maximum of 3 times with clearer instructions. If it still fails, say you cannot complete the task.

Segmentation
- Delegate all segmentation requests to the `segmentation-agent` subagent using the `task` tool.
- Do not call `imaging` or `monai` from the main agent; those tools are only available to the segmentation subagent.
- If multiple files are provided, pass them together in the delegated task.
- For segment then visualize or segment then radiomics workflows, first delegate segmentation, inspect the subagent result, then pass the resulting masks to `viz_slider` or `radiomics`.

Dataset and download rules
- Never fabricate IDC answers without `idc_query`.
- For IDC viewer-only requests, use `idc_query`; downloading is not needed.
- For DICOM Series or histopathology downloads, download one patient at a time.
- For clinical data downloads, first use `idc_query` to identify the correct collection.
- For TCIA downloads, use no-API-key download behavior. If the requested collection is not supported by `tcia_download`, say so.

UI and artifact rules
- File, image, plot, dataframe, slider, and download outputs are automatically rendered by the UI.
- Never expose local filesystem paths in final user-facing responses.
- You may mention text links returned by tools when relevant.
- Use `code_gen` for Plotly figures or custom generated visual/file outputs when no specialized tool exists.

Tool result handling
- Tool results arrive as JSON-like payloads: ok, outputs, ui, error, and related fields.
- If outputs.text exists, use it to answer.
- If outputs.df_preview exists, summarize key columns briefly; for a single row/column, return the plain value.
- Chain outputs manually between tools when needed.
"""


def _segmentation_policy() -> str:
    return f"""
You are VoxelInsight Segmentation, a specialized Deep Agents subagent for segmentation only.

Scope
- Your only job is to choose and run the correct segmentation tool.
- Use TotalSegmentator through `imaging` or MONAI bundles through `monai`.
- Do not answer metadata, downloads, radiomics, or visualization requests yourself.
- If the task is not segmentation, return a short internal explanation and do not call tools.

Tool selection
- Prefer `imaging` for TotalSegmentator-style anatomical segmentation requests.
- Prefer `monai` when the user asks for MONAI, a MONAI model, or a bundle-specific workflow.
- Keep tool instructions concise but include file paths and enough detail to avoid input-shape ambiguity.
- Retry a failing segmentation tool at most 2 additional times with clearer instructions.

TotalSegmentator contract for `imaging`
- Prefer `task_name=total` for CT and `task_name=total_mr` for MRI when valid ROI subsets satisfy the request.
- Prefer `roi_subset` or `roi_subsets` over a full-task run when the user requests specific structures.
- Use specialized non-total tasks only when the requested output cannot be produced by total/total_mr ROI subsets.
- Normalize task and ROI values to canonical lowercase underscore tokens.
- Never invent task names or ROI values outside the mapping table.
- Allowed mappings:
{_load_ts_mappings_for_prompt()}

Final response
- Return a terse internal status note for the main VoxelInsight agent.
- Include any mask or output artifact information from tool outputs so the main agent can chain visualization or radiomics.
- Do not write polished user-facing prose.
"""


def build_agent(checkpointer: Optional[MemorySaver] = None):
    if create_deep_agent is None:
        raise RuntimeError(
            "The `deepagents` package is not installed or could not be imported. "
            f"Original import error: {_DEEPAGENTS_IMPORT_ERROR}"
        )

    print("using top-level tools:", [t.name for t in USED_TOOLS])
    print("using segmentation subagent tools:", [t.name for t in SEGMENTATION_TOOLS])

    model = build_supervisor_llm(temperature=1, reasoning_effort="low")
    subagents = [
        {
            "name": "segmentation-agent",
            "description": (
                "Routes medical image segmentation requests to TotalSegmentator or MONAI and returns "
                "the produced segmentation artifacts for downstream visualization or radiomics."
            ),
            "system_prompt": _segmentation_policy(),
            "tools": SEGMENTATION_TOOLS,
            "model": model,
        }
    ]

    kwargs: Dict[str, Any] = {
        "model": model,
        "tools": USED_TOOLS,
        "system_prompt": _main_policy(),
        "subagents": subagents,
        "name": "voxelinsight-deepagent",
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    return create_deep_agent(**kwargs)


_CP = MemorySaver()
GRAPH = None


def _get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = build_agent(checkpointer=_CP)
    return GRAPH


async def _zip_paths(paths: List[str], zip_path: Path):
    def _worker():
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                pth = Path(p)
                if pth.is_dir():
                    base = pth.parent
                    for f in pth.rglob("*"):
                        if f.is_file():
                            zf.write(f, arcname=str(f.relative_to(base)))
                elif pth.is_file():
                    zf.write(pth, arcname=pth.name)
    await asyncio.to_thread(_worker)


def _collect_tool_payloads(messages: List[Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, dict):
                payload = dict(content)
                payload.setdefault("tool_name", getattr(m, "name", None))
                payloads.append(payload)
            else:
                try:
                    decoded = json.loads(content)
                    if isinstance(decoded, dict):
                        decoded.setdefault("tool_name", getattr(m, "name", None))
                        payloads.append(decoded)
                except Exception:
                    pass
    return payloads


async def _render_payload(payload: Dict[str, Any]):
    ok = payload.get("ok", True)
    if not ok:
        err = payload.get("error") or "Tool returned an error."
        await cl.Message(content=f"Warning: {err}").send()
        return

    outputs = payload.get("outputs", {}) or {}
    ui = payload.get("ui", []) or []

    code_text = outputs.get("code")
    if code_text:
        code_el = cl.CustomElement(
            name="IdcCodeView",
            props={"code": str(code_text), "title": "Generated code"},
            display="inline",
        )
        await cl.Message(content="Generated code (expand to inspect):", elements=[code_el]).send()

    for item in ui:
        kind = item.get("kind")
        if kind == "plotly_json_path":
            path = item.get("path")
            try:
                from plotly.io import from_json
                spec = Path(path).read_text()
                fig = from_json(spec)
                await cl.Message(
                    content="Interactive chart:",
                    elements=[cl.Plotly(name="plot", figure=fig)],
                ).send()
            except Exception:
                await cl.Message(content="(Plotly figure could not be rendered.)").send()
        elif kind == "image_path":
            path = item.get("path")
            if path and Path(path).exists():
                await cl.Message(
                    content="Here is your result:",
                    elements=[cl.Image(name=Path(path).name, path=path)],
                ).send()
        elif kind == "binary_path":
            path = item.get("path")
            if path and Path(path).exists():
                await cl.Message(
                    content="Here is your file:",
                    elements=[cl.File(name=Path(path).name, path=path)],
                ).send()

    files = outputs.get("files", [])
    output_dir = outputs.get("output_dir")
    tool = outputs.get("tool", "unknown_tool")
    if not files and output_dir and Path(output_dir).exists():
        files = [str(f) for f in Path(output_dir).rglob("*") if f.is_file()]
    if files:
        zip_tmpdir = get_temp_dir(prefix="vi_zip")
        zip_path = zip_tmpdir / "download.zip"
        await _zip_paths(files, zip_path)
        if tool == "dicom2nifti":
            output_content = f"**Dicom to Nifti conversion complete:**\n- Nifti Files: {len(files)}\n\nClick to download:"
        elif tool == "tcia_download":
            output_content = f"**TCIA Download complete:**\n- Items: {len(files)}\n\nClick to download:"
        elif tool == "midrc_download":
            output_content = f"**MIDRC Download complete:**\n- Items: {len(files)}\n\nClick to download:"
        else:
            output_content = f"**Files ready**\n- Items: {len(files)}\n\nClick to download:"

        await cl.Message(
            content=output_content,
            elements=[cl.File(name=zip_path.name, path=str(zip_path))],
        ).send()


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text") or "")
        return "".join(chunks)
    return str(content or "")


def _extract_stream_part(part: Any):
    if isinstance(part, dict) and {"type", "data"}.issubset(part.keys()):
        part_type = part.get("type")
        data = part.get("data")
        ns = tuple(part.get("ns") or ())
        if part_type == "messages" and isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1], ns
        if part_type == "updates":
            return None, data, ns
        return None, data, ns
    if isinstance(part, (tuple, list)) and len(part) == 2:
        return part[0], part[1], ()
    return part, {}, ()


TOOL_DESCRIPTIONS = {
    "task": "Segmentation Subagent",
    "idc_query": "IDC Query Tool",
    "bih_query": "BIH Query Tool",
    "imaging": "TotalSegmentator Segmentation - this may take a while",
    "monai": "MONAI Segmentation - this may take a while",
    "radiomics": "Radiomics Analysis",
    "viz_slider": "Slider Visualization Tool",
    "dicom_to_nifti": "DICOM to NIfTI Conversion",
    "code_gen": "Code Generation",
    "midrc_query": "MIDRC Query Tool",
    "midrc_download": "MIDRC Download Tool",
    "tcia_download": "TCIA Download Tool",
    "idc_download": "IDC Download Tool",
    "clinical_data_download": "Clinical Data Download",
    "image_registration": "Image Registration",
    "merlin_3d": "Merlin 3D Embedding",
    "universeg": "Universeg Segmentation",
}


@cl.on_message
async def on_message(message: cl.Message):
    file_elements = [el for el in (message.elements or []) if isinstance(el, cl.File)]
    files: List[str] = []
    for f in file_elements:
        tmpdir = get_temp_dir(prefix="uploads")
        new_path = tmpdir / f.name
        shutil.copy(f.path, new_path)
        files.append(str(new_path))

    config = {"configurable": {"thread_id": cl.context.session.id}}
    tool_descriptions = TOOL_DESCRIPTIONS
    announced_tool_calls: set[str] = set()

    streaming_reply: Optional[cl.Message] = None

    async def _ensure_streaming_reply() -> cl.Message:
        nonlocal streaming_reply
        if streaming_reply is None:
            streaming_reply = cl.Message(content="")
            await streaming_reply.send()
        return streaming_reply

    async def _finalize_streaming_reply():
        nonlocal streaming_reply
        if streaming_reply is not None:
            try:
                await streaming_reply.update()
            finally:
                streaming_reply = None

    user_content = message.content + ("" if not files else f" User uploaded files: {files}")
    initial_state = {"messages": [HumanMessage(content=user_content)]}

    async def _announce_tool_call(tool_name: Optional[str], call_id: Optional[str] = None):
        if not tool_name:
            return
        key = call_id or tool_name
        if key in announced_tool_calls:
            return
        announced_tool_calls.add(key)
        label = tool_descriptions.get(tool_name, tool_name)
        await _finalize_streaming_reply()
        await cl.Message(content=f"Running {label}...").send()

    try:
        graph = _get_graph()
        async for raw_part in graph.astream(
            initial_state,
            stream_mode="messages",
            subgraphs=True,
            version="v2",
            config=RunnableConfig(**config),
        ):
            event, meta, namespace = _extract_stream_part(raw_part)
            is_subagent = any(str(segment).startswith("tools:") for segment in namespace)

            if isinstance(event, ToolMessage):
                payloads = _collect_tool_payloads([event])
                for p in payloads:
                    await _render_payload(p)

            if isinstance(event, (AIMessage, AIMessageChunk)):
                content = _stringify_content(getattr(event, "content", None))
                tool_calls = getattr(event, "tool_calls", None) or []
                tool_call_chunks = getattr(event, "tool_call_chunks", None) or []
                has_tool_call = bool(tool_calls)
                has_tool_call_chunks = bool(tool_call_chunks)
                if has_tool_call or has_tool_call_chunks or is_subagent:
                    for call in tool_calls:
                        if isinstance(call, dict):
                            await _announce_tool_call(call.get("name"), call.get("id"))
                    for chunk in tool_call_chunks:
                        if isinstance(chunk, dict) and chunk.get("name"):
                            await _announce_tool_call(chunk.get("name"), chunk.get("id"))
                    if content and not is_subagent:
                        await cl.Message(content=content).send()
                    await _finalize_streaming_reply()
                    continue
                if content:
                    msg = await _ensure_streaming_reply()
                    await msg.stream_token(content)

        await _finalize_streaming_reply()

    except TypeError as e:
        if "subgraphs" not in str(e) and "version" not in str(e):
            raise
        await _finalize_streaming_reply()
        await cl.Message(
            content=(
                "This installed LangGraph/Deep Agents stack does not support the current "
                "`astream(..., subgraphs=True, version='v2')` streaming API. "
                f"Error: {e}"
            )
        ).send()
    except httpx.RemoteProtocolError:
        await _finalize_streaming_reply()
        await cl.Message(content="The model stream ended unexpectedly. Please retry the request.").send()
    except Exception as e:
        import traceback
        traceback.print_exc()
        await cl.Message(content=f"Error:\n```\n{type(e).__name__}: {e}\n```").send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    pass


@cl.action_callback("action_button")
async def on_action(action):
    await action.remove()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="What can you do?",
            message="Give me a quick tour of VoxelInsight features and common workflows.",
            icon="/public/info.svg",
        ),
        cl.Starter(
            label="How many patients are in IDC?",
            message="How many patients are currently in IDC?",
            icon="/public/database.svg",
        ),
        cl.Starter(
            label="Search IDC & Plot Histogram",
            message="Plot a histogram of the number patients for all breast collections in IDC.",
            icon="/public/search.svg",
        ),
        cl.Starter(
            label="Segment uploaded scan",
            message="Segment the liver in my uploaded CT scan and show me the result.",
            icon="/public/chart.svg",
        ),
    ]
