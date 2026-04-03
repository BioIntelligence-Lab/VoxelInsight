from __future__ import annotations

import os
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from idc_index import index

import tools.idc_query as dq_mod
import tools.imaging as img_mod
import tools.viz_slider as vz_mod
import tools.radiomics as rad_mod
import tools.monai_infer as monai_mod
import tools.segmentation_orchestrator as seg_mod
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
#import tools.merlin_3d as merlin3d_mod
from tools.shared import toolify_agent, TOOL_REGISTRY
from core.state import TaskResult
from core.supervisor_llm import build_supervisor_llm


_CONFIGURED: Dict[str, bool] = {"full": False, "idc": False}
_GRAPH_CACHE: Dict[str, Any] = {}
_CHECKPOINTERS: Dict[str, MemorySaver] = {}


def _configure_full_tools() -> None:
    if _CONFIGURED["full"]:
        return

    load_dotenv()
    idc_client = index.IDCClient()
    df_IDC = idc_client.index
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

    ts_ct = pd.DataFrame()
    ts_mri = pd.DataFrame()
    monai_instructions = ""
    if "imaging" in os.getenv("VOXELINSIGHT_TOOLS", ""):
        ts_ct = pd.read_csv("Data/TotalSegmentatorMappingsCT.tsv", sep="\t")
        ts_mri = pd.read_csv("Data/TotalSegmentatorMappingsMRI.tsv", sep="\t")
    if "monai" in os.getenv("VOXELINSIGHT_TOOLS", ""):
        monai_instructions = Path("Data/monai_bundles_instructions.txt").read_text()
    '''
    merlin3d_mod.configure_merlin_tool(
        device=None,
        cache_root=None,
        merlin_kwargs=None,
    )
    '''
    dq_mod.configure_idc_query_tool(
        df_IDC=df_IDC,
        df_BIH=df_BIH,
        system_prompt=Path("prompts/agent_systems/idc_query.txt").read_text(),
    )
    img_mod.configure_imaging_tool(ct_mappings="")
    vz_mod.configure_viz_slider_tool()
    rad_mod.configure_radiomics_tool(
        system_prompt=Path("prompts/agent_systems/radiomics.txt").read_text()
    )
    monai_mod.configure_monai_tool(
        system_prompt=Path("prompts/agent_systems/monai.txt").read_text(),
        additional_context=monai_instructions,
    )
    code_mod.configure_code_gen_tool(
        system_prompt=Path("prompts/agent_systems/code_gen.txt").read_text(),
        df_IDC=df_IDC,
    )
    '''
    midrc_mod.configure_midrc_query_tool(
        df_MIDRC=df_MIDRC,
        system_prompt=Path("prompts/agent_systems/midrc_query.txt").read_text(),
    )
    '''
    bih_mod.configure_bih_query_tool(
        df_BIH=df_BIH,
        system_prompt=Path("prompts/agent_systems/bih_query.txt").read_text(),
    )
    midrc_dl_mod.configure_midrc_download_tool()
    tcia_dl_mod.configure_tcia_download_tool()
    idc_dl_mod.configure_idc_download_tool()
    clin_mod.configure_clinical_data_tool()
    ir_mod.configure_image_registration_tool()
    _ = seg_mod.segmentation_orchestrator_runner

    _CONFIGURED["full"] = True


def _configure_idc_tools() -> None:
    if _CONFIGURED["idc"]:
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
    clin_mod.configure_clinical_data_tool()

    _CONFIGURED["idc"] = True


def _policy_text(pipeline: str) -> str:
    if pipeline == "idc":
        return (
            "You are **VoxelInsight IDC**, a multi-agent assistant for IDC metadata queries "
            "and clinical data exports.\n\n"
            "Core behavior\n"
            "- Only answer what the user asked; request clarifications solely when required to complete a tool call.\n"
            "- Only when asked about VoxelInsight, answer yourself otherwise always use tools. YOU ARE NOT ALLOWED TO ANSWER DIRECTLY.\n"
            "- Tools cannot see each other’s outputs—pass important values (SeriesInstanceUIDs, directories, etc.) yourself.\n"
            "- Keep tool instructions brief unless retrying an error. Retry at most 3 times (when it seems reasonable/necessary) with progressively clearer directions.\n"
            "- Before each tool call, tell the user what you are about to do in one concise sentence.\n"
            "- For llm based tools where you pass a reasoning_effort parameter, choose the lowest reasoning effort level that is likely to complete the task successfully.\n"
            "- When you output code snippets, ensure they are properly fenced with triple backticks and the appropriate language identifier.\n"
            "- Use `idc_query` for IDC metadata questions and use `clinical_data_download` only after identifying the correct collection.\n\n"
            "Output & chaining\n"
            "- This orchestrator is exposed through MCP, so the user only sees your final text response. No files, plots, tables, or other artifacts are rendered automatically.\n"
            "- Summarize relevant tool results directly in your final response instead of assuming the user can inspect artifacts.\n"
            "- Chain tools sequentially when needed rather than launching them all at once.\n"
            "- Never expose local filesystem paths in the final response—describe outcomes instead.\n"
        )

    return (
        "You are **VoxelInsight**, an AI radiology assistant/agent.\n"
        "You have access to many TOOLS. Use them carefully to answer user questions.\n\n"
        "You are an agent\n"
        "- Do not ask the user follow up questions unless absolutely necessary.\n"
        "- Do not do more than what the user requests.\n"
        "- Do not suggest next steps for the user unless they ask for them.\n"
        "- For downloading DICOM Series or histopathology tiles, always download one patient at a time.\n"
        "- When using llm based tools which generate code, be as concise as possible in your instructions, unless an error occurs.\n"
        "- Assume that tools cannot see each other's output or the conversation history. You must pass information between tools yourself.\n"
        "- If the user just wants to view a study from a collection in IDC, you do not need to download the study.\n\n"
        "General Principles\n"
        "1) Chain tools when needed; run one tool, inspect the result, then use another tool if needed.\n"
        "2) This orchestrator is primarily used through MCP, and the user only sees your final text response. Tool outputs are not rendered automatically in a UI.\n"
        "3) Summarize the important results from tools in your final response rather than assuming the user can inspect files, plots, or tables.\n"
        "4) Error handling: retry a failing tool max 3 times; refine instructions.\n"
        "5) File context: if user says “this file,” assume most recent upload.\n"
        "6) Other rules: keep instructions concise, no local file paths in final response.\n\n"
        "Tool Usage Rules\n"
        "- For reasoning_effort, start with 'low' and increase only if needed.\n"
        "- Before every tool call, tell the user what you are doing in concise language.\n"
        "- Use `segmentation_orchestrator` as the single entrypoint for segmentation requests.\n"
        "- For clinical_data_download, always identify the correct collection using idc_query first.\n"
    )


def _resolve_tools(pipeline: str, tool_names: Optional[List[str]]) -> List[Any]:
    tools = [t for t in TOOL_REGISTRY if t.name not in {"orchestrator", "imaging", "monai"}]
    default_full = {
        "idc_query",
        "segmentation_orchestrator",
        "viz_slider",
        "radiomics",
        "code_gen",
        "midrc_query",
        "bih_query",
        "midrc_download",
        "tcia_download",
        "idc_download",
        "clinical_data_download",
        "image_registration",
        "merlin_3d",
        "dicom2nifti",
        "universeg",
    }
    default_idc = {
        "idc_query",
        "clinical_data_download",
    }
    if pipeline == "idc":
        tools = [t for t in tools if t.name in default_idc]
    else:
        tools = [t for t in tools if t.name in default_full]

    if not tool_names:
        return tools

    requested = {n.strip() for n in tool_names if n and n.strip()}
    subset = [t for t in tools if t.name in requested]
    return subset or tools


def _build_graph(pipeline: str, tools: List[Any], checkpointer: Optional[MemorySaver]) -> Any:
    policy = SystemMessage(content=_policy_text(pipeline))
    base_llm = build_supervisor_llm(temperature=1, reasoning_effort="low")
    llm = base_llm.bind_tools(tools)
    tool_node = ToolNode(tools=tools)

    async def call_model(state: MessagesState):
        msgs = [policy] + state["messages"]
        for attempt in range(2):
            try:
                resp = await llm.ainvoke(msgs)
                return {"messages": [resp]}
            except httpx.RemoteProtocolError as e:
                if attempt == 0:
                    continue
                raise e

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "final"

    async def call_final(state: MessagesState):
        return {}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_node("tools", tool_node)
    g.add_node("final", call_final)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", "final": "final"})
    g.add_edge("tools", "agent")
    g.add_edge("final", END)
    return g.compile(checkpointer=checkpointer)


def _collect_tool_payloads(messages: List[Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, dict):
                payloads.append(content)
            else:
                try:
                    import json
                    payloads.append(json.loads(content))
                except Exception:
                    pass
    return payloads


def _extract_final_text(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return str(getattr(msg, "content", "") or "")
    return ""


class OrchestratorArgs(BaseModel):
    query: str = Field(..., description="User request to run through the VoxelInsight pipeline.")
    pipeline: str = Field("full", description="Pipeline: 'full' or 'idc'.")
    tool_names: Optional[List[str]] = Field(
        None,
        description="Optional tool subset by name. When omitted, uses the pipeline defaults.",
    )
    files: Optional[List[str]] = Field(
        None,
        description="Optional file paths to inject into the query context.",
    )
    thread_id: Optional[str] = Field(
        None,
        description="Optional thread id to keep state between calls within this process.",
    )
    include_tool_payloads: bool = Field(
        True,
        description="Include raw tool payloads in the output for downstream rendering.",
    )


@toolify_agent(
    name="orchestrator",
    description=(
        "IDC-focused supervisor for VoxelInsight MCP requests. Use this as the single entrypoint "
        "if you want VoxelInsight to plan and execute an IDC task across its internal tools, then return one final "
        "text answer. The orchestrator can handle IDC metadata and collection questions plus clinical data export "
        "requests, including multi-step tool chaining when needed. It decides which internal IDC tools to call, "
        "passes outputs between them, and summarizes the relevant results for the caller."
    ),
    args_schema=OrchestratorArgs,
    timeout_s=1200,
)
async def orchestrator_runner(
    query: str,
    pipeline: str = "full",
    tool_names: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
    include_tool_payloads: bool = True,
):
    pipeline = (pipeline or "full").strip().lower()
    if pipeline not in {"full", "idc"}:
        raise ValueError("pipeline must be 'full' or 'idc'.")

    if pipeline == "idc":
        _configure_idc_tools()
    else:
        _configure_full_tools()

    tools = _resolve_tools(pipeline, tool_names)

    cache_key = f"{pipeline}:{','.join(sorted(t.name for t in tools))}"
    if thread_id:
        cache_key = f"{cache_key}:mem"
    else:
        cache_key = f"{cache_key}:nomem"

    checkpointer = None
    if thread_id:
        if cache_key not in _CHECKPOINTERS:
            _CHECKPOINTERS[cache_key] = MemorySaver()
        checkpointer = _CHECKPOINTERS[cache_key]

    if cache_key not in _GRAPH_CACHE:
        _GRAPH_CACHE[cache_key] = _build_graph(pipeline, tools, checkpointer)

    graph = _GRAPH_CACHE[cache_key]

    user_content = query
    if files:
        user_content = f"{query} User uploaded files: {files}"

    initial_state = {"messages": [HumanMessage(content=user_content)]}
    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        initial_state,
        config=RunnableConfig(**config),
    )

    messages = result.get("messages", [])
    final_text = _extract_final_text(messages)
    payloads = _collect_tool_payloads(messages) if include_tool_payloads else []

    output = {"text": final_text}
    if include_tool_payloads:
        output["tool_payloads"] = payloads

    return TaskResult(output=output, artifacts={})
