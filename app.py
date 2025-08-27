# app.py  — LangGraph + LangChain Tools + Chainlit
import os
import asyncio
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any
import httpx

import chainlit as cl
from chainlit.types import ThreadDict
from dotenv import load_dotenv
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
)
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState          
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from idc_index import index

import tools.data_query as dq_mod
import tools.imaging as img_mod
import tools.viz_slider as vz_mod
import tools.radiomics as rad_mod
import tools.monai_infer as monai_mod
import tools.dicom_to_nifti as d2n_mod
import tools.code_exec as code_mod
import tools.universeg as ug_mod
import tools.midrc_query as midrc_mod
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

from pathlib import Path as _P

IDC_Client = index.IDCClient()
df_IDC = IDC_Client.index
df_BIH = pd.read_csv("Data/BIH_Cases_table.csv")
#df_MIDRC = pd.read_parquet("midrc_mirror/nodes/midrc_files_wide.parquet")
df_MIDRC = pd.DataFrame()
TS_CT = _P("Data/TotalSegmentatorMappingsCT.txt").read_text()
TS_MRI = _P("Data/TotalSegmentatorMappingsMRI.txt").read_text()
Monai_Instructions = _P("Data/monai_bundles_instructions.txt").read_text()

dq_mod.configure_data_query_tool(df_IDC=df_IDC, df_BIH=df_BIH, system_prompt=(_P("prompts/agent_systems/data_query.txt").read_text()))
img_mod.configure_imaging_tool(ct_mappings="")
vz_mod.configure_viz_slider_tool()
rad_mod.configure_radiomics_tool(system_prompt=(_P("prompts/agent_systems/radiomics.txt").read_text()))
monai_mod.configure_monai_tool(
    system_prompt=(_P("prompts/agent_systems/monai.txt").read_text()),
    additional_context=(_P("Data/monai_bundles_instructions.txt").read_text())
)
code_mod.configure_code_exec_tool(
    system_prompt=(_P("prompts/agent_systems/code_exec.txt").read_text()),
    df_IDC=df_IDC
)
midrc_mod.configure_midrc_query_tool(
    df_MIDRC=df_MIDRC,
    system_prompt=(_P("prompts/agent_systems/midrc_query.txt").read_text())
)

_ = dq_mod.data_query_runner
_ = img_mod.imaging_runner
_ = vz_mod.viz_slider_runner
_ = rad_mod.radiomics_runner
_ = monai_mod.monai_runner
_ = code_mod.code_exec_runner
_ = midrc_mod.midrc_query_runner

def build_graph(checkpointer=None):
    policy = SystemMessage(content=(
        f"""You are VoxelInsight, an AI radiology assistant.\n
        You have access to many TOOLS. Use them to answer user questions.\n

        NEVER use two tools at once. Use one tool, get the result, then if needed use another tool.\n
        All tools outputs are shown automatically in the the UI. You do not need to provide download links or try to display these outputs (like images, dataframes, sliders, etc) again.\n

        RULES FOR TOOL USE:\n

        - The data_query tool is designed to handle everything about IDC and BIH data. It can return dataframes, file download links, plots (it can make and output its own figures), and text. Use it for any questions about IDC data (counts, summaries, cohorts, df_IDC). You must use the data_query tool to answer any IDC questions. Never answer IDC questions without using the tool.\n
        - If users want plots of IDC data, use the data_query tool to generate them in one step since it can query the idc and make plots.\n
        - Most CT, MRI, and PET studies in IDC follow the standard DICOM hierarchy: Patient → Study → Series → Instances (slices/images). That means you’ll typically see a series (e.g., “T2w axial,” “CT lung window,” “PET SUV”), and inside each series you’ll have multiple DICOM files, one per slice or image. For this reason you may need to convert the series to a NIfTI volume to do segmentation or radiomics. Use the dicom_to_nifti tool to do this conversion.\n
        - You may use the llm based code_exec tool to generate and run arbitrary python code. Use it for general computation, data analysis, and visualization tasks that cannot be accomplished with other tools. You can use it to do radiomics analysis, segmentation using total segmentator, image processing, general data analysis and visualization, and many more general tasks which can't be done by other tools.\n
        - You can also use the code_exec to create and execute code for preprocessing and postprocessing of images and masks for other tools if needed.\n
        - For the monai_infer tool, make sure you provide the full paths to the image(s) to be used for inference via image_path.\n
        - For the imaging tool to segment the liver_tumor you have to use task name liver_vessels and roi_subset liver_tumor.\n
        - Use the radiomics tool to extract radiomics features like volume from images and masks using pyradiomics. Provide clear instructions about the features you want and any specific parameters. Make sure to provide the full paths to the image and mask files via image_path and mask_paths.\n
        - The radiomics tool can extract quantitative imaging features including first-order statistics, shape descriptors, and texture features (GLCM, GLRLM, GLSZM, NGTDM, GLDM), with optional computation on filtered images (e.g., wavelet, LoG).
        - The radiomics tool can only be run with one image mask pair at a time. If you have multiple masks for one image you can run the tool multiple times, once for each mask.\n
        - If a user wants you to do segmentation and then get radiomics features from the segmentation, make sure to do segmentation first and then use the produced masks to calculate radiomics features.\n
        - If a user want you to do segmentation and then visualize the result, make sure to do segmentation first and then use the viz_slider tool to visualize the result. If you do both the segmemntation step and visualization step at the same time the user won't be able to actually see the segmentations, they will only see the original image.\n
          
        RULES:\n
        If a tool returns an error, only retry a maximum of 3 times. If it still fails, inform the user you are unable to complete the task.\n
        Tool results arrive as JSON in a ToolMessage with schema:\n
          ok: bool, outputs: df_preview?: rows:[obj], nrows:int, text?: string, ... , summary?: string, ui?: [...], error?: string\n
          After tools run, READ that JSON and answer succinctly:\n
            - If outputs.text exists, use it.\n
            - Else if outputs.df_preview exists:\n
                - If exactly one row/one column, state that value plainly (e.g., 'Total patients: 12345').\n
                - Else summarize key columns briefly.\n
            - Outputs like files, images, plotly sliders, and dataframes made by tools are shown automatically in the UI. You do not need to provide download links or try to display these outputs again.\n
            - You may need to use the outputs of one tool as input to another tool. For example the imaging agents gives mask paths that can be visualized with the viz_slider agent\n
          -You must never provide download links or try to display outputs like images or sliders. Assume these are shown automatically in the UI when produced by tools.\n
          -Using the viz_slider tool will show an interactive slider in the UI automatically. You do not need to provide any download links or try to display the slider yourself.\n
          -If errors like shape mismatches occur when llm based agents execute their code, try to rerun the tool once giving more precise and strict instructions targeted at preventing the error.\n
          -When users ask for a segmentation task and then to visualize the result, make sure to use the appropriate tool to do the segmentation first and then use the viz slider or other tools to visualize the result.\n
          -You are allowed to make the instuctions for llm based tool more strict and lengthy to prevent errors. For example you can specify exact image shapes, data types, and other details in your instructions.\n
          -Be as precise and specific as possible in your instruction when rerunning a tool to fix an error. This will increase the likelihood of an llm based tool giving a correct output. For example in the case of a shape mismatch in the instruction include the original shape of the image and tell the tool what shape it must convert the image to.\n
          -The user may upload files at the start of the session. You are provided the paths of these files in the session context. You can use these paths when responding to the user, in your instructions to tools, or in the code_exec tool.\n
          -The user may be referring to these files if they say 'this file', 'the image', or similar. In this case use the most recently uploaded file.\n
          -Do not ask follow-ups unless a critical filter is missing.\n
          -Never fabricate IDC answers without the tool.\n

        - The Imaging tool can perform segmentation using TotalSegmentator. Be careful when selecting the task and roi_subset to use. Here are the mapping for TotalSegmentator tasks and roi_subset values:\n
        - CT Mappings:\n{TS_CT}\n
        - MRI Mappings:\n{TS_MRI}\n

        - The monai_infer tool can run MONAI bundles for segmentation. Note monai and totalsegmentator are completely different: do not mix them up. Here are some of the bundles you can run and instructions for each: {Monai_Instructions}\n

        """
    ))

    print("TOOLS:", [t.name for t in TOOL_REGISTRY])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOL_REGISTRY)
    tool_node = ToolNode(tools=TOOL_REGISTRY)  

    async def call_model(state: MessagesState):
        msgs = [policy] + state["messages"]
        for attempt in range(2):  
            try:
                resp = await llm.ainvoke(msgs)
                print("Agent tool_calls:", getattr(resp, "tool_calls", None))
                return {"messages": [resp]}
            except httpx.RemoteProtocolError as e:
                if attempt == 0:
                    print("Transient stream error, retrying once:", repr(e))
                    continue
                raise

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

_CP = MemorySaver()
GRAPH = build_graph(checkpointer=_CP)


# Helpers 
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
                payloads.append(content)
            else:
                try:
                    import json
                    payloads.append(json.loads(content))
                except Exception:
                    pass
    return payloads


async def _render_payload(payload: Dict[str, Any]):
    ok = payload.get("ok", True)
    if not ok:
        err = payload.get("error") or "Tool returned an error."
        await cl.Message(content=f"⚠️ {err}").send()
        return

    outputs = payload.get("outputs", {}) or {}
    ui = payload.get("ui", []) or []

    # UI items
    for item in ui:
        kind = item.get("kind")
        if kind == "plotly_json_path":
            path = item.get("path")
            try:
                import json
                from plotly.io import from_json
                spec = Path(path).read_text()
                fig = from_json(spec)
                await cl.Message(
                    content="Interactive chart:",
                    elements=[cl.Plotly(name="plot", figure=fig)]
                ).send()
            except Exception:
                await cl.Message(content="(Plotly figure could not be rendered.)").send()
        elif kind == "image_path":
            path = item.get("path")
            if path and Path(path).exists():
                await cl.Message(
                    content="Here is your result:",
                    elements=[cl.Image(name=Path(path).name, path=path)]
                ).send()

    # Tables
    if "df_preview" in outputs:
        prev = outputs["df_preview"]
        rows = prev.get("rows", [])
        if rows:
            df = pd.DataFrame(rows)
            await cl.Message(content=df.to_markdown(index=False)).send()
        else:
            await cl.Message(content="(No rows returned.)").send()

    # Files
    files = outputs.get("files", [])
    output_dir = outputs.get("output_dir")
    if not files and output_dir and Path(output_dir).exists():
        files = [str(f) for f in Path(output_dir).rglob("*") if f.is_file()]
    if files:
        zip_tmpdir = Path(tempfile.mkdtemp(prefix="vi_zip_"))
        zip_path = zip_tmpdir / "download.zip"
        await _zip_paths(files, zip_path)
        await cl.Message(
            content=f"📦 **Files ready**\n- Items: {len(files)}\n\nClick to download:",
            elements=[cl.File(name=zip_path.name, path=str(zip_path))]
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Collect any uploaded files
    file_elements = [el for el in (message.elements or []) if isinstance(el, cl.File)]
    files: List[str] = []
    for f in file_elements:
        tmpdir = Path(tempfile.mkdtemp())
        new_path = tmpdir / f.name
        shutil.copy(f.path, new_path)
        files.append(str(new_path))

    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    initial_state = {
        "messages": [HumanMessage(content=message.content + ("" if not files else f" User uploaded files: {files}"))],
    }

    # Stream the graph 
    try:
        async for event, meta in GRAPH.astream(
            initial_state,
            stream_mode="messages",
            config=RunnableConfig(callbacks=[cb], **config),
        ):
            print("STREAM:", type(event).__name__, meta.get("langgraph_node"))

            if isinstance(event, ToolMessage):
                print("TOOL MESSAGE CONTENT:", event.content) 
                payloads = _collect_tool_payloads([event])
                for p in payloads:
                    await _render_payload(p)

            if (
                getattr(event, "content", None)
                and isinstance(event, AIMessage)
                and not getattr(event, "tool_calls", None)
                and meta.get("langgraph_node") in ("final", "agent")
            ):
                await final_answer.stream_token(event.content)

        await final_answer.send()
    except Exception as e:
        import traceback; traceback.print_exc()
        await cl.Message(content=f"🚨 Error:\n```\n{type(e).__name__}: {e}\n```").send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    pass

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
            message="How many patients are currently in IDC? Please cite the source and date.",
            icon="/public/database.svg",
        ),
        cl.Starter(
            label="Search IDC & Plot Histogram",
            message="Plot a histogram of the number patients for all breast collections in IDC.",
            icon="/public/search.svg",
        ),
        cl.Starter(
            label="Radiomics guide",
            message="Explain which radiomics feature families VoxelInsight supports and how to export results as CSV.",
            icon="/public/chart.svg",
        ),
    ]
