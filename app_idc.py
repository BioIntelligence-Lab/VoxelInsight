import os
import asyncio
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any
import httpx

from core.llm_provider import choose_llm, OpenAISettings, AnthropicSettings

import chainlit as cl
from chainlit.types import ThreadDict
from dotenv import load_dotenv
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    AIMessageChunk,
)
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState          
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool

from idc_index import index

import tools.idc_query as dq_mod
import tools.dicom_to_nifti as d2n_mod
import tools.idc_download as idc_dl_mod
import tools.idc_web_qa as webqa_mod
import tools.pathology_download as path_mod
import tools.clinical_data as clin_mod
from tools.shared import TOOL_REGISTRY

'''
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user
'''

load_dotenv()

from pathlib import Path as _P

IDC_Client = index.IDCClient()
df_IDC = IDC_Client.index
try:
    df_BIH = pd.read_csv("Data/BIH_Cases_table.csv", low_memory=False)
except Exception as e:
    print(f"Warning: could not load BIH data ({e})")
    df_BIH = pd.DataFrame()

dq_mod.configure_idc_query_tool(
    df_IDC=df_IDC,
    df_BIH=df_BIH,
    system_prompt=(_P("prompts/agent_systems/idc_query.txt").read_text()),
)
idc_dl_mod.configure_idc_download_tool()
clin_mod.configure_clinical_data_tool()
webqa_mod.configure_idc_web_qa_tool()
path_mod.configure_pathology_download_tool()

_ = dq_mod.idc_query_runner
_ = d2n_mod.dicom2nifti_runner
_ = idc_dl_mod.idc_download_runner
_ = webqa_mod.idc_web_qa_runner
_ = clin_mod.clinical_data_download_runner
_ = path_mod.pathology_download_runner

ALL_TOOLS: tuple[BaseTool, ...] = tuple(TOOL_REGISTRY)

def build_graph(checkpointer=None):
    policy = SystemMessage(content=(
        f"""
        You are **VoxelInsight IDC**, a multi-agent assistant for IDC: metadata Q&A, web-grounded answers, radiology downloads,
        histopathology tiles via DICOMweb, clinical data exports (idc_index), and DICOM→NIfTI conversion.

        Core behavior
        - Only answer what the user asked; request clarifications solely when required to complete a tool call.
        - Only when asked about VoxelInsight, answer yourself otherwise always use tools. YOU ARE NOT ALLOWED TO ANSWER DIRECTLY.
        - When the user asks questions about IDC documentation, use the `idc_web_qa` tool to answer them based. These are questions like "What is the purpose of IDC?", "How to access IDC data?", "What collections are available in IDC?", etc.
        - Primarily if the user's question is a How to or what is, use the `idc_web_qa` tool to answer them based on IDC documentation.
        - When the user asks for IDC data (metadata, images, clinical data), use the idc_query tool. These include questions like "How many patients are in IDC?", "List all SeriesInstanceUIDs for CT scans in collection X", "Show me a summary of the IDC metadata tables", etc.
        - When the user requests downloads of DICOM series, histopathology tiles, or clinical data, use the respective download tools (`idc_download`, `pathology_download`, `clinical_data_download`).
        - For downloading DICOM Series or histopathology tiles, always download one patient at a time. If the user requests multiple patients, call the download tool multiple times, once per patient.
        - When the user requests DICOM to NIfTI conversion, use the `
        - Tools cannot see each other’s outputs—pass important values (SeriesInstanceUIDs, directories, etc.) yourself.
        - Keep tool instructions brief unless retrying an error. Retry at most 3 times (when it seems reasonable/necessary) with progressively clearer directions.
        - Before each tool call, tell the user what you are about to do in one concise sentence.
        - For llm based tools where you pass a reasoning_effort parameter, choose the lowest reasoning effort level that is likely to complete the task successfully. Start with 'low' for simple tasks and increase to 'medium' for more complex tasks or if previous attempts failed. Higher reasoning effort levels take longer (which is not prefferd) but may produce more accurate results.

        Available tools
        - `idc_query`: inspect IDC metadata, summarize tables, and surface SeriesInstanceUIDs. Never fabricate IDC answers—query first.
            -NEVER ask the idc_query tool to provide information beyond what the user has requested; this will waste time and resources. Efficiency is key.
            - Aim to get the most minimal information needed to satisfy the user's request.
            - For instance do not ask the tool for notes which you could have surmised. You will receive the tools code output and code itself so you can interpret it directly.
            - Aim to get the result in as few tool calls as possible. Do not split into multiple calls unless absolutely necessary.
            - If the user wants to view or visualize the radiology imaging data without downloading, the idc_query tool can provide links to online viewers.
        - `idc_web_qa`: answer general IDC questions grounded in learn.canceridc.dev (or a provided IDC doc URL). Use when the user asks doc questions.
        - `idc_download`: download DICOM series by UID. Use exactly the IDs produced by `idc_query` and respect user cancellations.
        - `pathology_download`: download histopathology tiles via DICOMweb. Default size 512x512; honor user-specified size. Needs study/series/sop instance UIDs.
        - `clinical_data_download`: download IDC clinical data by collection using idc_index (no BigQuery). Optionally select fields and/or filter on a field value.
        - `dicom2nifti`: convert downloaded DICOM folders to NIfTI files. Only run it after confirming the directory exists.
            - Automatically returns downloaded NIfTI files as download links in the UI.

        Output & chaining
        - The UI automatically renders files, plots, and tables, so never restate local file paths or download links in your response.
        - Chain tools sequentially (e.g., query → download → conversion) rather than launching them all at once.
        - Never expose local filesystem paths in the final response—describe outcomes instead.
        """
    ))

    print("using tools:", [t.name for t in ALL_TOOLS])
    base_llm = ChatOpenAI(model="gpt-5-nano", temperature=1, reasoning_effort="low")
    llm = base_llm.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(tools=ALL_TOOLS)  

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
        elif kind == "binary_path":
            path = item.get("path")
            if path and Path(path).exists():
                await cl.Message(
                    content="Here is your file:",
                    elements=[cl.File(name=Path(path).name, path=path)]
                ).send()

    # Tables
    '''
    if "df_preview" in outputs:
        prev = outputs["df_preview"]
        rows = prev.get("rows", [])
        if rows:
            df = pd.DataFrame(rows)
            await cl.Message(content=df.to_markdown(index=False)).send()
        else:
            await cl.Message(content="(No rows returned.)").send()
    '''
    # Files
    files = outputs.get("files", [])
    output_dir = outputs.get("output_dir")
    tool = outputs.get("tool", "unknown_tool")
    if not files and output_dir and Path(output_dir).exists():
        files = [str(f) for f in Path(output_dir).rglob("*") if f.is_file()]
    if files:
        zip_tmpdir = Path(tempfile.mkdtemp(prefix="vi_zip_"))
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
            elements=[cl.File(name=zip_path.name, path=str(zip_path))]
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

# Status Handler
class VoxelInsightHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.node_descriptions = {
            "agent": "VoxelInsight",
            "tools": "Tools",
            "final": "VoxelInsight Final",
        }
        self.tool_descriptions = {
            "idc_query": "IDC Query Tool",
            "bih_query": "BIH Query Tool",
            "imaging": "TotalSegmentator Segmentation - this may take a while",
            "monai_infer": "Monai Infer Tool - this may take a while",
            "radiomics": "Radiomics Analysis",
            "viz_slider": "Slider Visualization Tool",
            "dicom_to_nifti": "DICOM to NIfTI Conversion",
            "code_gen": "Code Generation",
            "midrc_query": "MIDRC Query Tool",
            "midrc_download": "MIDRC Download Tool",
            "tcia_download": "TCIA Download Tool",
            "universeg": "Universeg Segmentation",
            "idc_web_qa": "IDC Web Q&A",
            "pathology_download": "Histopathology Download",
            "clinical_data_download": "Clinical Data Download",
        }

    async def _rename_root(self, name: str):
        try:
            step = cl.context.current_step
            if step is not None:
                step.name = name
                await step.update()
        except Exception:
            pass

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        await self._rename_root("VoxelInsight")

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        await self._rename_root("VoxelInsight")

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        await self._rename_root("VoxelInsight")

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "tools")
        label = self.tool_descriptions.get(tool_name, f"{tool_name}")
        await self._rename_root(label)

    async def on_tool_end(self, output: str, **kwargs) -> None:
        pass

    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        await self._rename_root("VoxelInsight")

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

    status_handler = VoxelInsightHandler()

    await status_handler._rename_root("Initializing VoxelInsight…")

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
    initial_state = {
        "messages": [HumanMessage(content=message.content + ("" if not files else f" User uploaded files: {files}"))],
    }
   
    try:
        async for event, meta in GRAPH.astream(
            initial_state,
            stream_mode="messages",
            config=RunnableConfig(callbacks=[cb, status_handler], **config),
        ):
            node = meta.get("langgraph_node")
            if node:
                friendly = status_handler.node_descriptions.get(node, f"▶️ {node}")
                await status_handler._rename_root(friendly)

            if isinstance(event, ToolMessage):
                payloads = _collect_tool_payloads([event])
                for p in payloads:
                    await _render_payload(p)

            if isinstance(event, (AIMessage, AIMessageChunk)):
                content = _stringify_content(getattr(event, "content", None))
                has_tool_call = bool(getattr(event, "tool_calls", None))
                if has_tool_call:
                    if content:
                        await cl.Message(content=content).send()
                    await _finalize_streaming_reply()
                    continue
                if content and meta.get("langgraph_node") in ("final", "agent"):
                    msg = await _ensure_streaming_reply()
                    await msg.stream_token(content)

        await _finalize_streaming_reply()

    except Exception as e:
        import traceback; traceback.print_exc()
        await cl.Message(content=f"🚨 Error:\n```\n{type(e).__name__}: {e}\n```").send()


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
            message="Give me a quick tour of VoxelInsight IDC features and common workflows.",
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
    ]
