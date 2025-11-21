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
TS_CT = pd.DataFrame()
TS_MRI = pd.DataFrame()
Monai_Instructions = ""
if "imaging" in os.getenv("VOXELINSIGHT_TOOLS", ""):
    TS_CT = pd.read_csv("Data/TotalSegmentatorMappingsCT.tsv", sep="\t")
    TS_MRI = pd.read_csv("Data/TotalSegmentatorMappingsMRI.tsv", sep="\t")
if "monai" in os.getenv("VOXELINSIGHT_TOOLS", ""):
    Monai_Instructions = _P("Data/monai_bundles_instructions.txt").read_text()

dq_mod.configure_idc_query_tool(df_IDC=df_IDC, df_BIH=df_BIH, system_prompt=(_P("prompts/agent_systems/idc_query.txt").read_text()))
img_mod.configure_imaging_tool(ct_mappings="")
vz_mod.configure_viz_slider_tool()
rad_mod.configure_radiomics_tool(system_prompt=(_P("prompts/agent_systems/radiomics.txt").read_text()))
monai_mod.configure_monai_tool(
    system_prompt=(_P("prompts/agent_systems/monai.txt").read_text()),
    additional_context=(_P("Data/monai_bundles_instructions.txt").read_text())
)
code_mod.configure_code_gen_tool(
    system_prompt=(_P("prompts/agent_systems/code_gen.txt").read_text()),
    df_IDC=df_IDC
)
midrc_mod.configure_midrc_query_tool(
    df_MIDRC=df_MIDRC,
    system_prompt=(_P("prompts/agent_systems/midrc_query.txt").read_text())
)
bih_mod.configure_bih_query_tool(
    df_BIH=df_BIH,
    system_prompt=(_P("prompts/agent_systems/bih_query.txt").read_text())   
)
midrc_dl_mod.configure_midrc_download_tool()
tcia_dl_mod.configure_tcia_download_tool()
idc_dl_mod.configure_idc_download_tool()

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

ALL_TOOLS: tuple[BaseTool, ...] = tuple(TOOL_REGISTRY)
TOOL_NAMES = {tool.name: tool for tool in ALL_TOOLS}
DEFAULT_TOOL_NAMES = tuple(TOOL_NAMES)

def resolve_tool_subset(raw: str | None):
    if not raw:
        return list(ALL_TOOLS)
    requested = {name.strip() for name in raw.split(",") if name.strip()}
    subset = [TOOL_NAMES[name] for name in requested if name in TOOL_NAMES]
    return subset or list(ALL_TOOLS)  

USED_TOOLS = resolve_tool_subset(os.getenv("VOXELINSIGHT_TOOLS"))

def build_graph(checkpointer=None):
    policy = SystemMessage(content=(
        f"""
        You are **VoxelInsight**, an AI radiology assistant/agent.  
        You have access to many **TOOLS**. Use them carefully to answer user questions.

        You are an agent
        - Do not ask the user follow up questions unless absolutely necessary. 
        - Do not do more than what the user requests.
        - Do not suggest next steps for the user unless they ask for them.
        - When using llm based tools which generate code, be as concise as possible in your instructions, unless an error occurs, then be as specific as possible to fix the issue.
        - Assume that tools cannot see each other's output or the conversation history. You must pass information between tools yourself.
        ---

        ## General Principles
        1. **Chain Tool When Needed**  
        - Many times the output of one tool is required to run another. In these cases do not run both tools at once.  
        - Instead, run one tool, inspect the result, then (if needed) use another tool.

        2. **Automatic UI Outputs**  
        - All tool outputs (files, images, plots, dataframes, sliders, etc.) are automatically displayed in the UI.  
        - Never provide download paths, file paths, or attempt to re-display these outputs yourself.
        - For file downloads, tools return a path for a directory containing the file and the UI automatically zips and provides a download link.
        - For text based links and outputs, you may include them in your response.  
        - To send things to the user that is not text (for example images, files, plots, etc.), use the appropriate tool to generate these outputs instead of trying to do it yourself. You can use the `code_gen` tool to generate these if needed.
        - Tools are designed to output dictonaries with specific keys to create automatic UI outputs. We currently support displaying matplotlib images, plotly charts, and file download links (only for files stored locally). Any other outputs will not be displayed automatically and you must handle them yourself in your response.

        3. **Error Handling**  
        - Retry a failing tool a maximum of 3 times. 
        - When retrying llm powered tools, you must refine your instructions to attempt a fix for the issue. 
        - If it still fails, inform the user that you cannot complete the task.  
        - When retrying, make instructions stricter and more precise (e.g., include shapes, dtypes, conversions).  

        4. **File Context**  
        - Users may upload files at the start of a session. File paths are provided in context.  
        - If a user says “this file” or “the image,” assume they mean the most recent uploaded file. 

        5. **Other Rules**
        - Keep instructions for llm based agents concise and to the point for simple queries and be as clear as possible for complex queries like when using monai or code_gen agents.
        - Individual tools do not have a shared state and cannot see each other's outputs. As the supervisor you must pass outputs between tools yourself when necessary.
        - Tools do not have any context other than the instructions/arguments you provide to them. When using a new tool assume you're starting from scratch and provide any required context.
        - Don't show locally stored file paths to the user since they cannot access them anyways (although some testers may be able to). Some tools may automatically provide download links for files stored locally, but if not you can use the `code_gen` tool to generate the proper outputs if needed.
        - Plotly figures can only be generated using the `code_gen` tool or the "viz_slider" tool. The UI automatically displays plotly figures when generated properly.
        - When users want a file download, if possible always assume that they want you to download it dirrectly from the dataset using specialized download tools if available. If no specialized download tool is available for the dataset, inform the user.

        ---

        ## Tool Usage Rules

        - For llm based tools where you pass a reasoning_effort parameter, choose the lowest reasoning effort level that is likely to complete the task successfully. Start with 'low' for simple tasks and increase to 'medium' for more complex tasks or if previous attempts failed. Higher reasoning effort levels take longer (which is not prefferd) but may produce more accurate results.
        - Before every tool call, tell the user what you are doing in understandable and concise language. Don't explain tool call parameters in detail unless necessary. A quick + concise summary is sufficient.

        ### MONAI Inference (`monai_infer`)
        - Bundle-specific instructions are provided here (if blank ignore - the tools is unaivailable): {Monai_Instructions}.

        ### Imaging Segmentation (`imaging`)
        - Performs segmentation using TotalSegmentator.  
        - If multiple files are provided, pass them as a list instead of calling seperate instances of the imaging tool.
        - Mappings provided for TotalSegmentator tasks and subsets (if blank ignore - the tools is unaivailable):  
        - CT: {TS_CT}  
        - MRI: {TS_MRI}  

        ### TCIA Download (`tcia_download`)
        - You don't have an API key for this currently. Make sure to use the proper method for download without API.
        - These are the TCIA collections you can download directly from using the `tcia_download` tool (although there are more collections with metadata in the BIH for querying the datasets without download so refer to BIH if a user asks about TCIA as a whole): 4D-Lung, A091105, ACNS0332, ACRIN-6698, ACRIN-Contralateral-Breast-MR, ACRIN-FLT-Breast, ACRIN-HNSCC-FDG-PET-CT, ACRIN-NSCLC-FDG-PET, AHEP0731, AHOD0831.
        - If a user requests a download from a different collection, inform them that you cannot complete the task.
        ---

        ## Post-Tool Result Handling
        - Tool results arrive as JSON in a ToolMessage with schema:  
        ok: bool,
        outputs: 
        df_preview?: rows:[obj], nrows:int,
        text?: string,
        summary?: string,
        ui?: [...],
        error?: string,
        ...
        - Use results as follows:  
        1. If `outputs.text` exists → use it's information to make your resposne'.  
        2. If `outputs.df_preview` exists:  
        - If it’s a single row/column → return the plain value (e.g., “Total patients: 12345”).  
        - Otherwise → summarize key columns briefly.  
        3. Files, images, sliders, and dataframes are automatically shown in UI — do not re-display them.  
        4. You may chain outputs between tools by manually passing them in (e.g., use masks from `imaging` in `viz_slider` or `radiomics`).  

        ---

        ## Summary of Key Prohibitions
        - Never fabricate IDC answers without `idc_query`.   
        - Never provide download paths/links or try to display tool outputs already shown by the UI.  
        - Never provided direct paths to any files (e.g., NIfTI, DICOM) in your responses.
        - Never specify `roi_subset` for tasks other than `total` or `total_mr`.  
        - Never skip segmentation before visualization/radiomics.  
        - Never expose local filesystem paths in the final response—describe outcomes instead.
        ---
        """
    ))

    print("using tools:", [t.name for t in USED_TOOLS])
    base_llm = ChatOpenAI(model="gpt-5-nano", temperature=1, reasoning_effort="low")
    llm = base_llm.bind_tools(USED_TOOLS)
    tool_node = ToolNode(tools=USED_TOOLS)  

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
            label="Radiomics guide",
            message="Explain which radiomics feature families VoxelInsight supports and how to export results as CSV.",
            icon="/public/chart.svg",
        ),
    ]
