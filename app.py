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
from langchain_core.callbacks import BaseCallbackHandler

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
import tools.bih_query as bih_mod
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
df_BIH = pd.read_csv("Data/BIH_Cases_table.csv", low_memory=False)
df_MIDRC = pd.read_parquet("midrc_mirror/nodes/midrc_files_wide.parquet")
#df_MIDRC = pd.DataFrame()
TS_CT = pd.read_csv("Data/TotalSegmentatorMappingsCT.tsv", sep="\t")
TS_MRI = pd.read_csv("Data/TotalSegmentatorMappingsMRI.tsv", sep="\t")
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

_ = dq_mod.idc_query_runner
_ = img_mod.imaging_runner
_ = vz_mod.viz_slider_runner
_ = rad_mod.radiomics_runner
_ = monai_mod.monai_runner
_ = code_mod.code_gen_runner
_ = midrc_mod.midrc_query_runner
_ = bih_mod.bih_query_runner

def build_graph(checkpointer=None):
    policy = SystemMessage(content=(
        f"""
        You are **VoxelInsight**, an AI radiology assistant.  
        You have access to many **TOOLS**. Use them carefully to answer user questions.
        Assume that tools cannot see each other's output or the conversation history. You must pass information between tools yourself.

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

        ---

        ## Tool Usage Rules

        ### BIH Query Tool (`bih_query`)
        - Handles **all BIH tasks**.
        - Can be used to answer questions about datasets included in BIH like MIDRC, Stanford AIMI, IDC, NIHCC, TCIA, and ACRdart.
        - By default use this tools for any questions about the BIH and datasets included in it.
        - This tool cannot download files. For download requests you may use specialized tools specific to the dataset if available. These tools download files and return the path which is automatically displated to the user in the UI.
        - This tool can generate matplotlib plots and dataframes.
        - Capabilities: return dataframes, summaries, plots, and text. Note that it cannot return interactive plotly charts. Use the `code_gen` tool for that.

        ### IDC Query Tool (`idc_query`)
        - Handles **all IDC tasks which cannot be answered by bih_query**.  
        - Can be used for IDC related questions that bih_query cannot answer (by default use BIH query for IDC related questions).
        - Capabilities: return dataframes, summaries, plots, text, and download links (shown automatically).  
        - For IDC plots: request them directly from this tool or bih_query (it can query + plot in one step). 

        ### MIDRC Query Tool (`midrc_query`)
        - Handles **all MIDRC tasks which cannot be answered by bih_query**.  
        - Can be used for MIDRC related questions that bih_query cannot answer (by default use BIH query for MIDRC related questions).
        - Can download files from MIDRC using gen3.
        - Capabilities: return dataframes, summaries, plots, text, and download links (shown automatically).  
        - For MIDRC plots: request them directly from this tool or bih_query (it can query + plot in one step).   

        ### DICOM → NIfTI Conversion (`dicom_to_nifti`)
        - IDC studies follow the DICOM hierarchy: Patient → Study → Series → Instances.  
        - A series contains multiple DICOM slices.  
        - Use this tool to convert a series into a NIfTI volume before segmentation or radiomics.  

        ### Code Generation and Execution (`code_gen`)
        - Use for arbitrary Python code generation and execution.  
        - Applicable tasks:  
        - For creating UI outputs in the proper format (e.g., plotly charts, images, files). Outputs like plotly sliders and matplotlib images are automatically shown by the UI. 
        - Any task requiring python code generation and execution which cannot be answered by other tools.
        - For example:
            - Radiomics analysis  
            - Segmentation (e.g., TotalSegmentator)  
            - Image preprocessing / postprocessing  
            - Data analysis, statistics, and visualization not covered by other tools  
            - May also handle preprocessing or postprocessing for other tools.

        ### MONAI Inference (`monai_infer`)
        - Runs MONAI bundles for segmentation.  
        - Provide full image paths via `image_path`.  
        - Do not confuse MONAI with TotalSegmentator (different systems).  
        - Bundle-specific instructions are provided here: {Monai_Instructions}.

        ### Imaging Segmentation (`imaging`)
        - Performs segmentation using TotalSegmentator.  
        - **Task-specific rules**:  
        - If using `task="total"` or `task="total_mr"`: you may specify `roi_subset` values for specific organs/tissues.  
        - For all other tasks: **never** specify `roi_subset` (segmentation covers all ROIs by default).  
        - Incorrect use of `roi_subset` will cause errors.  
        - **Special rule**: For liver_tumor segmentation, use `task="liver_vessels"` with no `roi_subset`.  
        - Mappings provided for TotalSegmentator tasks and subsets:  
        - CT: {TS_CT}  
        - MRI: {TS_MRI}  

        ### Radiomics (`radiomics`)
        - Extracts quantitative features:  
        - First-order statistics  
        - Shape descriptors  
        - Texture features (GLCM, GLRLM, GLSZM, NGTDM, GLDM)  
        - Can also compute on filtered images (wavelet, LoG, etc.)  
        - Restrictions:  
        - Accepts exactly **one image–mask pair at a time**.  
        - For multiple masks on one image, run tool separately per mask.  

        ### Visualization (`viz_slider`)
        - Displays interactive slider for images/masks (nifti) automatically in UI (may not work for every case).  
        - Rules:  
        - If asked to **segment + visualize**:  
            - First run a segmentation tool,  
            - Then use `viz_slider` with the produced mask paths.  
        - Do not attempt segmentation and visualization in a single step (UI will only show the original image). 
        - Other kinds of interactive plots (e.g., plotly) and other visualizatoins like images can be generated using the `code_gen` tool. These will be automatically displayed in the UI.

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
        4. You may chain outputs between tools (e.g., use masks from `imaging` in `viz_slider` or `radiomics`).  

        ---

        ## Summary of Key Prohibitions
        - Never fabricate IDC answers without `idc_query`.   
        - Never provide download paths/links or try to display tool outputs already shown by the UI.  
        - Never provided direct paths to any files (e.g., NIfTI, DICOM) in your responses.
        - Never specify `roi_subset` for tasks other than `total` or `total_mr`.  
        - Never skip segmentation before visualization/radiomics.  

        ---
        """
    ))

    print("TOOLS:", [t.name for t in TOOL_REGISTRY])
    llm = ChatOpenAI(model="gpt-5-nano", reasoning_effort="low").bind_tools(TOOL_REGISTRY)
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
    if not files and output_dir and Path(output_dir).exists():
        files = [str(f) for f in Path(output_dir).rglob("*") if f.is_file()]
    if files:
        zip_tmpdir = Path(tempfile.mkdtemp(prefix="vi_zip_"))
        zip_path = zip_tmpdir / "download.zip"
        await _zip_paths(files, zip_path)
        await cl.Message(
            content=f"**Files ready**\n- Items: {len(files)}\n\nClick to download:",
            elements=[cl.File(name=zip_path.name, path=str(zip_path))]
        ).send()

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
            "imaging": "TotalSegmentator Segmentation",
            "monai_infer": "Monai Infer Tool",
            "radiomics": "Radiomics Analysis",
            "viz_slider": "Visualizing Slider",
            "dicom_to_nifti": "DICOM to NIfTI Conversion",
            "code_gen": "Code Generation",
            "midrc_query": "MIDRC Query Tool",
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

    final_answer = cl.Message(content="")
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

            if (
                getattr(event, "content", None)
                and isinstance(event, AIMessage)
                #and not getattr(event, "tool_calls", None)
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
