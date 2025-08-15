import os
from agents.universeg import UniversegAgent
import chainlit as cl
from dotenv import load_dotenv
from pathlib import Path
import tempfile, shutil, asyncio, zipfile, pandas as pd
import asyncio
import plotly.graph_objects as go
from typing import Dict, Optional
from chainlit.types import ThreadDict
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from idc_index import index

from core.state import Task, ConversationState
from orchestrators.vanilla import run_pipeline

from agents.router import RouterAgent
from agents.data_query import DataQueryAgent
from agents.imaging import ImagingAgent
from agents.monai_infer import MONAIAgent
from agents.radiomics import RadiomicsAgent
from agents.viz_slider import VizSliderAgent
from agents.code_exec import CodeExecAgent
from agents.dicom_to_nifti import Dicom2NiftiPyAgent
from agents.universeg import UniversegAgent

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    # Allow all users who pass OAuth authentication
    return default_user

load_dotenv()

from idc_index import index

IDC_Client = index.IDCClient()
df_IDC = IDC_Client.index
#df_BIH = pd.read_csv("Data/midrc_distributed_subjects.csv")
#df_MIDRC = pd.read_csv("Data/MIDRC_Cases_table.csv")
df_BIH = pd.DataFrame()
df_MIDRC = pd.DataFrame()

BASE_PROMPT   = Path("prompts/router_system.txt").read_text()
ct_mappings     = Path("Data/TotalSegmentatorMappingsCT.txt").read_text()          
mri_mappings    = Path("Data/TotalSegmentatorMappingsMRI.txt").read_text()  

#router
router = RouterAgent(
    available_agents=["data_query", "imaging", "monai", "radiomics", "viz", "code_exec", "viz_slider", "dicom2nifti", "universeg"],
    system_prompt=(
        BASE_PROMPT
        + "\n\n---\n### ROI to task mapping table for TotalSegmentator CT (tsv)\n" 
        + ct_mappings
        + "\n\n---\n### ROI to task mapping table for TotalSegmentator MRI (tsv)\n"
        + mri_mappings
    )
)

agents = {
    "data_query": DataQueryAgent(df_IDC=df_IDC,
                                 df_BIH=df_BIH,
                                 system_prompt=Path("prompts/agent_systems/data_query.txt").read_text()),
    "imaging": ImagingAgent(ct_mappings=ct_mappings),
    "radiomics": RadiomicsAgent(system_prompt=Path("prompts/agent_systems/radiomics.txt").read_text()),
    "code_exec": CodeExecAgent(system_prompt=Path("prompts/agent_systems/code_exec.txt").read_text()),
    "viz_slider": VizSliderAgent(),
    "monai": MONAIAgent(system_prompt=Path("prompts/agent_systems/monai.txt").read_text(), additional_context=Path("Data/monai_bundles_instructions.txt").read_text()),
    "dicom2nifti": Dicom2NiftiPyAgent(),
    "universeg": UniversegAgent(),
    }

async def _zip_paths(paths, zip_path: Path):
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


@cl.on_message
async def on_message(message: cl.Message):
    # Gather files
    file_elements = [el for el in (message.elements or []) if isinstance(el, cl.File)]
    files = []
    for f in file_elements:
        tmpdir = Path(tempfile.mkdtemp())
        new_path = tmpdir / f.name
        shutil.copy(f.path, new_path)
        files.append(str(new_path))

    task = Task(user_msg=message.content, files=files)
    state = ConversationState()

    if files:
        state.memory["files"] = files
        state.memory["image_path"] = files[0]

    res = await run_pipeline(router, agents, task, state)

    # Render final result
    if res is None:
        await cl.Message(content="No result.").send()
    elif isinstance(res, dict) and res.get("action") == "download":
        files_to_package = res.get("files") or []
        series_uids = res.get("series_uids", [])

        if not files_to_package:
            await cl.Message(content="No files were downloaded.").send()
        else:
            zip_tmpdir = Path(tempfile.mkdtemp(prefix="vi_zip_"))
            zip_path = zip_tmpdir / "download.zip"
            await _zip_paths(files_to_package, zip_path)
            await cl.Message(
                content=(
                    "📥 **Download complete**\n"
                    f"- Items: {len(files_to_package)}\n"
                    "Click to download:"
                ),
                elements=[cl.File(name=zip_path.name, path=str(zip_path))]
            ).send()
    elif isinstance(res, dict) and res.get("action") == "inference":
        output_dir = res.get("output_dir")
        segmentations = res.get("segmentations", [])

        if not output_dir and not segmentations:
            await cl.Message(content="No files were created.").send()
        else:
            if segmentations:
                files_to_package = segmentations
            else:
                output_path = Path(output_dir)
                if output_path.exists() and output_path.is_dir():
                    files_to_package = [str(f) for f in output_path.rglob("*") if f.is_file()]
                else:
                    files_to_package = []
            
            if not files_to_package:
                await cl.Message(content="No files were created.").send()
            else:
                zip_tmpdir = Path(tempfile.mkdtemp(prefix="vi_zip_"))
                zip_path = zip_tmpdir / "download.zip"
                await _zip_paths(files_to_package, zip_path)
            await cl.Message(
                content=(
                    "📥 **Inference complete**\n"
                    f"- Items: {len(files_to_package)}\n\n"
                    "Click to download the output:"
                ),
                elements=[cl.File(name=zip_path.name, path=str(zip_path))]
            ).send()
    elif isinstance(res, Figure):  # Matplotlib (pyplot) figure
        await cl.Message(
            content="Here is your result:",
            elements=[cl.Pyplot(name="plot", figure=res, display="inline")]
        ).send()
        plt.close(res)
    elif hasattr(res, "read"):  # BytesIO image
        await cl.Message(
            content="Here is your result:",
            elements=[cl.Image(name="plot.png", mime="image/png", content=res.read())]
        ).send()  
    elif isinstance(res, go.Figure):  # Plotly
        await cl.Message(
            content="Interactive viewer:",
            elements=[cl.Plotly(name="viewer", figure=res)]
        ).send()
    elif isinstance(res, pd.DataFrame):
        await cl.Message(content=res.head().to_markdown(index=False)).send()
    else:
        await cl.Message(content=str(res)).send()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")
