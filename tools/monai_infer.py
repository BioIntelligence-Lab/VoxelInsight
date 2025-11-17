import os, io, json, tempfile, shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal

import numpy as np
import torch
import nibabel as nib
from typing import List, Union, Optional
from pydantic import BaseModel, Field
from pathlib import Path

from monai.bundle import ConfigParser, download
from monai.transforms import EnsureChannelFirstd, Compose
from monai.data import MetaTensor
from monai.utils import convert_to_tensor

from core.utils import extract_code_block
from core.sandbox import run_user_code
from core.state import TaskResult  
from core.llm_provider import choose_llm
from tools.shared import toolify_agent 

def run_monai_bundle(
    inputs: dict,
    bundle_dir: str,
    *,
    device: Optional[torch.device] = None,
    auto_channel_first: bool = False,
    channel_dim: int = -1,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle_name = os.path.basename(bundle_dir.rstrip(os.sep))
    download(name=bundle_name, bundle_dir="./models", progress=True)

    parser = ConfigParser()
    try:
        parser.read_config(os.path.join(bundle_dir, "configs", "inference.json"))
    except FileNotFoundError:
        parser.read_config(os.path.join(bundle_dir, "configs", "inference.yaml"))
    try:
        parser.read_meta(os.path.join(bundle_dir, "configs", "metadata.json"))
    except FileNotFoundError:
        parser.read_meta(os.path.join(bundle_dir, "configs", "metadata.yaml"))

    model = parser.get_parsed_content("network_def")
    inferer = parser.get_parsed_content("inferer")
    preprocessing = parser.get_parsed_content("preprocessing")
    postprocessing = parser.get_parsed_content("postprocessing")

    if auto_channel_first:
        has_ecf = any(isinstance(t, EnsureChannelFirstd) for t in preprocessing.transforms)
        if not has_ecf and getattr(preprocessing, "transforms", None):
            first, *rest = preprocessing.transforms
            preprocessing = Compose([
                first,
                EnsureChannelFirstd(keys="image", channel_dim=channel_dim),
                *rest,
            ])

    ckpt_path = os.path.join(bundle_dir, "models", "model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()

    batch = preprocessing(inputs)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    img = batch["image"]
    model_in = img.unsqueeze(0) if img.ndim in {3, 4, 5} else img

    with torch.no_grad():
        preds = inferer(model_in, model)

    batch["pred"] = MetaTensor(
        convert_to_tensor(preds.squeeze(0).cpu()),
        meta=getattr(batch["image"], "meta", {}),
    )
    result = postprocessing(batch)
    return result


class MONAIAgent:
    name = "monai"
    model = "gpt-5"  

    def __init__(self, system_prompt: str, additional_context: str):
        self.system_prompt = system_prompt
        self.additional_context = additional_context
        try:
            self.llm = choose_llm()
        except Exception:
            self.llm = None

    async def run(
        self,
        *,
        instructions: str,
        image_paths: str,
        out_dir: Optional[str] = None,
    ) -> TaskResult:
        if not image_paths:
            return TaskResult(output="monai: 'image_paths' is required.", artifacts={})
        
        images = {}
        for image in ([image_paths] if isinstance(image_paths, str) else image_paths):
            path = Path(image)
            if path.suffix in [".nii", ".gz"]:
                try:
                    nii = nib.load(str(path))
                    shape = nii.shape 
                    images[str(path)] = {
                        "filename": path.name,
                        "path": str(path),
                        "dimensions": shape
                    }
                except Exception as e:
                    images[str(path)] = {"error": f"Failed to load: {e}"}
            else:
                images[str(path)] = {"error": "Not a recognized segmentation file"}


        out_dir = out_dir or tempfile.mkdtemp(prefix="vi_monai_")

        runtime_ctx = {
            "ARGS": {
                "instructions": instructions,
                "image_info": images,
                "out_dir": out_dir,
            }
        }

        print("MONAI AGENT RUNTIME CTX:\n", runtime_ctx)

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.system_prompt}\n\n"
                    "=== Monai Bundle Instructions ===\n"
                    f"{self.additional_context}\n"
                ),
            },
            {
                "role": "user",
                "content": f"{instructions}\n\n=== RUNTIME CONTEXT (JSON) ===\n{json.dumps(runtime_ctx, indent=2)}",
            },
        ]

        if self.llm is None:
            raise RuntimeError("LLM provider is not configured.")
        content = await self.llm.ainvoke(messages, temperature=1, reasoning_effort="medium")
        code = extract_code_block(content)

        print("MONAI AGENT GENERATED CODE:\n", code)

        def _save_pred_as_nifti(pred, out_path):
            arr = np.asarray(pred)
            if arr.dtype == np.int64:
                arr = arr.astype(np.uint8)
            aff = getattr(pred, "affine", np.eye(4))
            nib.save(nib.Nifti1Image(arr, aff), out_path)

        def _normalize_to_HWD(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.ndim == 5 and x.shape[0] == 1:  # (B,C,H,W,D)
                x = x.squeeze(0)
            if x.ndim == 4:  # (C,H,W,D)
                x = x[0] if x.shape[0] == 1 else x.argmax(axis=0).astype(np.uint8)
            elif x.ndim != 3:
                raise ValueError(f"Unexpected pred shape {x.shape}")
            return x

        local_env: Dict[str, Any] = {
            # MONAI execution utilities
            "run_monai_bundle": run_monai_bundle,
            "nib": nib,
            "np": np,
            "torch": torch,
            "os": os,
            "Path": Path,
            "tempfile": tempfile,
            "shutil": shutil,
            "_save_pred_as_nifti": _save_pred_as_nifti,
            "_normalize_to_HWD": _normalize_to_HWD,

            "IMAGE_PATHS": image_paths,
            "OUT_DIR": out_dir,
        }

        out = run_user_code(code, local_env)
        res = out.get("res_query")

        artifacts: Dict[str, Any] = {
            "code": code,
            "output_dir": out_dir,
            "image_paths": image_paths,
        }

        if isinstance(res, dict):
            for k in ("segmentations", "segmentations_map", "files", "nifti_paths", "output_dir", "image_paths"):
                if k in res:
                    artifacts[k] = res[k]

        return TaskResult(output=res, artifacts=artifacts)


_MONAI: Optional[MONAIAgent] = None

def configure_monai_tool(*, system_prompt: str, additional_context: str):
    """
    Call once at startup, e.g.:
      sys = Path('prompts/agent_systems/monai.txt').read_text()
      add = Path('Data/monai_bundles_instructions.txt').read_text()
      configure_monai_tool(system_prompt=sys, additional_context=add)
    """
    global _MONAI
    _MONAI = MONAIAgent(system_prompt=system_prompt, additional_context=additional_context)


class MonaiArgs(BaseModel):
    instructions: str = Field(..., description="Natural language instruction for running the bundle.")
    image_paths: Union[str, List[str]] = Field(..., description="NIfTI image path/s to run inference on. String or list of strings.")
    out_dir: Optional[str] = Field(None, description="Output directory; defaults to a temp dir.")

@toolify_agent(
    name="monai",
    description=(
        "Run MONAI bundle inference using llm to generate code which is executed."
        "Do not confuse MONAI with TotalSegmentator (different systems)." 
        "Returns segmentation files and/or output_dir when available. Output files stored locally will be shown as download links in the UI."
        "The MONAI agent can generate custom code for pre processing, inference, and post processing. You may pass specific instructions to the agent to guide this code generation and prevent errors."
    ),
    args_schema=MonaiArgs,
    timeout_s=1200, 
)
async def monai_runner(
    instructions: str,
    image_paths: str,
    out_dir: Optional[str] = None,
):
    if _MONAI is None:
        raise RuntimeError(
            "MONAI tool not configured. Call configure_monai_tool(system_prompt=..., additional_context=...) at startup."
        )
    return await _MONAI.run(
        instructions=instructions,
        image_paths=image_paths,
        out_dir=out_dir,
    )
