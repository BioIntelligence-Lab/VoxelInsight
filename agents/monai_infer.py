import os, io, json, tempfile, shutil, numpy as np
import torch
import nibabel as nib
from pathlib import Path
from openai import AsyncOpenAI

from core.utils import extract_code_block
from core.sandbox import run_user_code
from monai.bundle import ConfigParser, download
from monai.transforms import EnsureChannelFirstd, Compose
from monai.data import MetaTensor
from monai.utils import convert_to_tensor

from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code


def run_monai_bundle(
    inputs: dict,
    bundle_dir: str,
    *,
    device: torch.device | None = None,
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
        if not has_ecf:
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
    model = "gpt-4o"

    def __init__(self, system_prompt: str, additional_context: str):
        key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
        self.system_prompt = system_prompt
        self.additional_context = additional_context

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        user_text = task.kwargs.get("instructions") or task.user_msg
        context = self._build_llm_context(task, state)

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.system_prompt}\n\n"
                    "=== Monai Bundle Instructions ===\n"
                    f"{self.additional_context}\n\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{user_text}\n\n"
                    "=== RUNTIME CONTEXT ===\n"
                    + json.dumps(context, indent=2)
                ),
            },
        ]

        comp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=messages,
        )
        code = extract_code_block(comp.choices[0].message.content)

        local_env = self._build_local_env(task, state)
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        try:
            if hasattr(res, "to_csv"):
                state.memory["last_df"] = res
        except Exception:
            pass

        arts = {"code": code}
        if isinstance(res, dict):
            if "segmentations" in res:
                arts["segmentations"] = res["segmentations"]
                state.memory["segmentations"] = res["segmentations"]
            if "output_dir" in res:
                arts["output_dir"] = res["output_dir"]
                state.memory["last_output_dir"] = res["output_dir"]

        state.memory["monai_result"] = res
        return TaskResult(output=res, artifacts=arts)

    # helpers 
    def _build_llm_context(self, task: Task, state: ConversationState) -> dict:
        return {
            "INPUT_FILES": task.files or [],
            "KWARGS": task.kwargs,
            "STATE": {
                "image_path": state.memory.get("image_path"),
                "segmentations": state.memory.get("segmentations"),
                "mask_path": state.memory.get("mask_path"),
                "mask_paths": state.memory.get("mask_paths"),
                "last_output_dir": state.memory.get("last_output_dir"),
            },
        }

    def _build_local_env(self, task: Task, state: ConversationState) -> dict:
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

        return {
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

            "INPUT_FILES": task.files,
            "STATE_IMAGE_PATH": state.memory.get("image_path"),
            "STATE_SEGMENTATIONS": state.memory.get("segmentations"),
            "KWARGS": task.kwargs,
        }
