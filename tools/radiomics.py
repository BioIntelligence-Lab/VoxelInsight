import os
import json
import tempfile
from typing import Optional, List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field

from core.utils import extract_code_block
from core.sandbox import run_user_code
from core.state import TaskResult 
from core.llm_provider import choose_llm

from core.storage import get_run_dir
from tools.shared import toolify_agent  

class RadiomicsAgent:
    name = "radiomics"
    model = "gpt-5-nano" 

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        try:
            self.llm = choose_llm()
        except Exception:
            self.llm = None

    async def run(
        self,
        *,
        instructions: str,
        image_path: str,
        mask_paths: Optional[List[str]] = None,
        featureset: Optional[str] = None,
        out_dir: Optional[str] = None,
        reasoning_effort: str = "medium",
    ) -> TaskResult:
        if not image_path:
            return TaskResult(output="radiomics: 'image_path' is required.", artifacts={})

        mask_paths = mask_paths or []
        out_dir = out_dir or str(get_run_dir(self.name, persist=True))

        runtime_ctx = {
            "ARGS": {
                "instructions": instructions,
                "image_path": image_path,
                "mask_paths": mask_paths,
                "featureset": featureset,
                "out_dir": out_dir,
            }
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{instructions}\n\n=== RUNTIME CONTEXT (JSON) ===\n{json.dumps(runtime_ctx, indent=2)}",
            },
        ]

        if self.llm is None:
            raise RuntimeError("LLM provider is not configured.")
        content = await self.llm.ainvoke(messages, temperature=1, reasoning_effort=reasoning_effort)
        code = extract_code_block(content)

        local_env: Dict[str, Any] = {
            "os": os,
            "pd": pd,
            "tempfile": tempfile,
            "IMAGE_PATH": image_path,
            "MASK_PATHS": mask_paths,
            "FEATURESET": featureset,
            "OUT_DIR": out_dir,
        }
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        artifacts: Dict[str, Any] = {
            "code": code,
            "output_dir": out_dir,
            "image_path": image_path,
        }
        return TaskResult(output=res, artifacts=artifacts)

_RAD: Optional[RadiomicsAgent] = None
def configure_radiomics_tool(*, system_prompt: str):
    global _RAD; _RAD = RadiomicsAgent(system_prompt=system_prompt)

class RadiomicsArgs(BaseModel):
    instructions: str = Field(..., description="Natural language request.")
    image_path: str = Field(..., description="NIfTI image path.")
    mask_paths: Optional[List[str]] = Field(None, description="List of NIfTI masks.")
    featureset: Optional[str] = Field(None, description="Feature set/YAML path.")
    out_dir: Optional[str] = Field(None, description="Output dir (optional).")
    reasoning_effort: str = Field(
        ...,
        description="Reasoning effort level (select based on task complexity): 'minimal', 'low', 'medium', or 'high'."
    )

@toolify_agent(
    name="radiomics",
    description=(
        "Runs radiomics feature extraction (with pyradiomics) using llm to generate code which is executed."
        "- Extracts quantitative features: First-order statistics, Shape descriptors, Texture features (GLCM, GLRLM, GLSZM, NGTDM, GLDM0, Can also compute on filtered images (wavelet, LoG, etc.)" 
        "- Restrictions: Accepts exactly **one image–mask pair at a time. For multiple masks on one image, run tool separately per mask."
    ),
    args_schema=RadiomicsArgs,
    timeout_s=600,
)
async def radiomics_runner(
    instructions: str, image_path: str,
    mask_paths: Optional[List[str]] = None,
    featureset: Optional[str] = None,
    out_dir: Optional[str] = None,
    reasoning_effort: str = "medium",
):
    if _RAD is None:
        raise RuntimeError("Radiomics tool not configured.")
    return await _RAD.run(
        instructions=instructions,
        image_path=image_path,
        mask_paths=mask_paths,
        featureset=featureset,
        out_dir=out_dir,
        reasoning_effort=reasoning_effort,
    )
