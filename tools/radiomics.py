import os
import json
import tempfile
from typing import Optional, List, Dict, Any

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.utils import extract_code_block
from core.sandbox import run_user_code
from core.state import TaskResult 

from tools.shared import toolify_agent  

class RadiomicsAgent:
    name = "radiomics"
    model = "gpt-5-nano" 

    def __init__(self, system_prompt: str):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt

    async def run(
        self,
        *,
        instructions: str,
        image_path: str,
        mask_paths: Optional[List[str]] = None,
        featureset: Optional[str] = None,
        out_dir: Optional[str] = None,
    ) -> TaskResult:
        if not image_path:
            return TaskResult(output="radiomics: 'image_path' is required.", artifacts={})

        mask_paths = mask_paths or []
        out_dir = out_dir or tempfile.mkdtemp(prefix="vi_radiomics_")

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

        comp = await self.client.chat.completions.create(
            model=self.model,
            temperature=1,
            messages=messages,
        )
        code = extract_code_block(comp.choices[0].message.content)

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

@toolify_agent(
    name="radiomics",
    description=(
        "Run radiomics feature extraction ussing llm to generate code which is executed. "
        "Requires image_path; mask_paths optional. "
        "Returns a small preview and optional files."
    ),
    args_schema=RadiomicsArgs,
    timeout_s=600,
)
async def radiomics_runner(
    instructions: str, image_path: str,
    mask_paths: Optional[List[str]] = None,
    featureset: Optional[str] = None,
    out_dir: Optional[str] = None,
):
    if _RAD is None:
        raise RuntimeError("Radiomics tool not configured.")
    return await _RAD.run(
        instructions=instructions,
        image_path=image_path,
        mask_paths=mask_paths,
        featureset=featureset,
        out_dir=out_dir,
    )
