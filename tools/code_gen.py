# tools/code_gen.py
import os, io, json
from typing import List, Optional, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.utils import extract_code_block
from core.sandbox import run_user_code
from core.state import TaskResult 

from tools.shared import toolify_agent  

class CodeExecTool:
    name = "code_gen"
    model = "gpt-5"

    def __init__(self, system_prompt: str, df_IDC: Optional[pd.DataFrame] = None):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt
        self.df_IDC = df_IDC

    async def run(
        self,
        *,
        instructions: str,
        files: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        mask_paths: Optional[List[str]] = None,
        last_df_json: Optional[str] = None,  
    ) -> TaskResult:
        files = files or []
        mask_paths = mask_paths or []

        runtime_ctx = {
            "ARGS": {
                "instructions": instructions,
                "files": files,
                "image_path": image_path,
                "mask_paths": mask_paths,
                "last_df_json": bool(last_df_json),
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

        last_df: Optional[pd.DataFrame] = None
        if last_df_json:
            try:
                last_df = pd.read_json(last_df_json, orient="table")
            except Exception:
                pass

        local_env: Dict[str, Any] = {
            "pd": pd,
            "plt": plt,
            "io": io,
            "os": os,
            "nib": nib,
            "pydicom": pydicom,

            "df_IDC": self.df_IDC, 

            "FILES": files,
            "IMAGE_PATH": image_path,
            "MASK_PATHS": mask_paths,
            "LAST_DF": last_df,
        }

        out = run_user_code(code, local_env)
        res = out.get("res_query")

        artifacts: Dict[str, Any] = {"code": code}
        if isinstance(res, dict) and "files" in res:
            artifacts["files"] = res["files"]

        return TaskResult(output=res, artifacts=artifacts)


_CODE: Optional[CodeExecTool] = None

def configure_code_gen_tool(*, system_prompt: str, df_IDC: Optional[pd.DataFrame] = None):
    global _CODE
    _CODE = CodeExecTool(system_prompt=system_prompt, df_IDC=df_IDC)


class CodeExecArgs(BaseModel):
    instructions: str = Field(..., description="Natural language request for code to run.")
    files: Optional[List[str]] = Field(None, description="Optional file paths to use in the code.")
    image_path: Optional[str] = Field(None, description="Optional image path (e.g., NIfTI) for convenience.")
    mask_paths: Optional[List[str]] = Field(None, description="Optional mask paths.")
    last_df_json: Optional[str] = Field(
        None,
        description="Optional last table as JSON (orient='table') if you want continuity."
    )

@toolify_agent(
    name="code_gen",
    description=(
        "Use for arbitrary Python code generation and execution." 
        "Applicable tasks:"
        "- For creating UI outputs in the proper format (e.g., plotly charts, images, files). Outputs like plotly sliders and matplotlib images are automatically shown by the UI." 
        "- Any task requiring python code generation and execution which cannot be answered by other tools."
        "- For example: Radiomics analysis; Segmentation (e.g., TotalSegmentator); Image preprocessing / postprocessing; Data analysis, statistics, and visualization not covered by other tools; May also handle preprocessing or postprocessing for other tools."
        "Outputs like files stored locally can be shown as download links in the UI. Plots and plotly charts are automatically rendered in the UI."
        "Use the code_gen tool when the user requests tasks that require custom python code execution, data manipulation, or analysis beyond predefined tools."
    ),
    args_schema=CodeExecArgs,
    timeout_s=600,
)
async def code_gen_runner(
    instructions: str,
    files: Optional[List[str]] = None,
    image_path: Optional[str] = None,
    mask_paths: Optional[List[str]] = None,
    last_df_json: Optional[str] = None,
):
    if _CODE is None:
        raise RuntimeError("CodeExec tool not configured. Call configure_code_gen_tool(system_prompt=..., df_IDC=...).")
    return await _CODE.run(
        instructions=instructions,
        files=files,
        image_path=image_path,
        mask_paths=mask_paths,
        last_df_json=last_df_json,
    )
