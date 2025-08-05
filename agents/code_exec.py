# agents/code_exec.py
import os, io
import pandas as pd
import matplotlib.pyplot as plt
import json
from openai import AsyncOpenAI
import nibabel as nib
import pydicom 
from idc_index import index

from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code

class CodeExecAgent:
    name = "code_exec"
    model = "gpt-4o"

    def __init__(self, system_prompt: str):
        key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
        self.system_prompt = system_prompt

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        user_text = task.kwargs.get("instructions") or task.user_msg
        context = self._build_llm_context(task, state)

        messages = [
            {"role": "system", "content": self.system_prompt},
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

        local_env = self._build_local_env(state, task)
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        self._maybe_store_df_in_state(res, state)

        arts = {"code": code}
        if isinstance(res, dict) and "files" in res:
            arts["files"] = res["files"]

        return TaskResult(output=res, artifacts=arts)

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

    def _build_local_env(self, state: ConversationState, task: Task):
        env = {
            "pd": pd,
            "plt": plt,
            "io": io,
            "os": os,
            "nib": nib,
            "pydicom": pydicom,
            "index": index,  
            "INPUT_FILES": task.files,
            "STATE_FILES": state.memory.get("files", []),
            "STATE_IMAGE_PATH": state.memory.get("image_path"),
            "STATE_MASK_PATH": state.memory.get("mask_path"),
            "STATE_SEGMENTATIONS": state.memory.get("segmentations"),
            "STATE_MASK_PATHS": state.memory.get("mask_paths"),
            "CHAINED_FILES": state.memory.get("files"),
            "LAST_DF": state.memory.get("last_df"),
        }
        return env

    def _maybe_store_df_in_state(self, res, state: ConversationState):
        try:
            if hasattr(res, "to_csv"):
                state.memory["last_df"] = res
        except Exception:
            pass
