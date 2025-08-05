import os, tempfile, subprocess, pandas as pd
import json
from openai import AsyncOpenAI
from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code

class RadiomicsAgent:
    name = "radiomics"
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

        local_env = self._build_local_env(task, state)
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        try:
            if hasattr(res, "to_csv"):
                state.memory["last_df"] = res
        except Exception:
            pass

        arts = {"code": code}

        state.memory["radiomics"] = res
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

    def _build_local_env(self, task: Task, state: ConversationState) -> dict:

        return {
            "os": os,
            "tempfile": tempfile,

            "INPUT_FILES": task.files,
            "STATE_IMAGE_PATH": state.memory.get("image_path"),
            "STATE_SEGMENTATIONS": state.memory.get("segmentations"),
            "KWARGS": task.kwargs,
        }

