import os, io, pandas as pd, matplotlib.pyplot as plt
import duckdb  
from openai import AsyncOpenAI
from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code

class MIDRCQueryAgent:
    name = "midrc_query"
    model = "gpt-4o"

    def __init__(self, df_MIDRC: pd.DataFrame, system_prompt: str):
        key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
        self.df_MIDRC = df_MIDRC
        self.system_prompt = system_prompt

    async def run(self, task: Task, state: ConversationState) -> TaskResult:

        user_text = task.kwargs.get("instructions") or task.user_msg

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"{user_text}\n\n"
                    "=== df_MIDRC Columns ===\n"
                    + str(self.df_MIDRC.columns.tolist())
                ),
            },
        ]

        comp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=messages,
        )
        code = extract_code_block(comp.choices[0].message.content)
        local_env = {"df_MIDRC": self.df_MIDRC, "pd": pd, "plt": plt, "io": io, "os": os, "duckdb": duckdb}
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        if isinstance(res, pd.DataFrame):
            state.memory["last_df"] = res
        
        arts = {"code": code}
        if isinstance(res, dict) and "files" in res:
            arts["files"] = res["files"]    
        return TaskResult(output=res, artifacts=arts)

