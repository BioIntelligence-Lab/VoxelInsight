import os, io, pandas as pd, matplotlib.pyplot as plt
import duckdb  
from openai import AsyncOpenAI
from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code

class DataQueryAgent:
    name = "data_query"
    model = "gpt-4o"

    def __init__(self, df_IDC: pd.DataFrame, df_BIH: pd.DataFrame, system_prompt: str):
        key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
        self.df_IDC = df_IDC
        self.df_BIH = df_BIH
        self.system_prompt = system_prompt

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task.user_msg},
        ]
        comp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=messages,
        )
        code = extract_code_block(comp.choices[0].message.content)
        local_env = {"df_IDC": self.df_IDC, "df_BIH": self.df_BIH, "pd": pd, "plt": plt, "io": io, "os": os,}
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        if isinstance(res, pd.DataFrame):
            state.memory["last_df"] = res
        
        arts = {"code": code}
        if isinstance(res, dict) and "files" in res:
            arts["files"] = res["files"]    
        return TaskResult(output=res, artifacts=arts)

