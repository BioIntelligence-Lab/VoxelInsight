# tools/midrc_query.py
import os, io, pandas as pd, matplotlib.pyplot as plt
import duckdb
from openai import AsyncOpenAI

from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code
from pydantic import BaseModel, Field
from typing import Optional
from tools.shared import toolify_agent, _cs

class MIDRCQueryAgent:
    name = "midrc_query"
    model = "gpt-5"

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
                "content": f"{user_text}\n\n=== df_MIDRC Columns ===\n{self.df_MIDRC.columns.tolist()}",
            },
        ]
        comp = await self.client.chat.completions.create(
            model=self.model,
            temperature=1,
            messages=messages,
        )
        code = extract_code_block(comp.choices[0].message.content)
        print(code)
        local_env = {"df_MIDRC": self.df_MIDRC, "cred": "/Users/adhrith/Downloads/credentials.json", "pd": pd, "plt": plt, "io": io, "os": os, "duckdb": duckdb}
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        if isinstance(res, pd.DataFrame):
            state.memory["last_df"] = res

        arts = {"code": code}
        if isinstance(res, dict) and "files" in res:
            arts["files"] = res["files"]
        return TaskResult(output=res, artifacts=arts)

_DQ: Optional[MIDRCQueryAgent] = None

def configure_midrc_query_tool(*, df_MIDRC: pd.DataFrame, system_prompt: str):
    global _DQ
    _DQ = MIDRCQueryAgent(df_MIDRC=df_MIDRC, system_prompt=system_prompt)

class MIDRCQueryArgs(BaseModel):
    query: str = Field(..., description="Natural language query for the MIDRC table.")

@toolify_agent(
    name="midrc_query",
    description="Query the MIDRC dataframe using python. Returns previews or aggregates. Can also download files from MIDRC using gen3.",
    args_schema=MIDRCQueryArgs,
    timeout_s=120,
)
async def midrc_query_runner(query: str):
    if _DQ is None:
        raise RuntimeError("MIDRCQuery tool is not configured. Call configure_midrc_query_tool at startup.")
    task = Task(user_msg=query, files=[], kwargs={})
    return await _DQ.run(task, _cs())
