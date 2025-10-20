import os, io, pandas as pd, matplotlib.pyplot as plt
import duckdb
from openai import AsyncOpenAI

from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code
from pydantic import BaseModel, Field
from typing import Optional
from tools.shared import toolify_agent, _cs

class BIHQueryAgent:
    name = "bih_query"
    model = "gpt-5"

    def __init__(self, df_BIH: pd.DataFrame, system_prompt: str):
        key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
        self.df_BIH = df_BIH
        self.system_prompt = system_prompt

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        user_text = task.user_msg
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{user_text}\n\n=== df_BIH Columns ===\n{self.df_BIH.columns.tolist()}=== df_BIH Example Columns ===\n{self.df_BIH.head(3).to_dict(orient='records')}",
            },
        ]
        comp = await self.client.chat.completions.create(
            model=self.model,
            temperature=1,
            messages=messages,
        )
        code = extract_code_block(comp.choices[0].message.content)
        local_env = {"df_BIH": self.df_BIH, "pd": pd, "plt": plt, "io": io, "os": os, "duckdb": duckdb}
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        arts = {"code": code}
        if isinstance(res, dict) and "files" in res:
            arts["files"] = res["files"]
        return TaskResult(output=res, artifacts=arts)

_BIHQ: Optional[BIHQueryAgent] = None

def configure_bih_query_tool(*, df_BIH: pd.DataFrame, system_prompt: str):
    global _BIHQ
    _BIHQ = BIHQueryAgent(df_BIH=df_BIH, system_prompt=system_prompt)

class BIHQueryArgs(BaseModel):
    instructions: str = Field(..., description="Natural language instructions for querying BIH tables.")

@toolify_agent(
    name="bih_query",
    description=(
        "Handles all BIH tasks."
        "The MIDRC-BIH serves as a centralized data discovery and query infrastructure that links diverse and independent data repositories including MIDRC, Stanford AIMI, IDC, NIHCC, TCIA, and ACRdart."
        "The bih_query tool can query and answer questions about data repositories including MIDRC, Stanford AIMI, IDC, NIHCC, TCIA, and ACRdart using python. It can return dataframes, plots, and text. When dataframes or plots are returned, they will be shown in the chat UI automatically."
        "By default use this tools for any questions about the BIH, AIMI, NIHCC, TCIA, and ACRdart datasets. For questions about IDC use the `idc_query` tool and more midrc use the `midrc_query` tool."
        "This tool cannot download files from any of the repositories. For download requests you may use specialized tools specific to the dataset if available."
        "This tool can generate matplotlib plots"
        "matplotlib plots will automatically be rendered in the chat UI. Other outputs will not automatically be shown in the chat UI"
        "Capabilities: return dataframes, summaries, plots, and text. Note that it cannot return interactive plotly charts. Use the `code_gen` tool for that."
    ),
    args_schema=BIHQueryArgs,
    timeout_s=600,
)
async def bih_query_runner(instructions: str):
    if _BIHQ is None:
        raise RuntimeError(
            "BIHQuery tool is not configured."
        )
    task = Task(user_msg=instructions, files=[], kwargs={})
    return await _BIHQ.run(task, _cs())