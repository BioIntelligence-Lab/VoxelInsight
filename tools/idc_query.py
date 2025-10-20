import os, io, pandas as pd, matplotlib.pyplot as plt
from openai import AsyncOpenAI
from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code


class DataQueryAgent:
    name = "idc_query"
    model = "gpt-5"

    def __init__(self, df_IDC: pd.DataFrame, df_BIH: pd.DataFrame, system_prompt: str):
        key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
        self.df_IDC = df_IDC
        self.df_BIH = df_BIH
        self.system_prompt = system_prompt

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{task.user_msg}\n\n=== df_IDC Columns ===\n{self.df_IDC.columns.tolist()}\n\n=== df_IDC Example Rows ===\n{self.df_IDC.head(3).to_dict(orient='records')}",
            },
        ]
        comp = await self.client.chat.completions.create(
            model=self.model,
            temperature=1,
            messages=messages,
            reasoning_effort="medium",
        )
        code = extract_code_block(comp.choices[0].message.content)
        print(code)
        print(f"Running user code:\n{code}\n")
        local_env = {"df_IDC": self.df_IDC, "df_BIH": self.df_BIH, "pd": pd, "plt": plt, "io": io, "os": os,}
        out = run_user_code(code, local_env)
        res = out.get("res_query")

        if isinstance(res, pd.DataFrame):
            state.memory["last_df"] = res
        
        arts = {"code": code}
        if isinstance(res, dict) and "files" in res:
            arts["files"] = res["files"]    
        return TaskResult(output=res, artifacts=arts)
    
from pydantic import BaseModel, Field
from typing import Optional
from tools.shared import toolify_agent, _cs
from core.state import Task

_DQ: Optional[DataQueryAgent] = None

def configure_idc_query_tool(*, df_IDC: pd.DataFrame, df_BIH: pd.DataFrame, system_prompt: str):
    global _DQ
    _DQ = DataQueryAgent(df_IDC=df_IDC, df_BIH=df_BIH, system_prompt=system_prompt)

class DataQueryArgs(BaseModel):
    instructions: str = Field(..., description="Natural language for the IDC tables.")

@toolify_agent(
    name="idc_query",
    description=(
        "Handles all IDC tasks."  
        "Capabilities: return dataframes, summaries, plots, text, and download links (shown automatically)."
        "For IDC plots: request them directly from this tool (it can query + plot in one step)."
        "Matplotlib plots and file downloads (when files are downloaded directly from the IDC) will automatically be rendered in the chat UI. Other outputs will not automatically be shown in the chat UI." 
    ),
    args_schema=DataQueryArgs,
    timeout_s=600,
)
async def idc_query_runner(instructions: str):
    if _DQ is None:
        raise RuntimeError(
            "IDCQuery tool is not configured."
        )
    task = Task(user_msg=instructions, files=[], kwargs={})
    return await _DQ.run(task, _cs())