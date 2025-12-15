import os, io, pandas as pd, matplotlib.pyplot as plt
import chainlit as cl
from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code
from core.llm_provider import choose_llm


class DataQueryAgent:
    name = "idc_query"
    model = "gpt-5-mini"

    def __init__(self, df_IDC: pd.DataFrame, df_BIH: pd.DataFrame, system_prompt: str):
        self.df_IDC = df_IDC
        self.df_BIH = df_BIH
        self.system_prompt = system_prompt
        try:
            self.llm = choose_llm()
        except Exception:
            self.llm = None

    async def run(self, task: Task, state: ConversationState, reasoning_effort: str = "medium") -> TaskResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{task.user_msg}\n\n=== df_IDC Columns ===\n{self.df_IDC.columns.tolist()}\n\n=== df_IDC Example Rows ===\n{self.df_IDC.head(3).to_dict(orient='records')}",
            },
        ]
        if self.llm is None:
            raise RuntimeError("LLM provider is not configured.")
        content = await self.llm.ainvoke(messages, temperature=1, reasoning_effort=reasoning_effort)
        code = extract_code_block(content)
        print(code)
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
    reasoning_effort: str = Field(..., description="Reasoning effort level (select based on task complexity): 'minimal', 'low', 'medium'. Lower levels are faster (and preferred for most cases)but may produce less accurate results. When a result isn't satisfoctory, try increasing the reasoning effort to 'medium'.")

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
async def idc_query_runner(instructions: str, reasoning_effort: str = "medium"):
    if _DQ is None:
        raise RuntimeError(
            "IDCQuery tool is not configured."
        )
    task = Task(user_msg=instructions, files=[], kwargs={})
    return await _DQ.run(task, _cs(), reasoning_effort=reasoning_effort)
