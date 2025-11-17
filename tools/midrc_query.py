import os, io, pandas as pd, matplotlib.pyplot as plt
import duckdb

from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code
from pydantic import BaseModel, Field
from typing import Optional
from tools.shared import toolify_agent, _cs
from core.llm_provider import choose_llm

CRED_PATH = os.getenv("MIDRC_CRED", "~/midrc_credentials.json")

class MIDRCQueryAgent:
    name = "midrc_query"
    model = "gpt-5"

    def __init__(self, df_MIDRC: pd.DataFrame, system_prompt: str):
        self.df_MIDRC = df_MIDRC
        self.system_prompt = system_prompt
        try:
            self.llm = choose_llm()
        except Exception:
            self.llm = None

    async def run(self, task: Task, state: ConversationState, reasoning_effort: str = "medium") -> TaskResult:
        user_text = task.kwargs.get("instructions") or task.user_msg
        if self.df_MIDRC.empty:
            data_context = "df_MIDRC is not available. Use Gen3 for this task. MIDRC credentials can be accesssed via the local 'cred' variable."
        else:
            data_context = (
                f"=== df_MIDRC Columns ===\n{self.df_MIDRC.columns.tolist()}=== df_MIDRC Example Columns ===\n"
                f"{self.df_MIDRC.head(3).to_dict(orient='records')}"
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{user_text}\n\n{data_context}",
            },
        ]
        if self.llm is None:
            raise RuntimeError("LLM provider is not configured.")
        content = await self.llm.ainvoke(messages, temperature=1, reasoning_effort=reasoning_effort)
        code = extract_code_block(content)
        print(code)
        local_env = {"df_MIDRC": self.df_MIDRC, "cred": CRED_PATH, "pd": pd, "plt": plt, "io": io, "os": os, "duckdb": duckdb}
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
    reasoning_effort: str = Field(..., description="Reasoning effort level (select based on task complexity): 'low', 'medium'. Lower levels are faster but may produce less accurate results.")

@toolify_agent(
    name="midrc_query",
    description=(
    "Can query MIDRC, build cohorts, answer MIDRC related questions, and more."
    "Canot download files from MIDRC using gen3. Use the midrc_download tool for that."
    "Capabilities: return dataframes, summaries, plots, and text"
    "Plots will automatically be rendered in the chat UI. Other outputs will not automatically be shown in the chat UI."
    "For MIDRC plots: request them directly from this tool (it can query + plot in one step)"
    ),
    args_schema=MIDRCQueryArgs,
    timeout_s=120,
)
async def midrc_query_runner(query: str, reasoning_effort: str = "medium"):
    if _DQ is None:
        raise RuntimeError("MIDRCQuery tool is not configured. Call configure_midrc_query_tool at startup.")
    task = Task(user_msg=query, files=[], kwargs={})
    return await _DQ.run(task, _cs(), reasoning_effort=reasoning_effort)
