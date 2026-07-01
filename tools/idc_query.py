import os, io, json, pandas as pd, matplotlib.pyplot as plt
import chainlit as cl
from core.state import Task, TaskResult, ConversationState
from core.utils import extract_code_block
from core.sandbox import run_user_code
from core.llm_provider import choose_llm
from core.storage import get_run_dir


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
        out_dir = str(get_run_dir(self.name, persist=True))
        runtime_ctx = {"ARGS": {"out_dir": out_dir}}
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"{task.user_msg}\n\n"
                    f"=== RUNTIME CONTEXT (JSON) ===\n{json.dumps(runtime_ctx, indent=2)}\n\n"
                    f"=== df_IDC Columns ===\n{self.df_IDC.columns.tolist()}\n\n"
                    f"=== df_IDC Example Rows ===\n{self.df_IDC.head(3).to_dict(orient='records')}"
                ),
            },
        ]
        if self.llm is None:
            raise RuntimeError("LLM provider is not configured.")
        content = await self.llm.ainvoke(messages, temperature=1, reasoning_effort=reasoning_effort)
        code = extract_code_block(content)
        print(code)
        local_env = {
            "df_IDC": self.df_IDC,
            "df_BIH": self.df_BIH,
            "pd": pd,
            "plt": plt,
            "io": io,
            "os": os,
            "OUT_DIR": out_dir,
        }
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
        "Capabilities: return metadata dataframes, summaries, plots, text, viewer links, and SeriesInstanceUIDs for downstream acquisition."
        "For IDC plots: request them directly from this tool (it can query + plot in one step)."
        "Matplotlib plots will automatically be rendered in the chat UI. Other outputs will not automatically be shown in the chat UI." 
        "\nWhen the user asks for IDC imaging metadata, use the idc_query tool. These include questions like \"How many patients are in IDC?\", \"List all SeriesInstanceUIDs for CT scans in collection X\", \"Show me a summary of the IDC metadata tables\", etc."
        "\nIf the user just wants to view a study from a collection in IDC, you do not need to download the study. the tool idc_query can provide links to view images in the IDC viewer directly."
        "\n- `idc_query`: inspect IDC metadata, summarize tables, and surface SeriesInstanceUIDs. Never fabricate IDC answers—query first."
        "\n- Do not use idc_query to download imaging files or inspect IDC clinical tables. Use idc_download for imaging acquisition and clinical_data_download for clinical tables."
        "\n- Apply strict data minimization. For count, aggregate, table-summary, or chart requests, request only the scalar grouping and aggregate columns needed by the user."
        "\n- Never request viewer links, PatientID lists, StudyInstanceUID lists, SeriesInstanceUID lists, or list-valued detail fields unless the user explicitly asks for them or a confirmed downstream acquisition/viewing step requires them."
        "\n-NEVER ask the idc_query tool to provide information beyond what the user has requested; this will waste time and resources. Efficiency is key."
        "\n- Aim to get the most minimal information needed to satisfy the user's request."
        "\n- For instance do not ask the tool for notes which you could have surmised. You will receive the tools code output and code itself so you can interpret it directly."
        "\n- Aim to get the result in as few tool calls as possible. Do not split into multiple calls unless absolutely necessary."
        "\n- If the user wants to view or visualize the radiology imaging data without downloading, the idc_query tool can provide links to online viewers."
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
