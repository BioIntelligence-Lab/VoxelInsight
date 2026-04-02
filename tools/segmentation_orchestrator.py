from __future__ import annotations

import csv
import json
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field

from core.state import TaskResult
from core.supervisor_llm import build_supervisor_llm
from tools.shared import toolify_agent, TOOL_REGISTRY


_GRAPH_CACHE: Dict[str, Any] = {}


def _normalize_token(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def _load_ts_mappings() -> Dict[str, List[str]]:
    mapping_files = (
        Path("Data/TotalSegmentatorMappingsCT.tsv"),
        Path("Data/TotalSegmentatorMappingsMRI.tsv"),
    )
    rows: Dict[str, set[str]] = {}
    for mapping_file in mapping_files:
        if not mapping_file.exists():
            continue
        with mapping_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                task = _normalize_token(str(row.get("task_name", "")))
                roi = _normalize_token(str(row.get("roi_subset", "")))
                if not task or not roi:
                    continue
                rows.setdefault(task, set()).add(roi)
    return {task: sorted(rois) for task, rois in rows.items()}


def _format_ts_mappings_for_prompt() -> str:
    allowed = _load_ts_mappings()
    if not allowed:
        return "(No TotalSegmentator mappings available.)"
    lines: List[str] = []
    for task_name in sorted(allowed):
        rois = allowed.get(task_name) or []
        lines.append(f"- {task_name}: {', '.join(rois)}")
    return "\n".join(lines)


_TS_MAPPINGS_PROMPT = _format_ts_mappings_for_prompt()


def _segmentation_policy() -> str:
    return (
        "You are VoxelInsight Segmentation, a specialized sub-supervisor for segmentation only.\n\n"
        "Scope\n"
        "- Your only job is to choose and run the correct segmentation tool.\n"
        "- You may use TotalSegmentator through the `imaging` tool or MONAI bundles through the `monai` tool.\n"
        "- Do not answer imaging metadata, downloads, radiomics, or visualization requests yourself.\n"
        "- If the request is not a segmentation task, return a short explanation and do not call tools.\n\n"
        "Tool selection\n"
        "- Prefer `imaging` for TotalSegmentator-style anatomical segmentation requests.\n"
        "- Prefer `monai` when the user asks for a MONAI model or a bundle-specific workflow.\n"
        "- Keep tool instructions concise but include enough detail to avoid input-shape ambiguity.\n"
        "- Assume tools cannot see prior tool outputs unless you pass them explicitly.\n"
        "- Retry a failing tool at most 2 additional times with clearer instructions.\n\n"
        "TotalSegmentator mapping contract for `imaging`\n"
        "- You MUST adhere to the allowed TotalSegmentator task/ROI mappings below.\n"
        "- If the requested anatomy can be satisfied by `task_name=total` (or `total_mr` when appropriate) plus valid `roi_subset` / `roi_subsets`, you MUST use that route instead of a specialized non-total task.\n"
        "- Prefer `roi_subset` / `roi_subsets` over a full-task run whenever the user requests specific organs or structures and those ROIs are valid for the selected task.\n"
        "- Avoid running a full task with only `task_name` when a narrower valid ROI selection would satisfy the request, because full-task runs are more expensive.\n"
        "- Use a specialized non-total task only when the request requires task-specific output that `total` / `total_mr` ROI subsets cannot provide.\n"
        "- Favor `total` / `total_mr` ROI subsets because specialized tasks often do not support `fast` mode.\n"
        "- Example: for 'segment the liver', use `task_name=total` with `roi_subset=liver`, not the `liver_segments` task.\n"
        "- Example: only use a task like `liver_segments` when the user explicitly needs liver segment subdivision rather than the liver as a whole.\n"
        "- Only use `roi_subset` / `roi_subsets` values that appear under the selected task.\n"
        "- If a task is not in mappings, do not pass `roi_subset` / `roi_subsets`.\n"
        "- Normalize values to canonical mapping tokens (lowercase with underscores).\n"
        "- Never invent task names or ROI values outside this mapping table.\n"
        "- Call the `imaging` tool directly with the final canonical `task_name` and optional ROI list.\n"
        "- Allowed mappings:\n"
        f"{_TS_MAPPINGS_PROMPT}\n\n"
        "Final response behavior\n"
        "- Your final response is internal-only and will be consumed by the main VoxelInsight supervisor.\n"
        "- Do not write user-facing prose, explanations, or conversational text.\n"
        "- After tool execution, return only a terse internal status note.\n\n"
        "Files\n"
        "- If file paths are provided in the user message, pass them to the chosen tool."
    )

def _segmentation_tools() -> List[Any]:
    allowed = {"imaging", "monai"}
    return [t for t in TOOL_REGISTRY if t.name in allowed]


def _collect_tool_payloads(messages: List[Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, dict):
                payload = dict(content)
                payload.setdefault("tool_name", getattr(m, "name", None))
                payloads.append(payload)
            else:
                try:
                    payload = json.loads(content)
                    if isinstance(payload, dict):
                        payload.setdefault("tool_name", getattr(m, "name", None))
                        payloads.append(payload)
                except Exception:
                    pass
    return payloads


def _extract_final_text(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return str(getattr(msg, "content", "") or "")
    return ""


def _summarize_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": "no_action",
        "chosen_tool": None,
        "segmentations": [],
        "segmentations_map": {},
        "output_dir": None,
    }
    last_error: Optional[str] = None
    for payload in payloads:
        tool_name = payload.get("tool_name")
        if tool_name:
            summary["chosen_tool"] = tool_name
        outputs = payload.get("outputs") if isinstance(payload, dict) else None
        if payload.get("ok") and isinstance(outputs, dict):
            summary["status"] = "ok"
            if "segmentations" in outputs:
                summary["segmentations"] = outputs["segmentations"]
            if "segmentations_map" in outputs:
                summary["segmentations_map"] = outputs["segmentations_map"]
            if "output_dir" in outputs:
                summary["output_dir"] = outputs["output_dir"]
        elif payload.get("error"):
            summary["status"] = "error"
            last_error = str(payload["error"])

    if last_error:
        summary["error"] = last_error
    return summary


def _build_graph() -> Any:
    tools = _segmentation_tools()
    tool_map = {tool.name: tool for tool in tools}
    policy = SystemMessage(content=_segmentation_policy())
    llm = build_supervisor_llm(
        temperature=1,
        reasoning_effort="low",
    ).bind_tools(tools)

    async def call_model(state: MessagesState):
        msgs = [policy] + state["messages"]
        for attempt in range(2):
            try:
                resp = await llm.ainvoke(msgs)
                return {"messages": [resp]}
            except httpx.RemoteProtocolError as e:
                if attempt == 0:
                    continue
                raise e

    async def execute_tools(state: MessagesState):
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        tool_messages: List[ToolMessage] = []
        for call in last.tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})

            tool = tool_map.get(str(tool_name))
            if tool is None:
                payload = {"ok": False, "error": f"Unknown segmentation tool: {tool_name}"}
            else:
                try:
                    payload = await tool.ainvoke(tool_args)
                except Exception as e:
                    payload = {"ok": False, "error": f"{tool_name} failed. Error: {e}"}

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(payload),
                    tool_call_id=str(call.get("id", "")),
                    name=str(tool_name or "tool"),
                )
            )
        return {"messages": tool_messages}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "final"

    async def call_final(state: MessagesState):
        return {}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_node("tools", execute_tools)
    g.add_node("final", call_final)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", "final": "final"})
    g.add_edge("tools", "agent")
    g.add_edge("final", END)
    return g.compile()


class SegmentationOrchestratorArgs(BaseModel):
    query: str = Field(..., description="Natural-language segmentation request.")
    files: Optional[List[str]] = Field(
        None,
        description="Optional file paths to image volumes for the segmentation request.",
    )
    include_tool_payloads: bool = Field(
        False,
        description="Include normalized payloads from internal segmentation tool calls.",
    )


@toolify_agent(
    name="segmentation_orchestrator",
    description=(
        "Runs a specialized segmentation sub-supervisor that chooses between segmentation tools "
        "such as TotalSegmentator (`imaging`) and MONAI (`monai`). "
        "Use this instead of calling individual segmentation tools from the top-level supervisor."
    ),
    args_schema=SegmentationOrchestratorArgs,
    timeout_s=1200,
)
async def segmentation_orchestrator_runner(
    query: str,
    files: Optional[List[str]] = None,
    include_tool_payloads: bool = False,
):
    cache_key = "default"
    if cache_key not in _GRAPH_CACHE:
        _GRAPH_CACHE[cache_key] = _build_graph()
    graph = _GRAPH_CACHE[cache_key]

    user_content = query
    if files:
        user_content = f"{query} Input files: {files}"

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_content)]},
        config=RunnableConfig(),
    )

    messages = result.get("messages", [])
    payloads = _collect_tool_payloads(messages)
    summary: Dict[str, Any] = _summarize_payloads(payloads)

    artifacts: Dict[str, Any] = {}
    for key in ("segmentations", "segmentations_map"):
        value = summary.get(key)
        if value:
            artifacts[key] = value

    if include_tool_payloads:
        artifacts["tool_payloads"] = payloads

    if summary.get("status") == "error":
        return TaskResult(output=str(summary.get("error") or "Segmentation failed."), artifacts=artifacts)

    return TaskResult(output=None, artifacts=artifacts)
