from __future__ import annotations

from typing import Optional, List

from mcp.server.fastmcp import FastMCP

from tools.shared import normalize_task_result
import tools.orchestrator as orch_mod


mcp = FastMCP("VoxelInsight")


@mcp.tool()
async def orchestrator(
    query: str,
    pipeline: str = "idc",
    tool_names: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
    include_tool_payloads: bool = True,
) -> dict:
    res = await orch_mod.orchestrator_runner._runner(
        query=query,
        pipeline=pipeline,
        tool_names=tool_names,
        files=files,
        thread_id=thread_id,
        include_tool_payloads=include_tool_payloads,
    )
    return normalize_task_result(res)


if __name__ == "__main__":
    mcp.run()
