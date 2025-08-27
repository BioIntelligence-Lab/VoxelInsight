# tools/shared.py
from __future__ import annotations
import asyncio, time, json, tempfile
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Type, Callable
import pandas as pd
import plotly
import plotly.graph_objects as go
from matplotlib.figure import Figure
from pydantic import BaseModel, PrivateAttr
from langchain_core.tools import BaseTool

from core.state import Task, ConversationState

class ToolReturn(TypedDict, total=False):
    ok: bool
    outputs: Dict[str, Any]
    ui: List[Dict[str, Any]]
    memory_delta: Dict[str, Any]
    logs: List[str]
    error: str

def normalize_task_result(res) -> ToolReturn:
    outputs: Dict[str, Any] = {}
    ui: List[Dict[str, Any]] = []
    mem: Dict[str, Any] = {}
    logs: List[str] = []

    if res is None:
        return {"ok": False, "error": "Tool returned no result."}

    arts = getattr(res, "artifacts", {}) or {}
    out  = getattr(res, "output", None)

    for k in ("segmentations","segmentations_map","output_dir","image_path","files","nifti_paths","code"):
        if k in arts: outputs[k] = arts[k]
    for k in ("image_path","segmentations","segmentations_map"):
        if k in arts: mem[k] = arts[k]

    if isinstance(out, pd.DataFrame):
        outputs["df_preview"] = {"rows": out.head(50).to_dict("records"), "nrows": len(out)}
    elif isinstance(out, Figure):
        tmp = Path(tempfile.mkdtemp()) / "plot.png"
        out.savefig(tmp, bbox_inches="tight")
        ui.append({"kind":"image_path","path":str(tmp)})
    elif isinstance(out, go.Figure):
        tmp = Path(tempfile.mkdtemp()) / "plotly.json"
        tmp.write_text(json.dumps(out, cls=plotly.utils.PlotlyJSONEncoder))  # type: ignore
        ui.append({"kind":"plotly_json_path","path":str(tmp)})
    elif hasattr(out, "read"):
        tmp = Path(tempfile.mkdtemp()) / "blob.bin"
        tmp.write_bytes(out.read())
        ui.append({"kind":"binary_path","path":str(tmp)})
    elif isinstance(out, dict):
        outputs.update(out)
    elif out is not None:
        outputs["text"] = str(out)

    return {"ok": True, "outputs": outputs, "ui": ui, "memory_delta": mem, "logs": logs}

def _cs() -> ConversationState:
    s = ConversationState(); s.memory = {}; return s

TOOL_REGISTRY: List[BaseTool] = []

class AgentTool(BaseTool):
    """Generic wrapper that calls an async runner and normalizes TaskResult."""
    name: str
    description: str
    args_schema: Type[BaseModel]
    timeout_s: int = 300

    _runner = PrivateAttr(default=None)

    async def _arun(self, *args, **kwargs) -> ToolReturn:
        try:
            if args and not kwargs:
                field_names = list(self.args_schema.model_fields.keys()) if hasattr(self.args_schema, "model_fields") else list(self.args_schema.__fields__.keys())  # v2 vs v1
                if len(field_names) == 1:
                    kwargs = {field_names[0]: args[0]}
                else:
                    return {"ok": False, "error": f"{self.name}: positional args not supported for multi-field schema."}
        except Exception as e:
            return {"ok": False, "error": f"{self.name}: arg normalization failed: {e}"}

        if self._runner is None:
            return {"ok": False, "error": f"{self.name} runner not configured."}

        start = time.time()
        try:
            res = await asyncio.wait_for(self._runner(**kwargs), timeout=self.timeout_s)
            out = normalize_task_result(res)
            out.setdefault("logs", []).append(f"{self.name}: elapsed={time.time()-start:.2f}s")
            return out
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"{self.name} timed out after {self.timeout_s}s"}
        except Exception as e:
            return {"ok": False, "error": f"{self.name} failed. Error: {e}"}

    def _run(self, *args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(self._arun(*args, **kwargs))

def toolify_agent(*, name: str, description: str, args_schema: Type[BaseModel], timeout_s: int = 300):
    def _wrap(runner_fn: Callable[..., Any]):
        tool = AgentTool(
            name=name,
            description=description,
            args_schema=args_schema,
            timeout_s=timeout_s,
        )
        tool._runner = runner_fn
        TOOL_REGISTRY.append(tool)
        return tool
    return _wrap