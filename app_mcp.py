import asyncio
import os
import shlex
import sys
from contextlib import asynccontextmanager
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any

import chainlit as cl
from chainlit.types import ThreadDict
from dotenv import load_dotenv

from core.supervisor_llm import build_supervisor_llm
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    AIMessageChunk,
)
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, create_model

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

load_dotenv(override=True)


class MCPTool(BaseTool):
    name: str
    description: str
    args_schema: Optional[type[BaseModel]] = None
    _session: ClientSession = PrivateAttr()

    def __init__(self, *, name: str, description: str, session: ClientSession, args_schema: Optional[type[BaseModel]] = None):
        super().__init__(name=name, description=description or "", args_schema=args_schema)
        self._session = session

    def _run(self, *args, **kwargs):
        raise NotImplementedError("MCPTool is async-only")

    async def _arun(self, **kwargs):
        if "input" in kwargs and "query" not in kwargs:
            kwargs["query"] = kwargs.pop("input")
        return await self._session.call_tool(self.name, kwargs)


def _schema_type_to_python(schema: Dict[str, Any]):
    t = (schema or {}).get("type")
    if t == "string":
        return str
    if t == "number":
        return float
    if t == "integer":
        return int
    if t == "boolean":
        return bool
    if t == "array":
        item_t = _schema_type_to_python((schema or {}).get("items", {}))
        return List[item_t]
    if t == "object":
        return Dict[str, Any]
    return Any


def _schema_to_model(name: str, schema: Optional[Dict[str, Any]]):
    if name == "orchestrator":
        return create_model(
            "MCP_orchestrator_Args",
            query=(Optional[str], None),
            input=(Optional[str], None),
            pipeline=(Optional[str], "idc"),
            tool_names=(Optional[List[str]], None),
            files=(Optional[List[str]], None),
            thread_id=(Optional[str], None),
            include_tool_payloads=(Optional[bool], True),
        )

    if not schema or schema.get("type") != "object":
        return None

    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    fields: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_t = _schema_type_to_python(prop_schema)
        default = ... if prop_name in required else None
        fields[prop_name] = (py_t, Field(default=default))

    if not fields:
        return None

    return create_model(f"MCP_{name}_Args", **fields)


def _get_tool_field(tool_def: Any, key: str):
    if hasattr(tool_def, key):
        return getattr(tool_def, key)
    if isinstance(tool_def, dict):
        return tool_def.get(key)
    return None


def _build_mcp_tools_from_defs(session: ClientSession, tool_defs: List[Any]) -> List[BaseTool]:
    tools: List[BaseTool] = []
    for tool_def in tool_defs:
        name = _get_tool_field(tool_def, "name")
        if not name:
            print(f"Skipping MCP tool with missing name: {tool_def!r}")
            continue
        description = _get_tool_field(tool_def, "description") or ""
        input_schema = _get_tool_field(tool_def, "inputSchema")
        args_schema = _schema_to_model(name, input_schema)
        tools.append(MCPTool(name=name, description=description, session=session, args_schema=args_schema))
    return tools


MCP_TOOLS_BY_CONN: Dict[str, List[BaseTool]] = {}
_MCP_TASK: Optional[asyncio.Task] = None
_MCP_READY: asyncio.Event = asyncio.Event()
_MCP_STOP: asyncio.Event = asyncio.Event()


def build_graph(tools: Optional[List[BaseTool]] = None, checkpointer: Optional[MemorySaver] = None):
    def _orchestrator_tool_summary() -> str:
        return (
            "You can only call the MCP tool `orchestrator`.\n"
            "This MCP server exposes the IDC-oriented VoxelInsight orchestrator, which handles IDC metadata and "
            "collection questions, IDC documentation/code Q&A, clinical data export requests, pathology tile "
            "retrieval, IDC DICOM download workflows, and DICOM-to-NIfTI conversion.\n"
            "The MCP tool returns text for you to read. It does not automatically render files, plots, tables, or "
            "other artifacts to the user.\n"
            "Treat it as the single entrypoint for VoxelInsight work in this app and route requests through it.\n"
        )

    def _select_tools(all_tools: List[BaseTool]) -> List[BaseTool]:
        orchestrators = [t for t in all_tools if getattr(t, "name", None) == "orchestrator"]
        if not orchestrators and all_tools:
            print("Warning: orchestrator tool not found; no tools will be bound.")
            return []
        return orchestrators

    policy = SystemMessage(
        content=(
            "You are VoxelInsight MCP, a minimal test supervisor.\n"
            f"{_orchestrator_tool_summary()}\n"
            "Do not describe or rely on any other VoxelInsight pipelines or MCP tools.\n"
            "Do not assume the user can inspect tool artifacts directly; if a tool result matters, summarize it in text.\n"
            "Keep responses concise."
        )
    )

    tools = _select_tools(tools or [])
    base_llm = build_supervisor_llm(temperature=1, reasoning_effort="low")
    llm = base_llm.bind_tools(tools) if tools else base_llm
    tool_node = ToolNode(tools=tools)

    async def call_model(state: MessagesState):
        msgs = [policy] + state["messages"]
        resp = await llm.ainvoke(msgs)
        return {"messages": [resp]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "final"

    async def call_final(state: MessagesState):
        return {}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_node("tools", tool_node)
    g.add_node("final", call_final)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", "final": "final"})
    g.add_edge("tools", "agent")
    g.add_edge("final", END)
    return g.compile(checkpointer=checkpointer)


_CP = MemorySaver()
GRAPH = build_graph(checkpointer=_CP)


async def _zip_paths(paths: List[str], zip_path: Path):
    def _worker():
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                pth = Path(p)
                if pth.is_dir():
                    base = pth.parent
                    for f in pth.rglob("*"):
                        if f.is_file():
                            zf.write(f, arcname=str(f.relative_to(base)))
                elif pth.is_file():
                    zf.write(pth, arcname=pth.name)
    await asyncio.to_thread(_worker)


def _collect_tool_payloads(messages: List[Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, dict):
                payloads.append(content)
            else:
                try:
                    import json
                    payloads.append(json.loads(content))
                except Exception:
                    pass
    return payloads


async def _render_payload(payload: Dict[str, Any]):
    ok = payload.get("ok", True)
    if not ok:
        err = payload.get("error") or "Tool returned an error."
        await cl.Message(content=f"⚠️ {err}").send()
        return

    outputs = payload.get("outputs", {}) or {}
    ui = payload.get("ui", []) or []

    # Render nested tool payloads produced by the orchestrator.
    nested_payloads = outputs.get("tool_payloads", []) or []
    for p in nested_payloads:
        if isinstance(p, dict):
            await _render_payload(p)

    code_text = outputs.get("code")
    if code_text:
        code_el = cl.CustomElement(
            name="IdcCodeView",
            props={"code": str(code_text), "title": "Generated code"},
            display="inline",
        )
        await cl.Message(content="Generated code (expand to inspect):", elements=[code_el]).send()

    for item in ui:
        kind = item.get("kind")
        if kind == "plotly_json_path":
            path = item.get("path")
            try:
                import json
                from plotly.io import from_json
                spec = Path(path).read_text()
                fig = from_json(spec)
                await cl.Message(
                    content="Interactive chart:",
                    elements=[cl.Plotly(name="plot", figure=fig)]
                ).send()
            except Exception:
                await cl.Message(content="(Plotly figure could not be rendered.)").send()
        elif kind == "image_path":
            path = item.get("path")
            if path and Path(path).exists():
                await cl.Message(
                    content="Here is your result:",
                    elements=[cl.Image(name=Path(path).name, path=path)]
                ).send()
        elif kind == "binary_path":
            path = item.get("path")
            if path and Path(path).exists():
                await cl.Message(
                    content="Here is your file:",
                    elements=[cl.File(name=Path(path).name, path=path)]
                ).send()

    files = outputs.get("files", [])
    output_dir = outputs.get("output_dir")
    tool = outputs.get("tool", "unknown_tool")
    if not files and output_dir and Path(output_dir).exists():
        files = [str(f) for f in Path(output_dir).rglob("*") if f.is_file()]
    if files:
        zip_tmpdir = Path(tempfile.mkdtemp(prefix="vi_zip_"))
        zip_path = zip_tmpdir / "download.zip"
        await _zip_paths(files, zip_path)
        output_content = f"**Files ready**\n- Items: {len(files)}\n\nClick to download:"
        if tool == "dicom2nifti":
            output_content = f"**Dicom to Nifti conversion complete:**\n- Nifti Files: {len(files)}\n\nClick to download:"
        elif tool == "tcia_download":
            output_content = f"**TCIA Download complete:**\n- Items: {len(files)}\n\nClick to download:"
        elif tool == "midrc_download":
            output_content = f"**MIDRC Download complete:**\n- Items: {len(files)}\n\nClick to download:"

        await cl.Message(
            content=output_content,
            elements=[cl.File(name=zip_path.name, path=str(zip_path))]
        ).send()


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text") or "")
        return "".join(chunks)
    return str(content or "")


class VoxelInsightHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.node_descriptions = {
            "agent": "VoxelInsight MCP",
            "tools": "Tools",
            "final": "VoxelInsight MCP Final",
        }

    async def _rename_root(self, name: str):
        try:
            step = cl.context.current_step
            if step is not None:
                step.name = name
                await step.update()
        except Exception:
            pass

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        await self._rename_root("VoxelInsight MCP")

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        await self._rename_root("VoxelInsight MCP")

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        await self._rename_root("VoxelInsight MCP")

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "tools")
        await self._rename_root(tool_name)

    async def on_tool_end(self, output: str, **kwargs) -> None:
        pass

    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        await self._rename_root("VoxelInsight MCP")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    return val if val is None or val.strip() == "" else val.strip()


@asynccontextmanager
async def _connect_mcp_streams():
    transport = (_env("MCP_TRANSPORT", "stdio") or "stdio").lower()

    if transport == "stdio":
        server_path = Path(__file__).with_name("mcp_server.py")
        cmd = _env("MCP_COMMAND", sys.executable) or sys.executable
        args_raw = _env("MCP_ARGS", str(server_path)) or str(server_path)
        args = shlex.split(args_raw) if isinstance(args_raw, str) else [str(server_path)]
        cwd = _env("MCP_CWD", str(server_path.parent))

        server_params = StdioServerParameters(
            command=cmd,
            args=args,
            cwd=cwd,
        )
        async with stdio_client(server_params) as (read, write):
            yield read, write
        return

    url = _env("MCP_URL")
    if not url:
        raise RuntimeError("MCP_URL is required for non-stdio transports.")

    if transport == "sse":
        async with sse_client(url) as (read, write):
            yield read, write
        return

    if transport in {"streamable-http", "streamablehttp"}:
        async with streamablehttp_client(url) as (read, write, _get_session_id):
            yield read, write
        return

    raise RuntimeError(f"Unsupported MCP_TRANSPORT: {transport}")


async def _run_internal_mcp():
    transport = (_env("MCP_TRANSPORT", "stdio") or "stdio").lower()
    try:
        print(f"MCP starting (transport={transport})...")
        async with _connect_mcp_streams() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool_result = await session.list_tools()
                tools = _build_mcp_tools_from_defs(session, tool_result.tools)
                MCP_TOOLS_BY_CONN["internal"] = tools

                global GRAPH
                GRAPH = build_graph(tools=tools, checkpointer=_CP)
                print(f"MCP connected: tools={len(tools)} names={[t.name for t in tools]}")
                _MCP_READY.set()

                await _MCP_STOP.wait()
    except Exception as e:
        _MCP_READY.set()
        if isinstance(e, ExceptionGroup):
            print("MCP failed to start: ExceptionGroup")
            _log_exception_group(e, indent=2)
        else:
            print(f"MCP failed to start: {type(e).__name__}: {e}")
        raise


def _log_exception_group(eg: ExceptionGroup, indent: int = 0) -> None:
    pad = " " * indent
    for i, exc in enumerate(eg.exceptions, start=1):
        if isinstance(exc, ExceptionGroup):
            print(f"{pad}sub-exception {i}: ExceptionGroup")
            _log_exception_group(exc, indent=indent + 2)
        else:
            print(f"{pad}sub-exception {i}: {type(exc).__name__}: {exc}")


async def _ensure_internal_mcp():
    global _MCP_TASK
    if _MCP_TASK is None:
        print("MCP init requested; launching background task.")
        _MCP_TASK = asyncio.create_task(_run_internal_mcp())
    await _MCP_READY.wait()


@cl.on_message
async def on_message(message: cl.Message):
    await _ensure_internal_mcp()
    file_elements = [el for el in (message.elements or []) if isinstance(el, cl.File)]
    files: List[str] = []
    for f in file_elements:
        tmpdir = Path(tempfile.mkdtemp())
        new_path = tmpdir / f.name
        shutil.copy(f.path, new_path)
        files.append(str(new_path))

    config = {"configurable": {"thread_id": cl.context.session.id}}

    cb = cl.LangchainCallbackHandler()
    status_handler = VoxelInsightHandler()

    await status_handler._rename_root("Initializing VoxelInsight MCP…")

    streaming_reply: Optional[cl.Message] = None

    async def _ensure_streaming_reply() -> cl.Message:
        nonlocal streaming_reply
        if streaming_reply is None:
            streaming_reply = cl.Message(content="")
            await streaming_reply.send()
        return streaming_reply

    async def _finalize_streaming_reply():
        nonlocal streaming_reply
        if streaming_reply is not None:
            try:
                await streaming_reply.update()
            finally:
                streaming_reply = None

    initial_state = {
        "messages": [HumanMessage(content=message.content + ("" if not files else f" User uploaded files: {files}"))],
    }

    try:
        async for event, meta in GRAPH.astream(
            initial_state,
            stream_mode="messages",
            config=RunnableConfig(callbacks=[cb, status_handler], **config),
        ):
            node = meta.get("langgraph_node")
            if node:
                friendly = status_handler.node_descriptions.get(node, f"▶️ {node}")
                await status_handler._rename_root(friendly)

            if isinstance(event, ToolMessage):
                payloads = _collect_tool_payloads([event])
                for p in payloads:
                    await _render_payload(p)

            if isinstance(event, (AIMessage, AIMessageChunk)):
                content = _stringify_content(getattr(event, "content", None))
                has_tool_call = bool(getattr(event, "tool_calls", None))
                if has_tool_call:
                    if content:
                        await cl.Message(content=content).send()
                    await _finalize_streaming_reply()
                    continue
                if content and meta.get("langgraph_node") in ("final", "agent"):
                    msg = await _ensure_streaming_reply()
                    await msg.stream_token(content)

        await _finalize_streaming_reply()

    except Exception as e:
        import traceback; traceback.print_exc()
        await cl.Message(content=f"🚨 Error:\n```\n{type(e).__name__}: {e}\n```").send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    pass


@cl.action_callback("action_button")
async def on_action(action):
    await action.remove()
