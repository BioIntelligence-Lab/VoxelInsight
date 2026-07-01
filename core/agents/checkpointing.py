from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from core.storage import persist_root


_CHECKPOINTER: Optional[Any] = None
_CONNECTION: Optional[Any] = None
_LOCK = asyncio.Lock()


def checkpoint_path() -> Path:
    configured = os.getenv("VOXELINSIGHT_CHECKPOINT_DB")
    path = (
        Path(configured).expanduser().absolute()
        if configured
        else persist_root() / "state" / "voxelinsight-checkpoints.sqlite"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def get_durable_checkpointer() -> Any:
    """Return one process-wide async SQLite saver backed by a durable file."""
    global _CHECKPOINTER, _CONNECTION
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    async with _LOCK:
        if _CHECKPOINTER is not None:
            return _CHECKPOINTER
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except Exception as exc:
            raise RuntimeError(
                "Durable DeepAgents state requires langgraph-checkpoint-sqlite. "
                "Install the pinned project requirements."
            ) from exc

        connection = await aiosqlite.connect(str(checkpoint_path()))
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA synchronous=NORMAL")
        checkpointer = AsyncSqliteSaver(connection)
        await checkpointer.setup()
        _CONNECTION = connection
        _CHECKPOINTER = checkpointer
        return checkpointer


async def close_durable_checkpointer() -> None:
    global _CHECKPOINTER, _CONNECTION
    connection = _CONNECTION
    _CHECKPOINTER = None
    _CONNECTION = None
    if connection is not None:
        await connection.close()


async def seed_registry_from_child_checkpoints(thread_id: str) -> None:
    """One-time migration of tool artifacts stranded in child namespaces."""

    from core.agents.durable_registry import (
        mark_registry_seeded,
        persist_registry_delta,
        registry_seeded,
    )

    normalized_thread_id = str(thread_id).strip()
    if not normalized_thread_id or registry_seeded(normalized_thread_id):
        return

    checkpointer = await get_durable_checkpointer()
    latest_namespaces: set[str] = set()
    merged = {"artifact_registry": {}, "data_registry": {}, "tool_events": []}
    config = {"configurable": {"thread_id": normalized_thread_id}}
    async for item in checkpointer.alist(config):
        configurable = item.config.get("configurable", {})
        namespace = str(configurable.get("checkpoint_ns") or "")
        if not namespace or namespace in latest_namespaces:
            continue
        latest_namespaces.add(namespace)
        values = item.checkpoint.get("channel_values", {})
        if not isinstance(values, dict):
            continue
        merged["artifact_registry"].update(values.get("artifact_registry") or {})
        merged["data_registry"].update(values.get("data_registry") or {})
        merged["tool_events"].extend(values.get("tool_events") or [])

    if any(merged.values()):
        persist_registry_delta(normalized_thread_id, merged)
    else:
        mark_registry_seeded(normalized_thread_id)
