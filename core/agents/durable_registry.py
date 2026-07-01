from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from core.agents.checkpointing import checkpoint_path


_LOCK = threading.RLock()
_INITIALIZED_PATHS: set[str] = set()


def _database_path(db_path: Optional[Path | str] = None) -> Path:
    if db_path:
        return Path(db_path).expanduser().absolute()
    configured = os.getenv("VOXELINSIGHT_REGISTRY_DB")
    if configured:
        return Path(configured).expanduser().absolute()
    return checkpoint_path().with_name("voxelinsight-registry.sqlite")


def _connect(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    path = _database_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    path_key = str(path)
    if path_key not in _INITIALIZED_PATHS:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(
            """
        CREATE TABLE IF NOT EXISTS voxel_artifact_registry (
            thread_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY (thread_id, artifact_id)
        );
        CREATE TABLE IF NOT EXISTS voxel_data_registry (
            thread_id TEXT NOT NULL,
            data_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY (thread_id, data_id)
        );
        CREATE TABLE IF NOT EXISTS voxel_tool_events (
            thread_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY (thread_id, event_id)
        );
        CREATE TABLE IF NOT EXISTS voxel_registry_seed (
            thread_id TEXT PRIMARY KEY,
            seeded INTEGER NOT NULL DEFAULT 1
        );
            """
        )
        _INITIALIZED_PATHS.add(path_key)
    return connection


@contextmanager
def _connection(
    db_path: Optional[Path | str] = None,
) -> Iterator[sqlite3.Connection]:
    connection = _connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def persist_registry_delta(
    thread_id: str,
    delta: Dict[str, Any],
    *,
    db_path: Optional[Path | str] = None,
) -> None:
    """Persist verified tool state independently of subagent finalization."""

    normalized_thread_id = str(thread_id).strip()
    if not normalized_thread_id:
        return

    artifacts = delta.get("artifact_registry") or {}
    data_records = delta.get("data_registry") or {}
    tool_events = delta.get("tool_events") or []
    with _LOCK, _connection(db_path) as connection:
        for artifact_id, record in artifacts.items():
            if not isinstance(record, dict):
                continue
            if str(record.get("status") or "") != "verified":
                continue
            path_value = record.get("path")
            if not path_value or not Path(str(path_value)).exists():
                continue
            connection.execute(
                """
                INSERT INTO voxel_artifact_registry(thread_id, artifact_id, record_json)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id, artifact_id)
                DO UPDATE SET record_json=excluded.record_json
                """,
                (
                    normalized_thread_id,
                    str(artifact_id),
                    json.dumps(record, sort_keys=True, default=str),
                ),
            )
        for data_id, record in data_records.items():
            if not isinstance(record, dict):
                continue
            connection.execute(
                """
                INSERT INTO voxel_data_registry(thread_id, data_id, record_json)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id, data_id)
                DO UPDATE SET record_json=excluded.record_json
                """,
                (
                    normalized_thread_id,
                    str(data_id),
                    json.dumps(record, sort_keys=True, default=str),
                ),
            )
        for record in tool_events:
            if not isinstance(record, dict):
                continue
            event_id = str(record.get("event_id") or "").strip()
            if not event_id:
                continue
            connection.execute(
                """
                INSERT INTO voxel_tool_events(thread_id, event_id, record_json)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id, event_id)
                DO UPDATE SET record_json=excluded.record_json
                """,
                (
                    normalized_thread_id,
                    event_id,
                    json.dumps(record, sort_keys=True, default=str),
                ),
            )
        connection.execute(
            """
            INSERT INTO voxel_registry_seed(thread_id, seeded) VALUES (?, 1)
            ON CONFLICT(thread_id) DO UPDATE SET seeded=1
            """,
            (normalized_thread_id,),
        )


def load_registry_delta(
    thread_id: str,
    *,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Load durable records and ignore artifacts whose paths no longer exist."""

    normalized_thread_id = str(thread_id).strip()
    empty = {"artifact_registry": {}, "data_registry": {}, "tool_events": []}
    if not normalized_thread_id:
        return empty

    with _LOCK, _connection(db_path) as connection:
        artifact_rows = connection.execute(
            "SELECT artifact_id, record_json FROM voxel_artifact_registry WHERE thread_id=?",
            (normalized_thread_id,),
        ).fetchall()
        data_rows = connection.execute(
            "SELECT data_id, record_json FROM voxel_data_registry WHERE thread_id=?",
            (normalized_thread_id,),
        ).fetchall()
        event_rows = connection.execute(
            "SELECT event_id, record_json FROM voxel_tool_events WHERE thread_id=?",
            (normalized_thread_id,),
        ).fetchall()

    artifacts: Dict[str, Dict[str, Any]] = {}
    for artifact_id, encoded in artifact_rows:
        try:
            record = json.loads(encoded)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or str(record.get("status") or "") != "verified":
            continue
        path_value = record.get("path")
        if not path_value or not Path(str(path_value)).exists():
            continue
        artifacts[str(artifact_id)] = record

    data: Dict[str, Dict[str, Any]] = {}
    for data_id, encoded in data_rows:
        try:
            record = json.loads(encoded)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            data[str(data_id)] = record

    events = []
    for _event_id, encoded in event_rows:
        try:
            record = json.loads(encoded)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            events.append(record)
    return {
        "artifact_registry": artifacts,
        "data_registry": data,
        "tool_events": events,
    }


def registry_seeded(
    thread_id: str,
    *,
    db_path: Optional[Path | str] = None,
) -> bool:
    normalized_thread_id = str(thread_id).strip()
    if not normalized_thread_id:
        return True
    with _LOCK, _connection(db_path) as connection:
        row = connection.execute(
            "SELECT seeded FROM voxel_registry_seed WHERE thread_id=?",
            (normalized_thread_id,),
        ).fetchone()
    return bool(row and row[0])


def mark_registry_seeded(
    thread_id: str,
    *,
    db_path: Optional[Path | str] = None,
) -> None:
    normalized_thread_id = str(thread_id).strip()
    if not normalized_thread_id:
        return
    with _LOCK, _connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO voxel_registry_seed(thread_id, seeded) VALUES (?, 1)
            ON CONFLICT(thread_id) DO UPDATE SET seeded=1
            """,
            (normalized_thread_id,),
        )


def copy_registry_thread(
    source_thread_id: str,
    target_thread_id: str,
    *,
    db_path: Optional[Path | str] = None,
) -> None:
    """Copy durable records when migrating from a legacy session ID."""

    source = str(source_thread_id).strip()
    target = str(target_thread_id).strip()
    if not source or not target or source == target:
        return
    delta = load_registry_delta(source, db_path=db_path)
    if any(delta.values()):
        persist_registry_delta(target, delta, db_path=db_path)
    else:
        mark_registry_seeded(target, db_path=db_path)
