from __future__ import annotations

import os
import uuid
import contextlib
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

PERSIST_ENV = "VOXELINSIGHT_PERSIST_ROOT"
TEMP_ENV = "VOXELINSIGHT_TEMP_ROOT"

DEFAULT_PERSIST = Path("perm_files").absolute()
DEFAULT_TEMP = Path("tmp").absolute()


def _get_root(env_var: str, default: Path) -> Path:
    val = os.getenv(env_var)
    root = Path(val).expanduser().absolute() if val else default
    root.mkdir(parents=True, exist_ok=True)
    return root


def persist_root() -> Path:
    """user-facing or durable artifacts."""
    return _get_root(PERSIST_ENV, DEFAULT_PERSIST)


def temp_root() -> Path:
    """Temporary scratch work."""
    return _get_root(TEMP_ENV, DEFAULT_TEMP)


def _run_id(prefix: Optional[str] = None) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}-{stamp}-{short}" if prefix else f"{stamp}-{short}"


def get_run_dir(
    tool_name: str,
    *,
    persist: bool = True,
    prefix: Optional[str] = None,
    root: Optional[Path] = None,
) -> Path:
    base = root or (persist_root() if persist else temp_root())
    run_dir = base / tool_name / _run_id(prefix)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_temp_dir(prefix: str = "tmp", root: Optional[Path] = None) -> Path:
    """Allocate a temp dir under temp_root (not auto-cleaned here)."""
    base = root or temp_root()
    d = base / prefix / _run_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_temp_file(
    *,
    suffix: str = "",
    prefix: str = "tmp",
    root: Optional[Path] = None,
) -> Path:
    """Allocate a temp file path (parent created)"""
    dir_path = get_temp_dir(prefix=prefix, root=root)
    return dir_path / f"file{suffix}"


@contextmanager
def temp_dir(prefix: str = "tmp") -> Iterator[Path]:
    d = get_temp_dir(prefix=prefix)
    try:
        yield d
    finally:
        with contextlib.suppress(Exception):
            if not d.exists():
                return
            try:
                next(d.iterdir())  
                return
            except StopIteration:
                pass
            d.rmdir()


def cleanup(root: Path, *, older_than: timedelta) -> int:
    """Cleanup of directories older than the given age."""
    now = datetime.utcnow()
    removed = 0
    for path in root.glob("*"):
        if not path.is_dir():
            continue
        try:
            mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
            if now - mtime > older_than:
                for child in path.rglob("*"):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                for child_dir in sorted(
                    (p for p in path.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
                ):
                    child_dir.rmdir()
                path.rmdir()
                removed += 1
        except Exception:
            continue
    return removed
