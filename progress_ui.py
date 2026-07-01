# progress_ui.py  –  queue-based progress bridge
#
# Tools push (pct, label) events via update_progress().
# The Chainlit app drains the queue and renders the VoxelProgress
# custom element inside a context that has access to cl.context.

import asyncio
import contextvars
import threading
import uuid
from typing import Any, Dict, Optional, Tuple

_progress_token: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "voxelinsight_progress_token",
    default=None,
)
_progress_targets: Dict[
    str,
    Tuple[asyncio.Queue, asyncio.AbstractEventLoop],
] = {}
_progress_lock = threading.Lock()


def set_progress_queue(q: Optional[asyncio.Queue]) -> None:
    """Set or clear the progress target for the current async context."""
    current_token = _progress_token.get()
    with _progress_lock:
        if current_token:
            _progress_targets.pop(current_token, None)
        if q is None:
            _progress_token.set(None)
            return
        token = uuid.uuid4().hex
        _progress_targets[token] = (q, asyncio.get_running_loop())
        _progress_token.set(token)


def _put_latest(q: asyncio.Queue, event: Dict[str, Any]) -> None:
    """Enqueue an event, coalescing stale updates when the queue is bounded."""
    try:
        q.put_nowait(event)
        return
    except asyncio.QueueFull:
        pass

    try:
        q.get_nowait()
    except asyncio.QueueEmpty:
        pass

    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        # Another producer won the race. Its event is at least as recent as the
        # one we were trying to enqueue, so dropping this update is safe.
        pass


async def update_progress(pct: int, label: str, **details: Any) -> None:
    """Push the latest progress event. No-op when no queue is registered.

    ``details`` is intentionally generic so batch tools can expose structured
    counts (for example ``completed``, ``failed``, and ``total``) without each
    tool defining its own UI bridge.
    """
    token = _progress_token.get()
    with _progress_lock:
        target = _progress_targets.get(token or "")
        # A manually-created worker thread does not inherit contextvars. It is
        # safe to use the sole active target, but never guess when runs overlap.
        if target is None and len(_progress_targets) == 1:
            target = next(iter(_progress_targets.values()))
    if target is None:
        return
    q, loop = target

    event: Dict[str, Any] = {
        "pct": int(max(0, min(100, pct))),
        "label": str(label),
    }
    event.update({str(key): value for key, value in details.items() if value is not None})

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is loop:
        _put_latest(q, event)
        return

    try:
        loop.call_soon_threadsafe(_put_latest, q, event)
    except RuntimeError:
        return


async def progress_done() -> None:
    """Convenience shortcut for a 100 % / Done event."""
    await update_progress(100, "Done")
