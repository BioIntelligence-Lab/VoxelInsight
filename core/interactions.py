from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional


@dataclass(frozen=True)
class ConfirmationRequest:
    kind: str
    content: str
    details: Dict[str, Any]


ConfirmHandler = Callable[[ConfirmationRequest], Awaitable[bool]]
NotifyHandler = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class InteractionHandler:
    confirm: ConfirmHandler
    notify: Optional[NotifyHandler] = None


_ACTIVE_HANDLER: contextvars.ContextVar[Optional[InteractionHandler]] = (
    contextvars.ContextVar("voxelinsight_interaction_handler", default=None)
)


@contextmanager
def interaction_context(handler: InteractionHandler) -> Iterator[InteractionHandler]:
    token = _ACTIVE_HANDLER.set(handler)
    try:
        yield handler
    finally:
        _ACTIVE_HANDLER.reset(token)


async def confirm_operation(
    *,
    kind: str,
    content: str,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """Request confirmation; deny safely when no interactive policy is installed."""

    handler = _ACTIVE_HANDLER.get()
    if handler is None:
        return False
    return bool(
        await handler.confirm(
            ConfirmationRequest(
                kind=str(kind),
                content=str(content),
                details=dict(details or {}),
            )
        )
    )


async def notify_user(content: str) -> None:
    handler = _ACTIVE_HANDLER.get()
    if handler is not None and handler.notify is not None:
        await handler.notify(str(content))
