from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

@dataclass
class Task:
    user_msg: str
    files: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    output: Any
    artifacts: Dict[str, Any] = field(default_factory=dict)
    next_tasks: List["Task"] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    status: Optional[Literal["ok", "partial", "error", "no_action"]] = None
    errors: List[str] = field(default_factory=list)

@dataclass
class ConversationState:
    memory: Dict[str, Any] = field(default_factory=dict)
