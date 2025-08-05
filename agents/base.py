from typing import Protocol
from core.state import Task, TaskResult, ConversationState

class Agent(Protocol):
    name: str
    model: str | None

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        ...
