# agents/router.py
import json
import os
from openai import AsyncOpenAI
from core.state import Task, TaskResult, ConversationState


class RouterAgent:
    name = "router"
    model = "gpt-4o"

    def __init__(self, available_agents, system_prompt: str):
        key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
        self.available_agents = set(available_agents or [])
        self.system_prompt = system_prompt

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"User message: {task.user_msg}\n"
                    f"files={task.files}\n"
                    f"Available agents: {sorted(self.available_agents)}"
                ),
            },
        ]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        try:
            route = json.loads(raw)
        except Exception:
            return TaskResult(output={"error": "Router returned non-JSON", "raw": raw}, next_tasks=[])

        task.intent = route.get("intent", task.intent)

        next_tasks: list[Task] = []

        if isinstance(route.get("plan"), list):
            top_level_kwargs = route.get("kwargs", {}) or {}
            for i, step in enumerate(route["plan"]):
                agent_nm = step.get("agent")
                if not agent_nm or (self.available_agents and agent_nm not in self.available_agents):
                    continue
                step_id = step.get("id") or f"{agent_nm}_{i+1}"
                step_kwargs = {**top_level_kwargs, **(step.get("kwargs") or {})}
                next_tasks.append(
                    Task(
                        user_msg=task.user_msg,
                        files=task.files,
                        intent=step_id,
                        kwargs={**step_kwargs, "agent": agent_nm},
                    )
                )

        return TaskResult(output=route, next_tasks=next_tasks)
