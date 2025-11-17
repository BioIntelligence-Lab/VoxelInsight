import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Any

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


@dataclass
class OpenAISettings:
    model: str = "gpt-5"
    temperature: float = 1
    reasoning_effort: Optional[str] = "medium"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnthropicSettings:
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.7
    max_tokens: int = 1024
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(
        self,
        *,
        provider: str,
        client: Union[AsyncOpenAI, AsyncAnthropic],
        settings: Union[OpenAISettings, AnthropicSettings],
    ):
        self.provider = provider
        self.client = client
        self.settings = settings

    async def ainvoke(
        self,
        messages: List[Union[Dict[str, str], Any]],
        *,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        normalized = _normalize_messages(messages)
        target_temp = temperature if temperature is not None else getattr(self.settings, "temperature", 1)
        target_reasoning = reasoning_effort if reasoning_effort is not None else getattr(self.settings, "reasoning_effort", None)

        if self.provider == "openai":
            payload = {
                "model": getattr(self.settings, "model", "gpt-5"),
                "temperature": target_temp,
                "messages": normalized,
            }
            if target_reasoning is not None:
                payload["reasoning_effort"] = target_reasoning
            extra = getattr(self.settings, "extra", None)
            if extra:
                payload.update(extra)
            response = await self.client.chat.completions.create(**payload)
            return response.choices[0].message.content

        if self.provider == "anthropic":
            system_prompt = next((m["content"] for m in normalized if m["role"] == "system"), None)
            user_parts = [m["content"] for m in normalized if m["role"] == "user"]
            user_content = "\n\n".join(user_parts)
            kwargs = {
                "model": getattr(self.settings, "model", "claude-sonnet-4-5"),
                "max_tokens": getattr(self.settings, "max_tokens", 1024),
                "temperature": target_temp,
                "messages": [{"role": "user", "content": user_content}],
            }
            extra = getattr(self.settings, "extra", None)
            if extra:
                kwargs.update(extra)
            if system_prompt:
                kwargs["system"] = system_prompt
            response = await self.client.messages.create(**kwargs)
            parts = getattr(response, "content", None)
            if not parts:
                return None
            return "".join(getattr(part, "text", "") for part in parts if getattr(part, "text", ""))

        raise ValueError(f"Unsupported provider {self.provider}")


def _normalize_messages(messages: List[Union[Dict[str, str], Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", None) or getattr(msg, "type", None)
            content = getattr(msg, "content", "")
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        normalized.append({"role": role, "content": content})
    return normalized


def choose_llm(
    openai_settings: Optional[OpenAISettings] = None,
    anthropic_settings: Optional[AnthropicSettings] = None,
):
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")

        settings = anthropic_settings or AnthropicSettings()
        client = AsyncAnthropic(api_key=api_key)
        return LLMClient(
            provider=provider,
            client=client,
            settings=settings,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        settings = openai_settings or OpenAISettings()
        client = AsyncOpenAI(api_key=api_key)
        return LLMClient(
            provider=provider,
            client=client,
            settings=settings,
        )

    raise ValueError(f"Unknown LLM_PROVIDER {provider}. Use 'openai' or 'anthropic'.")
