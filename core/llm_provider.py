import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Settings

@dataclass
class OpenAISettings:
    model: str = "gpt-5"
    temperature: float = 1
    reasoning_effort: str = "medium"
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnthropicSettings:
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.7
    extra: Dict[str, Any] = field(default_factory=dict)

# Main chooser

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
        return ChatAnthropic(
            model=settings.model,
            temperature=settings.temperature,
            **settings.extra,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        settings = openai_settings or OpenAISettings()
        return ChatOpenAI(
            model=settings.model,
            temperature=settings.temperature,
            reasoning_effort=settings.reasoning_effort,
            **settings.extra,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER {provider}. Use 'openai' or 'anthropic'.")
