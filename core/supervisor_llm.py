from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except Exception:  
    ChatAnthropic = None

try:
    from langchain_aws import ChatBedrockConverse
except Exception:  
    ChatBedrockConverse = None


@dataclass
class OpenAISupervisorSettings:
    model: str = "gpt-5-nano"
    temperature: float = 1.0
    reasoning_effort: Optional[str] = "low"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnthropicSupervisorSettings:
    model: str = "claude-3-5-haiku-latest"
    temperature: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BedrockSupervisorSettings:
    model: str = "anthropic.claude-3-5-haiku-20241022-v1:0"
    temperature: float = 1.0
    region_name: Optional[str] = None
    credentials_profile_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def build_supervisor_llm(
    *,
    temperature: float = 1.0,
    reasoning_effort: str = "low",
    model_override: Optional[str] = None,
    openai_settings: Optional[OpenAISupervisorSettings] = None,
    anthropic_settings: Optional[AnthropicSupervisorSettings] = None,
    bedrock_settings: Optional[BedrockSupervisorSettings] = None,
) -> BaseChatModel:
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai is required when LLM_PROVIDER=openai. Install langchain-openai."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        settings = openai_settings or OpenAISupervisorSettings()
        kwargs: Dict[str, Any] = {
            "model": model_override or os.getenv("SUPERVISOR_LLM_MODEL") or settings.model,
            "temperature": temperature if temperature is not None else settings.temperature,
            "reasoning_effort": reasoning_effort if reasoning_effort is not None else settings.reasoning_effort,
        }
        if settings.extra:
            kwargs.update(settings.extra)
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError(
                "langchain-anthropic is required when LLM_PROVIDER=anthropic. Install langchain-anthropic."
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")

        settings = anthropic_settings or AnthropicSupervisorSettings()
        kwargs = {
            "model": model_override or os.getenv("SUPERVISOR_LLM_MODEL") or settings.model,
            "temperature": temperature if temperature is not None else settings.temperature,
            "max_tokens": settings.max_tokens,
        }
        if settings.extra:
            kwargs.update(settings.extra)
        return ChatAnthropic(**kwargs)

    if provider == "bedrock":
        if ChatBedrockConverse is None:
            raise RuntimeError(
                "langchain-aws is required when LLM_PROVIDER=bedrock. Install langchain-aws."
            )

        settings = bedrock_settings or BedrockSupervisorSettings()
        region_name = (
            settings.region_name
            or os.getenv("BEDROCK_AWS_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )
        kwargs = {
            "model": model_override or os.getenv("SUPERVISOR_LLM_MODEL") or settings.model,
            "temperature": temperature if temperature is not None else settings.temperature,
            "max_tokens": settings.max_tokens,
        }
        if region_name:
            kwargs["region_name"] = region_name
        if settings.credentials_profile_name:
            kwargs["credentials_profile_name"] = settings.credentials_profile_name
        if settings.extra:
            kwargs.update(settings.extra)
        return ChatBedrockConverse(**kwargs)

    raise ValueError(f"Unknown LLM_PROVIDER {provider}. Use 'openai', 'anthropic', or 'bedrock'.")
