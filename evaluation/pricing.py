from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def load_pricing(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {"version": "unconfigured", "models": {}}
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Pricing file must be a YAML object.")
    payload.setdefault("models", {})
    return payload


def estimate_llm_cost(
    events: Iterable[Dict[str, Any]],
    pricing: Dict[str, Any],
) -> tuple[Optional[float], list[str]]:
    rates_by_model = pricing.get("models") or {}
    total = 0.0
    unpriced: set[str] = set()
    saw_usage = False
    for event in events:
        if event.get("kind") != "llm":
            continue
        usage = event.get("usage") or {}
        total_tokens = int(usage.get("total_tokens") or 0)
        if not total_tokens:
            continue
        saw_usage = True
        model = str(event.get("model") or "unknown")
        rates = rates_by_model.get(model)
        if not isinstance(rates, dict):
            unpriced.add(model)
            continue
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cached_tokens = min(
            input_tokens,
            int(usage.get("cached_input_tokens") or 0),
        )
        cache_write_tokens = min(
            max(0, input_tokens - cached_tokens),
            int(usage.get("cache_write_input_tokens") or 0),
        )
        uncached_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)
        total += uncached_tokens * float(rates.get("input_per_million", 0)) / 1_000_000
        total += cached_tokens * float(
            rates.get("cached_input_per_million", rates.get("input_per_million", 0))
        ) / 1_000_000
        total += cache_write_tokens * float(
            rates.get("cache_write_per_million", rates.get("input_per_million", 0))
        ) / 1_000_000
        total += output_tokens * float(rates.get("output_per_million", 0)) / 1_000_000
    if unpriced or not saw_usage:
        return None, sorted(unpriced)
    return round(total, 8), []

