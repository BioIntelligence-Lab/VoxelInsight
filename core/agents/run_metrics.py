from __future__ import annotations

import contextvars
import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from langchain_core.callbacks import BaseCallbackHandler


_ACTIVE_RECORDER: contextvars.ContextVar[Optional["RunMetricsRecorder"]] = (
    contextvars.ContextVar("voxelinsight_run_metrics", default=None)
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): nested for key, nested in value.items()}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    result: Dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "inputTokens",
        "outputTokens",
        "totalTokens",
    ):
        nested = getattr(value, key, None)
        if nested is not None:
            result[key] = nested
    return result


def _int_value(mapping: Dict[str, Any], *names: str) -> int:
    for name in names:
        value = mapping.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def normalize_usage(value: Any) -> Dict[str, int]:
    raw = _safe_dict(value)
    input_details = _safe_dict(
        raw.get("input_token_details") or raw.get("prompt_tokens_details") or {}
    )
    output_details = _safe_dict(
        raw.get("output_token_details") or raw.get("completion_tokens_details") or {}
    )
    input_tokens = _int_value(raw, "input_tokens", "prompt_tokens", "inputTokens")
    output_tokens = _int_value(
        raw,
        "output_tokens",
        "completion_tokens",
        "outputTokens",
    )
    total_tokens = _int_value(raw, "total_tokens", "totalTokens")
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": _int_value(
            raw,
            "cache_read_input_tokens",
            "cached_tokens",
        )
        or _int_value(input_details, "cache_read", "cached_tokens"),
        "cache_write_input_tokens": _int_value(raw, "cache_creation_input_tokens")
        or _int_value(input_details, "cache_creation", "cache_write"),
        "reasoning_tokens": _int_value(raw, "reasoning_tokens")
        or _int_value(output_details, "reasoning"),
    }


def _usage_from_llm_result(response: Any) -> Dict[str, int]:
    candidates = []
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        candidates.extend(
            [
                llm_output.get("token_usage"),
                llm_output.get("usage"),
            ]
        )
    for generation_group in getattr(response, "generations", None) or []:
        for generation in generation_group or []:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            candidates.append(getattr(message, "usage_metadata", None))
            metadata = getattr(message, "response_metadata", None)
            if isinstance(metadata, dict):
                candidates.extend([metadata.get("token_usage"), metadata.get("usage")])
    for candidate in candidates:
        usage = normalize_usage(candidate)
        if usage["total_tokens"]:
            return usage
    return normalize_usage({})


def _model_from_llm_result(response: Any) -> str:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        model = llm_output.get("model_name") or llm_output.get("model")
        if model:
            return str(model)
    for generation_group in getattr(response, "generations", None) or []:
        for generation in generation_group or []:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "response_metadata", None)
            if isinstance(metadata, dict):
                model = metadata.get("model_name") or metadata.get("model")
                if model:
                    return str(model)
    return ""


def _hash_input(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        encoded = str(value).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


class RunMetricsRecorder(BaseCallbackHandler):
    """Local callback recorder for paper-facing run, model, and tool metrics."""

    def __init__(self) -> None:
        self.started_at = _utc_now()
        self.started_ns = time.perf_counter_ns()
        self.ended_at = ""
        self.ended_ns = 0
        self.events: list[Dict[str, Any]] = []
        self._active: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _start(
        self,
        kind: str,
        run_id: Any,
        parent_run_id: Any,
        **fields: Any,
    ) -> None:
        key = str(run_id or uuid.uuid4())
        now_ns = time.perf_counter_ns()
        record = {
            "event_id": key,
            "parent_event_id": str(parent_run_id or ""),
            "kind": kind,
            "status": "running",
            "started_at": _utc_now(),
            "started_ns": now_ns,
            **fields,
        }
        with self._lock:
            self._active[key] = record

    def _finish(self, run_id: Any, *, status: str, error: str = "", **fields: Any) -> None:
        key = str(run_id or "")
        now_ns = time.perf_counter_ns()
        with self._lock:
            record = self._active.pop(key, None)
            if record is None:
                return
            record.update(
                status=status,
                ended_at=_utc_now(),
                ended_ns=now_ns,
                duration_ms=(now_ns - int(record["started_ns"])) / 1_000_000,
                error=error,
                **fields,
            )
            self.events.append(record)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        params = invocation_params or {}
        model = str(params.get("model") or params.get("model_name") or "")
        provider = str((metadata or {}).get("ls_provider") or "")
        self._start("llm", run_id, parent_run_id, model=model, provider=provider)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        with self._lock:
            if key in self._active:
                return
        self.on_chat_model_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            invocation_params=invocation_params,
            metadata=metadata,
        )

    def on_llm_new_token(self, token: str, *, run_id: Any, **kwargs: Any) -> None:
        key = str(run_id)
        now_ns = time.perf_counter_ns()
        with self._lock:
            record = self._active.get(key)
            if record is not None and "first_token_ns" not in record:
                record["first_token_ns"] = now_ns
                record["time_to_first_token_ms"] = (
                    now_ns - int(record["started_ns"])
                ) / 1_000_000

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        fields: Dict[str, Any] = {"usage": _usage_from_llm_result(response)}
        model = _model_from_llm_result(response)
        if model:
            fields["model"] = model
        self._finish(run_id, status="ok", **fields)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, status="error", error=f"{type(error).__name__}: {error}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:
        name = str(serialized.get("name") or "")
        raw_input = inputs if inputs is not None else input_str
        subagent = ""
        task_input = raw_input
        if name == "task" and isinstance(task_input, str):
            try:
                decoded = json.loads(task_input)
            except (TypeError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, dict):
                task_input = decoded
        if name == "task" and isinstance(task_input, dict):
            subagent = str(
                task_input.get("subagent_type")
                or task_input.get("subagent_name")
                or task_input.get("subagent")
                or ""
            )
        self._start(
            "subagent" if name == "task" else "tool",
            run_id,
            parent_run_id,
            name=name,
            subagent=subagent,
            input_sha256=_hash_input(raw_input),
        )

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, status="ok")

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, status="error", error=f"{type(error).__name__}: {error}")

    def record_sdk_call(
        self,
        *,
        provider: str,
        model: str,
        started_ns: int,
        response: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        ended_ns = time.perf_counter_ns()
        usage_source = getattr(response, "usage", None)
        if usage_source is None and isinstance(response, dict):
            usage_source = response.get("usage")
        record = {
            "event_id": f"sdk-{uuid.uuid4().hex}",
            "parent_event_id": "",
            "kind": "llm",
            "source": "direct_sdk",
            "provider": provider,
            "model": str(getattr(response, "model", None) or model),
            "status": "error" if error else "ok",
            "started_at": "",
            "started_ns": started_ns,
            "ended_at": _utc_now(),
            "ended_ns": ended_ns,
            "duration_ms": (ended_ns - started_ns) / 1_000_000,
            "usage": normalize_usage(usage_source),
            "error": f"{type(error).__name__}: {error}" if error else "",
        }
        with self._lock:
            self.events.append(record)

    def finish(self) -> Dict[str, Any]:
        if not self.ended_ns:
            self.ended_ns = time.perf_counter_ns()
            self.ended_at = _utc_now()
        with self._lock:
            for key, record in list(self._active.items()):
                record.update(
                    status="incomplete",
                    ended_at=self.ended_at,
                    ended_ns=self.ended_ns,
                    duration_ms=(self.ended_ns - int(record["started_ns"])) / 1_000_000,
                    error="Run ended before the callback span completed.",
                )
                self.events.append(record)
                self._active.pop(key, None)
        events = self.snapshot_events()
        llm_events = [event for event in events if event.get("kind") == "llm"]
        tool_events = [event for event in events if event.get("kind") == "tool"]
        subagent_events = [event for event in events if event.get("kind") == "subagent"]
        usage_fields = (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "reasoning_tokens",
        )
        usage = {
            field: sum(int((event.get("usage") or {}).get(field) or 0) for event in llm_events)
            for field in usage_fields
        }
        first_token_times = [
            int(event["first_token_ns"])
            for event in llm_events
            if event.get("first_token_ns") is not None
        ]
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": (self.ended_ns - self.started_ns) / 1_000_000,
            "time_to_first_token_ms": (
                (min(first_token_times) - self.started_ns) / 1_000_000
                if first_token_times
                else None
            ),
            "usage": usage,
            "model_calls": len(llm_events),
            "tool_calls": len(tool_events),
            "subagent_calls": len(subagent_events),
            "errors": sum(
                event.get("status") in {"error", "incomplete"} for event in events
            ),
        }

    def snapshot_events(self) -> list[Dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self.events]


@contextmanager
def metrics_context(recorder: RunMetricsRecorder) -> Iterator[RunMetricsRecorder]:
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


def record_sdk_call(
    *,
    provider: str,
    model: str,
    started_ns: int,
    response: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is not None:
        recorder.record_sdk_call(
            provider=provider,
            model=model,
            started_ns=started_ns,
            response=response,
            error=error,
        )
