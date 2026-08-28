from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from core.agents.artifacts import VoxelAgentState, json_safe
from core.agents.durable_registry import load_registry_delta
from core.agents.schemas import VerifierResult


_VERIFIER_RESULT_TOOL = "VerifierResult"
_AUTO_REMEDIATE_ENV = "VOXELINSIGHT_AUTO_REMEDIATE"
_VERIFICATION_CYCLE_SCHEMA = "voxelinsight.verification-cycle.v1"
_DOMAIN_SUBAGENTS = frozenset(
    {
        "idc-agent",
        "cohort-agent",
        "acquisition-agent",
        "segmentation-agent",
        "analysis-agent",
    }
)
_LOCAL_PATH_PATTERN = re.compile(
    r"(?:^|\s)(?:/(?:Users|home|tmp|private|var|Volumes)/|~/|file://|[A-Za-z]:\\\\)",
    re.IGNORECASE,
)
_DESTRUCTIVE_PATTERN = re.compile(
    r"\b(?:delete|erase|overwrite|unlink|drop\s+table|rm\s+-rf)\b", re.IGNORECASE
)
_TRANSFER_PATTERN = re.compile(
    r"\b(?:download|acquire|transfer|fetch\s+(?:the\s+)?files?|retrieve\s+imaging)\b",
    re.IGNORECASE,
)
_AUTHORITY_PATTERN = re.compile(
    r"(?:\b(?:ask|confirm|obtain|request)\b.{0,40}\b(?:user|permission|approval|consent)\b|"
    r"\buser\b.{0,30}\b(?:choose|decide|select)\b)",
    re.IGNORECASE,
)
_NEGATED_ACTION_PATTERN = re.compile(
    r"(?:do\s+not|don't|never|without)\s+(?:[\w-]+\s+){0,4}$|"
    r"\bno\s+(?:[\w-]+\s+){0,3}(?:(?:or|and)\s+)?$",
    re.IGNORECASE,
)
_MAX_RECORDS = 80
_MAX_DATA_ROWS = 20
_MAX_COLUMNS = 80
_MAX_STRING_CHARS = 500
_EXECUTION_AUDIT_ID = "execution-audit-current-run"
_TOOL_EFFECT_CLASSES = {
    "idc_download": "imaging_download",
    "tcia_download": "imaging_download",
    "midrc_download": "imaging_download",
    "clinical_data_download": "clinical_metadata_download",
    "dicom2nifti_batch": "imaging_conversion",
}
_TRACKED_EFFECT_CLASSES = frozenset(_TOOL_EFFECT_CLASSES.values())
_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def auto_remediation_enabled() -> bool:
    """Return whether the bounded automatic repair cycle is enabled."""

    return str(os.getenv(_AUTO_REMEDIATE_ENV, "false")).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _contains_requested_action(text: str, pattern: re.Pattern[str]) -> bool:
    """Return true when an action keyword is not locally negated."""

    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 80) : match.start()]
        if _NEGATED_ACTION_PATTERN.search(prefix):
            continue
        return True
    return False


def evaluate_remediation_gate(
    state: dict[str, Any],
    verifier_payload: dict[str, Any],
    *,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Deterministically decide whether exactly one verifier repair may run.

    This gate is intentionally conservative. A rejected repair is reported honestly to
    the user; it is never broadened or rewritten by the supervisor.
    """

    reasons: list[str] = []
    is_enabled = auto_remediation_enabled() if enabled is None else bool(enabled)
    if verifier_payload.get("verdict") != "needs_remediation":
        return {
            "approved": False,
            "reasons": ["verdict is not needs_remediation"],
            "target_subagent": "",
            "remediation": {},
        }
    if not is_enabled:
        reasons.append("automatic remediation is disabled")
    if verifier_payload.get("allow_final") is not False:
        reasons.append("needs_remediation did not set allow_final=false")

    cycle = state.get("verification_cycle") or {}
    if cycle.get("repair_attempts") or cycle.get("phase") in {
        "repair_approved",
        "repair_attempted",
        "complete",
        "remediation_rejected",
    }:
        reasons.append("a remediation decision or attempt already exists for this run")

    remediations = verifier_payload.get("remediations") or []
    if len(remediations) != 1 or not isinstance(remediations[0], dict):
        reasons.append("exactly one structured remediation is required")
        remediation: dict[str, Any] = {}
    else:
        remediation = dict(remediations[0])

    target = str(remediation.get("target_subagent") or "")
    instructions = str(remediation.get("instructions") or "").strip()
    artifact_ids = [str(value) for value in remediation.get("input_artifact_ids") or []]
    data_ids = [str(value) for value in remediation.get("input_data_ids") or []]
    input_ids = [*artifact_ids, *data_ids]
    valid_ids = valid_verifier_evidence_ids(state)

    if target not in _DOMAIN_SUBAGENTS:
        reasons.append("remediation target is not an existing domain subagent")
    if target == "acquisition-agent":
        reasons.append("acquisition and transfer work is never automatically retried")
    if remediation.get("safe_to_retry") is not True:
        reasons.append("verifier did not mark the remediation safe_to_retry")
    if not instructions:
        reasons.append("remediation instructions are empty")
    elif len(instructions) > 1500:
        reasons.append("remediation instructions exceed the bounded size limit")
    if _LOCAL_PATH_PATTERN.search(instructions):
        reasons.append("remediation instructions contain a local filesystem path")
    if _contains_requested_action(instructions, _DESTRUCTIVE_PATTERN):
        reasons.append("remediation instructions request a destructive operation")
    if _contains_requested_action(instructions, _TRANSFER_PATTERN):
        reasons.append("remediation instructions request acquisition or transfer")
    if _contains_requested_action(instructions, _AUTHORITY_PATTERN):
        reasons.append("remediation requires a user choice, permission, or approval")
    named_other_agents = sorted(
        agent for agent in _DOMAIN_SUBAGENTS if agent != target and agent in instructions
    )
    if named_other_agents:
        reasons.append(
            "remediation requires additional domain agents: "
            + ", ".join(named_other_agents)
        )
    if not input_ids:
        reasons.append("remediation has no exact registered input reference")
    unknown_ids = sorted({value for value in input_ids if value not in valid_ids})
    if unknown_ids:
        reasons.append("remediation references unknown input IDs: " + ", ".join(unknown_ids))

    return {
        "approved": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "target_subagent": target,
        "remediation": remediation,
    }


def _bounded_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Make registry metadata safe and bounded for an evidence-only model context."""

    if any(marker in key.strip().lower() for marker in _SENSITIVE_KEY_MARKERS):
        return "<redacted>"
    if depth >= 4 and isinstance(value, (dict, list, tuple, set)):
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        try:
            if Path(value).expanduser().is_absolute():
                return "<local-path>"
        except (OSError, ValueError):
            pass
        if len(value) > _MAX_STRING_CHARS:
            return value[: _MAX_STRING_CHARS - 1] + "…"
        return value
    if isinstance(value, dict):
        items = list(value.items())
        result = {
            str(item_key): _bounded_value(
                nested,
                key=str(item_key),
                depth=depth + 1,
            )
            for item_key, nested in items[:40]
            if str(item_key).lower() not in {"path", "source_path"}
        }
        if len(items) > 40:
            result["_truncated_items"] = len(items) - 40
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        result = [
            _bounded_value(item, key=key, depth=depth + 1) for item in items[:40]
        ]
        if len(items) > 40:
            result.append(f"<truncated {len(items) - 40} item(s)>")
        return result
    return _bounded_value(json_safe(value), depth=depth)


def _all_current_run_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    run_id = str(state.get("current_run_id") or "")
    if not run_id:
        return []
    return [
        dict(event)
        for event in (state.get("tool_events") or [])
        if isinstance(event, dict)
        and str(event.get("run_id") or "") == run_id
    ]


def _current_run_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    return _all_current_run_events(state)[-_MAX_RECORDS:]


def _execution_audit(state: dict[str, Any]) -> dict[str, Any]:
    """Summarize the complete platform-mediated operational trace for this run."""

    run_id = str(state.get("current_run_id") or "")
    all_events = _all_current_run_events(state)
    observed_tools = sorted(
        {
            str(event.get("tool_name") or "")
            for event in all_events
            if event.get("tool_name")
        }
    )
    observed_effects = sorted(
        {
            _TOOL_EFFECT_CLASSES[tool_name]
            for tool_name in observed_tools
            if tool_name in _TOOL_EFFECT_CLASSES
        }
    )
    trace_available = isinstance(state.get("tool_events"), list)
    return {
        "evidence_id": _EXECUTION_AUDIT_ID,
        "scope": "platform-mediated operational tool calls in the current run",
        "trace_complete": bool(run_id) and trace_available,
        "operational_event_count": len(all_events),
        "snapshot_event_count": min(len(all_events), _MAX_RECORDS),
        "snapshot_truncated": len(all_events) > _MAX_RECORDS,
        "observed_tool_names": observed_tools,
        "observed_effect_classes": observed_effects,
        "unobserved_effect_classes": sorted(
            _TRACKED_EFFECT_CLASSES - set(observed_effects)
        ),
        "limitations": (
            "This audit supports claims only about actions mediated by registered "
            "VoxelInsight operational tools; it does not audit activity outside the platform."
        ),
    }


def _event_reference_ids(events: Iterable[dict[str, Any]]) -> tuple[set[str], set[str]]:
    artifact_ids: set[str] = set()
    data_ids: set[str] = set()
    for event in events:
        artifact_ids.update(str(value) for value in event.get("input_artifact_ids") or [])
        artifact_ids.update(str(value) for value in event.get("artifact_ids") or [])
        data_ids.update(str(value) for value in event.get("input_data_ids") or [])
        data_ids.update(str(value) for value in event.get("data_ids") or [])
    return artifact_ids, data_ids


def _artifact_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    raw_path = record.get("path")
    path_exists = False
    if raw_path:
        try:
            path_exists = Path(str(raw_path)).expanduser().exists()
        except OSError:
            path_exists = False
    return {
        "artifact_id": str(record.get("artifact_id") or ""),
        "kind": str(record.get("kind") or ""),
        "role": str(record.get("role") or ""),
        "name": str(record.get("name") or ""),
        "status": str(record.get("status") or ""),
        "source_tool": str(record.get("source_tool") or ""),
        "source_call_id": str(record.get("source_call_id") or ""),
        "run_id": str(record.get("run_id") or ""),
        "size_bytes": record.get("size_bytes"),
        "downloadable": bool(record.get("downloadable", True)),
        "path_exists": path_exists,
        "metadata": _bounded_value(record.get("metadata") or {}),
    }


def _data_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("rows") or []
    return {
        "data_id": str(record.get("data_id") or ""),
        "kind": str(record.get("kind") or ""),
        "visibility": str(record.get("visibility") or ""),
        "name": str(record.get("name") or ""),
        "source_tool": str(record.get("source_tool") or ""),
        "source_call_id": str(record.get("source_call_id") or ""),
        "run_id": str(record.get("run_id") or ""),
        "nrows": int(record.get("nrows") or 0),
        "complete": bool(record.get("complete", False)),
        "artifact_id": str(record.get("artifact_id") or ""),
        "columns": [str(column) for column in (record.get("columns") or [])][
            :_MAX_COLUMNS
        ],
        "rows": _bounded_value(list(rows)[:_MAX_DATA_ROWS]),
        "preview_row_count": min(len(rows), _MAX_DATA_ROWS),
        "metadata": _bounded_value(record.get("metadata") or {}),
    }


def build_verification_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Build the authoritative, current-run evidence supplied to verifier-agent."""

    run_id = str(state.get("current_run_id") or "")
    events = _current_run_events(state)
    execution_audit = _execution_audit(state)
    referenced_artifact_ids, referenced_data_ids = _event_reference_ids(events)
    artifacts = state.get("artifact_registry") or {}
    data = state.get("data_registry") or {}

    # Current uploads are valid evidence even before an operational tool references them.
    for artifact_id, record in artifacts.items():
        if not isinstance(record, dict):
            continue
        if (
            str(record.get("run_id") or "") == run_id
            and str(record.get("role") or "") == "input"
        ):
            referenced_artifact_ids.add(str(artifact_id))

    selected_artifacts = [
        _artifact_snapshot(record)
        for artifact_id, record in artifacts.items()
        if isinstance(record, dict)
        and (
            str(artifact_id) in referenced_artifact_ids
            or (run_id and str(record.get("run_id") or "") == run_id)
        )
    ][-_MAX_RECORDS:]
    selected_data = [
        _data_snapshot(record)
        for data_id, record in data.items()
        if isinstance(record, dict)
        and (
            str(data_id) in referenced_data_ids
            or (run_id and str(record.get("run_id") or "") == run_id)
        )
    ][-_MAX_RECORDS:]

    issues: list[str] = []
    if not run_id:
        issues.append("The current run has no run_id; execution evidence cannot be scoped safely.")
    for event in events:
        event_id = str(event.get("event_id") or "tool event")
        status = str(event.get("status") or "")
        if status in {"partial", "error", "no_action"}:
            issues.append(f"{event_id} has terminal status {status}.")
        for artifact_id in event.get("artifact_ids") or []:
            record = artifacts.get(str(artifact_id))
            if not isinstance(record, dict):
                issues.append(f"{event_id} references unknown output artifact {artifact_id}.")
            else:
                path_value = record.get("path")
                try:
                    path_exists = bool(path_value) and Path(
                        str(path_value)
                    ).expanduser().exists()
                except OSError:
                    path_exists = False
                if not path_exists:
                    issues.append(
                        f"{event_id} output artifact {artifact_id} is missing on disk."
                    )
        for data_id in event.get("data_ids") or []:
            if not isinstance(data.get(str(data_id)), dict):
                issues.append(f"{event_id} references unknown output data {data_id}.")
    for record in selected_data:
        if not record["complete"]:
            issues.append(
                f"{record['data_id']} contains a bounded preview rather than every table row."
            )

    event_snapshots = []
    for event in events:
        event_snapshots.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "tool_name": str(event.get("tool_name") or ""),
                "tool_call_id": str(event.get("tool_call_id") or ""),
                "run_id": str(event.get("run_id") or ""),
                "status": str(event.get("status") or ""),
                "input_artifact_ids": list(event.get("input_artifact_ids") or []),
                "input_data_ids": list(event.get("input_data_ids") or []),
                "arguments_summary": _bounded_value(
                    event.get("arguments_summary") or {}
                ),
                "artifact_ids": list(event.get("artifact_ids") or []),
                "data_ids": list(event.get("data_ids") or []),
                "outputs_summary": _bounded_value(event.get("outputs_summary") or {}),
                "errors": [str(error) for error in (event.get("errors") or [])],
            }
        )

    # A dangling registry reference is a deterministic issue, not valid evidence.
    # The corresponding tool event can still be cited to support that failure.
    evidence_ids = {
        *(str(event.get("event_id") or "") for event in events),
        *(str(record.get("artifact_id") or "") for record in selected_artifacts),
        *(str(record.get("data_id") or "") for record in selected_data),
    }
    if execution_audit["trace_complete"]:
        evidence_ids.add(str(execution_audit["evidence_id"]))
    evidence_ids.discard("")
    return {
        "schema_version": "voxelinsight.verification-context.v2",
        "current_run_id": run_id,
        "current_user_request": str(state.get("current_user_request") or ""),
        "requested_deliverables_hint": _bounded_value(
            state.get("requested_deliverables") or []
        ),
        "tool_events": event_snapshots,
        "artifacts": selected_artifacts,
        "data": selected_data,
        "execution_audit": execution_audit,
        "deterministic_issues": list(dict.fromkeys(issues)),
        "valid_evidence_ids": sorted(evidence_ids),
    }


def verification_context_text(state: dict[str, Any]) -> str:
    """Render the production verifier's authoritative bounded context block."""

    snapshot = build_verification_snapshot(state)
    return (
        "\n\n<voxelinsight_verification_json>\n"
        + json.dumps(snapshot, indent=2, sort_keys=True, default=str)
        + "\n</voxelinsight_verification_json>\n"
        "This verification block is authoritative. Cite only its valid_evidence_ids. "
        "A complete execution_audit may support a negative platform-action claim only "
        "within its stated scope. Other absence of evidence means unverifiable or missing, "
        "never successful."
    )


def valid_verifier_evidence_ids(state: dict[str, Any]) -> set[str]:
    snapshot = build_verification_snapshot(state)
    return {str(value) for value in snapshot.get("valid_evidence_ids") or []}


def _response_messages(response: ModelResponse | AIMessage) -> Iterable[BaseMessage]:
    if isinstance(response, AIMessage):
        return [response]
    return response.result


def _verifier_calls(
    response: ModelResponse | AIMessage,
) -> list[tuple[AIMessage, int, dict[str, Any]]]:
    calls: list[tuple[AIMessage, int, dict[str, Any]]] = []
    for message in _response_messages(response):
        if not isinstance(message, AIMessage):
            continue
        for index, call in enumerate(message.tool_calls or []):
            name = str(call.get("name") or "").rsplit(".", 1)[-1]
            if name == _VERIFIER_RESULT_TOOL:
                calls.append((message, index, call))
    return calls


def _negative_action_issue(
    *,
    text: str,
    status: str,
    evidence_ids: list[str],
    snapshot: dict[str, Any],
) -> str:
    """Validate supported no-download assertions against the scoped execution audit."""

    if status not in {"satisfied", "supported"}:
        return ""
    normalized = " ".join(str(text).casefold().split())
    no_imaging_download = (
        "no imaging" in normalized and "download" in normalized
    ) or (
        "without downloading" in normalized and "imaging" in normalized
    ) or "did not download imaging" in normalized
    if not no_imaging_download:
        return ""

    audit = snapshot.get("execution_audit") or {}
    audit_id = str(audit.get("evidence_id") or "")
    if audit_id not in evidence_ids:
        return "supported no-imaging-download assertion lacked execution-audit evidence"
    if not audit.get("trace_complete"):
        return "supported no-imaging-download assertion used an incomplete execution audit"
    if "imaging_download" not in set(audit.get("unobserved_effect_classes") or []):
        return "supported no-imaging-download assertion conflicts with the execution audit"
    return ""


def verifier_payload_evidence_issues(
    state: dict[str, Any], payload: dict[str, Any]
) -> list[str]:
    """Return evidence and verdict consistency issues for one verifier payload."""

    snapshot = build_verification_snapshot(state)
    valid_ids = {str(value) for value in snapshot.get("valid_evidence_ids") or []}
    issues: list[str] = []
    obligations = payload.get("obligations") or []
    if payload.get("verdict") == "pass" and not obligations:
        issues.append("pass verdict contained no independently assessed obligations")
    for obligation in obligations:
        if not isinstance(obligation, dict):
            issues.append("obligation entry was not an object")
            continue
        evidence = [str(value) for value in obligation.get("evidence_ids") or []]
        unknown = [value for value in evidence if value not in valid_ids]
        if unknown:
            issues.append(
                f"obligation {obligation.get('obligation_id') or '[unnamed]'} used "
                f"unknown evidence: {', '.join(unknown)}"
            )
        if obligation.get("status") == "satisfied" and not evidence:
            issues.append(
                f"satisfied obligation {obligation.get('obligation_id') or '[unnamed]'} "
                "had no deterministic evidence"
            )
        negative_issue = _negative_action_issue(
            text=str(obligation.get("description") or ""),
            status=str(obligation.get("status") or ""),
            evidence_ids=evidence,
            snapshot=snapshot,
        )
        if negative_issue:
            issues.append(negative_issue)
    for claim in payload.get("claim_checks") or []:
        if not isinstance(claim, dict):
            continue
        evidence = [str(value) for value in claim.get("evidence_ids") or []]
        unknown = [value for value in evidence if value not in valid_ids]
        if unknown:
            issues.append(f"claim check used unknown evidence: {', '.join(unknown)}")
        if claim.get("status") == "supported" and not evidence:
            issues.append("supported claim had no deterministic evidence")
        negative_issue = _negative_action_issue(
            text=str(claim.get("claim") or ""),
            status=str(claim.get("status") or ""),
            evidence_ids=evidence,
            snapshot=snapshot,
        )
        if negative_issue:
            issues.append(negative_issue)
    for remediation in payload.get("remediations") or []:
        if not isinstance(remediation, dict):
            continue
        references = [
            *[str(value) for value in remediation.get("input_artifact_ids") or []],
            *[str(value) for value in remediation.get("input_data_ids") or []],
        ]
        unknown = [value for value in references if value not in valid_ids]
        if unknown:
            issues.append(f"remediation used unknown inputs: {', '.join(unknown)}")
    if payload.get("verdict") == "pass":
        incomplete = [
            obligation
            for obligation in obligations
            if isinstance(obligation, dict)
            and obligation.get("status") not in {"satisfied", "not_required"}
        ]
        if incomplete:
            issues.append("pass verdict contained incomplete obligations")
        if payload.get("allow_final") is not True:
            issues.append("pass verdict set allow_final=false")
    if payload.get("verdict") == "needs_remediation" and payload.get("allow_final"):
        issues.append("needs_remediation verdict set allow_final=true")
    if payload.get("verdict") == "needs_remediation" and not payload.get("remediations"):
        issues.append("needs_remediation verdict contained no remediation instructions")
    return list(dict.fromkeys(issues))


def verifier_evidence_issues(
    state: dict[str, Any],
    response: ModelResponse | AIMessage,
) -> list[str]:
    """Return invalid or internally inconsistent verifier assertions."""

    issues: list[str] = []
    for _message, _index, call in _verifier_calls(response):
        payload = call.get("args") or {}
        if not isinstance(payload, dict):
            issues.append("VerifierResult arguments were not an object")
            continue
        issues.extend(verifier_payload_evidence_issues(state, payload))
    return list(dict.fromkeys(issues))


def _verification_error_response(
    response: ModelResponse | AIMessage,
    issues: list[str],
) -> ModelResponse | AIMessage:
    limitation = "Verifier evidence validation failed: " + "; ".join(issues)
    fallback = VerifierResult(
        verdict="verification_error",
        allow_final=True,
        obligations=[],
        claim_checks=[],
        completed_summary=[],
        incomplete_summary=[],
        limitations=[limitation],
        remediations=[],
    ).model_dump()

    def replace_message(message: BaseMessage) -> BaseMessage:
        if not isinstance(message, AIMessage):
            return message
        calls = [dict(call) for call in (message.tool_calls or [])]
        changed = False
        for index, call in enumerate(calls):
            name = str(call.get("name") or "").rsplit(".", 1)[-1]
            if name != _VERIFIER_RESULT_TOOL:
                continue
            call["args"] = dict(fallback)
            calls[index] = call
            changed = True
        return message.model_copy(update={"tool_calls": calls}) if changed else message

    if isinstance(response, AIMessage):
        return replace_message(response)  # type: ignore[return-value]
    return response.model_copy(
        update={"result": [replace_message(message) for message in response.result]}
    )


class VerificationContextMiddleware(AgentMiddleware):
    """Inject a bounded, read-only evidence snapshot only into verifier-agent."""

    state_schema = VoxelAgentState

    @staticmethod
    def _with_durable_evidence(request: ModelRequest) -> ModelRequest:
        runtime = getattr(request, "runtime", None)
        runtime_config = getattr(runtime, "config", {}) or {}
        configurable = runtime_config.get("configurable", {}) or {}
        thread_id = str(configurable.get("thread_id") or "").strip()
        if not thread_id:
            return request
        try:
            durable = load_registry_delta(thread_id)
        except Exception:
            return request

        state = dict(request.state or {})
        artifacts = dict(durable.get("artifact_registry") or {})
        artifacts.update(state.get("artifact_registry") or {})
        data = dict(durable.get("data_registry") or {})
        data.update(state.get("data_registry") or {})
        events_by_id: dict[str, dict[str, Any]] = {}
        for event in [
            *(durable.get("tool_events") or []),
            *(state.get("tool_events") or []),
        ]:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "")
            if event_id:
                events_by_id[event_id] = event
        state.update(
            artifact_registry=artifacts,
            data_registry=data,
            tool_events=list(events_by_id.values()),
        )
        return request.override(state=state)

    @classmethod
    def _with_context(cls, request: ModelRequest) -> ModelRequest:
        request = cls._with_durable_evidence(request)
        existing = request.system_message.text if request.system_message is not None else ""
        context = verification_context_text(dict(request.state or {}))
        return request.override(system_message=SystemMessage(content=existing + context))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        return handler(self._with_context(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        return await handler(self._with_context(request))


class VerifierEvidenceMiddleware(AgentMiddleware):
    """Downgrade invented verifier evidence without retries or workflow blocking."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        response = handler(request)
        issues = verifier_evidence_issues(dict(request.state or {}), response)
        return _verification_error_response(response, issues) if issues else response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        response = await handler(request)
        issues = verifier_evidence_issues(dict(request.state or {}), response)
        return _verification_error_response(response, issues) if issues else response


def _middleware_tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        name = tool.get("name")
        function = tool.get("function")
        if not name and isinstance(function, dict):
            name = function.get("name")
        return str(name or "")
    return str(getattr(tool, "name", "") or "")


def _task_target(request: ToolCallRequest) -> str:
    if str(request.tool_call.get("name") or "") != "task":
        return ""
    args = request.tool_call.get("args") or {}
    return str(args.get("subagent_type") or "") if isinstance(args, dict) else ""


def _tool_message_from_result(
    result: ToolMessage | Command[Any],
) -> ToolMessage | None:
    if isinstance(result, ToolMessage):
        return result
    update = result.update
    if not isinstance(update, dict):
        return None
    messages = update.get("messages") or []
    if isinstance(messages, BaseMessage):
        messages = [messages]
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return message
    return None


def _tool_message_payload(message: ToolMessage | None) -> dict[str, Any]:
    if message is None:
        return {}
    content: Any = message.content
    if not isinstance(content, str):
        content = message.text
    try:
        payload = json.loads(str(content))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _result_with_cycle(
    result: ToolMessage | Command[Any], cycle: dict[str, Any]
) -> Command[Any]:
    if isinstance(result, ToolMessage):
        return Command(update={"verification_cycle": cycle, "messages": [result]})
    update = dict(result.update) if isinstance(result.update, dict) else {}
    update["verification_cycle"] = cycle
    return Command(
        graph=result.graph,
        update=update,
        resume=result.resume,
        goto=result.goto,
    )


class VerificationRemediationMiddleware(AgentMiddleware):
    """Enforce one optional repair and one mandatory re-verification per turn."""

    state_schema = VoxelAgentState

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = enabled
        self._claim_lock = threading.Lock()
        self._claims: set[tuple[str, str]] = set()
        self._claim_order: deque[tuple[str, str]] = deque()

    def _is_enabled(self) -> bool:
        return auto_remediation_enabled() if self.enabled is None else bool(self.enabled)

    def _claim(self, state: dict[str, Any], action: str) -> bool:
        run_id = str(state.get("current_run_id") or "missing-run-id")
        key = (run_id, action)
        with self._claim_lock:
            if key in self._claims:
                return False
            self._claims.add(key)
            self._claim_order.append(key)
            while len(self._claim_order) > 4096:
                expired = self._claim_order.popleft()
                self._claims.discard(expired)
        return True

    def _release(self, state: dict[str, Any], action: str) -> None:
        run_id = str(state.get("current_run_id") or "missing-run-id")
        with self._claim_lock:
            self._claims.discard((run_id, action))

    @staticmethod
    def _blocked(request: ToolCallRequest, reason: str) -> ToolMessage:
        call_id = str(request.tool_call.get("id") or "")
        return ToolMessage(
            content=json.dumps(
                {
                    "ok": False,
                    "status": "error",
                    "tool_name": "task",
                    "tool_call_id": call_id,
                    "errors": [reason],
                    "error": reason,
                }
            ),
            name="task",
            tool_call_id=call_id,
            status="error",
        )

    @staticmethod
    def _canonical_repair_request(
        request: ToolCallRequest, cycle: dict[str, Any]
    ) -> ToolCallRequest:
        remediation = cycle.get("remediation") or {}
        args = dict(request.tool_call.get("args") or {})
        exact_inputs = {
            "artifact_ids": list(remediation.get("input_artifact_ids") or []),
            "data_ids": list(remediation.get("input_data_ids") or []),
        }
        args["subagent_type"] = str(cycle.get("repair_target") or "")
        args["description"] = (
            "Execute exactly this approved, one-call remediation and no additional work.\n"
            f"Original user request: {request.state.get('current_user_request') or ''}\n"
            f"Repair objective: {remediation.get('instructions') or ''}\n"
            "Use only these registered inputs: "
            + json.dumps(exact_inputs, sort_keys=True)
            + "\nReturn the normal structured SubagentResult. Do not broaden the task, "
            "perform acquisition, or request another subagent."
        )
        tool_call = dict(request.tool_call)
        tool_call["args"] = args
        return request.override(tool_call=tool_call)

    @staticmethod
    def _canonical_second_verifier_request(
        request: ToolCallRequest, cycle: dict[str, Any]
    ) -> ToolCallRequest:
        args = dict(request.tool_call.get("args") or {})
        args["subagent_type"] = "verifier-agent"
        args["description"] = (
            "Perform the final independent audit after the single bounded remediation. "
            "Reassess every obligation in the original user request against the updated "
            "authoritative current-run evidence, including whether the repair actually "
            "closed the verified gap. Return VerifierResult. This is verifier call two of "
            "two; do not recommend or expect another automatic attempt.\n"
            f"Original user request: {request.state.get('current_user_request') or ''}\n"
            f"Attempted repair target: {cycle.get('repair_target') or ''}\n"
            f"Repair status: {cycle.get('repair_status') or 'unknown'}"
        )
        tool_call = dict(request.tool_call)
        tool_call["args"] = args
        return request.override(tool_call=tool_call)

    def _prepare_task(
        self, request: ToolCallRequest
    ) -> tuple[str, ToolCallRequest] | ToolMessage | None:
        if str(request.tool_call.get("name") or "") != "task":
            return None
        state = dict(request.state or {})
        cycle = dict(state.get("verification_cycle") or {})
        phase = str(cycle.get("phase") or "not_started")
        target = _task_target(request)

        if phase in {"complete", "remediation_rejected"}:
            return self._blocked(request, "The bounded verification cycle has ended for this turn.")
        if phase == "repair_approved":
            expected = str(cycle.get("repair_target") or "")
            if target != expected:
                return self._blocked(
                    request,
                    f"Only the approved one-call repair target '{expected}' may run now.",
                )
            if not self._claim(state, "repair"):
                return self._blocked(request, "The single remediation call was already claimed.")
            return "repair", self._canonical_repair_request(request, cycle)
        if phase == "repair_attempted":
            if target != "verifier-agent":
                return self._blocked(request, "The repair must now be re-verified exactly once.")
            if not self._claim(state, "verify2"):
                return self._blocked(request, "The second verifier call was already claimed.")
            return "verify2", self._canonical_second_verifier_request(request, cycle)
        if target == "verifier-agent":
            if not self._claim(state, "verify1"):
                return self._blocked(request, "The first verifier call was already claimed.")
            return "verify1", request
        return "domain", request

    def _after_task(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
        action: str,
    ) -> ToolMessage | Command[Any]:
        if action == "domain":
            return result
        state = dict(request.state or {})
        prior = dict(state.get("verification_cycle") or {})
        payload = _tool_message_payload(_tool_message_from_result(result))

        if action == "repair":
            cycle = {
                **prior,
                "schema_version": _VERIFICATION_CYCLE_SCHEMA,
                "phase": "repair_attempted",
                "repair_attempts": 1,
                "repair_status": str(payload.get("status") or "unknown"),
            }
            return _result_with_cycle(result, cycle)

        try:
            verifier = VerifierResult.model_validate(payload).model_dump()
        except Exception as exc:
            cycle = {
                **prior,
                "schema_version": _VERIFICATION_CYCLE_SCHEMA,
                "phase": "complete",
                "verifier_calls": 2 if action == "verify2" else 1,
                "last_verdict": "verification_error",
                "cycle_error": f"Invalid verifier result: {type(exc).__name__}",
            }
            return _result_with_cycle(result, cycle)

        if action == "verify2":
            cycle = {
                **prior,
                "schema_version": _VERIFICATION_CYCLE_SCHEMA,
                "phase": "complete",
                "verifier_calls": 2,
                "second_verdict": verifier["verdict"],
                "second_allow_final": verifier["allow_final"],
            }
            return _result_with_cycle(result, cycle)

        gate = evaluate_remediation_gate(state, verifier, enabled=self._is_enabled())
        needs_repair = verifier["verdict"] == "needs_remediation"
        approved = needs_repair and gate["approved"]
        cycle = {
            "schema_version": _VERIFICATION_CYCLE_SCHEMA,
            "phase": (
                "repair_approved"
                if approved
                else "remediation_rejected"
                if needs_repair
                else "complete"
            ),
            "verifier_calls": 1,
            "repair_attempts": 0,
            "first_verdict": verifier["verdict"],
            "first_allow_final": verifier["allow_final"],
            "gate": {
                "approved": approved,
                "reasons": list(gate["reasons"]),
            },
        }
        if approved:
            cycle["repair_target"] = gate["target_subagent"]
            cycle["remediation"] = gate["remediation"]
        return _result_with_cycle(result, cycle)

    def _with_cycle_context(self, request: ModelRequest) -> ModelRequest:
        state = dict(request.state or {})
        cycle = dict(state.get("verification_cycle") or {})
        phase = str(cycle.get("phase") or "not_started")
        if phase in {"complete", "remediation_rejected"}:
            tools = [tool for tool in request.tools if _middleware_tool_name(tool) != "task"]
        else:
            tools = list(request.tools)
        context = {
            "auto_remediation_enabled": self._is_enabled(),
            "phase": phase,
            "verifier_calls": int(cycle.get("verifier_calls") or 0),
            "repair_attempts": int(cycle.get("repair_attempts") or 0),
            "next_required_action": (
                f"Call only {cycle.get('repair_target')} once for the approved repair."
                if phase == "repair_approved"
                else "Call verifier-agent exactly once to assess the repair."
                if phase == "repair_attempted"
                else "No more task calls are permitted; give the evidence-safe final answer."
                if phase in {"complete", "remediation_rejected"}
                else "Continue normal work; use verifier-agent only when the hard-task policy applies."
            ),
            "gate": cycle.get("gate") or {},
        }
        existing = request.system_message.text if request.system_message is not None else ""
        block = (
            "\n\n<verification_cycle_json>\n"
            + json.dumps(context, sort_keys=True)
            + "\n</verification_cycle_json>\n"
            "This block is code-enforced. Follow next_required_action exactly."
        )
        return request.override(
            tools=tools,
            system_message=SystemMessage(content=existing + block),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        return handler(self._with_cycle_context(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        return await handler(self._with_cycle_context(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        prepared = self._prepare_task(request)
        if prepared is None:
            return handler(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        action, prepared_request = prepared
        try:
            result = handler(prepared_request)
        except Exception:
            if action != "domain":
                self._release(dict(request.state or {}), action)
            raise
        return self._after_task(prepared_request, result, action)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        prepared = self._prepare_task(request)
        if prepared is None:
            return await handler(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        action, prepared_request = prepared
        try:
            result = await handler(prepared_request)
        except Exception:
            if action != "domain":
                self._release(dict(request.state or {}), action)
            raise
        return self._after_task(prepared_request, result, action)
