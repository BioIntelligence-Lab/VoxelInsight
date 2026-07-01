from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SubagentStatus = Literal["ok", "partial", "error", "no_action"]
VisibleOutputKind = Literal["image", "plotly", "file", "dataframe", "link", "code", "text"]
VisibleOutputSource = Literal["tool_ui", "tool_outputs", "verified_file"]
DeliverableStatus = Literal["satisfied", "missing", "failed"]
IDCValidationStatus = Literal["pass", "warn", "fail"]
IDCQueryLanguage = Literal["typed", "sql", "python"]


def _require_idc_audit_fields_in_json_schema(schema: Dict[str, Any]) -> None:
    """Keep IDC audit fields required for the model despite safe parse defaults."""

    required = list(schema.get("required") or [])
    for field in (
        "cohort_definition",
        "query_or_code",
        "counts",
        "validation_checks",
        "provenance",
        "warnings",
    ):
        if field not in required:
            required.append(field)
    schema["required"] = required


class VisibleOutput(BaseModel):
    """User-visible output derived from actual normalized tool payloads."""

    kind: VisibleOutputKind
    source: VisibleOutputSource
    title: str = ""
    path: Optional[str] = None
    url: Optional[str] = None

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> Any:
        if value is None:
            return "text"
        normalized = str(value).strip().lower()
        aliases = {
            "table": "dataframe",
            "data_frame": "dataframe",
            "chart": "plotly",
            "plot": "plotly",
            "directory": "file",
            "folder": "file",
        }
        return aliases.get(normalized, normalized)


class Deliverable(BaseModel):
    """Machine-checkable completion state for a user-requested deliverable."""

    name: str
    status: DeliverableStatus
    evidence: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> Any:
        if value is None:
            return "missing"
        status = str(value).strip().lower()
        if status in {"satisfied", "missing", "failed"}:
            return status
        if status in {"partial", "incomplete", "pending", "blocked", "no_action", "no action"}:
            return "missing"
        if status in {"ok", "done", "complete", "completed", "success"}:
            return "satisfied"
        if status in {"error", "errored", "failure"}:
            return "failed"
        return "missing"


class ToolCallSummary(BaseModel):
    """Minimal bookkeeping for a subagent tool call."""

    name: str
    status: str = ""
    call_id: str = ""

    @field_validator("call_id", mode="before")
    @classmethod
    def _coerce_null_call_id(cls, value: Any) -> Any:
        return "" if value is None else value


class ArtifactEvidence(BaseModel):
    """Backward-compatible deterministic artifact evidence.

    New subagent responses should use artifact_ids. This model remains for the
    verification tool and for reading older checkpoints.
    """

    artifact_id: str = ""
    type: str = ""
    name: str = ""
    path: str = ""
    role: str = ""
    exists: Optional[bool] = None
    files: List[str] = Field(default_factory=list)
    nifti_paths: List[str] = Field(default_factory=list)
    mask_paths: List[str] = Field(default_factory=list)
    segmentations: List[str] = Field(default_factory=list)
    segmentations_map: Dict[str, Any] = Field(default_factory=dict)
    output_dir: str = ""
    output_root: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubagentResult(BaseModel):
    """Shared structured response contract for VoxelInsight subagents."""

    status: SubagentStatus = Field(
        ...,
        description="Overall subagent status.",
    )
    tool_calls: List[ToolCallSummary] = Field(
        default_factory=list,
        description="Typed summaries of tools invoked by this subagent.",
    )
    artifact_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Artifact IDs from the deterministic artifact registry. Never invent IDs or paths."
        ),
    )
    data_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Data IDs from the deterministic data registry. Never copy or invent table rows."
        ),
    )
    artifacts: List[ArtifactEvidence] = Field(
        default_factory=list,
        description=(
            "Legacy deterministic artifact evidence. Subagents should normally leave this empty "
            "and return artifact_ids instead."
        ),
    )
    summary: str = Field(
        default="",
        description="Concise internal summary for the supervisor.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Errors or blocking issues encountered by this subagent.",
    )
    next_recommended_inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Values the supervisor may pass into downstream subagents/tools.",
    )
    visible_outputs: List[VisibleOutput] = Field(
        default_factory=list,
        description=(
            "Outputs the user can actually see, derived only from normalized tool payloads "
            "or verified files."
        ),
    )
    deliverables: List[Deliverable] = Field(
        default_factory=list,
        description="Machine-checkable status of requested deliverables.",
    )

    @field_validator("next_recommended_inputs", mode="before")
    @classmethod
    def _coerce_null_next_recommended_inputs(cls, value: Any) -> Any:
        if value is None:
            return {}
        return value

    @field_validator(
        "tool_calls",
        "artifact_ids",
        "data_ids",
        "artifacts",
        "visible_outputs",
        "deliverables",
        mode="before",
    )
    @classmethod
    def _coerce_null_lists(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value


class IDCCohortDefinition(BaseModel):
    """Explicit population definition used for an IDC result."""

    collection_ids: List[str] = Field(
        ...,
        description="Exact IDC collection_id values included in the result.",
    )
    inclusion_criteria: List[str] = Field(
        ...,
        description="Machine-grounded filters that define included records.",
    )
    exclusion_criteria: List[str] = Field(
        ...,
        description="Explicit exclusions applied to the cohort.",
    )
    unit_of_analysis: str = Field(
        ...,
        description="Counting unit such as patient, study, series, instance, or collection.",
    )


class IDCQueryEvidence(BaseModel):
    """Exact typed operation, SQL, or restricted Python used for a result."""

    tool_name: str = Field(..., description="IDC specialist tool that executed the operation.")
    language: IDCQueryLanguage = Field(..., description="Representation of the executed operation.")
    statement: str = Field(..., description="Exact query, typed filter description, or restricted code.")


class IDCCountEvidence(BaseModel):
    """Named numeric count with its denominator/counting definition."""

    name: str = Field(..., description="Count name, such as distinct_patients.")
    value: float = Field(..., description="Numeric count or aggregate value.")
    definition: str = Field(..., description="How the value was computed and de-duplicated.")


class IDCValidationEvidence(BaseModel):
    """A validation assertion performed against the IDC result."""

    name: str = Field(..., description="Validation check name.")
    status: IDCValidationStatus = Field(..., description="Pass, warning, or failure.")
    evidence: str = Field(..., description="Observed value or concise validation evidence.")

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_validation_status(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        aliases = {
            "pass": "pass",
            "passed": "pass",
            "ok": "pass",
            "success": "pass",
            "successful": "pass",
            "valid": "pass",
            "validated": "pass",
            "warn": "warn",
            "warning": "warn",
            "partial": "warn",
            "caution": "warn",
            "inconclusive": "warn",
            "fail": "fail",
            "failed": "fail",
            "error": "fail",
            "invalid": "fail",
        }
        return aliases.get(normalized, normalized)


class IDCProvenance(BaseModel):
    """Versioned source lineage for IDC discovery and cohort results."""

    source: str = Field(..., description="Authoritative source, normally NCI Imaging Data Commons.")
    idc_data_version: str = Field(..., description="IDC data release reported by idc-index.")
    skill_name: str = Field(..., description="Attached skill name.")
    skill_version: str = Field(..., description="Vendored official IDC skill version.")
    tables: List[str] = Field(..., description="IDC index tables used by the result.")


class IDCSubagentResult(SubagentResult):
    """Structured, auditable response contract for the dedicated IDC agent."""

    model_config = ConfigDict(json_schema_extra=_require_idc_audit_fields_in_json_schema)

    cohort_definition: IDCCohortDefinition = Field(
        ...,
        description="Explicit cohort definition, including for zero-result queries.",
    )
    query_or_code: List[IDCQueryEvidence] = Field(
        default_factory=list,
        description="Every executed typed operation, SQL statement, or restricted Python program.",
    )
    counts: List[IDCCountEvidence] = Field(
        default_factory=list,
        description=(
            "Named finite numeric counts/aggregates with their exact counting definitions. "
            "Unknown values belong in warnings, never as null values or invented zeros."
        ),
    )
    validation_checks: List[IDCValidationEvidence] = Field(
        default_factory=list,
        description="Checks for schema, row counts, uniqueness, missingness, and join integrity as applicable.",
    )
    provenance: IDCProvenance = Field(
        ...,
        description="IDC release, official skill, and source-table lineage.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Material limitations, missing metadata, ambiguity, truncation, or fallback-execution notices.",
    )

    @model_validator(mode="before")
    @classmethod
    def _move_unavailable_counts_to_audit_warnings(cls, value: Any) -> Any:
        """Keep ``counts`` numeric without failing the whole subagent result.

        Models occasionally represent an unavailable aggregate as a count whose
        value is null (or a common missing-value string).  Such a value is not a
        numeric count and must not be coerced to zero.  Omit it from ``counts``
        and retain the missingness explicitly in warnings and validation checks.
        """

        if not isinstance(value, dict):
            return value

        payload = dict(value)
        missing_audit_fields: List[str] = []
        if payload.get("status") not in {"ok", "partial", "error", "no_action"}:
            payload["status"] = "error" if payload.get("errors") else "partial"
            missing_audit_fields.append("status")
        if payload.get("cohort_definition") is None:
            payload["cohort_definition"] = {
                "collection_ids": [],
                "inclusion_criteria": [],
                "exclusion_criteria": [],
                "unit_of_analysis": "unspecified",
            }
            missing_audit_fields.append("cohort_definition")
        if payload.get("provenance") is None:
            payload["provenance"] = {
                "source": "NCI Imaging Data Commons",
                "idc_data_version": "unknown",
                "skill_name": "imaging-data-commons",
                "skill_version": "1.6.5",
                "tables": [],
            }
            missing_audit_fields.append("provenance")
        if missing_audit_fields:
            audit_warning = (
                "The model omitted or returned invalid required IDC result field(s): "
                f"{', '.join(missing_audit_fields)}. Explicit safe fallback values were inserted."
            )
            raw_warnings = payload.get("warnings") or []
            payload["warnings"] = (
                list(raw_warnings) if isinstance(raw_warnings, list) else [raw_warnings]
            ) + [audit_warning]
            raw_checks = payload.get("validation_checks") or []
            payload["validation_checks"] = (
                list(raw_checks) if isinstance(raw_checks, list) else [raw_checks]
            ) + [
                {
                    "name": "structured_audit_completeness",
                    "status": "warn",
                    "evidence": audit_warning,
                }
            ]
            if payload.get("status") == "ok":
                payload["status"] = "partial"

        missing_counts_field = "counts" not in payload
        raw_counts = payload.get("counts")
        unavailable_details: List[str] = []
        if missing_counts_field:
            count_entries: List[Any] = []
            unavailable_details.append("counts field was omitted")
        elif raw_counts is None:
            count_entries = []
            unavailable_details.append("counts field was null")
        elif isinstance(raw_counts, dict):
            count_entries = [raw_counts]
        elif isinstance(raw_counts, list):
            count_entries = raw_counts
        else:
            count_entries = []
            unavailable_details.append(
                f"counts had invalid top-level type {type(raw_counts).__name__}"
            )

        numeric_counts: List[Any] = []
        for index, entry in enumerate(count_entries):
            if isinstance(entry, IDCCountEvidence):
                numeric_counts.append(entry)
                continue
            if not isinstance(entry, dict):
                unavailable_details.append(
                    f"counts[{index}] had invalid type {type(entry).__name__}"
                )
                continue

            count_value = entry.get("value")
            is_finite_numeric = False
            if not isinstance(count_value, bool):
                if isinstance(count_value, (int, float)):
                    is_finite_numeric = math.isfinite(float(count_value))
                elif isinstance(count_value, str) and count_value.strip():
                    try:
                        is_finite_numeric = math.isfinite(float(count_value.strip()))
                    except ValueError:
                        is_finite_numeric = False

            name_value = entry.get("name")
            definition_value = entry.get("definition")
            has_name = isinstance(name_value, str) and bool(name_value.strip())
            has_definition = isinstance(definition_value, str) and bool(definition_value.strip())
            if not (is_finite_numeric and has_name and has_definition):
                name = str(name_value or f"counts[{index}]").strip()
                definition = (
                    definition_value.strip()
                    if isinstance(definition_value, str) and definition_value.strip()
                    else "no counting definition supplied"
                )
                unavailable_details.append(
                    f"{name} ({definition}; received value={count_value!r})"
                )
                continue
            numeric_counts.append(entry)

        if not unavailable_details:
            return payload

        payload["counts"] = numeric_counts
        warning = (
            "Unavailable or invalid aggregate value(s) were omitted from numeric counts: "
            f"{'; '.join(unavailable_details)}. They were not interpreted as zero."
        )

        raw_warnings = payload.get("warnings")
        if raw_warnings is None:
            warnings: List[Any] = []
        elif isinstance(raw_warnings, list):
            warnings = list(raw_warnings)
        else:
            warnings = [raw_warnings]
        warnings.append(warning)
        payload["warnings"] = warnings

        raw_checks = payload.get("validation_checks")
        if raw_checks is None:
            checks: List[Any] = []
        elif isinstance(raw_checks, list):
            checks = list(raw_checks)
        else:
            checks = [raw_checks]
        checks.append(
            {
                "name": "numeric_count_values",
                "status": "warn",
                "evidence": warning,
            }
        )
        payload["validation_checks"] = checks
        return payload

    @field_validator("query_or_code", "counts", "validation_checks", "warnings", mode="before")
    @classmethod
    def _coerce_idc_audit_lists(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value


def dump_subagent_result(result: SubagentResult) -> Dict[str, Any]:
    """Return a dict across Pydantic v1/v2."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result.dict()
