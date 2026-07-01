from __future__ import annotations

import json
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


WORKFLOW_PLAN_PATH = "/workflow/plan.json"
WORKFLOW_ARTIFACT_REGISTRY_PATH = "/workflow/artifact_registry.json"
WORKFLOW_PROVENANCE_PATH = "/workflow/provenance.json"
WORKFLOW_FAILURES_PATH = "/workflow/failures.json"

WORKFLOW_MEMORY_PATHS = (
    WORKFLOW_PLAN_PATH,
    WORKFLOW_ARTIFACT_REGISTRY_PATH,
    WORKFLOW_PROVENANCE_PATH,
    WORKFLOW_FAILURES_PATH,
)

WorkflowStepStatus = Literal["pending", "in_progress", "ok", "partial", "error", "blocked", "skipped"]
VerificationStatus = Literal["unverified", "ok", "partial", "error", "not_required"]


class WorkflowStep(BaseModel):
    """A decomposed supervisor step stored in the DeepAgents virtual filesystem."""

    step_id: str
    description: str
    assigned_subagent: str
    status: WorkflowStepStatus = "pending"
    dependencies: List[str] = Field(default_factory=list)


class WorkflowPlan(BaseModel):
    """Thread-scoped plan memory for the canonical DeepAgents supervisor."""

    current_user_goal: str = ""
    steps: List[WorkflowStep] = Field(default_factory=list)
    current_step: str = ""
    status: WorkflowStepStatus = "pending"


class WorkflowArtifact(BaseModel):
    """A file-backed or virtual artifact produced by a VoxelInsight subagent."""

    artifact_id: str
    artifact_type: str
    producing_subagent: str
    producing_tool: str = ""
    path: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    verification_status: VerificationStatus = "unverified"


class WorkflowArtifactRegistry(BaseModel):
    """Registry of artifacts available for downstream subagents."""

    artifacts: List[WorkflowArtifact] = Field(default_factory=list)


class WorkflowProvenanceEvent(BaseModel):
    """Ordered workflow event for delegation, tool calls, verification, and synthesis."""

    event_id: str
    actor: str
    action: str
    status: str
    inputs_summary: str = ""
    outputs_summary: str = ""
    artifact_ids: List[str] = Field(default_factory=list)


class WorkflowProvenance(BaseModel):
    """Ordered provenance log for one DeepAgents thread."""

    events: List[WorkflowProvenanceEvent] = Field(default_factory=list)


class WorkflowFailure(BaseModel):
    """Blocking or non-blocking workflow failure recorded for supervisor decisions."""

    failure_id: str
    actor: str
    message: str
    blocking: bool = True
    failed_step: str = ""
    artifact_id: str = ""
    retry_count: int = 0
    resolution_status: str = "open"


class WorkflowFailures(BaseModel):
    """Failure log for one DeepAgents thread."""

    failures: List[WorkflowFailure] = Field(default_factory=list)


def _dump_model(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def workflow_seed_documents() -> Dict[str, Dict[str, Any]]:
    """Return valid empty JSON documents for the virtual workflow filesystem."""
    return {
        WORKFLOW_PLAN_PATH: _dump_model(WorkflowPlan()),
        WORKFLOW_ARTIFACT_REGISTRY_PATH: _dump_model(WorkflowArtifactRegistry()),
        WORKFLOW_PROVENANCE_PATH: _dump_model(WorkflowProvenance()),
        WORKFLOW_FAILURES_PATH: _dump_model(WorkflowFailures()),
    }


def workflow_seed_json() -> Dict[str, str]:
    """Return pretty JSON seed documents keyed by virtual path."""
    return {
        path: json.dumps(document, indent=2, sort_keys=True)
        for path, document in workflow_seed_documents().items()
    }


def workflow_memory_policy() -> str:
    """Instructions for the code-maintained workflow state injected by middleware."""
    return """
Deterministic workflow state
- Application middleware injects `<voxelinsight_state_json>` before every supervisor and
  subagent model call.
- `uploaded_files` contains authoritative uploads for the thread.
- `artifacts` contains only existing paths registered from real normalized tool outputs.
- `data` contains machine-readable table rows or a durable table artifact reference.
- Tool provenance and failures are recorded by code in graph state and the durable
  checkpointer. Do not recreate or edit artifact/provenance registries with filesystem tools.
- Pass exact paths and rows from this state into domain tools. Return only existing
  artifact_ids/data_ids in SubagentResult.
- Use `write_todos` only when a complex workflow benefits from an explicit plan; it is not
  an artifact store.
- Never expose local filesystem paths in the final user-facing response.
"""
