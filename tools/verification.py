from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.agents.artifact_verification import verify_artifact_payload
from core.agents.schemas import dump_subagent_result
from core.state import TaskResult
from tools.shared import toolify_agent


class VerifyArtifactsArgs(BaseModel):
    artifacts: str = Field(
        ...,
        description="JSON string encoding a list of artifact dictionaries to verify.",
    )


@toolify_agent(
    name="verify_artifacts",
    description=(
        "Deterministically verifies file-backed VoxelInsight artifacts. "
        "Checks paths, output directories, CSV row presence, NIfTI loading, "
        "image/mask dimension agreement, and non-empty segmentation masks."
    ),
    args_schema=VerifyArtifactsArgs,
    timeout_s=120,
)
async def verify_artifacts_runner(
    artifacts: Any,
    context: Optional[Dict[str, Any]] = None,
):
    if isinstance(artifacts, str):
        parsed_artifacts = json.loads(artifacts)
    else:
        parsed_artifacts = artifacts
    if isinstance(parsed_artifacts, dict):
        parsed_artifacts = [parsed_artifacts]
    result = verify_artifact_payload(artifacts=parsed_artifacts, context=context)
    verified_files = [
        str(item.path)
        for item in result.artifacts
        if item.exists
        and item.path
        and not any(item.path in error for error in result.errors)
    ]
    return TaskResult(
        output=dump_subagent_result(result),
        artifacts={
            "subagent_result": dump_subagent_result(result),
            "files": verified_files,
        },
        status=result.status,
        errors=result.errors,
    )
