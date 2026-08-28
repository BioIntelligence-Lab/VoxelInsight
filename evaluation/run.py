from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from core.agents.deep_voxelinsight import (
    DEFAULT_DEEPAGENT_SUBAGENT_MODEL,
    DEFAULT_DEEPAGENT_SUPERVISOR_MODEL,
    DEFAULT_DEEPAGENT_VERIFIER_MODEL,
    DEEPAGENT_SUBAGENT_MODEL_ENV,
    DEEPAGENT_SUPERVISOR_MODEL_ENV,
    DEEPAGENT_VERIFIER_MODEL_ENV,
)
from core.agents.run_metrics import RunMetricsRecorder, metrics_context
from core.agents.run_turn import TurnResult, execute_turn
from core.interactions import (
    ConfirmationRequest,
    InteractionHandler,
    interaction_context,
)
from evaluation.pricing import estimate_llm_cost, load_pricing
from evaluation.schema import EvaluationCase, ExperimentSpec, load_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    temporary.replace(path)


def _jsonl_write(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _git_metadata() -> Dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    ).stdout
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _model_configuration() -> Dict[str, str]:
    return {
        "provider": (os.getenv("LLM_PROVIDER") or "openai").strip().lower(),
        "supervisor_model": os.getenv(
            DEEPAGENT_SUPERVISOR_MODEL_ENV,
            DEFAULT_DEEPAGENT_SUPERVISOR_MODEL,
        ),
        "subagent_model": os.getenv(
            DEEPAGENT_SUBAGENT_MODEL_ENV,
            DEFAULT_DEEPAGENT_SUBAGENT_MODEL,
        ),
        "verifier_model": os.getenv(
            DEEPAGENT_VERIFIER_MODEL_ENV,
            DEFAULT_DEEPAGENT_VERIFIER_MODEL,
        ),
    }


def _artifact_labels(record: Dict[str, Any]) -> set[str]:
    path = Path(str(record.get("path") or ""))
    suffixes = "".join(path.suffixes).lower()
    kind = str(record.get("kind") or "").lower()
    role = str(record.get("role") or "").lower()
    labels = {kind, role}
    if suffixes.endswith(".csv"):
        labels.add("csv")
    if suffixes.endswith((".nii", ".nii.gz")):
        labels.add("nifti")
    if kind in {"image", "plotly"} or role == "visualization":
        labels.add("plot")
    if role == "segmentation" or "mask" in path.name.lower():
        labels.update({"segmentation", "mask"})
    return {label for label in labels if label}


def _validate_case(
    case: EvaluationCase,
    artifacts: list[Dict[str, Any]],
    events: list[Dict[str, Any]],
    execution_success: bool,
) -> Dict[str, Any]:
    output_artifacts = [record for record in artifacts if record.get("role") != "input"]
    artifact_valid = all(
        record.get("status") == "verified"
        and bool(record.get("path"))
        and Path(str(record["path"])).exists()
        for record in output_artifacts
    )
    labels = set().union(*(_artifact_labels(record) for record in output_artifacts)) if output_artifacts else set()
    required_artifacts = {
        str(value).lower() for value in (case.expected.get("artifacts") or [])
    }
    observed_subagents = {
        str(event.get("subagent") or "")
        for event in events
        if event.get("kind") == "subagent" and event.get("subagent")
    }
    observed_tools = {
        str(event.get("name") or "")
        for event in events
        if event.get("kind") == "tool" and event.get("name")
    }
    required_subagents = set(case.expected.get("subagents") or [])
    forbidden_subagents = set(case.expected.get("forbidden_subagents") or [])
    required_tools = set(case.expected.get("tools") or [])
    forbidden_tools = set(case.expected.get("forbidden_tools") or [])
    checks = {
        "execution_success": execution_success,
        "artifact_valid": artifact_valid,
        "required_artifacts": required_artifacts.issubset(labels),
        "required_subagents": required_subagents.issubset(observed_subagents),
        "forbidden_subagents": not bool(forbidden_subagents & observed_subagents),
        "required_tools": required_tools.issubset(observed_tools),
        "forbidden_tools": not bool(forbidden_tools & observed_tools),
    }
    return {
        "checks": checks,
        "deliverables_satisfied": all(checks.values()),
        "required_artifacts": sorted(required_artifacts),
        "observed_artifact_labels": sorted(labels),
        "observed_subagents": sorted(observed_subagents),
        "observed_tools": sorted(observed_tools),
    }


async def _run_case(
    spec: ExperimentSpec,
    case: EvaluationCase,
    repetition: int,
    run_dir: Path,
    pricing: Dict[str, Any],
    timeout_override: Optional[int],
) -> Dict[str, Any]:
    recorder = RunMetricsRecorder()
    thread_id = (
        f"evaluation-{spec.experiment_id}-{case.case_id}-{repetition}-{uuid.uuid4().hex}"
    )
    turn_results: list[TurnResult] = []
    interactions: list[Dict[str, Any]] = []
    error = ""
    stack_trace = ""

    async def confirm(request: ConfirmationRequest) -> bool:
        configured = case.approvals.get(request.kind)
        if configured is None and request.kind.endswith("_download"):
            configured = case.approvals.get("downloads")
        approved = configured in {"approve", "approved", "allow", "yes", "true"}
        interactions.append(
            {
                "kind": "interaction",
                "operation": request.kind,
                "decision": "approve" if approved else "deny",
                "details": request.details,
            }
        )
        return approved

    async def notify(content: str) -> None:
        interactions.append({"kind": "notification", "content": content})

    execution_success = False
    try:
        with (
            metrics_context(recorder),
            interaction_context(InteractionHandler(confirm=confirm, notify=notify)),
        ):
            for turn_index, prompt in enumerate(case.turns):
                result = await asyncio.wait_for(
                    execute_turn(
                        prompt,
                        thread_id=thread_id,
                        uploaded_files=case.uploads if turn_index == 0 else [],
                        callbacks=[recorder],
                    ),
                    timeout=timeout_override or case.timeout_s,
                )
                turn_results.append(result)
        execution_success = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stack_trace = traceback.format_exc()

    metrics = recorder.finish()
    events = sorted(recorder.snapshot_events(), key=lambda item: int(item.get("started_ns") or 0))
    trace_records = [*events, *interactions]
    artifacts_by_id: Dict[str, Dict[str, Any]] = {}
    data_by_id: Dict[str, Dict[str, Any]] = {}
    for result in turn_results:
        for record in result.registry_payload.get("artifacts") or []:
            artifact_id = str(record.get("artifact_id") or "")
            if artifact_id:
                artifacts_by_id[artifact_id] = record
        for record in result.registry_payload.get("data") or []:
            data_id = str(record.get("data_id") or "")
            if data_id:
                data_by_id[data_id] = record
    artifacts = list(artifacts_by_id.values())
    validation = _validate_case(case, artifacts, events, execution_success)
    estimated_cost, unpriced_models = estimate_llm_cost(events, pricing)
    artifact_manifest = {
        "artifacts": artifacts,
        "data_records": list(data_by_id.values()),
    }
    _json_write(run_dir / "artifacts.json", artifact_manifest)
    _jsonl_write(run_dir / "trace.jsonl", trace_records)

    payload = {
        "schema_version": "voxelinsight.evaluation-run.v1",
        "experiment_id": spec.experiment_id,
        "case_id": case.case_id,
        "repetition": repetition,
        "thread_id": thread_id,
        "run_ids": [result.run_id for result in turn_results],
        "status": "complete" if execution_success else "error",
        "execution_success": execution_success,
        "deliverables_satisfied": validation["deliverables_satisfied"],
        "scientific_correctness": None,
        "turns": [
            {
                "prompt": result.prompt,
                "run_id": result.run_id,
                "final_text": result.final_text,
                "stream_event_count": result.stream_event_count,
            }
            for result in turn_results
        ],
        "final_text": turn_results[-1].final_text if turn_results else "",
        "metrics": metrics,
        "estimated_llm_cost_usd": estimated_cost,
        "pricing_version": pricing.get("version", "unconfigured"),
        "unpriced_models": unpriced_models,
        "validation": validation,
        "artifact_count": len(artifacts),
        "data_record_count": len(data_by_id),
        "interaction_decisions": interactions,
        "model_configuration": _model_configuration(),
        "git": _git_metadata(),
        "error": error,
        "traceback": stack_trace,
    }
    _json_write(run_dir / "run.json", payload)
    return payload


async def run_experiment(
    spec_path: str | Path,
    *,
    output_root: str | Path = "evaluation_results",
    selected_case: str = "",
    repetitions_override: Optional[int] = None,
    timeout_override: Optional[int] = None,
    resume: bool = False,
) -> Path:
    spec = load_experiment(spec_path)
    default_pricing = Path(__file__).with_name("pricing.yaml")
    pricing = load_pricing(spec.pricing_path or default_pricing)
    experiment_dir = Path(output_root).expanduser().resolve() / spec.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = experiment_dir / "runtime"
    os.environ["VOXELINSIGHT_PERSIST_ROOT"] = str(runtime_dir / "artifacts")
    os.environ["VOXELINSIGHT_TEMP_ROOT"] = str(runtime_dir / "tmp")
    os.environ["VOXELINSIGHT_CHECKPOINT_DB"] = str(
        runtime_dir / "state" / "checkpoints.sqlite"
    )
    os.environ["VOXELINSIGHT_REGISTRY_DB"] = str(
        runtime_dir / "state" / "registry.sqlite"
    )
    _json_write(
        experiment_dir / "experiment.json",
        {
            "experiment_id": spec.experiment_id,
            "source_spec": str(Path(spec_path).expanduser().resolve()),
            "git": _git_metadata(),
            "model_configuration": _model_configuration(),
            "pricing_version": pricing.get("version", "unconfigured"),
        },
    )
    for case in spec.cases:
        if selected_case and case.case_id != selected_case:
            continue
        repetitions = repetitions_override or case.repetitions
        for repetition in range(1, repetitions + 1):
            run_dir = experiment_dir / case.case_id / f"repetition_{repetition:03d}"
            run_path = run_dir / "run.json"
            if resume and run_path.exists():
                try:
                    previous = json.loads(run_path.read_text())
                except Exception:
                    previous = {}
                if previous.get("status") == "complete":
                    continue
            run_dir.mkdir(parents=True, exist_ok=True)
            await _run_case(
                spec,
                case,
                repetition,
                run_dir,
                pricing,
                timeout_override,
            )
    from evaluation.summarize import summarize_experiment

    summarize_experiment(experiment_dir)
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VoxelInsight evaluation scenarios.")
    parser.add_argument("spec", help="Path to an experiment YAML file.")
    parser.add_argument("--output-root", default="evaluation_results")
    parser.add_argument("--case", default="")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = asyncio.run(
        run_experiment(
            args.spec,
            output_root=args.output_root,
            selected_case=args.case,
            repetitions_override=args.repetitions,
            timeout_override=args.timeout,
            resume=args.resume,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
