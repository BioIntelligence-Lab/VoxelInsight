from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    turns: List[str]
    uploads: List[str] = field(default_factory=list)
    approvals: Dict[str, str] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    timeout_s: int = 900
    repetitions: int = 1


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    cases: List[EvaluationCase]
    repetitions: int = 1
    timeout_s: int = 900
    pricing_path: str = ""


def load_experiment(path: str | Path) -> ExperimentSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(spec_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Experiment specification must be a YAML object.")
    experiment_id = str(payload.get("experiment_id") or spec_path.stem).strip()
    if not experiment_id:
        raise ValueError("experiment_id is required.")
    default_repetitions = int(payload.get("repetitions", 1))
    default_timeout = int(payload.get("timeout_s", 900))
    cases: List[EvaluationCase] = []
    for index, raw_case in enumerate(payload.get("cases") or [], start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Case {index} must be a YAML object.")
        case_id = str(raw_case.get("id") or f"case-{index}").strip()
        raw_turns = raw_case.get("turns")
        if raw_turns is None and raw_case.get("prompt") is not None:
            raw_turns = [raw_case["prompt"]]
        turns = [str(turn).strip() for turn in (raw_turns or []) if str(turn).strip()]
        if not turns:
            raise ValueError(f"Case '{case_id}' has no prompts/turns.")
        uploads = []
        for upload in raw_case.get("uploads") or []:
            upload_path = Path(str(upload)).expanduser()
            if not upload_path.is_absolute():
                upload_path = spec_path.parent / upload_path
            uploads.append(str(upload_path.resolve()))
        cases.append(
            EvaluationCase(
                case_id=case_id,
                turns=turns,
                uploads=uploads,
                approvals={
                    str(key): str(value).lower()
                    for key, value in (raw_case.get("approvals") or {}).items()
                },
                expected=dict(raw_case.get("expected") or {}),
                timeout_s=int(raw_case.get("timeout_s", default_timeout)),
                repetitions=int(raw_case.get("repetitions", default_repetitions)),
            )
        )
    if not cases:
        raise ValueError("Experiment specification contains no cases.")
    pricing_path = str(payload.get("pricing_path") or "")
    if pricing_path:
        candidate = Path(pricing_path).expanduser()
        if not candidate.is_absolute():
            candidate = spec_path.parent / candidate
        pricing_path = str(candidate.resolve())
    return ExperimentSpec(
        experiment_id=experiment_id,
        cases=cases,
        repetitions=default_repetitions,
        timeout_s=default_timeout,
        pricing_path=pricing_path,
    )

