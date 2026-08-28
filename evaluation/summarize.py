from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict


SUMMARY_FIELDS = [
    "experiment_id",
    "case_id",
    "repetition",
    "status",
    "execution_success",
    "deliverables_satisfied",
    "scientific_correctness",
    "duration_s",
    "time_to_first_token_s",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "estimated_llm_cost_usd",
    "model_calls",
    "subagent_calls",
    "tool_calls",
    "errors",
    "artifact_count",
    "data_record_count",
]


def _summary_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") or {}
    usage = metrics.get("usage") or {}
    return {
        "experiment_id": payload.get("experiment_id"),
        "case_id": payload.get("case_id"),
        "repetition": payload.get("repetition"),
        "status": payload.get("status"),
        "execution_success": payload.get("execution_success"),
        "deliverables_satisfied": payload.get("deliverables_satisfied"),
        "scientific_correctness": payload.get("scientific_correctness"),
        "duration_s": round(float(metrics.get("duration_ms") or 0) / 1000, 3),
        "time_to_first_token_s": (
            round(float(metrics["time_to_first_token_ms"]) / 1000, 3)
            if metrics.get("time_to_first_token_ms") is not None
            else None
        ),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens", 0),
        "estimated_llm_cost_usd": payload.get("estimated_llm_cost_usd"),
        "model_calls": metrics.get("model_calls", 0),
        "subagent_calls": metrics.get("subagent_calls", 0),
        "tool_calls": metrics.get("tool_calls", 0),
        "errors": metrics.get("errors", 0),
        "artifact_count": payload.get("artifact_count", 0),
        "data_record_count": payload.get("data_record_count", 0),
    }


def summarize_experiment(experiment_dir: str | Path) -> Path:
    root = Path(experiment_dir).expanduser().resolve()
    rows = []
    for path in sorted(root.glob("*/repetition_*/run.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        rows.append(_summary_row(payload))
    output = root / "summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VoxelInsight evaluation runs.")
    parser.add_argument("experiment_dir")
    args = parser.parse_args()
    print(summarize_experiment(args.experiment_dir))


if __name__ == "__main__":
    main()
