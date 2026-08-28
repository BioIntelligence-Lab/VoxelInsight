# VoxelInsight evaluation guide

The evaluation infrastructure runs reproducible prompts through the same VoxelInsight
agent graph used by the application, records execution evidence and model usage, and
checks observable outcomes declared in YAML.

This document describes the general experiment runner used for end-to-end VoxelInsight
workflow evaluations.

## Quick start

Create a YAML file under `evaluation/cases/`, for example:

```yaml
experiment_id: my_first_evaluation
repetitions: 1
timeout_s: 900
pricing_path: ../pricing.yaml

cases:
  - id: collection_lookup
    prompt: "How many distinct patients are in the CPTAC-LUAD IDC collection?"
    expected:
      subagents: [idc-agent]
      tools: [idc_collection_summary]
      forbidden_subagents: [acquisition-agent]
```

Run it from the repository root:

```bash
python -m evaluation.run \
  evaluation/cases/my_evaluation.yaml
```

Run these commands after activating the Python environment containing VoxelInsight's
dependencies. If `python` does not refer to that environment, replace it with the correct
interpreter command or path for your installation.

The command prints the output directory when the experiment finishes. By default, results
are written under `evaluation_results/<experiment_id>/`.

## YAML structure

An experiment file is a YAML object with experiment-level defaults and a non-empty `cases`
list.

```yaml
experiment_id: example_v1
repetitions: 3
timeout_s: 1800
pricing_path: ../pricing.yaml

cases:
  - id: example_case
    turns:
      - "First user message."
      - "Follow-up message in the same conversation."
    uploads:
      - ../../example_data/input.csv
    approvals:
      downloads: approve
    expected:
      artifacts: [csv]
      subagents: [cohort-agent]
      forbidden_subagents: [segmentation-agent]
      tools: [clinical_data_download]
      forbidden_tools: [idc_download]
    timeout_s: 2400
    repetitions: 2
```

### Experiment fields

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `experiment_id` | No | YAML filename | Stable name used for the result directory. |
| `repetitions` | No | `1` | Default number of runs for each case. |
| `timeout_s` | No | `900` | Default timeout in seconds for each individual turn. |
| `pricing_path` | No | `evaluation/pricing.yaml` | Pricing snapshot used for estimated LLM cost. Relative paths resolve from the experiment YAML directory. |
| `cases` | Yes | — | Non-empty list of evaluation cases. |

### Case fields

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `id` | Recommended | `case-<number>` | Case name used in output paths and CLI selection. Keep it unique within the experiment. |
| `prompt` | One of `prompt` or `turns` | — | Shortcut for a single-turn case. |
| `turns` | One of `prompt` or `turns` | — | Ordered user messages sent to one persistent evaluation thread. |
| `uploads` | No | `[]` | Files attached on the first turn. Relative paths resolve from the YAML directory. |
| `approvals` | No | `{}` | Automatic decisions for confirmation requests. Unconfigured operations are denied. |
| `expected` | No | `{}` | Deterministic execution and deliverable checks. |
| `timeout_s` | No | experiment default | Per-turn timeout for this case. |
| `repetitions` | No | experiment default | Number of independent runs for this case. |

Use either `prompt` or `turns`. If both are present, `turns` takes precedence.

## Expected checks

The general runner supports these keys inside `expected`:

```yaml
expected:
  artifacts: [csv, plot]
  subagents: [idc-agent, analysis-agent]
  forbidden_subagents: [acquisition-agent, segmentation-agent]
  tools: [idc_collection_profile, table_chart]
  forbidden_tools: [idc_download]
```

All declared checks must pass for `deliverables_satisfied` to be `true`.

### `artifacts`

Requires registered, verified output artifacts whose paths exist on disk. Supported labels
are inferred from artifact metadata and filenames:

- `csv` for `.csv` files
- `nifti` for `.nii` or `.nii.gz` files
- `plot` for image or Plotly visualization artifacts
- `segmentation` or `mask` for segmentation artifacts and mask-like outputs
- Artifact `kind` and `role` values may also be used as labels

Input uploads do not satisfy output-artifact requirements.

### `subagents` and `forbidden_subagents`

These check which subagents were observed in the execution trace. Current production names
include:

- `idc-agent`
- `cohort-agent`
- `acquisition-agent`
- `segmentation-agent`
- `analysis-agent`
- `verifier-agent`

### `tools` and `forbidden_tools`

These check exact tool names recorded during the run. Use the registered tool name, such as
`idc_collection_profile`, `table_chart`, `idc_download`, or `radiomics`.

Only declare behavior that is necessary for the case. Overly specific routing expectations
can fail even when the user-visible result is correct.

## Automatic confirmations

Operations that request confirmation are denied unless the YAML explicitly approves them.
Approval values are case-insensitive; values such as `approve`, `allow`, `yes`, and `true`
are treated as approval.

To approve every download-style confirmation in a case:

```yaml
approvals:
  downloads: approve
```

The generic `downloads` key applies when the confirmation operation name ends in
`_download`. You may instead configure the exact operation name when a case needs narrower
control.

## Starter templates

### Metadata-only lookup

```yaml
experiment_id: metadata_lookup_v1
repetitions: 1
timeout_s: 900

cases:
  - id: idc_patient_count
    prompt: "Report the exact distinct patient count for CPTAC-LUAD in IDC."
    expected:
      subagents: [idc-agent]
      forbidden_subagents: [acquisition-agent]
```

### Generated visualization

```yaml
experiment_id: chart_workflow_v1
repetitions: 1
timeout_s: 1200

cases:
  - id: ct_sequence_chart
    prompt: >-
      Create and save a bar chart of CT SeriesDescription values in CPTAC-LUAD,
      using distinct patient count as the value.
    expected:
      artifacts: [plot]
      subagents: [idc-agent, analysis-agent]
      tools: [table_chart]
      forbidden_subagents: [acquisition-agent]
```

### Download requiring approval

```yaml
experiment_id: clinical_download_v1
repetitions: 1
timeout_s: 1800

cases:
  - id: download_clinical_table
    turns:
      - "Find the C4KC-KiTS collection in IDC and summarize it."
      - "Download its clinical data as a CSV."
    approvals:
      downloads: approve
    expected:
      artifacts: [csv]
```

### Uploaded-file workflow

```yaml
experiment_id: uploaded_table_v1
repetitions: 1
timeout_s: 900

cases:
  - id: inspect_uploaded_csv
    prompt: "Inspect the uploaded table and report its columns and missing-value counts."
    uploads:
      - ../../example_data/table.csv
    expected:
      subagents: [analysis-agent]
      tools: [tabular_inspection]
```

## Running selected cases

Run one case:

```bash
python -m evaluation.run \
  evaluation/cases/kidney_demo.yaml \
  --case c4kc_kits_sequence_plot
```

Override repetitions or timeout:

```bash
python -m evaluation.run \
  evaluation/cases/kidney_demo.yaml \
  --repetitions 1 \
  --timeout 1200
```

Resume an interrupted experiment:

```bash
python -m evaluation.run \
  evaluation/cases/kidney_demo.yaml \
  --resume
```

`--resume` skips an existing repetition whose `run.json` has `status: complete`. It does
not require that `deliverables_satisfied` is true.

Use a different output root:

```bash
python -m evaluation.run \
  evaluation/cases/kidney_demo.yaml \
  --output-root /path/to/evaluation_results
```

## Result files

A general experiment produces:

```text
evaluation_results/
  <experiment_id>/
    experiment.json
    summary.csv
    runtime/
      artifacts/
      state/
        checkpoints.sqlite
        registry.sqlite
    <case_id>/
      repetition_001/
        run.json
        trace.jsonl
        artifacts.json
```

- `experiment.json` records the source specification, Git state, model configuration, and
  pricing version.
- `run.json` records the final response, automatic checks, metrics, cost estimate,
  interactions, and errors for one repetition.
- `trace.jsonl` contains chronological LLM, subagent, tool, confirmation, and notification
  events.
- `artifacts.json` contains the final artifact and data registry records observed during
  the run.
- `summary.csv` contains one compact row per case repetition.

You can rebuild `summary.csv` from existing run files:

```bash
python -m evaluation.summarize \
  evaluation_results/<experiment_id>
```

## Metrics and estimated cost

The runner records:

- total duration and time to first token
- model, subagent, and tool call counts
- input, output, cached-input, cache-write, and reasoning tokens
- callback errors
- estimated LLM cost from `evaluation/pricing.yaml`

Pricing is a versioned snapshot rather than a live lookup. If usage is unavailable or a
model has no pricing entry, estimated cost is `null`; it is never silently reported as zero.

## What automatic evaluation does not prove

The general evaluator checks execution, routing, and registered deliverables. It does not
currently grade scientific or clinical correctness; `scientific_correctness` remains
`null` in general-run results.

For example, requiring a `plot` proves that a verified plot artifact exists. It does not by
itself prove that the plot used the correct cohort, modality, grouping, or statistical
definition. Encode important routing and tool constraints where helpful, inspect the trace
and artifacts, and use the verifier test kit or human review for semantic claims.

## Practical guidance

- Start with one repetition while developing a case, then increase repetitions to measure
  consistency.
- Give every case a stable, descriptive ID.
- Use multi-turn cases only when conversational state is part of what you are testing.
- Keep external downloads out of inexpensive calibration groups.
- Explicitly approve only the operations the test is intended to exercise.
- Prefer observable outcome checks over forcing one exact internal route.
- Version the experiment ID or YAML when changing prompts or expectations so results remain
  interpretable.
