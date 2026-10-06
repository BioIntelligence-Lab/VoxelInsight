from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from core.state import TaskResult
from tools.shared import toolify_agent
from tools.tabular_metadata import (
    demographic_metadata,
    infer_patient_id_column,
    resolve_column,
    table_schema,
)


class TabularInspectionArgs(BaseModel):
    file_path: str = Field(
        ...,
        description=(
            "Registered CSV, JSON, JSONL, or Parquet artifact to inspect. Pass its exact "
            "artifact_id or associated data_id; ArtifactRegistryMiddleware resolves either "
            "reference to the verified host path."
        ),
    )
    distribution_column: Optional[str] = Field(
        None,
        description=(
            "Optional categorical field to count. The alias 'sex' resolves to 'gender' "
            "when that is the stored column name."
        ),
    )
    unique_id_column: Optional[str] = Field(
        None,
        description=(
            "Optional patient/entity identifier used to deduplicate before counting. "
            "When omitted, a standard patient identifier is inferred when present."
        ),
    )
    columns: Optional[List[str]] = Field(
        None,
        description="Optional subset of columns to inspect.",
    )
    include_missing: bool = Field(
        True,
        description="Include missing distribution values as an 'Unknown' category.",
    )


def _load_table(path: Path) -> pd.DataFrame:
    lower_name = path.name.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if lower_name.endswith((".jsonl", ".ndjson")):
        return pd.read_json(path, lines=True)
    if lower_name.endswith(".json"):
        return pd.read_json(path)
    if lower_name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    raise ValueError(
        "Unsupported tabular artifact format. Expected CSV, JSON, JSONL, or Parquet."
    )


@toolify_agent(
    name="tabular_inspection",
    description=(
        "Inspect a registered tabular artifact through its artifact_id or associated data_id. Reads the resolved "
        "host file deterministically, returns schema and demographic metadata, and can "
        "produce a compact categorical distribution table for downstream table_chart use. "
        "Use this instead of DeepAgents read_file for CSV/JSON/Parquet artifacts."
    ),
    args_schema=TabularInspectionArgs,
    timeout_s=120,
)
async def tabular_inspection_runner(
    file_path: str,
    distribution_column: Optional[str] = None,
    unique_id_column: Optional[str] = None,
    columns: Optional[List[str]] = None,
    include_missing: bool = True,
):
    path = Path(file_path).expanduser()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Tabular artifact does not exist: {path}")

    dataframe = _load_table(path)
    if columns:
        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Requested columns missing: {', '.join(missing)}")
        dataframe = dataframe[columns].copy()

    demographics = demographic_metadata(dataframe)
    demographic_summary = pd.DataFrame(demographics)
    resolved_distribution_column = ""
    resolved_unique_id_column = ""
    distribution = None

    if distribution_column:
        resolved = resolve_column(dataframe, distribution_column)
        if resolved is None:
            raise ValueError(
                f"Distribution column '{distribution_column}' was not found."
            )
        resolved_distribution_column = resolved
        if unique_id_column:
            resolved_unique = resolve_column(dataframe, unique_id_column)
            if resolved_unique is None:
                raise ValueError(
                    f"Unique ID column '{unique_id_column}' was not found."
                )
        else:
            resolved_unique = infer_patient_id_column(dataframe)
        resolved_unique_id_column = resolved_unique or ""

        selected_columns = [resolved]
        if resolved_unique:
            selected_columns.insert(0, resolved_unique)
        distribution_source = dataframe[selected_columns].copy()
        if resolved_unique:
            distribution_source = distribution_source[
                distribution_source[resolved_unique].notna()
            ].drop_duplicates(subset=[resolved_unique], keep="first")

        values = distribution_source[resolved]
        if include_missing:
            values = values.astype("object").where(values.notna(), "Unknown")
        else:
            values = values.dropna()
        counts = values.astype(str).value_counts(dropna=False)
        distribution = pd.DataFrame(
            {
                resolved: [str(value) for value in counts.index],
                "count": [int(value) for value in counts.values],
            }
        )

    output = {
        "text": (
            f"Inspected tabular artifact: rows={len(dataframe)}, "
            f"columns={len(dataframe.columns)}"
        ),
        "row_count": int(len(dataframe)),
        "column_schema": table_schema(dataframe),
        "columns_with_missing": int(dataframe.isna().any().sum()),
        "missing_by_column": {
            str(column): int(count)
            for column, count in dataframe.isna().sum().sort_values(ascending=False).items()
            if count > 0
        },
        "demographic_fields": demographics,
        "resolved_distribution_column": resolved_distribution_column,
        "unique_id_column": resolved_unique_id_column,
        "demographic_summary": demographic_summary,
    }
    if distribution is not None:
        output["distribution"] = distribution
    return TaskResult(output=output)
