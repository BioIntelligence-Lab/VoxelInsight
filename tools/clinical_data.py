from __future__ import annotations

from typing import List, Optional

import pandas as pd
from idc_index import IDCClient
from pydantic import BaseModel, Field

from core.agents.artifacts import safe_slug
from core.storage import get_run_dir
from core.state import TaskResult
from tools.shared import toolify_agent
from tools.tabular_metadata import demographic_metadata, table_schema


class ClinicalDataArgs(BaseModel):
    collection_id: str = Field(..., description="IDC collection ID (e.g., tcga_brca).")
    fields: Optional[List[str]] = Field(
        None,
        description="Optional list of columns to keep. If omitted, all columns are returned.",
    )
    filter_field: Optional[str] = Field(
        None,
        description="Optional column to filter on (exact match).",
    )
    filter_value: Optional[str] = Field(
        None,
        description="Value to filter the filter_field by.",
    )
    limit_rows: int = Field(
        5000,
        ge=1,
        le=50000,
        description="Maximum rows to return (safety cap). Default 5000.",
    )


_CONFIGURED = False
_CLIENT: Optional[IDCClient] = None


def configure_clinical_data_tool():
    """Initialize the IDC client once."""
    global _CONFIGURED, _CLIENT
    if _CONFIGURED and _CLIENT is not None:
        return
    _CLIENT = IDCClient()
    _CLIENT.fetch_index("clinical_index")
    _CONFIGURED = True


def _available_tables_for_collection(collection_id: str) -> List[str]:
    if _CLIENT is None:
        raise RuntimeError("IDC client not initialized.")
    df = _CLIENT.clinical_index
    if "collection_id" not in df.columns:
        return []
    subset = df[df["collection_id"] == collection_id]
    if subset.empty:
        return []
    if "short_table_name" in subset.columns:
        tables = subset["short_table_name"].dropna().unique().tolist()
    elif "table_name" in subset.columns:
        tables = subset["table_name"].dropna().unique().tolist()
    else:
        tables = []
    return [t for t in tables if t]


@toolify_agent(
    name="clinical_data_download",
    description=(
        "Download IDC clinical data by collection using idc_index (no BigQuery). Can limit columns and apply an equality filter. Provides a CSV file download link in UI."
        "\nWhen the user requests downloads of DICOM series, histopathology tiles, or clinical data, use the respective download tools (`idc_download`, `pathology_download`, `clinical_data_download`)."
        "\n- `clinical_data_download`: download IDC clinical data by collection using idc_index (no BigQuery). Optionally select fields and/or filter on a field value."
        "\n- Use this tool to download clinical data tables from IDC for patients of interest."
        "\n- When an exact collection_id is supplied, use it directly without running a separate collection lookup."
        "\n- Collection discovery for partial names or descriptions must be completed before calling this tool."
    ),
    args_schema=ClinicalDataArgs,
    timeout_s=180,
)
async def clinical_data_download_runner(
    collection_id: str,
    fields: Optional[List[str]] = None,
    filter_field: Optional[str] = None,
    filter_value: Optional[str] = None,
    limit_rows: int = 5000,
):
    if not _CONFIGURED or _CLIENT is None:
        configure_clinical_data_tool()

    tables = _available_tables_for_collection(collection_id)
    if not tables:
        raise ValueError(f"No clinical tables found for collection '{collection_id}'.")

    loaded_tables: List[tuple[str, pd.DataFrame]] = []
    for tbl in tables:
        try:
            df_tbl = _CLIENT.get_clinical_table(tbl).copy()
            loaded_tables.append((str(tbl), df_tbl))
        except Exception as e:
            raise RuntimeError(f"Failed to load clinical table '{tbl}': {e}")

    if not loaded_tables:
        raise RuntimeError(f"Clinical tables for '{collection_id}' could not be loaded.")

    available_fields = {
        str(column)
        for _table_name, dataframe in loaded_tables
        for column in dataframe.columns
    }

    if filter_field:
        if filter_field not in available_fields:
            raise ValueError(f"Field '{filter_field}' not found in clinical data.")

    if fields:
        missing = [f for f in fields if f not in available_fields]
        if missing:
            raise ValueError(f"Requested fields missing: {', '.join(missing)}")

    out_root = get_run_dir("clinical_data_download", persist=True)
    table_data = []
    table_metadata = []
    csv_paths: List[str] = []
    remaining_rows = int(limit_rows)
    for table_name, source_dataframe in loaded_tables:
        dataframe = source_dataframe
        if filter_field:
            if filter_field not in dataframe.columns:
                continue
            if filter_value is not None:
                dataframe = dataframe[dataframe[filter_field] == filter_value]
        if fields:
            selected_fields = [field for field in fields if field in dataframe.columns]
            if not selected_fields:
                continue
            dataframe = dataframe[selected_fields]

        source_rows = int(len(dataframe))
        dataframe = dataframe.head(remaining_rows).copy()
        remaining_rows -= len(dataframe)
        if dataframe.empty and source_rows:
            break

        csv_path = out_root / f"{safe_slug(table_name, 'clinical-table')}.csv"
        dataframe.to_csv(csv_path, index=False)
        csv_paths.append(str(csv_path))
        demographics = demographic_metadata(dataframe)
        schema = table_schema(dataframe)
        metadata = {
            "source_table": table_name,
            "source_rows": source_rows,
            "returned_rows": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "schema": schema,
            "demographic_fields": demographics,
        }
        table_metadata.append(metadata)
        table_data.append(
            {
                "name": table_name,
                "dataframe": dataframe,
                "artifact_path": str(csv_path),
                "visibility": "user",
                "metadata": metadata,
            }
        )
        if remaining_rows <= 0:
            break

    if not table_data:
        raise RuntimeError(
            f"Clinical tables for '{collection_id}' contained no rows after filtering."
        )

    total_rows = sum(int(item["returned_rows"]) for item in table_metadata)
    summary = (
        "Clinical data downloaded via idc_index: "
        f"collection={collection_id}, tables={len(table_data)}, rows={total_rows}"
    )
    outputs = {
        "text": summary,
        "tool": "clinical_data_download",
        "collection_id": collection_id,
        "source_tables": [item["source_table"] for item in table_metadata],
        "tables": table_metadata,
        "table_data": table_data,
    }
    artifacts = {"files": csv_paths}
    return TaskResult(output=outputs, artifacts=artifacts)
