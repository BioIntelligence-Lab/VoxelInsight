from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
from idc_index import IDCClient
from pydantic import BaseModel, Field

from core.state import TaskResult
from tools.shared import toolify_agent


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
    description="Download IDC clinical data by collection using idc_index (no BigQuery). Can limit columns and apply an equality filter.",
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

    frames: List[pd.DataFrame] = []
    for tbl in tables:
        try:
            df_tbl = _CLIENT.get_clinical_table(tbl)
            df_tbl["__source_table"] = tbl
            frames.append(df_tbl)
        except Exception as e:
            raise RuntimeError(f"Failed to load clinical table '{tbl}': {e}")

    if not frames:
        raise RuntimeError(f"Clinical tables for '{collection_id}' could not be loaded.")

    df = pd.concat(frames, ignore_index=True)

    if filter_field:
        if filter_field not in df.columns:
            raise ValueError(f"Field '{filter_field}' not found in clinical data.")
        if filter_value is not None:
            df = df[df[filter_field] == filter_value]

    if fields:
        missing = [f for f in fields if f not in df.columns]
        if missing:
            raise ValueError(f"Requested fields missing: {', '.join(missing)}")
        df = df[fields]

    df = df.head(limit_rows)

    tmp_dir = Path(tempfile.mkdtemp(prefix="clinical_"))
    csv_path = tmp_dir / f"{collection_id}_clinical.csv"
    df.to_csv(csv_path, index=False)

    summary = f"Clinical data downloaded via idc_index: collection={collection_id}, rows={len(df)}, columns={len(df.columns)}"
    outputs = {
        "text": summary,
        "tool": "clinical_data_download",
        "df_preview": {"rows": df.head(50).to_dict("records"), "nrows": len(df)},
    }
    artifacts = {"files": [str(csv_path)]}
    return TaskResult(output=outputs, artifacts=artifacts)
