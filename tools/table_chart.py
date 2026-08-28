from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, field_validator

from core.state import TaskResult
from tools.shared import toolify_agent


CHART_TYPES = {"bar", "line", "scatter"}
ORIENTATIONS = {"vertical", "horizontal"}
SORT_DIRECTIONS = {"asc", "desc"}


def _normalized_sort_direction(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    aliases = {
        "": "desc",
        "ascending": "asc",
        "ascending order": "asc",
        "descending": "desc",
        "descending order": "desc",
    }
    return aliases.get(normalized, normalized)


class TableChartArgs(BaseModel):
    rows: str = Field(
        ...,
        description=(
            "Machine-readable table rows as a JSON string encoding a list of JSON "
            "objects, e.g. '[{\"collection\": \"alpha\", \"patients\": 12}]'. "
            "Use an empty string when file_path supplies the complete table."
        ),
    )
    file_path: str = Field(
        default="",
        description=(
            "Optional registered CSV, JSON, JSONL, or Parquet artifact/data reference. "
            "Use this when registry rows are only a bounded preview; middleware resolves "
            "the reference to its verified file."
        ),
    )
    source_data_id: str = Field(
        ...,
        description=(
            "Data registry ID that supplied these rows. Use an empty string only when "
            "the rows came directly from the user's request and no data_id exists."
        ),
    )
    chart_type: Literal["bar", "line", "scatter"] = Field(
        ...,
        description="Chart type: bar, line, or scatter.",
    )
    category_column: str = Field(
        ...,
        description=(
            "Column containing category/group labels. This remains the category column "
            "for both vertical and horizontal bars."
        ),
    )
    value_column: str = Field(
        ...,
        description=(
            "Column containing numeric values. This remains the value column for both "
            "vertical and horizontal bars."
        ),
    )
    title: str = Field(..., description="Chart title. Use an empty string for the default title.")
    category_axis_title: str = Field(
        ...,
        description="Category-axis label. Use an empty string for the category column name.",
    )
    value_axis_title: str = Field(
        ...,
        description="Value-axis label. Use an empty string for the value column name.",
    )
    orientation: Literal["vertical", "horizontal"] = Field(
        ...,
        description="Bar orientation: vertical or horizontal. Use vertical for line/scatter.",
    )
    sort_by: str = Field(
        ...,
        description="Column to sort rows by before plotting. Use an empty string for no sorting.",
    )
    sort_direction: Literal["asc", "desc"] = Field(..., description="Sort direction: asc or desc.")
    limit: int = Field(
        ...,
        description="Maximum number of rows to plot. Use 0 for no limit.",
    )
    show_value_labels: bool = Field(
        ...,
        description="Whether to show y/value labels on bars.",
    )
    tick_angle: int = Field(
        ...,
        description="X-axis tick angle in degrees. Use 0 for the default angle.",
    )

    @field_validator("sort_direction", mode="before")
    @classmethod
    def _normalize_sort_direction(cls, value: Any) -> Any:
        return _normalized_sort_direction(value)


def _parse_rows(rows: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(rows)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"rows must be a JSON string encoding a list of objects: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError("rows must be a JSON string encoding a list of objects.")
    return parsed


def _validate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("rows must contain at least one object.")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("rows must be a list of JSON objects.")
    return pd.DataFrame(rows)


def _load_table_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path).expanduser()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Chart source artifact does not exist: {path}")
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
        "Unsupported chart source artifact. Expected CSV, JSON, JSONL, or Parquet."
    )


def _chart_dataframe(rows: str, file_path: str) -> pd.DataFrame:
    if rows.strip():
        return _validate_rows(_parse_rows(rows))
    if file_path.strip():
        dataframe = _load_table_file(file_path)
        if dataframe.empty:
            raise ValueError("Chart source artifact contains no rows.")
        return dataframe
    raise ValueError("Provide non-empty rows or a registered file_path.")


def _require_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _validate_options(chart_type: str, orientation: str, sort_direction: str) -> None:
    if chart_type not in CHART_TYPES:
        raise ValueError(f"chart_type must be one of: {', '.join(sorted(CHART_TYPES))}")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of: {', '.join(sorted(ORIENTATIONS))}")
    if sort_direction not in SORT_DIRECTIONS:
        raise ValueError(f"sort_direction must be one of: {', '.join(sorted(SORT_DIRECTIONS))}")


def _clean_limit(limit: int) -> int | None:
    limit_int = int(limit)
    if limit_int < 0:
        raise ValueError("limit must be 0 or a positive integer.")
    if limit_int == 0:
        return None
    return limit_int


def _numeric_values(df: pd.DataFrame, value_column: str) -> pd.Series:
    numeric = pd.to_numeric(df[value_column], errors="coerce")
    invalid = numeric.isna()
    if invalid.any():
        examples = [
            str(value)
            for value in df.loc[invalid, value_column].head(3).tolist()
        ]
        raise ValueError(
            f"value_column '{value_column}' must contain only numeric values; "
            f"invalid value(s): {', '.join(examples)}"
        )
    return numeric


def _build_chart(
    df: pd.DataFrame,
    *,
    chart_type: str,
    category_column: str,
    value_column: str,
    title: str,
    category_axis_title: str,
    value_axis_title: str,
    orientation: str,
    show_value_labels: bool,
    tick_angle: int,
) -> go.Figure:
    if chart_type == "bar":
        if orientation == "horizontal":
            text = df[value_column].astype(str).tolist() if show_value_labels else None
            fig = go.Figure(
                go.Bar(
                    x=df[value_column].tolist(),
                    y=df[category_column].astype(str).tolist(),
                    orientation="h",
                    text=text,
                    textposition="outside" if text else None,
                )
            )
            fig.update_layout(
                xaxis_title=value_axis_title or value_column.replace("_", " "),
                yaxis_title=category_axis_title or category_column.replace("_", " "),
                yaxis={"automargin": True, "autorange": "reversed"},
            )
        else:
            text = df[value_column].astype(str).tolist() if show_value_labels else None
            fig = go.Figure(
                go.Bar(
                    x=df[category_column].astype(str).tolist(),
                    y=df[value_column].tolist(),
                    text=text,
                    textposition="outside" if text else None,
                )
            )
            fig.update_layout(
                xaxis_title=category_axis_title or category_column.replace("_", " "),
                yaxis_title=value_axis_title or value_column.replace("_", " "),
            )
            if tick_angle:
                fig.update_xaxes(tickangle=int(tick_angle), automargin=True)
    elif chart_type == "line":
        fig = go.Figure(
            go.Scatter(
                x=df[category_column].tolist(),
                y=df[value_column].tolist(),
                mode="lines+markers",
            )
        )
        fig.update_layout(
            xaxis_title=category_axis_title or category_column.replace("_", " "),
            yaxis_title=value_axis_title or value_column.replace("_", " "),
        )
        if tick_angle:
            fig.update_xaxes(tickangle=int(tick_angle), automargin=True)
    elif chart_type == "scatter":
        fig = go.Figure(
            go.Scatter(
                x=df[category_column].tolist(),
                y=df[value_column].tolist(),
                mode="markers",
            )
        )
        fig.update_layout(
            xaxis_title=category_axis_title or category_column.replace("_", " "),
            yaxis_title=value_axis_title or value_column.replace("_", " "),
        )
        if tick_angle:
            fig.update_xaxes(tickangle=int(tick_angle), automargin=True)
    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    fig.update_layout(
        title=title or f"{value_column} by {category_column}",
        margin={"l": 80, "r": 40, "t": 70, "b": 120},
    )
    return fig


@toolify_agent(
    name="table_chart",
    description=(
        "Create deterministic Plotly charts from explicit table rows or a registered tabular "
        "artifact. Use this for bar, line, and scatter charts when the data are already known. "
        "The tool does not query data sources or infer hidden context; pass rows as JSON, or "
        "pass file_path as an artifact/data ID when inline registry rows are a bounded preview. "
        "Also pass category/value columns, sorting, labels, and title explicitly. "
        "category_column always contains labels and value_column always contains numeric "
        "values, regardless of bar orientation."
    ),
    args_schema=TableChartArgs,
    timeout_s=60,
)
async def table_chart_runner(
    rows: str,
    file_path: str = "",
    source_data_id: str = "",
    chart_type: str = "bar",
    category_column: str = "",
    value_column: str = "",
    title: str = "",
    category_axis_title: str = "",
    value_axis_title: str = "",
    orientation: str = "vertical",
    sort_by: str = "",
    sort_direction: str = "desc",
    limit: int = 0,
    show_value_labels: bool = False,
    tick_angle: int = 0,
):
    sort_direction = _normalized_sort_direction(sort_direction)
    _validate_options(chart_type, orientation, sort_direction)
    df = _chart_dataframe(rows, file_path)
    required = [category_column, value_column] + ([sort_by] if sort_by else [])
    _require_columns(df, [column for column in required if column])
    if not category_column or not value_column:
        raise ValueError(
            "Both category_column and value_column are required."
        )

    df = df.copy()
    df[value_column] = _numeric_values(df, value_column)

    if sort_by:
        ascending = sort_direction == "asc"
        df = df.sort_values(sort_by, ascending=ascending, kind="mergesort")

    clean_limit = _clean_limit(limit)
    if clean_limit is not None:
        df = df.head(clean_limit)

    fig = _build_chart(
        df,
        chart_type=chart_type,
        category_column=category_column,
        value_column=value_column,
        title=title,
        category_axis_title=category_axis_title,
        value_axis_title=value_axis_title,
        orientation=orientation,
        show_value_labels=show_value_labels,
        tick_angle=tick_angle,
    )

    return TaskResult(
        output={
            "figure": fig,
            "summary": f"Rendered {chart_type} chart from {len(df)} row(s).",
            "source_data_id": source_data_id,
        },
        artifacts={
            "chart_type": chart_type,
            "row_count": len(df),
            "category_column": category_column,
            "value_column": value_column,
            "source_data_id": source_data_id,
        },
    )
