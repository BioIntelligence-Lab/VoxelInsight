from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from core.state import TaskResult
from core.agents.external_links import validated_idc_viewer_url
from tools.idc_python_worker import validate_restricted_idc_code
from tools.shared import toolify_agent


OFFICIAL_SKILL_VERSION = "1.6.5"
OFFICIAL_IDC_INDEX_MIN_VERSION = "0.12.3"
DEFAULT_MAX_ROWS = 200
ABSOLUTE_MAX_ROWS = 500
MAX_MANIFEST_SERIES = 20_000

_IDC_CLIENT: Optional[Any] = None

_SQL_FORBIDDEN = re.compile(
    r"\b(ALTER|ATTACH|CALL|COPY|CREATE|DELETE|DETACH|DROP|EXPORT|IMPORT|INSERT|"
    r"INSTALL|LOAD|MERGE|PRAGMA|REPLACE|TRUNCATE|UPDATE|VACUUM)\b",
    re.IGNORECASE,
)
_SQL_EXTERNAL_IO = re.compile(
    r"\b(csv_scan|delta_scan|http_get|httpfs|iceberg_scan|json_scan|parquet_scan|"
    r"read_avro|read_blob|read_csv|read_json|read_ndjson|read_parquet|read_text|"
    r"read_xlsx|sqlite_scan|postgres_scan|mysql_scan)\s*\(",
    re.IGNORECASE,
)
_SQL_TABLE_REF = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_SQL_QUOTED_SOURCE = re.compile(r"\b(?:FROM|JOIN)\s*''", re.IGNORECASE)
_SQL_TABLE_FUNCTION = re.compile(
    r"\b(?:FROM|JOIN)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
    re.IGNORECASE,
)

_GENERIC_DISCOVERY_WORDS = {
    "collection",
    "collections",
    "data",
    "dataset",
    "datasets",
    "exam",
    "exams",
    "image",
    "images",
    "imaging",
    "scan",
    "scans",
    "series",
    "study",
    "studies",
}
_MODALITY_WORDS = {
    "ct": {"ct", "computed", "tomography"},
    "mr": {"mr", "mri", "magnetic", "resonance"},
    "pt": {"pet", "pt", "positron", "emission"},
    "nm": {"nm", "nuclear", "medicine"},
    "sm": {"microscopy", "pathology", "slide", "sm"},
}
_ANATOMY_ALIASES = (
    {"kidney", "nephro", "renal"},
    {"lung", "pulmonary"},
    {"liver", "hepatic"},
    {"brain", "cerebral", "intracranial"},
    {"breast", "mammary"},
    {"pancreas", "pancreatic"},
    {"prostate", "prostatic"},
    {"colon", "colorectal"},
)
_DERIVED_NON_ACQUISITION_MODALITIES = {
    "DOC",
    "KO",
    "PR",
    "REG",
    "RTDOSE",
    "RTPLAN",
    "RTSTRUCT",
    "SEG",
    "SR",
}


def configure_idc_specialist_tools(*, client: Any) -> None:
    global _IDC_CLIENT
    _IDC_CLIENT = client


def _client() -> Any:
    if _IDC_CLIENT is None:
        raise RuntimeError("IDC specialist tools are not configured.")
    return _IDC_CLIENT


def _idc_version(client: Any) -> str:
    try:
        return str(client.get_idc_version())
    except Exception:
        return str(getattr(client, "idc_version", "unknown"))


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _semantic_search_terms(text: str, modality: str = "") -> list[str]:
    """Normalize natural discovery text into deterministic metadata search terms."""
    tokens = [token.casefold() for token in re.findall(r"[A-Za-z0-9]+", text)]
    modality_tokens: set[str] = set()
    normalized_modality = modality.strip().casefold()
    for modality_name, aliases in _MODALITY_WORDS.items():
        if normalized_modality in aliases or normalized_modality == modality_name:
            modality_tokens.update(aliases)
    meaningful = [
        token
        for token in tokens
        if token not in _GENERIC_DISCOVERY_WORDS and token not in modality_tokens
    ]
    if not meaningful:
        return []

    terms: list[str] = [" ".join(meaningful)]
    raw_normalized = text.strip().casefold()
    if "_" in raw_normalized or "-" in raw_normalized:
        terms.append(raw_normalized)
    meaningful_set = set(meaningful)
    for aliases in _ANATOMY_ALIASES:
        if meaningful_set & aliases:
            terms.extend(sorted(aliases))
    return list(dict.fromkeys(term for term in terms if term))


def _semantic_term_pattern(term: str) -> str:
    """Build a boundary-safe pattern with equivalent common ID separators."""

    pieces = re.findall(r"[a-z0-9]+", term.casefold())
    escaped = r"[-_\s]+".join(re.escape(piece) for piece in pieces)
    if term.casefold() == "nephro":
        escaped = r"nephro[a-z0-9]*"
    return rf"(^|[^a-z0-9]){escaped}([^a-z0-9]|$)"


def _semantic_match_sql(columns: list[str], terms: list[str]) -> str:
    def pattern(term: str) -> str:
        return _semantic_term_pattern(term)

    return "(" + " OR ".join(
        f"regexp_matches(LOWER(COALESCE(CAST({column} AS VARCHAR), '')), {_sql_literal(pattern(term))})"
        for term in terms
        for column in columns
    ) + ")"


def _collection_id_filter_sql(column: str, collection_id: str) -> str:
    """Match collection IDs exactly while ignoring case and common separators."""

    normalized = "_".join(re.findall(r"[a-z0-9]+", collection_id.casefold()))
    if not normalized:
        raise ValueError("collection_id must contain at least one letter or number.")
    separator_pattern = _sql_literal(r"[-_\s]+")
    normalized_column = (
        f"regexp_replace(LOWER(TRIM(CAST({column} AS VARCHAR))), "
        f"{separator_pattern}, '_', 'g')"
    )
    return f"{normalized_column} = {_sql_literal(normalized)}"


def _strip_sql_strings_and_comments(sql: str) -> str:
    without_comments = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"'(?:''|[^'])*'", "''", without_comments)


def validate_readonly_sql(sql: str, available_tables: set[str]) -> tuple[str, set[str]]:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a non-empty string.")
    normalized = sql.strip()
    if len(normalized) > 50_000:
        raise ValueError("SQL exceeds the 50,000-character limit.")
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    scrubbed = _strip_sql_strings_and_comments(normalized)
    if ";" in scrubbed:
        raise ValueError("Only one SQL statement is allowed.")
    if not re.match(r"^\s*(SELECT|WITH)\b", scrubbed, flags=re.IGNORECASE):
        raise ValueError("Only read-only SELECT or WITH queries are allowed.")
    if _SQL_FORBIDDEN.search(scrubbed):
        raise ValueError("SQL contains a forbidden mutating or administrative statement.")
    if _SQL_EXTERNAL_IO.search(scrubbed):
        raise ValueError("SQL functions that read external files or URLs are not allowed.")
    if _SQL_QUOTED_SOURCE.search(scrubbed):
        raise ValueError("Quoted file/URL sources in FROM or JOIN are not allowed.")
    if _SQL_TABLE_FUNCTION.search(scrubbed):
        raise ValueError("SQL table functions in FROM or JOIN are not allowed.")

    referenced = {match.lower() for match in _SQL_TABLE_REF.findall(scrubbed)}
    cte_names = {
        name.lower()
        for name in re.findall(r"(?:\bWITH|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", scrubbed, re.IGNORECASE)
    }
    physical_tables = referenced - cte_names
    allowed_lower = {table.lower() for table in available_tables}
    unknown = sorted(physical_tables - allowed_lower)
    if unknown:
        raise ValueError(f"SQL references unavailable IDC tables: {', '.join(unknown)}")
    return normalized, physical_tables


def _available_tables(client: Any) -> set[str]:
    overview = getattr(client, "indices_overview", {})
    tables = set(overview.keys()) if isinstance(overview, dict) else set()
    for table_name in ("collections_index", "clinical_index"):
        if getattr(client, table_name, None) is not None:
            tables.add(table_name)
    tables.update({"index", "prior_versions_index"})
    tables.update(_clinical_table_names(client))
    return tables


def _clinical_table_names(client: Any) -> set[str]:
    """Short names of clinical tables, available once clinical_index is loaded."""
    clinical = getattr(client, "clinical_index", None)
    if not isinstance(clinical, pd.DataFrame) or "short_table_name" not in clinical.columns:
        return set()
    return {str(name) for name in clinical["short_table_name"].dropna().unique()}


def _get_index_schema(client: Any, table_name: str) -> dict[str, Any]:
    """Return a table schema across old and current idc-index releases."""
    getter = getattr(client, "get_index_schema", None)
    if callable(getter):
        try:
            schema = getter(table_name)
            if isinstance(schema, dict):
                return schema
        except Exception:
            pass

    overview = getattr(client, "indices_overview", {})
    if isinstance(overview, dict):
        info = overview.get(table_name)
        if isinstance(info, dict) and isinstance(info.get("schema"), dict):
            return info["schema"]

    dataframe = getattr(client, table_name, None)
    if isinstance(dataframe, pd.DataFrame):
        return {
            "columns": [
                {"name": str(column), "type": str(dataframe[column].dtype)}
                for column in dataframe.columns
            ]
        }
    if table_name == "index" and isinstance(getattr(client, "index", None), pd.DataFrame):
        dataframe = client.index
        return {
            "columns": [
                {"name": str(column), "type": str(dataframe[column].dtype)}
                for column in dataframe.columns
            ]
        }
    return {}


def _schema_column_names(client: Any, table_name: str) -> set[str]:
    schema = _get_index_schema(client, table_name)
    names = {
        str(item.get("name"))
        for item in schema.get("columns", [])
        if isinstance(item, dict) and item.get("name")
    }
    dataframe = getattr(client, table_name, None)
    if isinstance(dataframe, pd.DataFrame):
        names.update(str(column) for column in dataframe.columns)
    return names


def _canonical_table_name(client: Any, lower_name: str) -> str:
    for name in _available_tables(client):
        if name.lower() == lower_name:
            return name
    return lower_name


def _prepare_tables(client: Any, tables: set[str]) -> None:
    clinical_tables = {name.lower(): name for name in _clinical_table_names(client)}
    for lower_name in sorted(tables - {"index", "prior_versions_index"}):
        if lower_name in clinical_tables:
            table_name = clinical_tables[lower_name]
            frame = client.get_clinical_table(table_name)
            connection = getattr(client, "_duckdb_conn", None)
            if frame is None or connection is None:
                raise RuntimeError(f"IDC clinical table {table_name!r} could not be loaded.")
            connection.register(table_name, frame)
            continue
        table_name = _canonical_table_name(client, lower_name)
        client.fetch_index(table_name)
        if getattr(client, table_name, None) is None:
            raise RuntimeError(f"IDC table {table_name!r} could not be loaded.")


def _execute_readonly_sql(
    client: Any,
    sql: str,
    max_rows: int,
    *,
    absolute_max_rows: int = ABSOLUTE_MAX_ROWS,
) -> tuple[pd.DataFrame, str, list[str]]:
    validated, tables = validate_readonly_sql(sql, _available_tables(client))
    _prepare_tables(client, tables)
    bounded_rows = max(1, min(int(max_rows), int(absolute_max_rows)))
    bounded_sql = f"SELECT * FROM ({validated}) AS voxelinsight_idc_result LIMIT {bounded_rows}"
    frame = client.sql_query(bounded_sql)
    return frame, bounded_sql, sorted(tables)


def _expected_columns(value: str) -> list[str]:
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("expected_columns must be a JSON array of column names.") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("expected_columns must be a JSON array of column names.")
    return parsed


def _result(
    frame: pd.DataFrame,
    *,
    query: str,
    tables: list[str],
    expected_columns: list[str] | None = None,
    warnings: list[str] | None = None,
    partial: bool = False,
) -> TaskResult:
    expected = expected_columns or []
    missing = [column for column in expected if column not in frame.columns]
    validation = {
        "row_count": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "expected_columns": expected,
        "missing_expected_columns": missing,
        "row_limit_applied": len(frame) >= ABSOLUTE_MAX_ROWS,
    }
    errors = [f"Missing expected output columns: {', '.join(missing)}"] if missing else []
    return TaskResult(
        output={
            "result": frame,
            "query": query,
            "tables": tables,
            "counts": {"returned_rows": int(len(frame))},
            "validation": validation,
            "provenance": {
                "source": "NCI Imaging Data Commons",
                "idc_data_version": _idc_version(_client()),
                "skill_version": OFFICIAL_SKILL_VERSION,
                "idc_index_min_version": OFFICIAL_IDC_INDEX_MIN_VERSION,
            },
            "warnings": warnings or [],
        },
        status="partial" if errors or partial else "ok",
        errors=errors,
    )


class IDCSchemaArgs(BaseModel):
    table_names: str = Field(
        ...,
        description="Comma-separated IDC index table names. Use an empty string to list every table.",
    )
    include_columns: bool = Field(..., description="Whether to include column names, types, and descriptions.")


@toolify_agent(
    name="idc_schema",
    description=(
        "Inspect authoritative idc-index table descriptions and schemas before composing queries. "
        "This is the preferred first tool when columns or join keys are uncertain."
    ),
    args_schema=IDCSchemaArgs,
    timeout_s=120,
)
async def idc_schema_runner(table_names: str, include_columns: bool) -> TaskResult:
    client = _client()
    overview = getattr(client, "indices_overview", {})
    requested = [name.strip() for name in table_names.split(",") if name.strip()]
    selected = requested or sorted(overview)
    rows: list[dict[str, Any]] = []
    for table_name in selected:
        if table_name not in overview:
            raise ValueError(f"Unknown IDC index table: {table_name}")
        info = overview[table_name]
        row: dict[str, Any] = {
            "table_name": table_name,
            "description": info.get("description", ""),
            "installed": bool(info.get("installed")),
        }
        if include_columns:
            row["columns"] = info.get("schema", {}).get("columns", [])
        rows.append(row)
    frame = pd.DataFrame(rows)
    return _result(frame, query="indices_overview metadata lookup", tables=selected)


class IDCCollectionSearchArgs(BaseModel):
    search_text: str = Field(..., description="Case-insensitive text to match in curated collection metadata; empty for all.")
    modality: str = Field(..., description="Exact DICOM Modality filter such as CT or MR; empty for any.")
    body_part: str = Field(
        ...,
        description=(
            "Semantic anatomy term such as kidney, renal, chest, or lung. It is matched "
            "case-insensitively against curated tumor locations and DICOM BodyPartExamined; "
            "empty for any anatomy. Do not assume it is an exact DICOM value."
        ),
    )
    limit: int = Field(
        ...,
        ge=1,
        le=500,
        description=(
            "Maximum collection rows. Use at least 100 when the user asks which/what/all "
            "datasets without an explicit numeric cap; use a smaller value only when requested."
        ),
    )


@toolify_agent(
    name="idc_collection_search",
    description="Find IDC collections using typed text, modality, and anatomy filters with patient/series counts.",
    args_schema=IDCCollectionSearchArgs,
    timeout_s=180,
)
async def idc_collection_search_runner(
    search_text: str,
    modality: str,
    body_part: str,
    limit: int,
) -> TaskResult:
    client = _client()
    curated_columns: set[str] = set()
    curated_available = "collections_index" in _available_tables(client)
    if curated_available:
        try:
            client.fetch_index("collections_index")
            curated_columns = _schema_column_names(client, "collections_index")
            curated_available = (
                "collection_id" in curated_columns
                and getattr(client, "collections_index", None) is not None
            )
        except Exception:
            curated_available = False

    descriptive_columns = [
        column
        for column in (
            "collection_title",
            "cancer_types",
            "tumor_locations",
            "species",
            "supporting_data",
        )
        if column in curated_columns
    ]
    filters = []
    search_terms = _semantic_search_terms(search_text, modality)
    if search_terms:
        if curated_available:
            search_columns = [
                "c.collection_id",
                *[f"c.{column}" for column in descriptive_columns],
            ]
        else:
            index_columns = set(getattr(client.index, "columns", []))
            search_columns = [
                f"i.{column}"
                for column in (
                    "collection_id",
                    "BodyPartExamined",
                    "StudyDescription",
                    "SeriesDescription",
                )
                if column in index_columns
            ]
        filters.append(_semantic_match_sql(search_columns, search_terms))
    if modality.strip():
        filters.append(f"i.Modality = {_sql_literal(modality.strip())}")
    body_terms = _semantic_search_terms(body_part, modality)
    if body_terms:
        anatomy_columns = ["i.BodyPartExamined"]
        if curated_available and "tumor_locations" in curated_columns:
            anatomy_columns.append("c.tumor_locations")
        filters.append(_semantic_match_sql(anatomy_columns, body_terms))
    where = " WHERE " + " AND ".join(filters) if filters else ""
    curated_select = (
        "".join(f", c.{column}" for column in descriptive_columns)
        if curated_available
        else ""
    )
    curated_group = (
        "".join(f", c.{column}" for column in descriptive_columns)
        if curated_available
        else ""
    )
    collection_column = "c.collection_id" if curated_available else "i.collection_id"
    collection_source = (
        "FROM index i JOIN collections_index c ON i.collection_id = c.collection_id"
        if curated_available
        else "FROM index i"
    )
    sql = f"""
        SELECT {collection_column} AS collection_id
               {curated_select},
               COUNT(DISTINCT i.PatientID) AS patient_count,
               COUNT(DISTINCT i.StudyInstanceUID) AS study_count,
               COUNT(DISTINCT i.SeriesInstanceUID) AS series_count,
               STRING_AGG(DISTINCT i.Modality, ', ' ORDER BY i.Modality) AS modalities,
               ROUND(SUM(i.series_size_MB) / 1000.0, 3) AS size_gb
        {collection_source}
        {where}
        GROUP BY {collection_column} {curated_group}
        ORDER BY patient_count DESC, collection_id
        LIMIT {int(limit)}
    """
    frame, query, tables = _execute_readonly_sql(client, sql, limit)
    warnings: list[str] = []
    partial = False
    runtime_version = _idc_version(client)
    if runtime_version != "v24":
        warnings.append(
            f"Runtime IDC data version is {runtime_version}, while the attached official skill "
            "targets v24. Restart VoxelInsight after upgrading idc-index/idc-index-data."
        )
        partial = True
    if (search_text.strip() or body_part.strip()) and not curated_available:
        warnings.append(
            "collections_index was unavailable, so semantic discovery fell back to inconsistent "
            "DICOM collection/body-part/description fields. A zero-row result is not evidence "
            "that no relevant IDC collection exists."
        )
        partial = True
    normalized_terms = list(dict.fromkeys([*search_terms, *body_terms]))
    raw_terms = [value.strip().casefold() for value in (search_text, body_part) if value.strip()]
    if normalized_terms and any(term not in normalized_terms for term in raw_terms):
        warnings.append(
            "Semantic discovery normalized the supplied text to: "
            + ", ".join(normalized_terms)
            + ". Modality and generic scan/dataset words are filtered separately."
        )
    if len(frame) >= limit:
        warnings.append(
            f"The collection result reached limit={limit} and may be truncated. Increase the "
            "limit for an exhaustive dataset list unless the user explicitly requested this cap."
        )
    return _result(
        frame,
        query=query,
        tables=tables,
        expected_columns=["collection_id", "patient_count"],
        warnings=warnings,
        partial=partial,
    )


class IDCCollectionSummaryArgs(BaseModel):
    collection_id: str = Field(..., min_length=1, description="Exact IDC collection_id.")
    modality: str = Field(..., description="Optional exact DICOM Modality filter; empty for all modalities.")
    include_demographics: bool = Field(..., description="Include de-duplicated DICOM PatientSex and PatientAge summaries.")


@toolify_agent(
    name="idc_collection_summary",
    description=(
        "Summarize one IDC collection with distinct patient/study/series counts, modalities, size, "
        "and optional de-duplicated DICOM demographics."
    ),
    args_schema=IDCCollectionSummaryArgs,
    timeout_s=180,
)
async def idc_collection_summary_runner(
    collection_id: str,
    modality: str,
    include_demographics: bool,
) -> TaskResult:
    filters = [_collection_id_filter_sql("collection_id", collection_id)]
    if modality.strip():
        filters.append(f"Modality = {_sql_literal(modality.strip())}")
    where = " AND ".join(filters)
    demographics = ""
    demographic_columns = ""
    warnings: list[str] = []
    if include_demographics:
        demographics = f""",
        patient_level AS (
            SELECT PatientID,
                   MAX(NULLIF(TRIM(PatientSex), '')) AS patient_sex,
                   MAX(
                       CASE RIGHT(TRIM(PatientAge), 1)
                           WHEN 'Y' THEN TRY_CAST(LEFT(TRIM(PatientAge), LENGTH(TRIM(PatientAge))-1) AS DOUBLE)
                           WHEN 'M' THEN TRY_CAST(LEFT(TRIM(PatientAge), LENGTH(TRIM(PatientAge))-1) AS DOUBLE) / 12.0
                           WHEN 'W' THEN TRY_CAST(LEFT(TRIM(PatientAge), LENGTH(TRIM(PatientAge))-1) AS DOUBLE) / 52.1429
                           WHEN 'D' THEN TRY_CAST(LEFT(TRIM(PatientAge), LENGTH(TRIM(PatientAge))-1) AS DOUBLE) / 365.25
                           ELSE TRY_CAST(TRIM(PatientAge) AS DOUBLE)
                       END
                   ) AS age_years
            FROM filtered
            GROUP BY PatientID
        ),
        demographic_summary AS (
            SELECT COUNT(*) FILTER (WHERE UPPER(patient_sex) = 'M') AS male_patients,
                   COUNT(*) FILTER (WHERE UPPER(patient_sex) = 'F') AS female_patients,
                   COUNT(*) FILTER (WHERE patient_sex IS NULL OR UPPER(patient_sex) NOT IN ('M','F')) AS sex_unknown_or_other,
                   COUNT(age_years) AS patients_with_age,
                   ROUND(AVG(age_years), 2) AS average_age_years,
                   ROUND(MEDIAN(age_years), 2) AS median_age_years,
                   ROUND(MIN(age_years), 2) AS minimum_age_years,
                   ROUND(MAX(age_years), 2) AS maximum_age_years
            FROM patient_level
        )"""
        demographic_columns = ", d.male_patients, d.female_patients, d.sex_unknown_or_other, d.patients_with_age, d.average_age_years, d.median_age_years, d.minimum_age_years, d.maximum_age_years"
        warnings.append(
            "Demographics are de-identified DICOM PatientSex/PatientAge metadata and may differ from authoritative clinical tables."
        )
    sql = f"""
        WITH filtered AS (
            SELECT * FROM index WHERE {where}
        ),
        imaging_summary AS (
            SELECT collection_id,
                   COUNT(DISTINCT PatientID) AS patient_count,
                   COUNT(DISTINCT StudyInstanceUID) AS study_count,
                   COUNT(DISTINCT SeriesInstanceUID) AS series_count,
                   STRING_AGG(DISTINCT Modality, ', ' ORDER BY Modality) AS modalities,
                   ROUND(SUM(series_size_MB) / 1000.0, 3) AS size_gb
            FROM filtered
            GROUP BY collection_id
        )
        {demographics}
        SELECT i.* {demographic_columns}
        FROM imaging_summary i
        {"CROSS JOIN demographic_summary d" if include_demographics else ""}
    """
    frame, query, tables = _execute_readonly_sql(_client(), sql, 10)
    return _result(
        frame,
        query=query,
        tables=tables,
        expected_columns=["collection_id", "patient_count", "series_count"],
        warnings=warnings,
    )


class IDCSeriesSearchArgs(BaseModel):
    collection_id: str = Field(..., description="Exact collection_id; empty for any collection.")
    modality: str = Field(..., description="Exact Modality; empty for any.")
    body_part: str = Field(..., description="Exact BodyPartExamined; empty for any.")
    series_description_contains: str = Field(..., description="Case-insensitive substring in SeriesDescription; empty for any.")
    study_description_contains: str = Field(..., description="Case-insensitive substring in StudyDescription; empty for any.")
    patient_id: str = Field(..., description="Exact PatientID; empty for any patient.")
    limit: int = Field(..., ge=1, le=500, description="Maximum number of series rows.")
    include_patient_id: bool = Field(..., description="Include PatientID only when needed by the request or downstream step.")
    include_study_uid: bool = Field(..., description="Include StudyInstanceUID only when needed.")
    include_series_uid: bool = Field(..., description="Include SeriesInstanceUID only when needed.")
    include_viewer_url: bool = Field(
        ...,
        description=(
            "Generate a validated IDC viewer URL for every returned series. Use true only "
            "when the user explicitly asks to view, display, or open imaging without downloading."
        ),
    )


@toolify_agent(
    name="idc_series_search",
    description=(
        "Search IDC series with typed collection, modality, anatomy, description, and patient "
        "filters, optionally returning validated OHIF/SLIM viewer links."
    ),
    args_schema=IDCSeriesSearchArgs,
    timeout_s=180,
)
async def idc_series_search_runner(
    collection_id: str,
    modality: str,
    body_part: str,
    series_description_contains: str,
    study_description_contains: str,
    patient_id: str,
    limit: int,
    include_patient_id: bool,
    include_study_uid: bool,
    include_series_uid: bool,
    include_viewer_url: bool,
) -> TaskResult:
    if include_viewer_url and int(limit) > 20:
        raise ValueError(
            "Viewer-link requests must use limit <= 20. Request only the smallest number "
            "of examples needed; use limit=1 for a single example series."
        )
    filters = []
    if collection_id.strip():
        filters.append(_collection_id_filter_sql("collection_id", collection_id))
    exact = {
        "Modality": modality,
        "BodyPartExamined": body_part,
        "PatientID": patient_id,
    }
    for column, value in exact.items():
        if value.strip():
            filters.append(f"{column} = {_sql_literal(value.strip())}")
    for column, value in {
        "SeriesDescription": series_description_contains,
        "StudyDescription": study_description_contains,
    }.items():
        if value.strip():
            filters.append(f"LOWER(COALESCE({column}, '')) LIKE LOWER({_sql_literal('%' + value.strip() + '%')})")
    columns = [
        "collection_id",
        "Modality",
        "BodyPartExamined",
        "StudyDescription",
        "SeriesDescription",
        "instanceCount",
        "series_size_MB",
    ]
    if include_patient_id:
        columns.append("PatientID")
    if include_study_uid or include_viewer_url:
        columns.append("StudyInstanceUID")
    if include_series_uid or include_viewer_url:
        columns.append("SeriesInstanceUID")
    where = " WHERE " + " AND ".join(filters) if filters else ""
    sql = f"SELECT {', '.join(columns)} FROM index{where} ORDER BY collection_id, PatientID, StudyInstanceUID LIMIT {int(limit)}"
    client = _client()
    frame, query, tables = _execute_readonly_sql(client, sql, limit)
    warnings: list[str] = []
    viewer_failures = 0
    if include_viewer_url:
        viewer_urls: list[str] = []
        viewer_types: list[str] = []
        for raw_uid in frame.get("SeriesInstanceUID", pd.Series(dtype=str)).tolist():
            try:
                generated = client.get_viewer_URL(seriesInstanceUID=str(raw_uid))
                url = validated_idc_viewer_url(generated)
            except Exception:
                url = ""
            if not url:
                viewer_failures += 1
                viewer_urls.append("")
                viewer_types.append("")
            else:
                viewer_urls.append(url)
                viewer_types.append("slim" if "/slim/" in url else "ohif_v3")
        frame["viewer_url"] = viewer_urls
        frame["viewer_type"] = viewer_types
        if viewer_failures:
            warnings.append(
                f"IDC viewer URL generation failed for {viewer_failures} returned series. "
                "Do not construct replacement URLs from memory."
            )
    if len(frame) >= limit and not include_viewer_url:
        warnings.append(
            f"The detail result reached limit={limit} and may be truncated. Do not use "
            "these rows to compute a complete aggregate distribution; use "
            "idc_series_category_summary instead."
        )
    result = _result(
        frame,
        query=query,
        tables=tables,
        expected_columns=(
            ["collection_id", "Modality", "StudyInstanceUID", "SeriesInstanceUID", "viewer_url"]
            if include_viewer_url
            else ["collection_id", "Modality"]
        ),
        warnings=warnings,
        partial=(len(frame) >= limit and not include_viewer_url) or viewer_failures > 0,
    )
    result.output["validation"]["viewer_urls_requested"] = bool(include_viewer_url)
    result.output["validation"]["viewer_urls_generated"] = int(
        len(frame) - viewer_failures if include_viewer_url else 0
    )
    return result


class IDCSeriesManifestArgs(BaseModel):
    collection_id: str = Field(..., min_length=1, description="Exact IDC collection_id.")
    modality: str = Field(..., description="Exact Modality such as CT; empty for any modality.")
    body_part: str = Field(..., description="Exact BodyPartExamined filter; empty for any.")
    series_description_contains: str = Field(
        ...,
        description="Case-insensitive substring in SeriesDescription; empty for any.",
    )
    study_description_contains: str = Field(
        ...,
        description="Case-insensitive substring in StudyDescription; empty for any.",
    )
    patient_id: str = Field(..., description="Exact PatientID; empty to select across patients.")
    patient_count: int = Field(
        ...,
        ge=1,
        le=500,
        description="Exact number of distinct patients requested for the acquisition manifest.",
    )
    selection_strategy: Literal["first", "random"] = Field(
        ...,
        description="Use first for stable lexical selection or random for deterministic hash sampling.",
    )
    random_seed: int = Field(
        ...,
        ge=0,
        le=2_147_483_647,
        description="Seed used by deterministic random patient sampling; ignored for first.",
    )
    series_scope: Literal[
        "representative",
        "all_matching",
        "all_patient_series",
    ] = Field(
        default="representative",
        description=(
            "Series coverage after selecting patients: representative keeps one matching "
            "series per patient; all_matching keeps every series matching the supplied "
            "filters; all_patient_series uses the filters only to select patients and then "
            "keeps every series for those patients in the collection."
        ),
    )


@toolify_agent(
    name="idc_series_manifest",
    description=(
        "Create an exact acquisition manifest for a deterministic patient sample. Supports "
        "one representative series, all matching series, or all collection series for each "
        "selected patient. It never generates viewer URLs and validates patient and series "
        "coverage."
    ),
    args_schema=IDCSeriesManifestArgs,
    timeout_s=180,
)
async def idc_series_manifest_runner(
    collection_id: str,
    modality: str,
    body_part: str,
    series_description_contains: str,
    study_description_contains: str,
    patient_id: str,
    patient_count: int,
    selection_strategy: Literal["first", "random"],
    random_seed: int,
    series_scope: Literal[
        "representative",
        "all_matching",
        "all_patient_series",
    ] = "representative",
) -> TaskResult:
    filters = [_collection_id_filter_sql("collection_id", collection_id)]
    for column, value in {
        "Modality": modality,
        "BodyPartExamined": body_part,
        "PatientID": patient_id,
    }.items():
        if value.strip():
            filters.append(f"{column} = {_sql_literal(value.strip())}")
    for column, value in {
        "SeriesDescription": series_description_contains,
        "StudyDescription": study_description_contains,
    }.items():
        if value.strip():
            filters.append(
                f"LOWER(COALESCE({column}, '')) LIKE "
                f"LOWER({_sql_literal('%' + value.strip() + '%')})"
            )
    filters.extend(
        [
            "COALESCE(CAST(PatientID AS VARCHAR), '') <> ''",
            "COALESCE(CAST(SeriesInstanceUID AS VARCHAR), '') <> ''",
        ]
    )
    patient_order_expression = "PatientID"
    if selection_strategy == "random":
        patient_order_expression = (
            "MD5(CONCAT(CAST(PatientID AS VARCHAR), "
            f"{_sql_literal(':' + str(int(random_seed)))}) )"
        )

    common_ctes = f"""
        WITH eligible_series AS (
            SELECT
                collection_id,
                Modality,
                BodyPartExamined,
                StudyDescription,
                SeriesDescription,
                instanceCount,
                series_size_MB,
                PatientID,
                StudyInstanceUID,
                SeriesInstanceUID
            FROM index
            WHERE {' AND '.join(filters)}
        ),
        eligible_patients AS (
            SELECT DISTINCT PatientID
            FROM eligible_series
        ),
        selected_patients AS (
            SELECT
                PatientID,
                {patient_order_expression} AS patient_selection_key
            FROM eligible_patients
            ORDER BY patient_selection_key, PatientID
            LIMIT {int(patient_count)}
        )
    """
    columns = """
        collection_id,
        Modality,
        BodyPartExamined,
        StudyDescription,
        SeriesDescription,
        instanceCount,
        series_size_MB,
        PatientID,
        StudyInstanceUID,
        SeriesInstanceUID
    """
    if series_scope == "representative":
        sql = common_ctes + f""",
        ranked_selected_series AS (
            SELECT
                eligible_series.*,
                selected_patients.patient_selection_key,
                ROW_NUMBER() OVER (
                    PARTITION BY eligible_series.PatientID
                    ORDER BY eligible_series.StudyInstanceUID,
                             eligible_series.SeriesInstanceUID
                ) AS patient_series_rank
            FROM eligible_series
            JOIN selected_patients USING (PatientID)
        )
        SELECT
            {columns}
        FROM ranked_selected_series
        WHERE patient_series_rank = 1
        ORDER BY patient_selection_key, PatientID, StudyInstanceUID, SeriesInstanceUID
        """
        query_limit = int(patient_count)
    elif series_scope == "all_matching":
        sql = common_ctes + f"""
        SELECT
            {columns}
        FROM eligible_series
        JOIN selected_patients USING (PatientID)
        ORDER BY patient_selection_key, PatientID, StudyInstanceUID, SeriesInstanceUID
        """
        query_limit = MAX_MANIFEST_SERIES + 1
    else:
        selected_collection_filter = _collection_id_filter_sql(
            "selected_series.collection_id", collection_id
        )
        sql = common_ctes + f"""
        SELECT
            selected_series.collection_id,
            selected_series.Modality,
            selected_series.BodyPartExamined,
            selected_series.StudyDescription,
            selected_series.SeriesDescription,
            selected_series.instanceCount,
            selected_series.series_size_MB,
            selected_series.PatientID,
            selected_series.StudyInstanceUID,
            selected_series.SeriesInstanceUID
        FROM index AS selected_series
        JOIN selected_patients
          ON selected_series.PatientID = selected_patients.PatientID
        WHERE {selected_collection_filter}
          AND COALESCE(CAST(selected_series.SeriesInstanceUID AS VARCHAR), '') <> ''
        ORDER BY patient_selection_key, selected_series.PatientID,
                 selected_series.StudyInstanceUID, selected_series.SeriesInstanceUID
        """
        query_limit = MAX_MANIFEST_SERIES + 1

    frame, query, tables = _execute_readonly_sql(
        _client(),
        sql,
        query_limit,
        absolute_max_rows=(
            ABSOLUTE_MAX_ROWS
            if series_scope == "representative"
            else MAX_MANIFEST_SERIES + 1
        ),
    )
    distinct_patients = int(frame["PatientID"].nunique()) if "PatientID" in frame else 0
    distinct_series = (
        int(frame["SeriesInstanceUID"].nunique())
        if "SeriesInstanceUID" in frame
        else 0
    )
    series_limit_exceeded = len(frame) > MAX_MANIFEST_SERIES
    unique_series = distinct_series == len(frame)
    exact_patients = distinct_patients == int(patient_count)
    exact = exact_patients and unique_series and not series_limit_exceeded
    if series_scope == "representative":
        exact = exact and len(frame) == int(patient_count)
    series_per_patient = (
        frame.groupby("PatientID")["SeriesInstanceUID"].nunique().astype(int).to_dict()
        if {"PatientID", "SeriesInstanceUID"}.issubset(frame.columns)
        else {}
    )
    warnings: list[str] = []
    if not exact_patients:
        warnings.append(
            f"Requested {patient_count} distinct patients but only {distinct_patients} "
            "matching patients were available. Do not start acquisition "
            "unless this reduced cardinality is acceptable to the user."
        )
    if not unique_series:
        warnings.append(
            "The manifest contains duplicate SeriesInstanceUID rows. Do not start acquisition."
        )
    if series_limit_exceeded:
        warnings.append(
            f"The selected cohort contains more than {MAX_MANIFEST_SERIES} series, which "
            "exceeds the supported batch limit. Refine the request before acquisition."
        )
    frame = frame.copy()
    frame["manifest_series_scope"] = series_scope
    frame["manifest_complete"] = bool(exact)
    frame["manifest_requested_patient_count"] = int(patient_count)
    frame["manifest_distinct_patient_count"] = distinct_patients
    frame["manifest_distinct_series_count"] = distinct_series
    result = _result(
        frame,
        query=query,
        tables=tables,
        expected_columns=["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
        warnings=warnings,
        partial=not exact,
    )
    result.output["counts"].update(
        requested_patients=int(patient_count),
        distinct_patients=distinct_patients,
        distinct_series=distinct_series,
        series_per_patient={str(key): int(value) for key, value in series_per_patient.items()},
    )
    result.output["validation"].update(
        row_limit_applied=series_limit_exceeded,
        requested_cardinality_satisfied=exact,
        one_series_per_patient=(distinct_patients == distinct_series == len(frame)),
        manifest_complete=exact,
        unique_series=unique_series,
        series_limit_exceeded=series_limit_exceeded,
        manifest_series_limit=MAX_MANIFEST_SERIES,
        series_scope=series_scope,
        all_matching_series=(series_scope == "all_matching" and exact),
        all_series_for_selected_patients=(
            series_scope == "all_patient_series" and exact
        ),
        viewer_urls_generated=0,
        selection_strategy=selection_strategy,
        random_seed=int(random_seed),
    )
    return result


_SERIES_CATEGORY_COLUMNS = {
    "SeriesDescription",
    "StudyDescription",
    "BodyPartExamined",
    "Modality",
}


class IDCSeriesCategorySummaryArgs(BaseModel):
    collection_id: str = Field(..., min_length=1, description="Exact IDC collection_id.")
    modality: str = Field(..., description="Optional exact DICOM Modality filter; empty for all modalities.")
    category: str = Field(
        ...,
        description=(
            "Series-level category to aggregate: SeriesDescription, StudyDescription, "
            "BodyPartExamined, or Modality. Use SeriesDescription for sequence/protocol plots."
        ),
    )
    include_missing: bool = Field(..., description="Include an explicit [missing] category row.")
    limit: int = Field(..., ge=1, le=500, description="Maximum aggregate category rows.")


@toolify_agent(
    name="idc_series_category_summary",
    description=(
        "Create a complete compact aggregate distribution over a series metadata category, "
        "including distinct patient, study, and series counts. Use this instead of fetching "
        "raw series rows for sequence/protocol count tables and plots. When aggregating "
        "SeriesDescription without an explicit modality, derived/non-acquisition modalities "
        "such as SEG, SR, and RTSTRUCT are excluded."
    ),
    args_schema=IDCSeriesCategorySummaryArgs,
    timeout_s=180,
)
async def idc_series_category_summary_runner(
    collection_id: str,
    modality: str,
    category: str,
    include_missing: bool,
    limit: int,
) -> TaskResult:
    if category not in _SERIES_CATEGORY_COLUMNS:
        raise ValueError(
            "category must be one of: " + ", ".join(sorted(_SERIES_CATEGORY_COLUMNS))
        )
    filters = [_collection_id_filter_sql("collection_id", collection_id)]
    if modality.strip():
        filters.append(f"Modality = {_sql_literal(modality.strip())}")
    exclude_derived_modalities = category == "SeriesDescription" and not modality.strip()
    if exclude_derived_modalities:
        derived_values = ", ".join(
            _sql_literal(value)
            for value in sorted(_DERIVED_NON_ACQUISITION_MODALITIES)
        )
        filters.append(f"UPPER(COALESCE(Modality, '')) NOT IN ({derived_values})")
    category_sql = (
        f"COALESCE(NULLIF(TRIM(CAST({category} AS VARCHAR)), ''), '[missing]')"
    )
    if not include_missing:
        filters.append(f"NULLIF(TRIM(CAST({category} AS VARCHAR)), '') IS NOT NULL")
    where = " AND ".join(filters)
    sql = f"""
        SELECT {category_sql} AS category,
               COUNT(DISTINCT PatientID) AS patient_count,
               COUNT(DISTINCT StudyInstanceUID) AS study_count,
               COUNT(DISTINCT SeriesInstanceUID) AS series_count
        FROM index
        WHERE {where}
        GROUP BY {category_sql}
        ORDER BY patient_count DESC, category
        LIMIT {int(limit)}
    """
    frame, query, tables = _execute_readonly_sql(_client(), sql, limit)
    warnings = [
        f"{category} is free-text DICOM metadata and may contain synonymous or site-specific labels; "
        "rows preserve exact trimmed labels rather than inferring clinical phases.",
        "Distinct-patient counts are computed independently per category and categories may "
        "overlap. Do not sum category patient_count values or compare their sum with the "
        "collection-level distinct-patient count unless a separate query proves the categories "
        "are mutually exclusive.",
    ]
    if exclude_derived_modalities:
        warnings.append(
            "No modality filter was supplied for the SeriesDescription distribution, so "
            "derived/non-acquisition modalities were excluded: "
            + ", ".join(sorted(_DERIVED_NON_ACQUISITION_MODALITIES))
            + "."
        )
    partial = len(frame) >= limit
    if partial:
        warnings.append(
            f"The aggregate result reached limit={limit} and may be truncated; increase the limit "
            "before treating the distribution as exhaustive."
        )
    return _result(
        frame,
        query=query,
        tables=tables,
        expected_columns=["category", "patient_count", "study_count", "series_count"],
        warnings=warnings,
        partial=partial,
    )


class IDCCollectionProfileArgs(BaseModel):
    collection_id: str = Field(..., min_length=1, description="Exact IDC collection_id.")
    sequence_modality: str = Field(
        ...,
        description=(
            "Exact DICOM modality used only for the sequence distribution, normally CT or MR. "
            "The collection summary always covers all modalities."
        ),
    )
    sequence_category: Literal[
        "SeriesDescription", "StudyDescription", "BodyPartExamined", "Modality"
    ] = Field(
        ...,
        description="Series metadata category to aggregate; use SeriesDescription for sequences/protocols.",
    )
    include_missing_sequences: bool = Field(
        ...,
        description="Include an explicit [missing] sequence-category row.",
    )
    include_demographics: bool = Field(
        ...,
        description="Include de-duplicated DICOM PatientSex and PatientAge statistics.",
    )
    sequence_limit: int = Field(..., ge=1, le=500, description="Maximum sequence-category rows.")


@toolify_agent(
    name="idc_collection_profile",
    description=(
        "Atomically characterize one IDC collection for a sequence/protocol plot plus dataset "
        "summary. Returns a modality-filtered sequence_summary and an all-modality "
        "collection_summary with optional DICOM demographics. Prefer this single tool when a "
        "request combines sequence counts with patients/sex/age/study/series summary."
    ),
    args_schema=IDCCollectionProfileArgs,
    timeout_s=300,
)
async def idc_collection_profile_runner(
    collection_id: str,
    sequence_modality: str,
    sequence_category: str,
    include_missing_sequences: bool,
    include_demographics: bool,
    sequence_limit: int,
) -> TaskResult:
    sequence_result = await idc_series_category_summary_runner._runner(
        collection_id=collection_id,
        modality=sequence_modality,
        category=sequence_category,
        include_missing=include_missing_sequences,
        limit=sequence_limit,
    )
    collection_result = await idc_collection_summary_runner._runner(
        collection_id=collection_id,
        modality="",
        include_demographics=include_demographics,
    )

    sequence_output = sequence_result.output or {}
    collection_output = collection_result.output or {}
    warnings = list(dict.fromkeys([
        *[str(value) for value in sequence_output.get("warnings", [])],
        *[str(value) for value in collection_output.get("warnings", [])],
        (
            f"sequence_summary is filtered to Modality={sequence_modality}; "
            "collection_summary covers all modalities."
        ),
    ]))
    errors = [
        *[str(value) for value in (sequence_result.errors or [])],
        *[str(value) for value in (collection_result.errors or [])],
    ]
    statuses = {str(sequence_result.status), str(collection_result.status)}
    if "error" in statuses:
        status = "error"
    elif "partial" in statuses:
        status = "partial"
    else:
        status = "ok"

    sequence_frame = sequence_output.get("result")
    collection_frame = collection_output.get("result")
    return TaskResult(
        output={
            "sequence_summary": sequence_frame,
            "collection_summary": collection_frame,
            "queries": {
                "sequence_summary": sequence_output.get("query", ""),
                "collection_summary": collection_output.get("query", ""),
            },
            "tables": sorted(set([
                *[str(value) for value in sequence_output.get("tables", [])],
                *[str(value) for value in collection_output.get("tables", [])],
            ])),
            "counts": {
                "sequence_categories": int(len(sequence_frame)) if isinstance(sequence_frame, pd.DataFrame) else 0,
                "collection_rows": int(len(collection_frame)) if isinstance(collection_frame, pd.DataFrame) else 0,
            },
            "validation": {
                "sequence_modality": sequence_modality,
                "collection_summary_modality_scope": "all",
                "sequence_status": str(sequence_result.status),
                "collection_status": str(collection_result.status),
            },
            "provenance": {
                "source": "NCI Imaging Data Commons",
                "idc_data_version": _idc_version(_client()),
                "skill_version": OFFICIAL_SKILL_VERSION,
            },
            "warnings": warnings,
        },
        status=status,
        errors=errors,
    )


class IDCClinicalCatalogArgs(BaseModel):
    collection_id: str = Field(..., description="Exact collection_id; empty to inspect the complete clinical catalog.")
    include_columns: bool = Field(..., description="Include the actual clinical column labels in addition to table names.")
    limit: int = Field(..., ge=1, le=500, description="Maximum catalog rows.")


@toolify_agent(
    name="idc_clinical_catalog",
    description=(
        "Inspect IDC clinical_index for real clinical table names and column labels. "
        "Use this before selecting a clinical table or making clinical-field claims."
    ),
    args_schema=IDCClinicalCatalogArgs,
    timeout_s=300,
)
async def idc_clinical_catalog_runner(collection_id: str, include_columns: bool, limit: int) -> TaskResult:
    client = _client()
    client.fetch_index("clinical_index")
    schema = _get_index_schema(client, "clinical_index")
    columns = [str(item.get("name")) for item in schema.get("columns", []) if item.get("name")]
    selected = [name for name in ("collection_id", "table_name", "column_label") if name in columns]
    if "table_name" not in selected:
        raise RuntimeError("clinical_index does not expose the expected table_name column.")
    if not include_columns and "column_label" in selected:
        selected.remove("column_label")
    where = ""
    if collection_id.strip() and "collection_id" in columns:
        where = f" WHERE {_collection_id_filter_sql('collection_id', collection_id)}"
    sql = f"SELECT DISTINCT {', '.join(selected)} FROM clinical_index{where} ORDER BY {', '.join(selected)} LIMIT {int(limit)}"
    frame, query, tables = _execute_readonly_sql(client, sql, limit)
    return _result(frame, query=query, tables=tables, expected_columns=["table_name"])


class IDCSQLQueryArgs(BaseModel):
    sql: str = Field(..., description="One read-only SELECT/WITH query over named idc-index tables.")
    expected_columns: str = Field(..., description="JSON array of required output columns, or [] if none.")
    max_rows: int = Field(..., ge=1, le=500, description="Maximum returned rows.")


@toolify_agent(
    name="idc_sql_query",
    description=(
        "Execute a validated read-only SQL query over authoritative idc-index tables. "
        "Clinical tables listed by idc_clinical_catalog can also be queried by their "
        "short_table_name (for example SELECT COUNT(*) FROM c4kc_kits_clinical). "
        "External-file functions, mutations, multiple statements, and unknown tables are blocked."
    ),
    args_schema=IDCSQLQueryArgs,
    timeout_s=300,
)
async def idc_sql_query_runner(sql: str, expected_columns: str, max_rows: int) -> TaskResult:
    expected = _expected_columns(expected_columns)
    frame, query, tables = _execute_readonly_sql(_client(), sql, max_rows)
    return _result(frame, query=query, tables=tables, expected_columns=expected)


class ExecuteIDCPythonArgs(BaseModel):
    code: str = Field(
        ...,
        description=(
            "Restricted Python authored by the IDC subagent, with prebound client, pd, np, and "
            "math. No imports, file access, arbitrary network, downloads, or dynamic execution. "
            "Direct SQL execution is not available; use idc_sql_query for SQL. Assign the final "
            "JSON-serializable or pandas value to result."
        ),
    )
    expected_columns: str = Field(..., description="JSON array of required DataFrame columns, or [] if none.")
    max_rows: int = Field(..., ge=1, le=500, description="Maximum serialized DataFrame rows.")
    timeout_seconds: int = Field(..., ge=1, le=60, description="Hard subprocess timeout in seconds.")


@toolify_agent(
    name="execute_idc_python",
    description=(
        "Execute restricted Python written directly by the IDC subagent as a last resort for "
        "multi-step local metadata transformations that typed tools and read-only SQL cannot "
        "express. This tool only validates and executes supplied code; it does not generate code. "
        "It runs in a resource-limited subprocess where imports, filesystem access, arbitrary "
        "network, direct SQL, dynamic execution, private attributes, and download methods are "
        "blocked."
    ),
    args_schema=ExecuteIDCPythonArgs,
    timeout_s=75,
)
async def execute_idc_python_runner(
    code: str,
    expected_columns: str,
    max_rows: int,
    timeout_seconds: int,
) -> TaskResult:
    validate_restricted_idc_code(code)
    expected = _expected_columns(expected_columns)
    worker = Path(__file__).with_name("idc_python_worker.py")
    payload = json.dumps(
        {
            "code": code,
            "max_rows": min(max_rows, ABSOLUTE_MAX_ROWS),
            "timeout_seconds": timeout_seconds,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-I", str(worker)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout_seconds + 5,
        check=False,
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Restricted IDC worker returned invalid output.") from exc
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "Restricted IDC worker failed."))
    output = response["output"]
    kind = output.get("kind")
    if kind == "dataframe":
        frame = pd.DataFrame(output.get("rows") or [], columns=output.get("columns") or None)
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            return TaskResult(
                output={"result": frame, "code": code, "worker": output},
                artifacts={"code": code},
                status="partial",
                errors=[f"Missing expected output columns: {', '.join(missing)}"],
            )
        result_value: Any = frame
    else:
        if expected:
            raise ValueError("expected_columns can only be used when restricted code returns a DataFrame.")
        result_value = output.get("value")
    return TaskResult(
        output={
            "result": result_value,
            "code": code,
            "counts": {
                "returned_rows": int(output.get("nrows", len(result_value) if isinstance(result_value, list) else 1))
            },
            "validation": {
                "restricted_ast_validated": True,
                "subprocess_exit_code": completed.returncode,
                "truncated": bool(output.get("truncated", False)),
                "expected_columns": expected,
            },
            "provenance": {
                "source": "NCI Imaging Data Commons",
                "idc_data_version": output.get("idc_data_version", "unknown"),
                "skill_version": OFFICIAL_SKILL_VERSION,
                "execution": "resource-limited restricted subprocess",
            },
            "warnings": [
                "Restricted Python is a fallback; prefer typed IDC tools or idc_sql_query when possible."
            ],
        },
        artifacts={"code": code},
        status="ok",
    )
