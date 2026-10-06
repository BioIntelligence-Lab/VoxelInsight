from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


PATIENT_ID_CANDIDATES = (
    "case_barcode",
    "case_id",
    "patient_id",
    "patientid",
    "dicom_patient_id",
    "PatientID",
    "subject_id",
)


def demographic_group(column: str) -> Optional[str]:
    normalized = str(column).strip().lower()
    if (
        normalized == "age"
        or normalized.startswith("age_")
        or normalized.endswith("_age")
    ):
        return "age"
    if (
        normalized in {"sex", "gender"}
        or normalized.startswith(("sex_", "gender_"))
        or normalized.endswith(("_sex", "_gender"))
    ):
        return "sex"
    if normalized == "race" or normalized.startswith("race_") or normalized.endswith("_race"):
        return "race"
    if "ethnicity" in normalized or normalized.startswith("ethnic"):
        return "ethnicity"
    if (
        normalized == "dob"
        or "date_of_birth" in normalized
        or "year_of_birth" in normalized
        or "days_to_birth" in normalized
        or normalized.startswith("birth_")
        or normalized.endswith("_birth")
    ):
        return "birth"
    return None


def table_schema(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "name": str(column),
            "dtype": str(dataframe[column].dtype),
            "non_null": int(dataframe[column].notna().sum()),
            "missing": int(dataframe[column].isna().sum()),
        }
        for column in dataframe.columns
    ]


def _safe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def demographic_metadata(
    dataframe: pd.DataFrame,
    *,
    max_top_values: int = 5,
) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    for column in dataframe.columns:
        group = demographic_group(str(column))
        if group is None:
            continue
        series = dataframe[column]
        non_null = series.dropna()
        record: Dict[str, Any] = {
            "field": str(column),
            "group": group,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "unique": int(non_null.nunique(dropna=True)),
        }
        if not non_null.empty and pd.api.types.is_numeric_dtype(non_null):
            record["numeric_range"] = {
                "min": _safe_scalar(non_null.min()),
                "max": _safe_scalar(non_null.max()),
            }
        if not non_null.empty and record["unique"] <= 50:
            counts = non_null.astype(str).value_counts().head(max_top_values)
            record["top_values"] = {
                str(value): int(count)
                for value, count in counts.items()
            }
        fields.append(record)
    return fields


def resolve_column(dataframe: pd.DataFrame, requested: str) -> Optional[str]:
    requested_normalized = str(requested).strip().lower()
    by_lower = {str(column).lower(): str(column) for column in dataframe.columns}
    if requested_normalized in by_lower:
        return by_lower[requested_normalized]
    aliases = {
        "sex": ("gender",),
        "gender": ("sex",),
    }
    for alias in aliases.get(requested_normalized, ()):
        if alias in by_lower:
            return by_lower[alias]
    return None


def infer_patient_id_column(dataframe: pd.DataFrame) -> Optional[str]:
    by_lower = {str(column).lower(): str(column) for column in dataframe.columns}
    for candidate in PATIENT_ID_CANDIDATES:
        match = by_lower.get(candidate.lower())
        if match:
            return match
    return None
