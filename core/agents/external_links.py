from __future__ import annotations

from typing import Any, Iterable
from urllib.parse import urlparse


IDC_VIEWER_HOST = "viewer.imaging.datacommons.cancer.gov"


def validated_idc_viewer_url(value: Any) -> str:
    """Return a normalized IDC viewer URL, or an empty string when invalid."""

    if not isinstance(value, str) or not value.strip():
        return ""
    url = value.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if parsed.scheme != "https" or parsed.hostname != IDC_VIEWER_HOST:
        return ""
    if not parsed.path.startswith(("/v3/viewer/", "/slim/studies/", "/viewer/")):
        return ""
    return url


def idc_viewer_urls_from_rows(rows: Iterable[Any]) -> list[str]:
    """Extract unique, validated IDC viewer URLs from table-like rows."""

    urls: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = validated_idc_viewer_url(row.get("viewer_url"))
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls
