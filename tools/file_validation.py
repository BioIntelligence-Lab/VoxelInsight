from __future__ import annotations

import pathlib
from typing import List


PLACEHOLDER_PATH_MARKERS = (
    "user_uploaded_image_path",
    "uploaded_image_path",
    "uploaded_file_path",
    "provided_uploaded_file",
    "attached_file",
    "path/to/",
    "example/path",
)


def looks_like_placeholder_path(path: str) -> bool:
    value = str(path).strip()
    lowered = value.lower()
    if not value:
        return True
    if (value.startswith("<") and value.endswith(">")) or (value.startswith("{") and value.endswith("}")):
        return True
    return any(marker in lowered for marker in PLACEHOLDER_PATH_MARKERS)


def validate_existing_input_files(files: List[str], *, tool_name: str = "tool") -> None:
    if not files:
        raise ValueError(
            f"No input file was provided to {tool_name}. "
            "Pass an exact uploaded NIfTI path as file_path or file_paths."
        )

    for file_value in files:
        if looks_like_placeholder_path(file_value):
            raise ValueError(
                f"Invalid {tool_name} input path {file_value!r}: placeholder paths are not allowed. "
                "Use the exact uploaded file path from uploaded_files."
            )

        path = pathlib.Path(file_value)
        if not path.exists():
            raise ValueError(f"Invalid {tool_name} input path {file_value!r}: file does not exist.")
