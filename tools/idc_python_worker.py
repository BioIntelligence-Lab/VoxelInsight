from __future__ import annotations

import ast
import json
import math
import resource
import sys
from typing import Any


MAX_CODE_CHARS = 20_000
MAX_RESULT_BYTES = 2_000_000

_BLOCKED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "quit",
    "setattr",
    "vars",
}

_BLOCKED_NODES = (
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)

_SAFE_BUILTIN_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}

_SAFE_CLIENT_CALLS = {
    "fetch_index",
    "get_clinical_table",
    "get_idc_version",
    "get_index_schema",
    "get_viewer_URL",
    "sql_query",
}

_SAFE_LIBRARY_CALLS = {
    "DataFrame",
    "Series",
    "array",
    "asarray",
    "concat",
    "crosstab",
    "isna",
    "isnan",
    "notna",
    "to_datetime",
    "to_numeric",
    "where",
}

_SAFE_MATH_CALLS = {
    "ceil",
    "exp",
    "fabs",
    "floor",
    "isfinite",
    "isinf",
    "isnan",
    "log",
    "log10",
    "sqrt",
}

_SAFE_OBJECT_METHODS = {
    "agg",
    "aggregate",
    "all",
    "any",
    "astype",
    "between",
    "clip",
    "contains",
    "copy",
    "count",
    "drop",
    "drop_duplicates",
    "dropna",
    "endswith",
    "fillna",
    "first",
    "groupby",
    "head",
    "isin",
    "join",
    "last",
    "lower",
    "map",
    "max",
    "mean",
    "median",
    "merge",
    "min",
    "nunique",
    "pivot",
    "pivot_table",
    "rename",
    "replace",
    "reset_index",
    "round",
    "size",
    "sort_index",
    "sort_values",
    "startswith",
    "strip",
    "sum",
    "tail",
    "to_dict",
    "tolist",
    "unique",
    "upper",
    "value_counts",
}


def _root_name(node: ast.AST) -> str:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def validate_restricted_idc_code(code: str) -> ast.Module:
    """Validate model-authored IDC analysis code before isolated execution."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Code must be a non-empty string.")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"Code exceeds the {MAX_CODE_CHARS}-character limit.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc

    assigned_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODES):
            raise ValueError(f"Python construct {type(node).__name__} is not allowed.")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in _BLOCKED_NAMES:
                raise ValueError(f"Name {node.id!r} is not allowed.")
            if isinstance(node.ctx, ast.Store):
                assigned_names.add(node.id)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private and dunder attributes are not allowed.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_BUILTIN_CALLS:
                    raise ValueError(f"Call to {node.func.id!r} is not allowed.")
            elif isinstance(node.func, ast.Attribute):
                root = _root_name(node.func)
                method = node.func.attr
                direct_root_call = isinstance(node.func.value, ast.Name)
                if direct_root_call and root == "client":
                    if method not in _SAFE_CLIENT_CALLS:
                        raise ValueError(f"IDCClient method {method!r} is not allowed.")
                elif direct_root_call and root in {"pd", "np"}:
                    if method not in _SAFE_LIBRARY_CALLS:
                        raise ValueError(f"Library call {root}.{method} is not allowed.")
                elif direct_root_call and root == "math":
                    if method not in _SAFE_MATH_CALLS:
                        raise ValueError(f"Math call math.{method} is not allowed.")
                elif method not in _SAFE_OBJECT_METHODS:
                    raise ValueError(f"Object method {method!r} is not allowed.")
            else:
                raise ValueError("Dynamic call targets are not allowed.")

    if "result" not in assigned_names:
        raise ValueError("Restricted code must assign its final value to `result`.")
    return tree


def _json_safe(value: Any, *, max_rows: int) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        frame = value.head(max_rows).copy()
        frame = frame.where(pd.notna(frame), None)
        return {
            "kind": "dataframe",
            "columns": [str(column) for column in frame.columns],
            "rows": frame.to_dict(orient="records"),
            "nrows": int(len(value)),
            "truncated": len(value) > len(frame),
        }
    if isinstance(value, pd.Series):
        return _json_safe(value.reset_index(), max_rows=max_rows)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    if isinstance(value, (list, tuple, dict)):
        encoded = json.dumps(value, default=str, allow_nan=False)
        return {"kind": "json", "value": json.loads(encoded)}
    raise ValueError(
        "`result` must be a pandas DataFrame/Series, JSON object/list, or scalar value."
    )


def _safe_builtins() -> dict[str, Any]:
    return {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }


def _apply_resource_limits(timeout_seconds: int) -> None:
    cpu_seconds = max(1, min(timeout_seconds, 60))
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except (OSError, ValueError):
        pass
    try:
        memory_bytes = 4 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (OSError, ValueError):
        pass


def execute_payload(payload: dict[str, Any]) -> dict[str, Any]:
    code = str(payload.get("code") or "")
    max_rows = max(1, min(int(payload.get("max_rows") or 200), 500))
    timeout_seconds = max(1, min(int(payload.get("timeout_seconds") or 30), 60))
    tree = validate_restricted_idc_code(code)
    import numpy as np
    import pandas as pd
    from idc_index import IDCClient

    client = IDCClient()
    _apply_resource_limits(timeout_seconds)
    environment: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "client": client,
        "math": math,
        "np": np,
        "pd": pd,
    }
    exec(compile(tree, "<restricted-idc-code>", "exec"), environment, environment)
    output = _json_safe(environment["result"], max_rows=max_rows)
    output["idc_data_version"] = client.get_idc_version()
    encoded = json.dumps(output, default=str, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError(f"Serialized result exceeds {MAX_RESULT_BYTES} bytes.")
    return output


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        output = execute_payload(payload)
        sys.stdout.write(json.dumps({"ok": True, "output": output}, default=str))
        return 0
    except Exception as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
