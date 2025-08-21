# midrc_graph_mirror.py
from __future__ import annotations
import argparse, re
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import duckdb

from gen3.auth import Gen3Auth
from gen3.submission import Gen3Submission

API = "https://data.midrc.org"

def _read_tsv_to_df(tsv: str) -> pd.DataFrame:
    if not tsv:
        return pd.DataFrame()
    return pd.read_csv(StringIO(tsv), sep="\t", dtype=str)

def _export_node(sub: Gen3Submission, program: str, project: str, node: str) -> pd.DataFrame:
    tsv = sub.export_node(program=program, project=project, node_type=node, fileformat="tsv")
    return _read_tsv_to_df(tsv)

def _snake_set(cols: List[str]) -> set:
    return {c.lower().strip() for c in cols}

def _guess_has_object_id(df: pd.DataFrame) -> bool:
    return "object_id" in _snake_set(df.columns)

def _best_col(cols: List[str], candidates: List[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def _project_codes_from_get_projects(obj, program: str) -> List[str]:
    if isinstance(obj, dict):
        if isinstance(obj.get("data"), list):
            return [d["code"] for d in obj["data"] if isinstance(d, dict) and d.get("code")]
        if isinstance(obj.get("links"), list):
            needle = f"/v0/submission/{program}/"
            out = []
            for link in obj["links"]:
                if isinstance(link, str) and needle in link:
                    out.append(link.split(needle, 1)[1].strip("/"))
            return out
    if isinstance(obj, list):
        return [d["code"] for d in obj if isinstance(d, dict) and d.get("code")]
    return []

def _sql_safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if re.match(r"^[0-9]", s):
        s = "_" + s
    return s

def build_midrc_graph(credentials_json: str,
                      out_dir: str,
                      program: str = "Open",
                      wide_out: str | None = None) -> str:
    out_root = Path(out_dir)
    out_nodes = out_root / "nodes"
    wide_out = wide_out or str(out_root / "midrc_files_wide.parquet")

    auth = Gen3Auth(API, refresh_file=credentials_json)
    sub = Gen3Submission(API, auth)

    raw_projects = sub.get_projects(program) or []
    projects = _project_codes_from_get_projects(raw_projects, program)
    if not projects:
        raise RuntimeError(f"No projects visible for program '{program}'. Raw get_projects()={raw_projects!r}")

    ddict = sub.get_dictionary_all()
    if not ddict:
        raise RuntimeError("Dictionary empty or unavailable.")
    meta_nodes = {"metaschema", "root", "data_release", "_settings", "_definitions", "_terms"}
    all_nodes = sorted([k for k in ddict.keys() if not k.startswith("_") and k not in meta_nodes])

    exported_tables: List[Tuple[str, Path]] = []
    for prj in projects:
        for node in all_nodes:
            df = _export_node(sub, program, prj, node)
            if df.empty:
                continue
            df.columns = [c.strip() for c in df.columns]
            p = out_nodes / f"{program}-{prj}" / f"{node}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            sql_name = _sql_safe(f"{node}_{program}_{prj}")
            exported_tables.append((sql_name, p))

    if not exported_tables:
        raise RuntimeError("No node exports produced rows. Check credentials and access.")

    con = duckdb.connect()
    for sql_name, parquet_path in exported_tables:
        con.execute(f"CREATE VIEW {sql_name} AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')")

    case_views = [name for name, _ in exported_tables if name.startswith("case_")]
    case_df = con.execute(" UNION ALL BY NAME ".join([f"SELECT * FROM {v}" for v in case_views])).df() if case_views else pd.DataFrame()

    study_views = [name for name, _ in exported_tables if name.startswith("imaging_study_")]
    study_df = con.execute(" UNION ALL BY NAME ".join([f"SELECT * FROM {v}" for v in study_views])).df() if study_views else pd.DataFrame()

    file_views: List[str] = []
    for name, _ in exported_tables:
        head = con.execute(f"SELECT * FROM {name} LIMIT 0").df()
        if _guess_has_object_id(head):
            file_views.append(name)

    if not file_views:
        Path(wide_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(wide_out, index=False)
        return wide_out

    parts = []
    for v in file_views:
        node_label = v.split("_")[0]
        parts.append(f"SELECT *, '{node_label}' AS file_node FROM {v}")
    df_files = con.execute(" UNION ALL BY NAME ".join(parts)).df()

    def pick_case_key(cols: List[str]) -> str | None:
        return _best_col(cols, ["case_submitter_id", "case_ids", "case_id", "case"])

    def pick_study_key(cols: List[str]) -> str | None:
        return _best_col(cols, ["study_submitter_id", "study_ids", "study_uid", "imaging_study_id"])

    case_submitter_col = _best_col(list(case_df.columns), ["submitter_id", "case_submitter_id"]) if not case_df.empty else None
    study_submitter_col = _best_col(list(study_df.columns), ["submitter_id", "study_uid"]) if not study_df.empty else None

    files_cols = list(df_files.columns)
    f_case_col = pick_case_key(files_cols)
    f_study_col = pick_study_key(files_cols)

    if not case_df.empty and case_submitter_col:
        keep_case = [c for c in ["project_id", case_submitter_col, "race", "sex", "ethnicity", "age_at_index", "covid19_positive"] if c in case_df.columns]
        case_slim = case_df[keep_case].drop_duplicates()
    else:
        case_slim = pd.DataFrame()

    if not study_df.empty and study_submitter_col:
        keep_study = [c for c in ["project_id", study_submitter_col, "study_modality", "body_part_examined", "study_description"] if c in study_df.columns]
        study_slim = study_df[keep_study].drop_duplicates()
    else:
        study_slim = pd.DataFrame()

    wide = df_files.copy()

    if not study_slim.empty and f_study_col and study_submitter_col and "project_id" in wide.columns and "project_id" in study_slim.columns:
        wide = wide.merge(
            study_slim,
            left_on=["project_id", f_study_col],
            right_on=["project_id", study_submitter_col],
            how="left",
            suffixes=("", "_study"),
        )

    if not case_slim.empty and f_case_col and case_submitter_col and "project_id" in wide.columns and "project_id" in case_slim.columns:
        wide = wide.merge(
            case_slim,
            left_on=["project_id", f_case_col],
            right_on=["project_id", case_submitter_col],
            how="left",
            suffixes=("", "_case"),
        )

    core_cols = [c for c in [
        "project_id",
        f_case_col, "race", "sex", "ethnicity", "age_at_index", "covid19_positive",
        f_study_col, "study_modality", "body_part_examined", "study_description",
        "series_uid", "modality", "file_node", "file_name", "object_id",
    ] if c and c in wide.columns]
    rest = [c for c in wide.columns if c not in core_cols]
    wide = wide[core_cols + rest] if core_cols else wide

    Path(wide_out).parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(wide_out, index=False)
    return wide_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creds", required=True, help="Path to MIDRC credentials JSON (Gen3 refresh file)")
    ap.add_argument("--out_dir", required=True, help="Output directory for mirror (parquet files)")
    ap.add_argument("--program", default="Open", help="Program name (default: Open)")
    ap.add_argument("--wide_out", default=None, help="Final wide parquet path (default: <out_dir>/midrc_files_wide.parquet)")
    args = ap.parse_args()

    path = build_midrc_graph(args.creds, args.out_dir, program=args.program, wide_out=args.wide_out)
    print(f"[OK] Wrote wide view: {path}")

if __name__ == "__main__":
    main()
