from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from core.state import ConversationState, Task, TaskResult
from core.interactions import confirm_operation, notify_user
from core.storage import persist_root
from progress_ui import update_progress
from tools.shared import _cs, toolify_agent


ChecksumMode = Literal["none", "manifest", "files"]
MAX_BATCH_SERIES = 20_000
DEFAULT_DIR_TEMPLATE = "%collection_id/%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe_number(value: Any, *, integer: bool = False) -> Optional[int | float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if integer else round(number, 6)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _read_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"IDC series manifest does not exist: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).where(pd.notna, None).to_dict("records")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("IDC series manifest must be valid JSON or CSV.") from exc
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("series") or payload.get("rows") or []
    else:
        rows = []
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("IDC series manifest must contain a list of row objects.")
    return rows


def _manifest_uids(rows: List[Dict[str, Any]]) -> List[str]:
    aliases = ("SeriesInstanceUID", "series_instance_uid", "series_uid")
    uids: List[str] = []
    for row in rows:
        value = next((row.get(key) for key in aliases if row.get(key)), None)
        if value:
            uids.append(str(value).strip())
    return uids


class IDCDownloadAgent:
    name = "idc_download"
    model = None

    def __init__(
        self,
        client_factory: Optional[Callable[[], Any]] = None,
        download_runner: Optional[
            Callable[[str, Path, bool, int], Any]
        ] = None,
    ) -> None:
        self._client_factory = client_factory or self._default_client_factory
        self._download_runner = download_runner

    @staticmethod
    def _default_client_factory() -> Any:
        from idc_index import IDCClient

        return IDCClient()

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        del state
        kw = task.kwargs or {}
        timeout_s = int(kw.get("timeout_s", 3600))
        max_concurrency = max(1, min(16, int(kw.get("max_concurrency", 4))))
        max_retries = max(0, min(5, int(kw.get("max_retries", 2))))
        resume = bool(kw.get("resume", True))
        checksum_mode: ChecksumMode = str(kw.get("checksum_mode", "manifest"))  # type: ignore[assignment]
        if checksum_mode not in {"none", "manifest", "files"}:
            raise ValueError("checksum_mode must be one of: none, manifest, files")

        try:
            supplied_uids = self._selection_uids(
                series_uid=kw.get("series_uid"),
                series_uids=kw.get("series_uids") or [],
                manifest_path=kw.get("manifest_path"),
            )
            expected_series_count = kw.get("expected_series_count")
            if (
                expected_series_count is not None
                and len(supplied_uids) != int(expected_series_count)
            ):
                raise ValueError(
                    "IDC manifest cardinality mismatch: expected "
                    f"{int(expected_series_count)} series, found {len(supplied_uids)}. "
                    "Create an exact manifest; prose cannot narrow a registered table."
                )
            client = self._client_factory()
            manifest_rows = self._authoritative_manifest(client, supplied_uids)
            expected_patient_count = kw.get("expected_patient_count")
            actual_patient_count = len(
                {str(row.get("PatientID") or "") for row in manifest_rows}
                - {""}
            )
            if (
                expected_patient_count is not None
                and actual_patient_count != int(expected_patient_count)
            ):
                raise ValueError(
                    "IDC manifest patient cardinality mismatch: expected "
                    f"{int(expected_patient_count)} patients, found {actual_patient_count}."
                )
            idc_data_version = str(getattr(client, "idc_version", None) or "unknown")
        except Exception as exc:
            return TaskResult(
                output=f"IDC download manifest preparation failed: {exc}",
                status="error",
                errors=[f"execution_failed: {exc}"],
            )

        selection_hash = hashlib.sha256(
            (
                idc_data_version
                + "\n"
                + "\n".join(
                    sorted(row["SeriesInstanceUID"] for row in manifest_rows)
                )
            ).encode()
        ).hexdigest()
        job_id = f"idc-{selection_hash[:20]}"
        job_root = persist_root() / self.name / "jobs" / job_id
        download_root = job_root / "dicom"
        manifest_path = job_root / "series_manifest.json"
        status_path = job_root / "status.json"
        series_status_path = job_root / "series_status.csv"
        patient_status_path = job_root / "patient_status.csv"
        download_root.mkdir(parents=True, exist_ok=True)

        manifest_payload = {
            "schema_version": "voxelinsight.idc-download-manifest.v1",
            "job_id": job_id,
            "selection_sha256": selection_hash,
            "idc_data_version": idc_data_version,
            "created_at": _utc_now(),
            "series_count": len(manifest_rows),
            "patient_count": len({row["PatientID"] for row in manifest_rows}),
            "estimated_size_mb": round(
                sum(float(row.get("series_size_MB") or 0) for row in manifest_rows), 3
            ),
            "series": manifest_rows,
        }
        if not manifest_path.exists():
            _atomic_write_json(manifest_path, manifest_payload)

        ledger = self._initial_ledger(
            job_id=job_id,
            selection_hash=selection_hash,
            manifest_rows=manifest_rows,
            previous=self._load_ledger(status_path) if resume else {},
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            checksum_mode=checksum_mode,
        )
        if resume:
            self._refresh_completed_entries(ledger, download_root, checksum_mode)
        self._write_job_outputs(
            ledger,
            status_path=status_path,
            series_status_path=series_status_path,
            patient_status_path=patient_status_path,
        )

        completed_before = sum(
            entry.get("status") == "completed" for entry in ledger["series"].values()
        )
        estimated_gb = manifest_payload["estimated_size_mb"] / 1000.0
        confirmation_content = (
            "Start this IDC cohort download?\n\n"
            f"- Patients: {manifest_payload['patient_count']}\n"
            f"- Series: {manifest_payload['series_count']}\n"
            f"- Estimated size: {estimated_gb:.2f} GB\n"
            f"- Already complete/resumable: {completed_before}\n"
            f"- Concurrent series downloads: {max_concurrency}\n"
            f"- Retries per series: {max_retries}\n"
            f"- Verification: instance counts + {checksum_mode} checksum mode\n\n"
            "One confirmation submits the complete deterministic batch job."
        )
        confirmed = await confirm_operation(
            kind="idc_download",
            content=confirmation_content,
            details={
                "patients": manifest_payload["patient_count"],
                "series": manifest_payload["series_count"],
                "estimated_size_gb": estimated_gb,
                "completed_before": completed_before,
                "max_concurrency": max_concurrency,
                "max_retries": max_retries,
                "checksum_mode": checksum_mode,
            },
        )

        if not confirmed:
            for entry in ledger["series"].values():
                if entry.get("status") != "completed":
                    entry["status"] = "cancelled"
                    entry["last_error"] = (
                        "Batch was cancelled before transfer; completed files were preserved."
                    )
            ledger["job_status"] = "cancelled"
            ledger["updated_at"] = _utc_now()
            self._write_job_outputs(
                ledger,
                status_path=status_path,
                series_status_path=series_status_path,
                patient_status_path=patient_status_path,
            )
            await notify_user(
                "IDC cohort download cancelled; any previously completed series were preserved."
            )
            return self._job_result(
                ledger,
                download_root=download_root,
                manifest_path=manifest_path,
                status_path=status_path,
                series_status_path=series_status_path,
                patient_status_path=patient_status_path,
                cancelled=True,
            )

        await notify_user("Starting the IDC cohort batch download...")
        ledger["job_status"] = "running"
        ledger["updated_at"] = _utc_now()
        self._write_job_outputs(
            ledger,
            status_path=status_path,
            series_status_path=series_status_path,
            patient_status_path=patient_status_path,
        )
        await update_progress(
            int(100 * completed_before / max(1, len(manifest_rows))),
            f"IDC batch: {completed_before}/{len(manifest_rows)} series complete",
        )

        pending = [
            uid
            for uid, entry in ledger["series"].items()
            if entry.get("status") != "completed"
        ]
        semaphore = asyncio.Semaphore(max_concurrency)
        ledger_lock = asyncio.Lock()

        async def download_one(uid: str) -> None:
            async with semaphore:
                entry = ledger["series"][uid]
                for _attempt in range(max_retries + 1):
                    async with ledger_lock:
                        entry["status"] = "running"
                        entry["attempts"] = int(entry.get("attempts") or 0) + 1
                        entry["started_at"] = entry.get("started_at") or _utc_now()
                        entry["last_error"] = ""
                        ledger["updated_at"] = _utc_now()
                        self._write_job_outputs(
                            ledger,
                            status_path=status_path,
                            series_status_path=series_status_path,
                            patient_status_path=patient_status_path,
                        )
                    try:
                        await self._download_series(
                            uid,
                            download_root,
                            resume=resume,
                            timeout_s=timeout_s,
                        )
                        verification = self._verify_series(
                            download_root,
                            uid,
                            expected_instance_count=entry.get("expected_instance_count"),
                            checksum_mode=checksum_mode,
                        )
                        if not verification["verified"]:
                            raise RuntimeError(verification["error"])
                        async with ledger_lock:
                            entry.update(verification)
                            entry["status"] = "completed"
                            entry["finished_at"] = _utc_now()
                            entry["last_error"] = ""
                            await self._persist_progress(
                                ledger,
                                status_path=status_path,
                                series_status_path=series_status_path,
                                patient_status_path=patient_status_path,
                            )
                        return
                    except asyncio.CancelledError:
                        async with ledger_lock:
                            entry["status"] = "cancelled"
                            entry["last_error"] = "Batch task cancelled; completed files were preserved."
                            ledger["updated_at"] = _utc_now()
                            self._write_job_outputs(
                                ledger,
                                status_path=status_path,
                                series_status_path=series_status_path,
                                patient_status_path=patient_status_path,
                            )
                        raise
                    except Exception as exc:
                        async with ledger_lock:
                            entry["status"] = "failed"
                            entry["last_error"] = str(exc)[:1000]
                            ledger["updated_at"] = _utc_now()
                            self._write_job_outputs(
                                ledger,
                                status_path=status_path,
                                series_status_path=series_status_path,
                                patient_status_path=patient_status_path,
                            )
                        if _attempt >= max_retries:
                            return
                        await asyncio.sleep(min(8, 2 ** max(0, int(entry["attempts"]) - 1)))

        tasks = [asyncio.create_task(download_one(uid)) for uid in pending]
        try:
            if tasks:
                await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for running_task in tasks:
                running_task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.gather(*tasks, return_exceptions=True)
            for entry in ledger["series"].values():
                if entry.get("status") in {"pending", "running"}:
                    entry["status"] = "cancelled"
                    entry["last_error"] = "Batch task cancelled; completed files were preserved."
            ledger["job_status"] = "cancelled"
            ledger["updated_at"] = _utc_now()
            self._write_job_outputs(
                ledger,
                status_path=status_path,
                series_status_path=series_status_path,
                patient_status_path=patient_status_path,
            )
            raise

        failed = sum(entry.get("status") == "failed" for entry in ledger["series"].values())
        completed = sum(
            entry.get("status") == "completed" for entry in ledger["series"].values()
        )
        ledger["job_status"] = "completed" if failed == 0 else "partial"
        ledger["updated_at"] = _utc_now()
        self._write_job_outputs(
            ledger,
            status_path=status_path,
            series_status_path=series_status_path,
            patient_status_path=patient_status_path,
        )
        await update_progress(
            int(100 * completed / max(1, len(manifest_rows))),
            f"IDC batch finished: {completed} complete, {failed} failed",
        )
        return self._job_result(
            ledger,
            download_root=download_root,
            manifest_path=manifest_path,
            status_path=status_path,
            series_status_path=series_status_path,
            patient_status_path=patient_status_path,
            cancelled=False,
        )

    def _selection_uids(
        self,
        *,
        series_uid: Optional[str],
        series_uids: List[str],
        manifest_path: Optional[str],
    ) -> List[str]:
        supplied: List[str] = []
        if manifest_path:
            supplied.extend(_manifest_uids(_read_manifest_rows(Path(manifest_path))))
        if series_uid:
            supplied.append(str(series_uid))
        supplied.extend(str(uid) for uid in series_uids if uid)
        deduped = list(dict.fromkeys(uid.strip() for uid in supplied if uid.strip()))
        if not deduped:
            raise ValueError("Provide series_uid, series_uids, or manifest_path.")
        if len(deduped) > MAX_BATCH_SERIES:
            raise ValueError(
                f"IDC batch contains {len(deduped)} series; maximum is {MAX_BATCH_SERIES}."
            )
        return deduped

    @staticmethod
    def _authoritative_manifest(client: Any, requested_uids: List[str]) -> List[Dict[str, Any]]:
        index = client.index
        required = {
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "PatientID",
            "collection_id",
            "Modality",
        }
        missing_columns = sorted(required - set(index.columns))
        if missing_columns:
            raise ValueError(
                "IDC index is missing required manifest columns: " + ", ".join(missing_columns)
            )
        selected = index[index["SeriesInstanceUID"].astype(str).isin(requested_uids)].copy()
        selected = selected.drop_duplicates(subset=["SeriesInstanceUID"], keep="first")
        by_uid = {
            str(row["SeriesInstanceUID"]): row
            for row in selected.to_dict("records")
        }
        unknown = [uid for uid in requested_uids if uid not in by_uid]
        if unknown:
            preview = ", ".join(unknown[:5])
            raise ValueError(
                f"{len(unknown)} SeriesInstanceUID value(s) are not present in this IDC release: {preview}"
            )
        rows: List[Dict[str, Any]] = []
        for uid in requested_uids:
            row = by_uid[uid]
            rows.append(
                {
                    "collection_id": str(row.get("collection_id") or ""),
                    "PatientID": str(row.get("PatientID") or ""),
                    "StudyInstanceUID": str(row.get("StudyInstanceUID") or ""),
                    "SeriesInstanceUID": uid,
                    "Modality": str(row.get("Modality") or ""),
                    "SeriesDescription": str(row.get("SeriesDescription") or ""),
                    "expected_instance_count": _json_safe_number(
                        row.get("instanceCount"), integer=True
                    ),
                    "series_size_MB": _json_safe_number(row.get("series_size_MB")) or 0.0,
                }
            )
        return rows

    @staticmethod
    def _load_ledger(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _initial_ledger(
        *,
        job_id: str,
        selection_hash: str,
        manifest_rows: List[Dict[str, Any]],
        previous: Dict[str, Any],
        max_concurrency: int,
        max_retries: int,
        checksum_mode: ChecksumMode,
    ) -> Dict[str, Any]:
        previous_series = previous.get("series") if isinstance(previous.get("series"), dict) else {}
        entries: Dict[str, Dict[str, Any]] = {}
        for row in manifest_rows:
            uid = row["SeriesInstanceUID"]
            prior = previous_series.get(uid) if isinstance(previous_series.get(uid), dict) else {}
            entries[uid] = {
                "SeriesInstanceUID": uid,
                "StudyInstanceUID": row["StudyInstanceUID"],
                "PatientID": row["PatientID"],
                "collection_id": row["collection_id"],
                "Modality": row["Modality"],
                "SeriesDescription": row["SeriesDescription"],
                "expected_instance_count": row["expected_instance_count"],
                "actual_instance_count": prior.get("actual_instance_count"),
                "series_size_MB": row["series_size_MB"],
                "status": prior.get("status", "pending"),
                "attempts": int(prior.get("attempts") or 0),
                "output_dir": prior.get("output_dir", ""),
                "checksum_sha256": prior.get("checksum_sha256", ""),
                "started_at": prior.get("started_at", ""),
                "finished_at": prior.get("finished_at", ""),
                "last_error": prior.get("last_error", ""),
            }
        return {
            "schema_version": "voxelinsight.idc-download-status.v1",
            "job_id": job_id,
            "selection_sha256": selection_hash,
            "job_status": previous.get("job_status", "pending"),
            "created_at": previous.get("created_at", _utc_now()),
            "updated_at": _utc_now(),
            "configuration": {
                "max_concurrency": max_concurrency,
                "max_retries": max_retries,
                "checksum_mode": checksum_mode,
            },
            "series": entries,
            "patients": {},
        }

    def _refresh_completed_entries(
        self,
        ledger: Dict[str, Any],
        download_root: Path,
        checksum_mode: ChecksumMode,
    ) -> None:
        for uid, entry in ledger["series"].items():
            if entry.get("status") != "completed":
                if entry.get("status") in {"running", "cancelled"}:
                    entry["status"] = "pending"
                continue
            verification = self._verify_series(
                download_root,
                uid,
                expected_instance_count=entry.get("expected_instance_count"),
                checksum_mode=checksum_mode,
            )
            if verification["verified"]:
                entry.update(verification)
            else:
                entry["status"] = "pending"
                entry["last_error"] = "Resume verification failed: " + verification["error"]

    async def _download_series(
        self,
        uid: str,
        download_root: Path,
        *,
        resume: bool,
        timeout_s: int,
    ) -> None:
        if self._download_runner is not None:
            result = self._download_runner(uid, download_root, resume, timeout_s)
            if asyncio.iscoroutine(result):
                await result
            return

        executable = shutil.which("idc")
        if not executable:
            candidate = Path(sys.executable).with_name("idc")
            executable = str(candidate) if candidate.is_file() else ""
        if not executable:
            raise RuntimeError("The idc-index CLI executable `idc` is unavailable.")
        command = [
            executable,
            "download-from-selection",
            "--download-dir",
            str(download_root),
            "--series-instance-uid",
            uid,
            "--quiet",
            "true",
            "--show-progress-bar",
            "false",
            "--use-s5cmd-sync",
            "true" if resume else "false",
            "--dir-template",
            DEFAULT_DIR_TEMPLATE,
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError as exc:
            await self._terminate_process_group(process)
            raise RuntimeError(
                f"IDC series download timed out after {timeout_s} seconds."
            ) from exc
        except asyncio.CancelledError:
            await self._terminate_process_group(process)
            raise
        if process.returncode != 0:
            detail = stderr.decode(errors="ignore").strip()
            if not detail:
                detail = stdout.decode(errors="ignore").strip()
            raise RuntimeError(
                f"IDC CLI failed with rc={process.returncode}: {detail[:1000]}"
            )

    @staticmethod
    async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
            return
        except (asyncio.TimeoutError, ProcessLookupError):
            pass
        with contextlib.suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        with contextlib.suppress(Exception):
            await process.wait()

    @staticmethod
    def _find_series_dir(root: Path, uid: str) -> Optional[Path]:
        matches = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_dir()
                and (path.name == uid or path.name.endswith(f"_{uid}"))
            ),
            key=lambda path: (len(path.parts), str(path)),
        )
        return matches[0] if matches else None

    def _verify_series(
        self,
        root: Path,
        uid: str,
        *,
        expected_instance_count: Optional[int],
        checksum_mode: ChecksumMode,
    ) -> Dict[str, Any]:
        series_dir = self._find_series_dir(root, uid)
        if series_dir is None:
            return {
                "verified": False,
                "error": "Downloaded series directory was not found.",
                "actual_instance_count": 0,
                "output_dir": "",
                "checksum_sha256": "",
            }
        files = sorted(path for path in series_dir.rglob("*") if path.is_file())
        actual = len(files)
        expected = int(expected_instance_count) if expected_instance_count is not None else None
        if actual == 0:
            error = "Downloaded series directory contains no files."
        elif expected is not None and actual != expected:
            error = f"Instance-count mismatch: expected {expected}, found {actual}."
        else:
            error = ""
        checksum = self._series_checksum(series_dir, files, checksum_mode) if not error else ""
        return {
            "verified": not bool(error),
            "error": error,
            "actual_instance_count": actual,
            "output_dir": str(series_dir.resolve()),
            "checksum_sha256": checksum,
        }

    @staticmethod
    def _series_checksum(root: Path, files: List[Path], mode: ChecksumMode) -> str:
        if mode == "none":
            return ""
        digest = hashlib.sha256()
        for path in files:
            relative = str(path.relative_to(root))
            digest.update(relative.encode())
            digest.update(str(path.stat().st_size).encode())
            if mode == "files":
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _patient_rows(ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for entry in ledger["series"].values():
            grouped.setdefault(str(entry.get("PatientID") or "[missing]"), []).append(entry)
        rows: List[Dict[str, Any]] = []
        for patient_id, entries in sorted(grouped.items()):
            statuses = [str(entry.get("status") or "pending") for entry in entries]
            if all(status == "completed" for status in statuses):
                patient_status = "completed"
            elif any(status == "failed" for status in statuses):
                patient_status = "partial" if any(status == "completed" for status in statuses) else "failed"
            elif any(status == "cancelled" for status in statuses):
                patient_status = "partial" if any(status == "completed" for status in statuses) else "cancelled"
            elif any(status == "running" for status in statuses):
                patient_status = "running"
            else:
                patient_status = "pending"
            rows.append(
                {
                    "PatientID": patient_id,
                    "status": patient_status,
                    "series_total": len(entries),
                    "series_completed": sum(status == "completed" for status in statuses),
                    "series_failed": sum(status == "failed" for status in statuses),
                    "expected_instances": sum(
                        int(entry.get("expected_instance_count") or 0) for entry in entries
                    ),
                    "actual_instances": sum(
                        int(entry.get("actual_instance_count") or 0) for entry in entries
                    ),
                }
            )
        return rows

    def _write_job_outputs(
        self,
        ledger: Dict[str, Any],
        *,
        status_path: Path,
        series_status_path: Path,
        patient_status_path: Path,
    ) -> None:
        patient_rows = self._patient_rows(ledger)
        ledger["patients"] = {
            row["PatientID"]: {key: value for key, value in row.items() if key != "PatientID"}
            for row in patient_rows
        }
        _atomic_write_json(status_path, ledger)
        pd.DataFrame(list(ledger["series"].values())).to_csv(series_status_path, index=False)
        pd.DataFrame(patient_rows).to_csv(patient_status_path, index=False)

    async def _persist_progress(
        self,
        ledger: Dict[str, Any],
        *,
        status_path: Path,
        series_status_path: Path,
        patient_status_path: Path,
    ) -> None:
        ledger["updated_at"] = _utc_now()
        self._write_job_outputs(
            ledger,
            status_path=status_path,
            series_status_path=series_status_path,
            patient_status_path=patient_status_path,
        )
        total = len(ledger["series"])
        complete = sum(
            entry.get("status") == "completed" for entry in ledger["series"].values()
        )
        failed = sum(entry.get("status") == "failed" for entry in ledger["series"].values())
        await update_progress(
            int(100 * complete / max(1, total)),
            f"IDC batch: {complete}/{total} complete, {failed} failed",
        )

    def _job_result(
        self,
        ledger: Dict[str, Any],
        *,
        download_root: Path,
        manifest_path: Path,
        status_path: Path,
        series_status_path: Path,
        patient_status_path: Path,
        cancelled: bool,
    ) -> TaskResult:
        series_entries = list(ledger["series"].values())
        patient_rows = self._patient_rows(ledger)
        completed_dirs = [
            str(entry["output_dir"])
            for entry in series_entries
            if entry.get("status") == "completed" and entry.get("output_dir")
        ]
        complete = sum(entry.get("status") == "completed" for entry in series_entries)
        failed = sum(entry.get("status") == "failed" for entry in series_entries)
        cancelled_count = sum(entry.get("status") == "cancelled" for entry in series_entries)
        summary = (
            f"IDC cohort job {ledger['job_id']}: {complete}/{len(series_entries)} series "
            f"complete, {failed} failed, {cancelled_count} cancelled."
        )
        files = [
            str(manifest_path),
            str(status_path),
            str(series_status_path),
            str(patient_status_path),
        ]
        if cancelled:
            status = "partial" if complete else "no_action"
            errors = ["cancelled: IDC batch stopped; completed series and status were preserved."]
        elif failed:
            status = "partial"
            errors = [f"execution_failed: {failed} series failed after configured retries."]
        else:
            status = "ok"
            errors = []
        return TaskResult(
            output={
                "text": summary,
                "job_id": ledger["job_id"],
                "selection_sha256": ledger["selection_sha256"],
                "series_total": len(series_entries),
                "series_completed": complete,
                "series_failed": failed,
                "series_cancelled": cancelled_count,
                "patient_total": len(patient_rows),
                "series_status": pd.DataFrame(series_entries),
                "patient_status": pd.DataFrame(patient_rows),
                "manifest_path": str(manifest_path),
                "status_path": str(status_path),
                "files": files,
                "dicom_dirs": completed_dirs,
                "dicom_dir": completed_dirs[0] if len(completed_dirs) == 1 else None,
                "output_dir": str(download_root) if complete else None,
                "tool": self.name,
            },
            artifacts={
                "files": files,
                "dicom_dirs": completed_dirs,
                "dicom_dir": completed_dirs[0] if len(completed_dirs) == 1 else None,
                "output_dir": str(download_root) if complete else None,
            },
            status=status,
            errors=errors,
        )


_DL: Optional[IDCDownloadAgent] = None


def configure_idc_download_tool() -> None:
    global _DL
    _DL = IDCDownloadAgent()


class IDCDownloadArgs(BaseModel):
    series_uid: Optional[str] = Field(None, description="Single SeriesInstanceUID.")
    series_uids: Optional[List[str]] = Field(
        None,
        description="Exact SeriesInstanceUID values for one deterministic batch job.",
    )
    manifest_path: Optional[str] = Field(
        None,
        description=(
            "JSON/CSV series manifest path, or a registered IDC series-search data/artifact "
            "reference resolved by middleware. Must contain SeriesInstanceUID values."
        ),
    )
    expected_series_count: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_BATCH_SERIES,
        description=(
            "Expected exact series cardinality from the registered manifest's nrows/counts. "
            "This is not the requested patient count unless scope is representative. The tool "
            "refuses mismatches."
        ),
    )
    expected_patient_count: Optional[int] = Field(
        default=None,
        ge=1,
        le=500,
        description=(
            "Expected exact distinct-patient cardinality from the registered manifest. "
            "Use this independently of expected_series_count."
        ),
    )
    timeout_s: int = Field(
        default=3600,
        ge=30,
        le=86400,
        description="Timeout seconds for each individual series attempt.",
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum concurrent series downloads inside the deterministic batch job.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries after the first attempt for each failed or incomplete series.",
    )
    resume: bool = Field(
        default=True,
        description="Reuse completed series and resume partial downloads for the same UID manifest.",
    )
    checksum_mode: ChecksumMode = Field(
        default="manifest",
        description=(
            "Verification checksum mode: none, manifest (paths and sizes), or files "
            "(content hashing). Instance counts are always validated when available."
        ),
    )


@toolify_agent(
    name="idc_download",
    description=(
        "Submit one confirmed, resumable IDC download job for one series or an entire exact "
        "SeriesInstanceUID manifest. The tool performs bounded concurrency, per-series retries, "
        "separate expected patient/series cardinality checks, per-patient status tracking, "
        "instance-count/checksum verification, and durable resume internally. For cohorts, call "
        "this tool exactly once; never loop over patients or series."
    ),
    args_schema=IDCDownloadArgs,
    timeout_s=86400,
)
async def idc_download_runner(
    series_uid: Optional[str] = None,
    series_uids: Optional[List[str]] = None,
    manifest_path: Optional[str] = None,
    expected_series_count: Optional[int] = None,
    expected_patient_count: Optional[int] = None,
    timeout_s: int = 3600,
    max_concurrency: int = 4,
    max_retries: int = 2,
    resume: bool = True,
    checksum_mode: ChecksumMode = "manifest",
) -> TaskResult:
    if _DL is None:
        raise RuntimeError(
            "IDC download tool not configured. Call configure_idc_download_tool() first."
        )
    task = Task(
        user_msg="Download IDC files",
        files=[],
        kwargs={
            "series_uid": series_uid,
            "series_uids": series_uids,
            "manifest_path": manifest_path,
            "expected_series_count": expected_series_count,
            "expected_patient_count": expected_patient_count,
            "timeout_s": timeout_s,
            "max_concurrency": max_concurrency,
            "max_retries": max_retries,
            "resume": resume,
            "checksum_mode": checksum_mode,
        },
    )
    return await _DL.run(task, _cs())
