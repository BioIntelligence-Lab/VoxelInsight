from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class TSMappingResolver:
    def __init__(self) -> None:
        self.allowed_by_task = self._load_mappings()
        self._llm = ChatOpenAI(model="gpt-5-nano", temperature=0, reasoning_effort="low")

    @staticmethod
    def _normalize_token(text: str) -> str:
        return text.strip().lower().replace(" ", "_").replace("-", "_")

    def _load_mappings(self) -> Dict[str, List[str]]:
        mapping_files = (
            Path("Data/TotalSegmentatorMappingsCT.tsv"),
            Path("Data/TotalSegmentatorMappingsMRI.tsv"),
        )
        rows: Dict[str, set[str]] = {}
        for mapping_file in mapping_files:
            if not mapping_file.exists():
                continue
            with mapping_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    task = self._normalize_token(str(row.get("task_name", "")))
                    roi = self._normalize_token(str(row.get("roi_subset", "")))
                    if not task or not roi:
                        continue
                    rows.setdefault(task, set()).add(roi)
        return {k: sorted(v) for k, v in rows.items()}

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        s = (text or "").strip()
        if s.startswith("```"):
            s = s.strip("`")
            if "\n" in s:
                s = s.split("\n", 1)[1]
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Resolver returned non-JSON output.")
        return json.loads(s[start : end + 1])

    def _format_mapping_text(self) -> str:
        lines: List[str] = []
        for task, rois in sorted(self.allowed_by_task.items()):
            lines.append(f"{task}: {', '.join(rois)}")
        return "\n".join(lines)

    def _collect_requested_rois(self, args: Dict[str, Any]) -> List[str]:
        rois: List[str] = []
        roi_subset = args.get("roi_subset")
        roi_subsets = args.get("roi_subsets")
        if isinstance(roi_subsets, list):
            rois.extend([str(x) for x in roi_subsets if x is not None and str(x).strip()])
        elif isinstance(roi_subsets, str) and roi_subsets.strip():
            rois.append(roi_subsets)
        if isinstance(roi_subset, list):
            rois.extend([str(x) for x in roi_subset if x is not None and str(x).strip()])
        elif isinstance(roi_subset, str) and roi_subset.strip():
            rois.append(roi_subset)
        return rois

    async def resolve_imaging_args(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        task_name = str(args.get("task_name", "total"))
        requested_rois = self._collect_requested_rois(args)

        system_prompt = (
            "You resolve TotalSegmentator mapping inputs. "
            "Use only the allowed mapping values provided by the user message. "
            "Output strict JSON only with keys: task_name (string), roi_subsets (array of strings). "
            "If no ROI should be used, set roi_subsets to []."
        )
        user_prompt = (
            "Allowed mappings:\n"
            f"{self._format_mapping_text()}\n\n"
            "Requested call:\n"
            f"task_name={task_name}\n"
            f"roi_inputs={requested_rois}\n\n"
            "Rules:\n"
            "- Return canonical mapping values only.\n"
            "- If task_name is not in mapping keys, keep the task_name but return roi_subsets=[].\n"
            "- For mapped tasks, include only valid ROIs from allowed mappings.\n"
            "- Remove duplicates.\n"
        )

        raw = await self._llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        payload = self._extract_json(getattr(raw, "content", str(raw)))

        resolved_task = self._normalize_token(str(payload.get("task_name", task_name))) or self._normalize_token(task_name)
        resolved_rois_raw = payload.get("roi_subsets", [])
        resolved_rois: List[str] = []
        if isinstance(resolved_rois_raw, list):
            for item in resolved_rois_raw:
                token = self._normalize_token(str(item))
                if token:
                    resolved_rois.append(token)
        resolved_rois = list(dict.fromkeys(resolved_rois))

        updated = dict(args)
        updated["task_name"] = resolved_task
        if resolved_rois:
            updated["roi_subsets"] = resolved_rois
            updated["roi_subset"] = None
        else:
            updated["roi_subsets"] = None
            updated["roi_subset"] = None

        resolution_note = (
            f"task_name: {task_name} -> {resolved_task}; "
            f"rois: {requested_rois} -> {resolved_rois}"
        )
        return updated, resolution_note

