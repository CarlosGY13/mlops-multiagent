from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.services.eda import eda_for_dataset
from app.services.storage import BASE_DATA, load_json


def _safe_str(x: Any, max_len: int = 90) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def build_dataset_context(dataset_id: str) -> Dict[str, Any]:
    schema_path = BASE_DATA / "curated" / f"{dataset_id}_schema.json"
    reasons_path = BASE_DATA / "quarantine" / f"{dataset_id}_reasons.json"
    curated_path = BASE_DATA / "curated" / f"{dataset_id}.csv"

    schema = _load_optional_json(schema_path)
    quality = _load_optional_json(reasons_path)

    preview: Dict[str, Any] = {"columns": [], "rows": []}
    if curated_path.exists():
        try:
            df = pd.read_csv(curated_path).head(8)
            preview["columns"] = df.columns.tolist()
            preview["rows"] = [
                {str(k): _safe_str(v) for k, v in row.items()} for row in df.to_dict(orient="records")
            ]
        except Exception:
            preview = {"columns": [], "rows": []}

    eda = {}
    try:
        eda = eda_for_dataset(dataset_id, target_column=None, bins=12)
    except Exception:
        eda = {}

    eda_brief: Dict[str, Any] = {}
    if eda:
        miss = eda.get("missingness") or {}
        miss_sorted = sorted(miss.items(), key=lambda kv: kv[1], reverse=True)[:8]
        feat = eda.get("features") if isinstance(eda.get("features"), dict) else {}
        corr = eda.get("correlation") if isinstance(eda.get("correlation"), dict) else {}
        top_pairs = (corr.get("top_pairs") or []) if isinstance(corr, dict) else []

        eda_brief = {
            "overview": eda.get("overview") or {},
            "top_missingness": [{"column": k, "rate": float(v)} for k, v in miss_sorted],
            "numeric_columns": feat.get("numeric_columns") or [],
            "categorical_columns": feat.get("categorical_columns") or [],
            "id_like_columns": feat.get("id_like_columns") or [],
            "target_candidates": feat.get("target_candidates") or [],
            "top_correlations": top_pairs[:6],
        }

    # Keep payload compact for LLM prompts.
    return {
        "dataset_id": dataset_id,
        "schema": schema or {},
        "quality": {
            "total_rows": quality.get("total_rows"),
            "curated_rows": quality.get("curated_rows"),
            "quarantine_rows": quality.get("quarantine_rows"),
            "quarantine_reasons": quality.get("quarantine_reasons", [])[:12],
            "input_filename": quality.get("input_filename"),
            "input_format": quality.get("input_format"),
        }
        if quality
        else {},
        "eda_brief": eda_brief,
        "preview": preview,
    }


def dataset_context_json(dataset_id: str, max_chars: int = 8000) -> str:
    ctx = build_dataset_context(dataset_id)
    txt = json.dumps(ctx, ensure_ascii=False, indent=2)
    if len(txt) <= max_chars:
        return txt
    return txt[: max_chars - 3] + "..."
