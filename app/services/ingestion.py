from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from app.services.storage import BASE_DATA, save_json


@dataclass
class IngestionArtifacts:
    dataset_id: str
    raw_path: Path
    curated_path: Path
    quarantine_path: Path
    schema: Dict[str, Any]
    quality_summary: Dict[str, Any]


def _dataset_id_from_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    columns = {}
    for col in df.columns:
        series = df[col]
        col_lower = col.lower()
        unit = None
        for token in ("mg", "kg", "cm", "mm", "ms", "s", "min", "%"):
            if token in col_lower:
                unit = token
                break

        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
            observed_range = [float(series.min(skipna=True)), float(series.max(skipna=True))]
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = "datetime"
            observed_range = [str(series.min(skipna=True)), str(series.max(skipna=True))]
        else:
            col_type = "categorical"
            observed_range = [str(series.dropna().astype(str).nunique()), "unique_values"]

        columns[col] = {
            "type": col_type,
            "observed_range": observed_range,
            "detected_unit": unit,
            "null_rate": float(series.isna().mean()),
        }
    return {"rows": int(df.shape[0]), "columns": columns}


def _quality_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    invalid_mask = df.isna().all(axis=1)
    reasons: List[Dict[str, Any]] = []

    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        invalid_mask = invalid_mask | outlier_mask.fillna(False)
        if outlier_mask.any():
            reasons.append({"column": col, "rule": "iqr_3x", "lower": float(lower), "upper": float(upper)})

    quarantine = df[invalid_mask].copy()
    curated = df[~invalid_mask].copy()

    return curated, quarantine, reasons


def ingest_csv(content: bytes, filename: str) -> IngestionArtifacts:
    dataset_id = _dataset_id_from_bytes(content)
    raw_path = BASE_DATA / "raw" / f"{dataset_id}_{filename}"
    raw_path.write_bytes(content)

    df = pd.read_csv(raw_path)
    schema = _infer_schema(df)
    curated, quarantine, reasons = _quality_split(df)

    curated_path = BASE_DATA / "curated" / f"{dataset_id}.csv"
    quarantine_path = BASE_DATA / "quarantine" / f"{dataset_id}.csv"

    curated.to_csv(curated_path, index=False)
    quarantine.to_csv(quarantine_path, index=False)

    quality_summary = {
        "total_rows": int(df.shape[0]),
        "curated_rows": int(curated.shape[0]),
        "quarantine_rows": int(quarantine.shape[0]),
        "quarantine_reasons": reasons,
    }
    save_json(BASE_DATA / "curated" / f"{dataset_id}_schema.json", schema)
    save_json(BASE_DATA / "quarantine" / f"{dataset_id}_reasons.json", quality_summary)

    return IngestionArtifacts(
        dataset_id=dataset_id,
        raw_path=raw_path,
        curated_path=curated_path,
        quarantine_path=quarantine_path,
        schema=schema,
        quality_summary=quality_summary,
    )
