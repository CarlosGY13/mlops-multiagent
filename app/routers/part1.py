from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models import IngestResponse
from app.services.eda import eda_for_dataset
from app.services.ingestion import ingest_dataset
from app.services.storage import BASE_DATA

import pandas as pd

router = APIRouter(prefix="/api/part1", tags=["part1-ingestion-quality"])


@router.get("/curated/sample")
def curated_sample(dataset_id: str, limit: int = 12):
    path = BASE_DATA / "curated" / f"{dataset_id}.csv"
    if not path.exists():
        return {
            "investigator": {"summary": "No curated file available for this dataset."},
            "technical": {"columns": [], "rows": [], "limit": limit},
        }

    df = pd.read_csv(path).head(limit)
    return {
        "investigator": {"summary": f"Showing up to {limit} curated rows."},
        "technical": {"columns": df.columns.tolist(), "rows": df.to_dict(orient="records"), "limit": limit},
    }


@router.get("/quarantine/sample")
def quarantine_sample(dataset_id: str, limit: int = 12):
    path = BASE_DATA / "quarantine" / f"{dataset_id}.csv"
    if not path.exists():
        return {
            "investigator": {"summary": "No quarantine file available for this dataset."},
            "technical": {"columns": [], "rows": [], "limit": limit},
        }

    df = pd.read_csv(path).head(limit)
    return {
        "investigator": {"summary": f"Showing up to {limit} quarantined rows."},
        "technical": {"columns": df.columns.tolist(), "rows": df.to_dict(orient="records"), "limit": limit},
    }


@router.get("/eda")
def eda(dataset_id: str, target_column: Optional[str] = None, bins: int = 12):
    if bins < 5 or bins > 60:
        raise HTTPException(status_code=400, detail="bins must be between 5 and 60")

    try:
        report = eda_for_dataset(dataset_id, target_column=target_column, bins=bins)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    inv = {
        "summary": "Exploratory analysis of your curated dataset.",
        "high_missing_columns": [k for k, v in sorted(report.get("missingness", {}).items(), key=lambda kv: kv[1], reverse=True)[:5] if v >= 0.05],
        "top_correlations": report.get("correlation", {}).get("top_pairs", [])[:5],
    }

    return {"investigator": inv, "technical": report}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    content = await file.read()
    artifacts = ingest_dataset(content=content, filename=file.filename or "dataset.csv")

    quality = {
        "investigator": {
            "summary": (
                f"We reviewed {artifacts.quality_summary['total_rows']} rows. "
                f"{artifacts.quality_summary['quarantine_rows']} were moved to quarantine with documented reasons."
            ),
            "plain_language": {
                "rows in quarantine": "rows separated for review (anomalous or incomplete); nothing was silently dropped",
            },
        },
        "technical": artifacts.quality_summary,
    }

    return IngestResponse(dataset_id=artifacts.dataset_id, schema_info=artifacts.schema, quality=quality)
