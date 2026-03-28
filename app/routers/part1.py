from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models import IngestResponse
from app.services.eda import eda_for_dataset
from app.services.ingestion import ingest_dataset

router = APIRouter(prefix="/api/part1", tags=["part1-ingestion-quality"])


@router.get("/eda")
def eda(dataset_id: str):
    try:
        report = eda_for_dataset(dataset_id)
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
