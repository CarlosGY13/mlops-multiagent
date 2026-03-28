from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.models import IngestResponse
from app.services.ingestion import ingest_csv

router = APIRouter(prefix="/api/part1", tags=["part1-ingestion-quality"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    content = await file.read()
    artifacts = ingest_csv(content=content, filename=file.filename or "dataset.csv")

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
