from __future__ import annotations

from fastapi import APIRouter

from app.models import DriftRequest, TrainRequest, TrainResponse
from app.services.ml_pipeline import evaluate_drift, train_dataset

router = APIRouter(prefix="/api/part2", tags=["part2-ml-pipeline"])


@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    artifacts = train_dataset(
        dataset_id=req.dataset_id,
        target_column=req.target_column,
        task=req.task,
        sensitive_column=req.sensitive_column,
    )
    gate = {
        "investigator": {
            "summary": "The model was trained on your experiment data. "
            + ("It passed" if artifacts.gate["pass"] else "It did not pass")
            + " the promotion gate.",
            "explanation": "We check overall performance, recall of positives, latency, and group fairness.",
        },
        "technical": artifacts.gate,
    }
    return TrainResponse(
        run_id=artifacts.run_id,
        dataset_hash=artifacts.dataset_hash,
        metrics=artifacts.metrics,
        production_gate=gate,
    )


@router.post("/drift")
def drift(req: DriftRequest):
    report = evaluate_drift(
        reference_dataset_id=req.reference_dataset_id,
        current_dataset_id=req.current_dataset_id,
        numeric_columns=req.numeric_columns,
    )
    return {
        "investigator": {
            "summary": "We detected changes in the incoming data distribution." if report["drift_detected"] else "No concerning drift was detected.",
            "drift_detected": report["drift_detected"],
        },
        "technical": report,
    }


@router.post("/deploy/canary")
def canary_deploy(run_id: str):
    return {
        "investigator": {
            "summary": "The new model will enter a controlled canary at 10% traffic for 48 hours before full promotion."
        },
        "technical": {
            "run_id": run_id,
            "strategy": "canary",
            "traffic": "10%",
            "validation_window_hours": 48,
            "next": "promote_if_pass",
            "champion_challenger_visible": True,
        },
    }
