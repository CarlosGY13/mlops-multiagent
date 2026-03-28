from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import (
    AzureMLDeployRequest,
    AzureMLDeployResponse,
    AzureMLJobStatusResponse,
    AzureMLTrainRequest,
    AzureMLTrainResponse,
    DriftRequest,
    TrainRequest,
    TrainResponse,
)
from app.services.azure_ml import AzureMLNotConfigured, deploy_from_job, get_job_status, submit_training_job
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


@router.post("/aml/train", response_model=AzureMLTrainResponse)
def aml_train(req: AzureMLTrainRequest) -> AzureMLTrainResponse:
    try:
        out = submit_training_job(
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            drop_columns=req.drop_columns,
            task=req.task,
            model_candidates=req.model_candidates,
        )
        return AzureMLTrainResponse(**out)
    except AzureMLNotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/aml/jobs/{job_id}", response_model=AzureMLJobStatusResponse)
def aml_job_status(job_id: str) -> AzureMLJobStatusResponse:
    try:
        out = get_job_status(job_id)
        return AzureMLJobStatusResponse(**out)
    except AzureMLNotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/aml/deploy", response_model=AzureMLDeployResponse)
def aml_deploy(req: AzureMLDeployRequest) -> AzureMLDeployResponse:
    try:
        out = deploy_from_job(job_id=req.job_id, model_id=req.model_id, endpoint_name=req.endpoint_name)
        return AzureMLDeployResponse(**out)
    except AzureMLNotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
