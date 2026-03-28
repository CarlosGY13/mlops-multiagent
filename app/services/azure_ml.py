from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.config import get_settings
from app.services.storage import BASE_DATA


class AzureMLNotConfigured(RuntimeError):
    pass


def _lazy_imports():
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore
        from azure.ai.ml import MLClient  # type: ignore
        from azure.ai.ml.entities import Environment, AmlCompute  # type: ignore
        from azure.ai.ml import command, Input  # type: ignore
        from azure.core.exceptions import ResourceNotFoundError  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Azure ML SDK import failed. Ensure you're running the server from the same venv where dependencies were installed. "
            "Try: `source .venv/bin/activate && pip install -r requirements.txt` and start with `python -m uvicorn app.main:app --reload`. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    return DefaultAzureCredential, MLClient, Environment, AmlCompute, ResourceNotFoundError, command, Input


@dataclass
class AzureMLJobResult:
    metrics: Dict[str, Any]
    models: Dict[str, Any]
    best_model_id: Optional[str] = None


def _require_workspace_settings() -> Tuple[str, str, str]:
    s = get_settings()
    if not (s.azure_subscription_id and s.azure_resource_group and s.azure_ml_workspace_name):
        raise AzureMLNotConfigured(
            "Azure ML workspace is not configured. Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE_NAME."
        )
    return s.azure_subscription_id, s.azure_resource_group, s.azure_ml_workspace_name


def get_ml_client():
    DefaultAzureCredential, MLClient, *_ = _lazy_imports()
    sub, rg, ws = _require_workspace_settings()
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return MLClient(credential=cred, subscription_id=sub, resource_group_name=rg, workspace_name=ws)


def ensure_compute(ml_client, *, name: str) -> None:
    """Ensure the configured AML compute exists; create it if missing."""
    s = get_settings()
    _, _, _, AmlCompute, ResourceNotFoundError, *_ = _lazy_imports()

    try:
        ml_client.compute.get(name)
        return
    except ResourceNotFoundError:
        pass

    compute = AmlCompute(
        name=name,
        size=s.azure_ml_compute_vm_size,
        min_instances=s.azure_ml_compute_min_nodes,
        max_instances=s.azure_ml_compute_max_nodes,
        idle_time_before_scale_down=120,
    )
    ml_client.compute.begin_create_or_update(compute).result()


def submit_training_job(
    *,
    dataset_id: str,
    target_column: str,
    drop_columns: list[str],
    task: Optional[str],
    model_candidates: list[str],
) -> Dict[str, Any]:
    s = get_settings()
    if s.use_local_mock:
        return {"job_id": f"mock-{dataset_id}", "status": "mock", "studio_url": None}

    _, _, Environment, _, _, command, Input = _lazy_imports()
    ml_client = get_ml_client()

    # Auto-create compute if missing (saves manual setup for demos)
    ensure_compute(ml_client, name=s.azure_ml_compute_name)

    data_path = BASE_DATA / "curated" / f"{dataset_id}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Curated dataset not found: {dataset_id}")

    code_dir = Path(__file__).resolve().parents[1] / "azureml"
    env = Environment(
        name="labnotebookai-train-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file=str(code_dir / "conda.yml"),
    )

    drop_json = json.dumps(drop_columns or [])
    models_json = json.dumps(model_candidates or [])

    job = command(
        code=str(code_dir),
        command=(
            "python train_multi.py "
            "--data ${{inputs.data}} "
            "--target ${{inputs.target}} "
            "--task ${{inputs.task}} "
            "--drop_cols_json ${{inputs.drop_cols_json}} "
            "--models_json ${{inputs.models_json}} "
            "--out_dir ${{outputs.out_dir}}"
        ),
        inputs={
            "data": Input(type="uri_file", path=str(data_path)),
            "target": target_column,
            "task": task or "",
            "drop_cols_json": drop_json,
            "models_json": models_json,
        },
        outputs={"out_dir": {"type": "uri_folder"}},
        environment=env,
        compute=s.azure_ml_compute_name,
        display_name=f"labnotebookai-train-{dataset_id}",
        description="Feature engineering (drop columns) + multi-model training.",
    )

    created = ml_client.jobs.create_or_update(job)
    studio_url = getattr(created, "studio_url", None)
    return {"job_id": created.name, "status": str(created.status), "studio_url": studio_url}


def get_job_status(job_id: str) -> Dict[str, Any]:
    s = get_settings()
    if s.use_local_mock:
        return {"job_id": job_id, "status": "mock", "details": {"mode": "mock"}, "results": None}

    ml_client = get_ml_client()
    job = ml_client.jobs.get(job_id)
    status = str(job.status)

    results = None
    if status.lower() in {"completed", "finished", "succeeded"}:
        try:
            results = _download_and_parse_results(ml_client, job_id)
        except Exception as e:
            results = {"error": str(e)}

    return {"job_id": job_id, "status": status, "details": job.as_dict(), "results": results}


def _download_and_parse_results(ml_client, job_id: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        ml_client.jobs.download(job_id, download_path=str(out_dir), all=True)

        # Try common locations
        candidates = list(out_dir.rglob("results.json")) + list(out_dir.rglob("metrics.json"))
        if not candidates:
            raise FileNotFoundError("No results.json/metrics.json found in downloaded job outputs")

        p = candidates[0]
        return json.loads(p.read_text(encoding="utf-8"))


def deploy_from_job(*, job_id: str, model_id: str, endpoint_name: Optional[str]) -> Dict[str, Any]:
    s = get_settings()
    if s.use_local_mock:
        return {
            "endpoint_name": endpoint_name or "mock-endpoint",
            "deployment_name": "mock-deploy",
            "status": "mock",
            "scoring_uri": None,
        }

    _, _, Environment, *_ = _lazy_imports()
    from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model  # type: ignore

    ml_client = get_ml_client()

    ep_name = (endpoint_name or f"labnotebookai-{job_id[:12]}").lower().replace("_", "-")
    deployment_name = "blue"

    # Model artifact path: outputs/out_dir/models/<model_id>/model.pkl
    # We register the whole model folder so scoring can load it.
    model_path = f"azureml://jobs/{job_id}/outputs/out_dir/paths/models/{model_id}"
    model = Model(path=model_path, name=f"labnotebookai-{model_id}")
    reg = ml_client.models.create_or_update(model)

    code_dir = Path(__file__).resolve().parents[1] / "azureml"
    env = Environment(
        name="labnotebookai-infer-env",
        image="mcr.microsoft.com/azureml/minimal-ubuntu22.04-py310-cpu-inference:latest",
        conda_file=str(code_dir / "conda_infer.yml"),
    )

    endpoint = ManagedOnlineEndpoint(
        name=ep_name,
        description="Lab Notebook AI endpoint",
        auth_mode="key",
        location=s.azure_region or None,
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=ep_name,
        model=reg,
        environment=env,
        code_configuration={"code": str(code_dir), "scoring_script": "score.py"},
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    ml_client.online_endpoints.begin_update(
        ManagedOnlineEndpoint(name=ep_name, traffic={deployment_name: 100})
    ).result()

    ep = ml_client.online_endpoints.get(ep_name)
    return {
        "endpoint_name": ep_name,
        "deployment_name": deployment_name,
        "status": "deployed",
        "scoring_uri": getattr(ep, "scoring_uri", None),
    }
