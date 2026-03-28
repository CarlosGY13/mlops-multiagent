from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.config import get_settings
from app.services.storage import BASE_DATA

_ML_CLIENT_LOCK = threading.Lock()
_ML_CLIENT_CACHE: Dict[Tuple[Optional[str], str, str, str], Any] = {}


_ANSI_RE = re.compile(r"\x1B\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s or "")


def _sanitize_endpoint_name(name: Optional[str], *, fallback: str) -> str:
    candidate = (name or "").strip() or (fallback or "").strip()

    s = candidate.lower().replace("_", "-").replace(" ", "-")
    s = re.sub(r"[^a-z0-9-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")

    if not s:
        s = "lab-endpoint"

    if not re.match(r"^[a-z]", s):
        s = f"lab-{s}".lower()

    # Azure ML endpoint names are typically limited (commonly 32 chars). Keep it safe.
    s = s[:32].rstrip("-")
    if len(s) < 3:
        s = "lab-endpoint"

    return s


class AzureMLNotConfigured(RuntimeError):
    pass


def _raise_friendly_tenant_error(e: Exception, *, tenant_id: str, subscription_id: str) -> None:
    msg = str(e)
    if "Azure CLI not found on path" in msg:
        raise RuntimeError(
            "Azure CLI (`az`) is not installed or not on PATH. "
            "Option A: install Azure CLI and run `az login --tenant ...`. "
            "Option B: restart the server and the app will prompt an interactive browser login for the configured AZURE_TENANT_ID."
        ) from e
    if "InvalidAuthenticationTokenTenant" in msg:
        raise RuntimeError(
            "Azure tenant mismatch for this subscription. "
            f"Please run: az logout && az login --tenant {tenant_id} && az account set --subscription {subscription_id} "
            "then restart the server."
        ) from e


@dataclass(frozen=True)
class _AzureMLImports:
    AzureCliCredential: Any
    ChainedTokenCredential: Any
    DefaultAzureCredential: Any
    InteractiveBrowserCredential: Any
    MLClient: Any
    Environment: Any
    AmlCompute: Any
    ResourceNotFoundError: Any
    HttpResponseError: Any
    command: Any
    Input: Any
    Output: Any


def _lazy_imports() -> _AzureMLImports:
    try:
        from azure.identity import (  # type: ignore
            AzureCliCredential,
            ChainedTokenCredential,
            DefaultAzureCredential,
            InteractiveBrowserCredential,
        )
        from azure.ai.ml import MLClient  # type: ignore
        from azure.ai.ml.entities import Environment, AmlCompute  # type: ignore
        from azure.ai.ml import command, Input, Output  # type: ignore
        from azure.core.exceptions import ResourceNotFoundError, HttpResponseError  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Azure ML SDK import failed. Ensure you're running the server from the same venv where dependencies were installed. "
            "Try: `source .venv/bin/activate && pip install -r requirements.txt` and start with `python -m uvicorn app.main:app --reload`. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    return _AzureMLImports(
        AzureCliCredential=AzureCliCredential,
        ChainedTokenCredential=ChainedTokenCredential,
        DefaultAzureCredential=DefaultAzureCredential,
        InteractiveBrowserCredential=InteractiveBrowserCredential,
        MLClient=MLClient,
        Environment=Environment,
        AmlCompute=AmlCompute,
        ResourceNotFoundError=ResourceNotFoundError,
        HttpResponseError=HttpResponseError,
        command=command,
        Input=Input,
        Output=Output,
    )


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
    imp = _lazy_imports()

    s = get_settings()
    sub, rg, ws = _require_workspace_settings()

    cache_key = (s.azure_tenant_id, sub, rg, ws)
    with _ML_CLIENT_LOCK:
        cached = _ML_CLIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # If a tenant is provided, hard-pin authentication to that tenant.
        # Prefer Azure CLI when available, but fall back to interactive browser login.
        if s.azure_tenant_id:
            cred = imp.ChainedTokenCredential(
                imp.AzureCliCredential(tenant_id=s.azure_tenant_id),
                imp.InteractiveBrowserCredential(tenant_id=s.azure_tenant_id, timeout=600),
            )
        else:
            cred = imp.DefaultAzureCredential(exclude_interactive_browser_credential=False)

        client = imp.MLClient(credential=cred, subscription_id=sub, resource_group_name=rg, workspace_name=ws)
        _ML_CLIENT_CACHE[cache_key] = client
        return client


def ensure_compute(ml_client, *, name: str) -> None:
    """Ensure the configured AML compute exists; create it if missing."""
    s = get_settings()
    imp = _lazy_imports()
    AmlCompute = imp.AmlCompute
    ResourceNotFoundError = imp.ResourceNotFoundError

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

    imp = _lazy_imports()
    Environment = imp.Environment
    command = imp.command
    Input = imp.Input
    Output = imp.Output
    ml_client = get_ml_client()

    # Auto-create compute if missing (saves manual setup for demos)
    try:
        ensure_compute(ml_client, name=s.azure_ml_compute_name)
    except Exception as e:
        if s.azure_tenant_id:
            _raise_friendly_tenant_error(e, tenant_id=s.azure_tenant_id, subscription_id=_require_workspace_settings()[0])
        raise

    data_path = BASE_DATA / "curated" / f"{dataset_id}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Curated dataset not found: {dataset_id}")

    code_dir = Path(__file__).resolve().parents[1] / "azureml"
    env = Environment(
        name="labnotebookai-train-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file=str(code_dir / "conda.yml"),
    )

    drop_json = json.dumps(drop_columns or [], separators=(",", ":"))
    models_json = json.dumps(model_candidates or [], separators=(",", ":"))

    cmd = [
        "python train_multi.py",
        "--data '${{inputs.data}}'",
        "--target '${{inputs.target}}'",
        "--drop_cols_json '${{inputs.drop_cols_json}}'",
        "--models_json '${{inputs.models_json}}'",
        "--out_dir '${{outputs.out_dir}}'",
    ]

    inputs: Dict[str, Any] = {
        "data": Input(type="uri_file", path=str(data_path)),
        "target": target_column,
        "drop_cols_json": drop_json,
        "models_json": models_json,
    }
    if task and task.strip():
        cmd.insert(3, "--task '${{inputs.task}}'")
        inputs["task"] = task.strip()

    job = command(
        code=str(code_dir),
        command=" ".join(cmd),
        inputs=inputs,
        outputs={"out_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=env,
        compute=s.azure_ml_compute_name,
        display_name=f"labnotebookai-train-{dataset_id}",
        description="Feature engineering (drop columns) + multi-model training.",
    )

    try:
        created = ml_client.jobs.create_or_update(job)
    except Exception as e:
        if s.azure_tenant_id:
            _raise_friendly_tenant_error(e, tenant_id=s.azure_tenant_id, subscription_id=_require_workspace_settings()[0])
        # Improve AzureML REST diagnostics
        imp = _lazy_imports()
        if isinstance(e, imp.HttpResponseError) and getattr(e, "response", None) is not None:
            resp = e.response
            try:
                body = resp.text()
            except Exception:
                body = None
            raise RuntimeError(
                f"Azure ML request failed (HTTP {getattr(resp,'status_code', '?')}). "
                f"{str(e)}" + (f"\nResponse body:\n{body}" if body else "")
            ) from e
        raise
    studio_url = getattr(created, "studio_url", None)
    return {"job_id": created.name, "status": str(created.status), "studio_url": studio_url}


def get_job_status(job_id: str) -> Dict[str, Any]:
    s = get_settings()
    if s.use_local_mock:
        return {"job_id": job_id, "status": "mock", "details": {"mode": "mock"}, "results": None}

    ml_client = get_ml_client()
    try:
        job = ml_client.jobs.get(job_id)
    except Exception as e:
        if s.azure_tenant_id:
            _raise_friendly_tenant_error(e, tenant_id=s.azure_tenant_id, subscription_id=_require_workspace_settings()[0])
        raise
    status = str(job.status)

    results = None
    if status.lower() in {"completed", "finished", "succeeded"}:
        try:
            results = _download_and_parse_results(ml_client, job_id)
        except Exception as e:
            results = {"error": str(e)}

    # `job.as_dict` can be a method or a property depending on azure-ai-ml version.
    # Keep this endpoint resilient so the UI can always poll status.
    details: Dict[str, Any]
    try:
        as_dict = getattr(job, "as_dict", None)
        if callable(as_dict):
            details = as_dict()
        elif isinstance(as_dict, dict):
            details = as_dict
        else:
            to_dict = getattr(job, "_to_dict", None)
            details = to_dict() if callable(to_dict) else {"repr": repr(job)}
    except Exception:
        details = {"repr": repr(job)}

    return {"job_id": job_id, "status": status, "details": details, "results": results}


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

    imp = _lazy_imports()
    Environment = imp.Environment
    from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model  # type: ignore

    ml_client = get_ml_client()

    fallback = f"labnotebookai-{job_id[:12]}".lower()
    ep_name = _sanitize_endpoint_name(endpoint_name, fallback=fallback)
    deployment_name = "blue"

    # Model artifact path: outputs/out_dir/models/<model_id>/model.pkl
    # We register the whole model folder so scoring can load it.
    model_path = f"azureml://jobs/{job_id}/outputs/out_dir/paths/models/{model_id}"
    model = Model(path=model_path, name=f"labnotebookai-{model_id}")
    try:
        reg = ml_client.models.create_or_update(model)
    except Exception as e:
        if s.azure_tenant_id:
            _raise_friendly_tenant_error(e, tenant_id=s.azure_tenant_id, subscription_id=_require_workspace_settings()[0])
        raise

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
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    except Exception as e:
        raise RuntimeError(_strip_ansi(str(e))) from e

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=ep_name,
        model=reg,
        environment=env,
        code_configuration={"code": str(code_dir), "scoring_script": "score.py"},
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    try:
        ml_client.online_deployments.begin_create_or_update(deployment).result()
    except Exception as e:
        raise RuntimeError(_strip_ansi(str(e))) from e

    try:
        ml_client.online_endpoints.begin_update(
            ManagedOnlineEndpoint(name=ep_name, traffic={deployment_name: 100})
        ).result()
    except Exception as e:
        raise RuntimeError(_strip_ansi(str(e))) from e

    ep = ml_client.online_endpoints.get(ep_name)
    return {
        "endpoint_name": ep_name,
        "deployment_name": deployment_name,
        "status": "deployed",
        "scoring_uri": getattr(ep, "scoring_uri", None),
    }
