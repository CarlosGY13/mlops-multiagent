from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DualView(BaseModel):
    investigator: Dict[str, Any]
    technical: Dict[str, Any]


class IngestResponse(BaseModel):
    dataset_id: str
    schema_info: Dict[str, Any]
    quality: DualView


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    sensitive_column: Optional[str] = None
    task: Optional[str] = Field(default=None, description="classification|regression")


class AzureMLTrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    drop_columns: list[str] = []
    task: Optional[str] = Field(default=None, description="classification|regression")
    model_candidates: list[str] = ["logreg", "rf", "gbrt"]


class AzureMLTrainResponse(BaseModel):
    job_id: str
    status: str
    studio_url: Optional[str] = None


class AzureMLJobStatusResponse(BaseModel):
    job_id: str
    status: str
    details: dict
    results: Optional[dict] = None


class AzureMLDeployRequest(BaseModel):
    job_id: str
    model_id: str
    endpoint_name: Optional[str] = None


class AzureMLDeployResponse(BaseModel):
    endpoint_name: str
    deployment_name: str
    status: str
    scoring_uri: Optional[str] = None


class TrainResponse(BaseModel):
    run_id: str
    dataset_hash: str
    metrics: Dict[str, Any]
    production_gate: DualView


class DriftRequest(BaseModel):
    reference_dataset_id: str
    current_dataset_id: str
    numeric_columns: List[str]


class LiteratureSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ResearchSearchRequest(BaseModel):
    session_id: Optional[str] = None
    dataset_id: Optional[str] = None
    user_context: Optional[str] = None
    query: str
    top_k: int = 10


class AgentMessageRequest(BaseModel):
    session_id: Optional[str] = None
    dataset_id: Optional[str] = None
    user_context: Optional[str] = None
    message: str
    rag_active: bool = False


class DataFeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    dataset_id: str
    user_context: Optional[str] = None


class AgentMessageResponse(BaseModel):
    answer: str
    rationale: str
    citations: List[Dict[str, str]] = []
    side_panel: DualView
