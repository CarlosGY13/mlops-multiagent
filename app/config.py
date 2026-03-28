from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Lab Notebook AI"
    env: str = "development"
    use_local_mock: bool = True

    # Azure (shared)
    azure_subscription_id: Optional[str] = None
    azure_resource_group: Optional[str] = None
    azure_region: Optional[str] = None
    azure_tenant_id: Optional[str] = None

    # Azure ML workspace
    azure_ml_workspace_name: Optional[str] = None
    azure_ml_compute_name: str = "cpu-cluster"

    # Azure ML compute defaults (used when auto-creating compute)
    azure_ml_compute_vm_size: str = "Standard_DS3_v2"
    azure_ml_compute_min_nodes: int = 0
    azure_ml_compute_max_nodes: int = 2

    azure_key_vault_url: Optional[str] = None
    azure_ai_search_endpoint: Optional[str] = None
    azure_ai_search_index: Optional[str] = None

    # Microsoft Foundry / OpenAI-compatible model endpoint configuration.
    # Option A (OpenAI-compatible):
    #   FOUNDRY_OPENAI_BASE_URL=https://.../v1
    #   FOUNDRY_OPENAI_API_KEY=...
    #   FOUNDRY_OPENAI_MODEL=...
    foundry_openai_base_url: Optional[str] = None
    foundry_openai_api_key: Optional[str] = None
    foundry_openai_model: Optional[str] = None

    # Option B (Azure OpenAI-style deployment endpoint):
    #   FOUNDRY_AZURE_OPENAI_ENDPOINT=https://...openai.azure.com
    #   FOUNDRY_AZURE_OPENAI_DEPLOYMENT=your_deployment
    #   FOUNDRY_AZURE_OPENAI_API_VERSION=2024-05-01-preview
    foundry_azure_openai_endpoint: Optional[str] = None
    foundry_azure_openai_deployment: Optional[str] = None
    foundry_azure_openai_api_version: str = "2024-05-01-preview"

    llm_max_history_messages: int = 24

    content_safety_blocklist: str = "patogeno,patógeno,cb rn,cbrn,diagnostico clinico,diagnóstico clínico,pathogen,diagnosis,clinical diagnosis,cbrn"

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Settings reads from environment variables and .env (see model_config.env_file)
    return Settings()
