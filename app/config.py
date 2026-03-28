from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Lab Notebook AI"
    env: str = "development"
    use_local_mock: bool = True

    azure_key_vault_url: Optional[str] = None
    azure_ai_search_endpoint: Optional[str] = None
    azure_ai_search_index: Optional[str] = None

    content_safety_blocklist: str = "patogeno,patógeno,cb rn,cbrn,diagnostico clinico,diagnóstico clínico,pathogen,diagnosis,clinical diagnosis,cbrn"

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(
        use_local_mock=os.getenv("USE_LOCAL_MOCK", "true").lower() == "true",
    )
    return settings
