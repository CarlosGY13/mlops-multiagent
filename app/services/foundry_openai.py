from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


@dataclass
class FoundryOpenAIConfig:
    # Supports both OpenAI-compatible and Azure OpenAI-style endpoints.
    base_url: Optional[str] = None  # e.g. https://your-gateway/v1
    api_key: Optional[str] = None
    model: Optional[str] = None

    azure_endpoint: Optional[str] = None  # e.g. https://{resource}.openai.azure.com
    azure_deployment: Optional[str] = None
    azure_api_version: str = "2024-05-01-preview"


def _normalize_openai_base_url(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def is_configured(cfg: FoundryOpenAIConfig) -> bool:
    if cfg.azure_endpoint and cfg.azure_deployment and cfg.api_key:
        return True
    if cfg.base_url and cfg.api_key and cfg.model:
        return True
    return False


def _resolve_url_and_headers(cfg: FoundryOpenAIConfig) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    if cfg.azure_endpoint and cfg.azure_deployment:
        endpoint = (cfg.azure_endpoint or "").strip().rstrip("/")
        url = f"{endpoint}/openai/deployments/{cfg.azure_deployment}/chat/completions"
        headers = {"api-key": cfg.api_key or "", "Content-Type": "application/json"}
        extra_params: Dict[str, Any] = {"api-version": cfg.azure_api_version}
        return url, headers, extra_params

    base = _normalize_openai_base_url(cfg.base_url or "")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    return url, headers, {}


def chat_completions(
    cfg: FoundryOpenAIConfig,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 700,
    timeout_s: int = 40,
) -> str:
    if not is_configured(cfg):
        raise RuntimeError("Foundry/OpenAI config not set")

    url, headers, extra_params = _resolve_url_and_headers(cfg)

    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # OpenAI-style requires model; Azure deployment style doesn't.
    if not (cfg.azure_endpoint and cfg.azure_deployment):
        payload["model"] = cfg.model

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, headers=headers, params=extra_params, json=payload)
        r.raise_for_status()
        js = r.json()

    choices = js.get("choices") or []
    if not choices:
        raise RuntimeError("LLM returned no choices")
    msg = (choices[0] or {}).get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise RuntimeError("LLM returned empty content")
    return content
