from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.env,
        "use_local_mock": settings.use_local_mock,
        "azure_ml_compute_name": settings.azure_ml_compute_name,
        "zones": {
            "development": "fases 1-6",
            "production": "fases 7-10",
        },
    }
