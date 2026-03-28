from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.services.storage import BASE_DATA


def dataset_hash_from_path(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def log_run(payload: Dict[str, Any]) -> str:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d%H%M%S%f")
    payload = {"run_id": run_id, **payload, "logged_at": datetime.now(timezone.utc).isoformat()}
    out = BASE_DATA / "mlflow_mock" / f"{run_id}.json"
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return run_id
