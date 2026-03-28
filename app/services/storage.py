from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BASE_DATA = Path("data")
BASE_DATA.mkdir(exist_ok=True)
for zone in ("raw", "curated", "quarantine", "ground_truth", "mlflow_mock", "rag_cache"):
    (BASE_DATA / zone).mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
