from __future__ import annotations

import hashlib
from typing import Dict


def sha256_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def anonymize_ids(payload: Dict[str, str]) -> Dict[str, str]:
    safe = payload.copy()
    for key in ("researcher_id", "sample_id"):
        if key in safe and safe[key]:
            safe[key] = sha256_hash(str(safe[key]))
    return safe


class ContentSafetyError(ValueError):
    pass


def enforce_content_safety(message: str, blocklist_csv: str) -> None:
    lowered = message.lower()
    blocked = [token.strip().lower() for token in blocklist_csv.split(",") if token.strip()]
    if any(token in lowered for token in blocked):
        raise ContentSafetyError("This request was blocked by Content Safety.")
