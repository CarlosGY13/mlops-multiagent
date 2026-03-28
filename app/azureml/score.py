from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import joblib
import pandas as pd


_model = None


def init():
    global _model
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")

    candidates = [
        os.path.join(model_dir, "model.joblib"),
        os.path.join(model_dir, "outputs", "model.joblib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            _model = joblib.load(p)
            return

    for root, _, files in os.walk(model_dir):
        if "model.joblib" in files:
            _model = joblib.load(os.path.join(root, "model.joblib"))
            return

    raise FileNotFoundError("model.joblib not found in AZUREML_MODEL_DIR")


def run(raw_data):
    if _model is None:
        init()

    try:
        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    except Exception:
        payload = raw_data

    rows: List[Dict[str, Any]] = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {"error": "Expected JSON with key 'data': [ {col: val}, ... ]"}

    df = pd.DataFrame(rows)

    if hasattr(_model, "predict_proba"):
        p = _model.predict_proba(df)
        return {"pred_proba": p[:, 1].tolist() if p.shape[1] >= 2 else p[:, 0].tolist()}

    preds = _model.predict(df)
    return {"pred": preds.tolist()}
