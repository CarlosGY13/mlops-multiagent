from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.storage import BASE_DATA
from app.services.tracking import dataset_hash_from_path, log_run


@dataclass
class TrainArtifacts:
    run_id: str
    dataset_hash: str
    metrics: Dict[str, Any]
    gate: Dict[str, Any]


def _build_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    features = df.drop(columns=[target_col])
    num_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in features.columns if c not in num_cols]

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def _equalized_odds_proxy(y_true: np.ndarray, y_pred: np.ndarray, sensitive: Optional[pd.Series]) -> float:
    if sensitive is None:
        return 0.0
    groups = sensitive.astype(str).fillna("unknown").unique().tolist()
    if len(groups) < 2:
        return 0.0
    tprs = []
    for g in groups[:2]:
        mask = sensitive.astype(str).fillna("unknown") == g
        positives = (y_true[mask] == 1).sum()
        if positives == 0:
            continue
        tp = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
        tprs.append(tp / positives)
    if len(tprs) < 2:
        return 0.0
    return float(abs(tprs[0] - tprs[1]))


def train_dataset(dataset_id: str, target_column: str, task: Optional[str], sensitive_column: Optional[str]) -> TrainArtifacts:
    path = BASE_DATA / "curated" / f"{dataset_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset curated no encontrado: {dataset_id}")

    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"target_column `{target_column}` no existe en dataset")

    inferred_task = task
    if inferred_task is None:
        nunique = df[target_column].nunique(dropna=True)
        inferred_task = "classification" if nunique <= 20 else "regression"

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    preprocessor = _build_preprocessor(df, target_column)

    if inferred_task == "classification":
        model = LogisticRegression(max_iter=300, class_weight="balanced")
    else:
        model = RandomForestRegressor(n_estimators=120, random_state=42)

    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    start = time.perf_counter()
    preds = pipeline.predict(X_val)
    elapsed = (time.perf_counter() - start) * 1000
    latency_p95 = float(np.percentile([elapsed], 95))

    if inferred_task == "classification":
        y_pred = (preds > 0.5).astype(int) if preds.dtype != int else preds
        try:
            auc = float(roc_auc_score(y_val, preds))
        except Exception:
            auc = 0.5
        recall = float(recall_score(y_val, y_pred, zero_division=0))
        fair = _equalized_odds_proxy(
            y_true=np.asarray(y_val),
            y_pred=np.asarray(y_pred),
            sensitive=val_df[sensitive_column] if sensitive_column and sensitive_column in val_df.columns else None,
        )
        metrics = {
            "task": inferred_task,
            "auc": auc,
            "recall": recall,
            "latency_p95_ms": latency_p95,
            "equalized_odds_difference": fair,
        }
        gate = {
            "pass": auc >= 0.7 and recall >= 0.6 and latency_p95 <= 500 and fair < 0.05,
            "thresholds": {
                "auc": ">= 0.7",
                "recall": ">= 0.6",
                "latency_p95_ms": "<= 500",
                "equalized_odds_difference": "< 0.05",
            },
        }
    else:
        r2 = float(r2_score(y_val, preds))
        metrics = {
            "task": inferred_task,
            "r2": r2,
            "latency_p95_ms": latency_p95,
            "equalized_odds_difference": 0.0,
        }
        gate = {
            "pass": r2 >= 0.4 and latency_p95 <= 500,
            "thresholds": {"r2": ">= 0.4", "latency_p95_ms": "<= 500"},
        }

    d_hash = dataset_hash_from_path(path)
    run_id = log_run(
        {
            "dataset_id": dataset_id,
            "dataset_hash": d_hash,
            "target_column": target_column,
            "sensitive_column": sensitive_column,
            "metrics": metrics,
            "production_gate": gate,
            "trigger_reason": "manual_train",
            "model_version": "experiment_specific_v1",
        }
    )

    return TrainArtifacts(run_id=run_id, dataset_hash=d_hash, metrics=metrics, gate=gate)


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref_hist, edges = np.histogram(ref, bins=bins)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_ratio = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, 1)
    cur_ratio = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, 1)
    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))


def evaluate_drift(reference_dataset_id: str, current_dataset_id: str, numeric_columns: list[str]) -> Dict[str, Any]:
    ref = pd.read_csv(BASE_DATA / "curated" / f"{reference_dataset_id}.csv")
    cur = pd.read_csv(BASE_DATA / "curated" / f"{current_dataset_id}.csv")

    report: Dict[str, Any] = {}
    drift_detected = False
    for col in numeric_columns:
        if col not in ref.columns or col not in cur.columns:
            continue
        ref_vals = ref[col].dropna().to_numpy()
        cur_vals = cur[col].dropna().to_numpy()
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        psi = _psi(ref_vals, cur_vals)
        ks_stat, ks_p = ks_2samp(ref_vals, cur_vals)
        col_drift = psi > 0.2 or ks_p < 0.05
        drift_detected = drift_detected or col_drift
        report[col] = {
            "psi": float(psi),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "drift": col_drift,
        }

    return {
        "drift_detected": drift_detected,
        "details": report,
        "monitoring_action": "trigger_retraining" if drift_detected else "continue_monitoring",
    }
