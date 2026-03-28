from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    features = df.drop(columns=[target])
    num_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in features.columns if c not in num_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")


def infer_task(y: pd.Series, task: str | None) -> str:
    if task and task.strip():
        return task.strip()
    nunique = y.nunique(dropna=True)
    return "classification" if nunique <= 20 else "regression"


def train_one(model_key: str, task: str):
    if task == "classification":
        if model_key == "logreg":
            return LogisticRegression(max_iter=400, class_weight="balanced")
        if model_key == "rf":
            return RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
        return GradientBoostingClassifier(random_state=42)

    if model_key == "rf":
        return RandomForestRegressor(n_estimators=250, random_state=42)
    return GradientBoostingRegressor(random_state=42)


def metric_bundle(task: str, y_true, y_score) -> Dict[str, Any]:
    if task == "classification":
        y_pred = (y_score > 0.5).astype(int) if hasattr(y_score, "dtype") else y_score
        out: Dict[str, Any] = {}
        try:
            out["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["auc"] = 0.5
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        return out

    y_pred = y_score
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def choose_best(task: str, metrics_by_model: Dict[str, Dict[str, Any]]) -> str:
    if not metrics_by_model:
        return ""
    if task == "classification":
        return sorted(
            metrics_by_model.items(),
            key=lambda kv: (kv[1].get("auc", 0), kv[1].get("recall", 0)),
            reverse=True,
        )[0][0]
    return sorted(metrics_by_model.items(), key=lambda kv: kv[1].get("r2", -1e9), reverse=True)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", default="")
    ap.add_argument("--drop_cols_json", default="[]")
    ap.add_argument("--models_json", default='["logreg","rf","gbrt"]')
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    drop_cols = json.loads(args.drop_cols_json or "[]")
    models = json.loads(args.models_json or "[]")

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"target column not found: {args.target}")

    # Feature engineering: drop requested columns (except target)
    drop_cols = [c for c in drop_cols if c in df.columns and c != args.target]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    y = df[args.target]
    task = infer_task(y, args.task)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    pre = build_preprocessor(df, args.target)

    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target]
    X_val = val_df.drop(columns=[args.target])
    y_val = val_df[args.target]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    metrics_by_model: Dict[str, Dict[str, Any]] = {}

    for key in models:
        key = str(key)
        m = train_one(key, task)
        pipe = Pipeline([("prep", pre), ("model", m)])
        pipe.fit(X_train, y_train)

        if task == "classification" and hasattr(pipe, "predict_proba"):
            score = pipe.predict_proba(X_val)[:, 1]
        else:
            score = pipe.predict(X_val)

        met = metric_bundle(task, y_val, score)
        metrics_by_model[key] = met

        mdl_path = models_dir / key
        mdl_path.mkdir(exist_ok=True)
        joblib.dump(pipe, mdl_path / "model.joblib")
        (mdl_path / "meta.json").write_text(
            json.dumps({"task": task, "target": args.target, "drop_columns": drop_cols}, indent=2)
        )

    best = choose_best(task, metrics_by_model)

    results = {
        "task": task,
        "target": args.target,
        "dropped_columns": drop_cols,
        "models": {k: {"metrics": v} for k, v in metrics_by_model.items()},
        "best_model_id": best,
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
