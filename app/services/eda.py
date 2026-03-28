from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.services.storage import BASE_DATA


def _safe_sample(df: pd.DataFrame, max_rows: int = 200_000) -> pd.DataFrame:
    if df.shape[0] <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _histogram(series: pd.Series, bins: int = 12) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.shape[0] == 0:
        return {"bins": [], "counts": []}

    counts, edges = np.histogram(s.to_numpy(), bins=bins)
    return {
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    out = {
        "missing": int(s.isna().sum()),
        "missing_rate": float(s.isna().mean()),
    }

    s2 = s.dropna()
    if s2.shape[0] == 0:
        return {**out, "count": 0}

    qs = s2.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
    return {
        **out,
        "count": int(s2.shape[0]),
        "mean": float(s2.mean()),
        "std": float(s2.std(ddof=1)) if s2.shape[0] > 1 else 0.0,
        "min": float(qs.get(0.0, s2.min())),
        "p25": float(qs.get(0.25, s2.min())),
        "median": float(qs.get(0.5, s2.median())),
        "p75": float(qs.get(0.75, s2.max())),
        "max": float(qs.get(1.0, s2.max())),
    }


def _categorical_summary(series: pd.Series, top_k: int = 5) -> Dict[str, Any]:
    s = series.astype(str)
    missing = int(series.isna().sum())
    missing_rate = float(series.isna().mean())

    vc = s.value_counts(dropna=False).head(top_k)
    total = int(series.shape[0])
    top_values = [
        {"value": str(idx), "count": int(cnt), "ratio": float(cnt / total) if total else 0.0}
        for idx, cnt in vc.items()
    ]

    nunique = int(series.dropna().astype(str).nunique())
    return {
        "missing": missing,
        "missing_rate": missing_rate,
        "unique": nunique,
        "top_values": top_values,
    }


def _corr_matrix(df: pd.DataFrame, max_cols: int = 20) -> Dict[str, Any]:
    num = df.select_dtypes(include=["number"]).copy()
    cols = num.columns.tolist()
    if len(cols) == 0:
        return {"columns": [], "matrix": [], "top_pairs": []}

    if len(cols) > max_cols:
        cols = cols[:max_cols]
        num = num[cols]

    corr = num.corr(numeric_only=True).fillna(0.0)
    matrix = [[float(corr.loc[i, j]) for j in cols] for i in cols]

    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = [
        {"a": a, "b": b, "corr": c}
        for a, b, c in pairs[: min(12, len(pairs))]
        if abs(c) >= 0.3
    ]

    return {"columns": cols, "matrix": matrix, "top_pairs": top_pairs}


def eda_for_dataset(dataset_id: str) -> Dict[str, Any]:
    curated_path = BASE_DATA / "curated" / f"{dataset_id}.csv"
    quarantine_path = BASE_DATA / "quarantine" / f"{dataset_id}.csv"

    if not curated_path.exists():
        raise FileNotFoundError(f"Curated dataset not found: {dataset_id}")

    df = pd.read_csv(curated_path)
    df = _safe_sample(df)

    quarant_rows = 0
    if quarantine_path.exists():
        try:
            quarant_rows = int(pd.read_csv(quarantine_path).shape[0])
        except Exception:
            quarant_rows = 0

    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "quarantine_rows": quarant_rows,
    }

    missingness = (
        df.isna().mean().sort_values(ascending=False).head(20).to_dict()
        if df.shape[1] > 0
        else {}
    )
    missingness = {k: float(v) for k, v in missingness.items()}

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    numeric = {
        c: {
            "summary": _numeric_summary(df[c]),
            "hist": _histogram(df[c], bins=12),
        }
        for c in numeric_cols[:25]
    }
    categorical = {c: _categorical_summary(df[c]) for c in cat_cols[:25]}

    corr = _corr_matrix(df, max_cols=18)

    return {
        "overview": overview,
        "missingness": missingness,
        "numeric": numeric,
        "categorical": categorical,
        "correlation": corr,
        "notes": {
            "sampling": "sampled" if df.shape[0] >= 200_000 else "full",
            "numeric_columns_shown": int(min(len(numeric_cols), 25)),
            "categorical_columns_shown": int(min(len(cat_cols), 25)),
        },
    }
