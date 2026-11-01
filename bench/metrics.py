"""Metric helpers: ANN recall, latency statistics, ranking quality, CIs."""
from __future__ import annotations

import numpy as np


def recall_at_k(found: list[np.ndarray], gt: np.ndarray, k: int) -> np.ndarray:
    """Per-query |ANN top-k ∩ exact top-k| / k. Queries whose exact list has
    fewer than k valid ids (tiny filtered sets) are scored against what exists."""
    out = np.empty(len(found), dtype="float64")
    for i, f in enumerate(found):
        g = gt[i, :k]
        g = g[g >= 0]
        denom = len(g)
        out[i] = len(np.intersect1d(f[:k], g, assume_unique=False)) / denom if denom else 1.0
    return out


def latency_stats(lat_s: np.ndarray) -> dict:
    ms = np.asarray(lat_s) * 1000.0
    return {"lat_mean_ms": float(ms.mean()), "lat_p50_ms": float(np.percentile(ms, 50)),
            "lat_p95_ms": float(np.percentile(ms, 95)), "lat_p99_ms": float(np.percentile(ms, 99)),
            "qps_1thread": float(len(ms) / (ms.sum() / 1000.0))}


# ---- ranking quality against human relevance labels (ESCI) ----------------
ESCI_GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}  # gains used in the ESCI paper


def ndcg_at_k(ranked: np.ndarray, gains: dict[int, float], k: int = 10) -> float:
    """nDCG@k. `gains` maps product row id -> graded gain; unjudged = 0."""
    g = np.array([gains.get(int(r), 0.0) for r in ranked[:k]])
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    dcg = float((g * disc).sum())
    ideal = np.sort(np.fromiter(gains.values(), dtype="float64"))[::-1][:k]
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum())
    return dcg / idcg if idcg > 0 else np.nan


def exact_recall_at_k(ranked: np.ndarray, exact_ids: set[int], k: int = 100) -> float:
    """Share of human-labelled Exact products that appear in the top-k."""
    if not exact_ids:
        return np.nan
    return len(exact_ids.intersection(int(r) for r in ranked[:k])) / len(exact_ids)


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
