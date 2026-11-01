"""Responsible-AI evaluation on ESCI: does retrieval quality differ across
languages (locales) and product categories, and does the ANN index or the
filtering strategy widen those gaps?

    python -m bench.esci_eval --run results/esci --emb data/esci_emb

For every engine's saved top-100 lists (unfiltered and locale-filtered) plus
the exact-search baseline, per query:
  ndcg@10        graded relevance vs human ESCI labels (E=1, S=.1, C=.01, I=0)
  exact_recall@100  share of the query's Exact-labelled products retrieved
  ann_recall@10  overlap with exact search under the same mode (index fidelity)
  foreign@10     share of top-10 results from another locale (unfiltered only)
  underfilled    fewer than 10 results returned (filtered only)

Outputs (in --run):
  esci_per_query.parquet  one row per engine x mode x query
  esci_slices.csv         mean + 95% bootstrap CI per engine x mode x slice
  esci_disparity.csv      gap between best and worst slice, and how much of
                          it the ANN index adds on top of exact search
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from .metrics import ESCI_GAIN, bootstrap_ci, exact_recall_at_k, ndcg_at_k

METRICS = ["ndcg@10", "exact_recall@100", "ann_recall@10", "foreign@10", "underfilled"]


def per_query(ids: np.ndarray, exact: np.ndarray, qmeta: pd.DataFrame, qrels_by_q: dict,
              row_locale: np.ndarray, mode: str) -> pd.DataFrame:
    rows = []
    for i in range(len(qmeta)):
        qidx = int(qmeta.qidx.iat[i]); loc = qmeta.locale.iat[i]
        r = ids[i]; r = r[r >= 0]
        gains, exact_set = qrels_by_q.get(qidx, ({}, set()))
        e10 = exact[i, :10]; e10 = e10[e10 >= 0]
        rows.append({
            "qidx": qidx, "locale": loc, "category": qmeta.category_name.iat[i],
            "ndcg@10": ndcg_at_k(r, gains, 10) if gains else np.nan,
            "exact_recall@100": exact_recall_at_k(r, exact_set, 100),
            "ann_recall@10": len(np.intersect1d(r[:10], e10)) / len(e10) if len(e10) else np.nan,
            "foreign@10": float(np.mean(row_locale[r[:10]] != loc)) if (mode == "unfiltered" and len(r)) else np.nan,
            "underfilled": float(len(r) < 10) if mode == "filtered" else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/esci")
    ap.add_argument("--emb", default="data/esci_emb")
    ap.add_argument("--min-queries", type=int, default=30, help="slices smaller than this are flagged")
    a = ap.parse_args()

    meta = pd.read_parquet(os.path.join(a.emb, "products_meta.parquet"), columns=["row", "locale"])
    row_locale = meta.sort_values("row").locale.to_numpy()
    qmeta = pd.read_parquet(os.path.join(a.run, "queries_used.parquet"))
    qrels = pd.read_parquet(os.path.join(a.emb, "qrels.parquet"))
    qrels = qrels[qrels.qidx.isin(qmeta.qidx)]
    qrels_by_q = {}
    for qidx, g in qrels.groupby("qidx"):
        gains = {int(r): ESCI_GAIN[l] for r, l in zip(g.row, g.esci_label)}
        qrels_by_q[int(qidx)] = (gains, set(int(r) for r in g.row[g.esci_label == "E"]))

    frames = []
    for path in sorted(glob.glob(os.path.join(a.run, "ids", "*_k100.npy"))):
        base = os.path.basename(path)[:-len("_k100.npy")]
        engine, mode = base.rsplit("_", 1)
        exact = np.load(os.path.join(a.run, "ids", f"exact_{mode}_k100.npy"))
        df = per_query(np.load(path), exact, qmeta, qrels_by_q, row_locale, mode)
        df["engine"], df["mode"] = engine, mode
        frames.append(df)
    pq_df = pd.concat(frames, ignore_index=True)
    pq_df.to_parquet(os.path.join(a.run, "esci_per_query.parquet"), index=False)

    # ---- slices with bootstrap CIs ------------------------------------
    out = []
    for (eng, mode), g in pq_df.groupby(["engine", "mode"]):
        for dim in ("all", "locale", "category"):
            groups = [("ALL", g)] if dim == "all" else list(g.groupby(dim))
            for sl, gg in groups:
                row = {"engine": eng, "mode": mode, "slice_type": dim, "slice": sl,
                       "n_queries": len(gg), "small_slice": len(gg) < a.min_queries}
                for m in METRICS:
                    mean, lo, hi = bootstrap_ci(gg[m].to_numpy())
                    row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
                out.append(row)
    slices = pd.DataFrame(out)
    slices.to_csv(os.path.join(a.run, "esci_slices.csv"), index=False)

    # ---- disparity: worst vs best slice, and the part added by ANN ------
    disp = []
    s = slices[~slices.small_slice & slices.slice_type.isin(["locale", "category"])]
    exact_ref = s[s.engine == "exact"].set_index(["mode", "slice_type", "slice"])
    for (eng, mode, st), g in s.groupby(["engine", "mode", "slice_type"]):
        for m in ("ndcg@10", "exact_recall@100", "ann_recall@10"):
            v = g.set_index("slice")[m].dropna()
            if len(v) < 2:
                continue
            row = {"engine": eng, "mode": mode, "slice_type": st, "metric": m,
                   "best_slice": v.idxmax(), "best": v.max(), "worst_slice": v.idxmin(), "worst": v.min(),
                   "gap": v.max() - v.min(), "ratio_worst_best": v.min() / v.max() if v.max() else np.nan}
            if eng != "exact" and m != "ann_recall@10":
                ref = exact_ref.loc[(mode, st)][m].reindex(v.index)
                loss = ref - v                         # quality lost to approximation, per slice
                row["ann_loss_max_slice"] = loss.idxmax(); row["ann_loss_max"] = loss.max()
                row["ann_loss_min"] = loss.min()
                row["gap_exact"] = ref.max() - ref.min()
                row["gap_added_by_engine"] = row["gap"] - row["gap_exact"]
            disp.append(row)
    pd.DataFrame(disp).to_csv(os.path.join(a.run, "esci_disparity.csv"), index=False)

    pd.set_option("display.width", 160)
    view = slices[slices.slice_type.isin(["all", "locale"])]
    print(view[["engine", "mode", "slice", "n_queries", "ndcg@10", "exact_recall@100", "ann_recall@10",
                "foreign@10", "underfilled"]].round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
