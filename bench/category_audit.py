"""Audit the zero-shot department labels against human judgement.

Step 1 - draw a stratified sample to label by hand (equal per locale):
    python -m bench.category_audit sample --emb data/esci_emb --per-locale 100
    -> data/esci_emb/category_audit.csv  (fill the `human_category` column)

Step 2 - score it:
    python -m bench.category_audit score --emb data/esci_emb
    -> accuracy overall and per locale, with 95% Wilson intervals

If accuracy differs a lot between locales, per-category results for the weaker
locale should be read with that caveat (reported in the data card).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import pandas as pd

from .taxonomy import NAMES


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--emb", default="data/esci_emb")
    ap.add_argument("--src", default="data/esci")
    ap.add_argument("--per-locale", type=int, default=100)
    a = ap.parse_args()
    path = os.path.join(a.emb, "category_audit.csv")

    if a.cmd == "sample":
        meta = pd.read_parquet(os.path.join(a.emb, "products_meta.parquet"))
        s = meta.sample(frac=1, random_state=1).groupby("locale").head(a.per_locale)
        prod = pd.read_parquet(os.path.join(a.src, "shopping_queries_dataset_products.parquet"),
                               columns=["product_id", "product_locale", "product_title"])
        s = s.merge(prod, left_on=["product_id", "locale"], right_on=["product_id", "product_locale"])
        s["human_category"] = ""
        s[["row", "locale", "product_title", "category_name", "human_category"]].to_csv(path, index=False)
        print(f"wrote {len(s)} rows to {path}\nallowed labels: {', '.join(NAMES)}")
        return

    df = pd.read_csv(path).dropna(subset=["human_category"])
    df = df[df.human_category.str.strip() != ""]
    df["ok"] = df.human_category.str.strip() == df.category_name
    rows = []
    for loc, g in [("ALL", df), *df.groupby("locale")]:
        lo, hi = wilson(int(g.ok.sum()), len(g))
        rows.append({"locale": loc, "n": len(g), "accuracy": g.ok.mean(), "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.emb, "category_audit_scores.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
