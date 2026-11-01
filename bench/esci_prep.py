"""Turn the raw Amazon ESCI parquet files into benchmark inputs.

    python -m bench.esci_prep --src data/esci --out data/esci_emb \
        --embedder st:intfloat/multilingual-e5-small

Writes to --out:
  products.npy           (n, d) float32, L2-normalised product embeddings
  products_meta.parquet  row, product_id, locale, category, category_name, category_score
  queries.npy            (nq, d) float32, test-split query embeddings
  queries_meta.parquet   qidx, query_id, query, locale, category, category_name
  qrels.parquet          qidx, row, esci_label   (human relevance judgements)
  prep_info.json         embedder, sizes, wall time and throughput

Embedders:
  st:<hf-model>   sentence-transformers model (default multilingual-e5-small,
                  384-d, covers English/Spanish/Japanese). Use a GPU for the
                  full 1.8M catalogue; CPU works with --limit.
  hash            character n-gram hashing + SVD. No download, no GPU. Only
                  for smoke-testing the pipeline: it is lexical, not semantic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from .taxonomy import DEPARTMENTS, NAMES


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def product_text(df: pd.DataFrame, max_bullet=300) -> list[str]:
    def s(col):
        return df[col].fillna("").astype(str)
    bullets = s("product_bullet_point").str.replace(r"\s+", " ", regex=True).str.slice(0, max_bullet)
    parts = s("product_title") + ". " + s("product_brand") + " " + s("product_color") + ". " + bullets
    return parts.str.strip().tolist()


# ---------------------------------------------------------------------------
class STEmbedder:
    def __init__(self, model: str, device: str | None, batch: int, max_len: int):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device=device)
        self.m.max_seq_length = max_len
        self.batch = batch
        e5 = "e5" in model.lower()
        self.qp, self.pp = ("query: ", "passage: ") if e5 else ("", "")
        self.name = f"st:{model}"
        self.dim = self.m.get_sentence_embedding_dimension()

    def fit(self, texts):  # no-op, pretrained
        pass

    def _enc(self, texts):
        return self.m.encode(texts, batch_size=self.batch, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype("float32")

    def passages(self, texts):
        return self._enc([self.pp + t for t in texts])

    def queries(self, texts):
        return self._enc([self.qp + t for t in texts])


class HashEmbedder:
    """Lexical smoke-test embedder: char 2-4 grams -> hashing -> TF-IDF -> SVD."""

    def __init__(self, dim: int = 128):
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 18,
                                    alternate_sign=False, norm=None)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.dim = dim
        self.svd = None
        self.name = "hash-char-ngram-svd"

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        X = self.tfidf.fit_transform(self.hv.transform(texts))
        self.svd = TruncatedSVD(self.dim, random_state=0, algorithm="randomized", n_iter=4).fit(X)

    def _enc(self, texts):
        Z = self.svd.transform(self.tfidf.transform(self.hv.transform(texts))).astype("float32")
        n = np.linalg.norm(Z, axis=1, keepdims=True); n[n == 0] = 1
        return Z / n

    passages = queries = _enc


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/esci")
    ap.add_argument("--out", default="data/esci_emb")
    ap.add_argument("--embedder", default="st:intfloat/multilingual-e5-small")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap catalogue size; judged products are always kept")
    ap.add_argument("--queries-per-locale", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--fit-sample", type=int, default=50_000, help="hash embedder: products used to fit SVD")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    # ---- queries + judgements (test split of the large version) ----------
    ex = pd.read_parquet(os.path.join(a.src, "shopping_queries_dataset_examples.parquet"))
    ex = ex[(ex["split"] == "test") & (ex["large_version"] == 1)]
    qs = ex[["query_id", "query", "product_locale"]].drop_duplicates("query_id")
    if a.queries_per_locale:
        qs = qs.sample(frac=1, random_state=0).groupby("product_locale").head(a.queries_per_locale)
    ex = ex[ex["query_id"].isin(qs["query_id"])]
    log(f"test queries: {len(qs):,}  judgements: {len(ex):,}")

    # ---- catalogue -------------------------------------------------------
    cols = ["product_id", "product_title", "product_brand", "product_color",
            "product_bullet_point", "product_locale"]
    prod = pd.read_parquet(os.path.join(a.src, "shopping_queries_dataset_products.parquet"), columns=cols)
    log(f"catalogue: {len(prod):,} products")
    if a.limit and a.limit < len(prod):
        key = prod["product_id"] + "|" + prod["product_locale"]
        judged = set(ex["product_id"] + "|" + ex["product_locale"])
        is_j = key.isin(judged).to_numpy()
        rest = np.flatnonzero(~is_j)
        rng = np.random.default_rng(0)
        take = rng.choice(rest, max(0, a.limit - is_j.sum()), replace=False)
        prod = prod.iloc[np.sort(np.concatenate([np.flatnonzero(is_j), take]))]
        log(f"limited catalogue to {len(prod):,} (all {is_j.sum():,} judged products kept)")
    prod = prod.reset_index(drop=True)

    # ---- embedder ------------------------------------------------------
    if a.embedder == "hash":
        emb = HashEmbedder()
        sample = prod.sample(min(len(prod), a.fit_sample), random_state=0)
        log(f"fitting hashing+SVD embedder on {len(sample):,} products")
        emb.fit(product_text(sample))
    elif a.embedder.startswith("st:"):
        emb = STEmbedder(a.embedder[3:], a.device, a.batch, a.max_len)
    else:
        raise SystemExit(f"unknown embedder {a.embedder}")

    # ---- category prototypes (mean of en/es/ja descriptions) -------------
    protos = np.stack([emb.passages(list(d[1:])).mean(axis=0) for d in DEPARTMENTS])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)

    # ---- embed catalogue in chunks into a memmap --------------------------
    n = len(prod)
    P = np.lib.format.open_memmap(os.path.join(a.out, "products.npy"), mode="w+",
                                  dtype="float32", shape=(n, emb.dim))
    cat = np.empty(n, dtype="int16"); cat_score = np.empty(n, dtype="float32")
    t0 = time.time()
    for s in range(0, n, a.chunk):
        e = emb.passages(product_text(prod.iloc[s:s + a.chunk]))
        P[s:s + len(e)] = e
        sims = e @ protos.T
        cat[s:s + len(e)] = sims.argmax(1); cat_score[s:s + len(e)] = sims.max(1)
        done = s + len(e)
        rate = done / (time.time() - t0)
        log(f"embedded {done:,}/{n:,}  ({rate:,.0f} products/s, eta {(n - done) / rate / 60:.1f} min)")
    P.flush()
    embed_s = time.time() - t0

    meta = pd.DataFrame({"row": np.arange(n), "product_id": prod["product_id"],
                         "locale": prod["product_locale"], "category": cat,
                         "category_name": np.array(NAMES)[cat], "category_score": cat_score})
    meta.to_parquet(os.path.join(a.out, "products_meta.parquet"), index=False)

    # ---- queries + qrels ---------------------------------------------------
    qs = qs.reset_index(drop=True)
    qs["qidx"] = np.arange(len(qs))
    Q = emb.queries(qs["query"].astype(str).tolist())
    np.save(os.path.join(a.out, "queries.npy"), Q)
    rowkey = pd.Series(meta["row"].to_numpy(), index=meta["product_id"] + "|" + meta["locale"])
    ex = ex.merge(qs[["query_id", "qidx"]], on="query_id")
    ex["row"] = (ex["product_id"] + "|" + ex["product_locale"]).map(rowkey)
    qrels = ex.dropna(subset=["row"]).astype({"row": "int64"})[["qidx", "row", "esci_label"]]
    qrels.to_parquet(os.path.join(a.out, "qrels.parquet"), index=False)

    # query category = majority department of its Exact-labelled products
    e_rows = qrels[qrels.esci_label == "E"].merge(meta[["row", "category"]], on="row")
    maj = e_rows.groupby("qidx")["category"].agg(lambda c: c.value_counts().idxmax())
    qcat = (Q @ protos.T).argmax(1).astype("int16")
    qcat[maj.index.to_numpy()] = maj.to_numpy()
    qs["locale"] = qs["product_locale"]
    qs["category"] = qcat
    qs["category_name"] = np.array(NAMES)[qcat]
    qs[["qidx", "query_id", "query", "locale", "category", "category_name"]].to_parquet(
        os.path.join(a.out, "queries_meta.parquet"), index=False)

    info = {"embedder": emb.name, "dim": emb.dim, "n_products": n, "n_queries": len(qs),
            "n_qrels": len(qrels), "embed_seconds": embed_s, "products_per_second": n / embed_s,
            "total_seconds": time.time() - t_start,
            "locale_counts": meta["locale"].value_counts().to_dict(),
            "category_counts": meta["category_name"].value_counts().to_dict()}
    with open(os.path.join(a.out, "prep_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    log(json.dumps({k: info[k] for k in ("embedder", "n_products", "n_queries", "n_qrels",
                                          "products_per_second")}))


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
