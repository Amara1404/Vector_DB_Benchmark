"""Dataset loaders.

  ann:<file>.hdf5   ANN-Benchmarks format (SIFT1M, GIST1M, GloVe...). Ships
                    train/test vectors and exact neighbours.
  synthetic         Clustered Gaussian vectors, for smoke tests only.
  esci              Amazon ESCI product embeddings produced by bench.esci_prep,
                    with locale + category payload for sliced evaluation.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .engines.base import Payload


@dataclass
class Dataset:
    name: str
    base: np.ndarray                 # (n, d) float32
    queries: np.ndarray              # (nq, d) float32
    metric: str = "l2"
    payload: Optional[Payload] = None
    query_locale: Optional[np.ndarray] = None   # (nq,) str, for filtered search
    shipped_gt: Optional[np.ndarray] = None     # (nq, k) shipped neighbours, if any
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.base.shape[0]

    @property
    def dim(self):
        return self.base.shape[1]

    @property
    def ids(self):
        return np.arange(self.n, dtype="int64")


def _normalize(x):
    x = np.ascontiguousarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_ann_hdf5(path: str, n_queries: int | None = None, limit: int | None = None) -> Dataset:
    import h5py
    with h5py.File(path, "r") as f:
        base = f["train"][: limit or None].astype("float32")
        queries = f["test"][: n_queries or None].astype("float32")
        gt = f["neighbors"][: n_queries or None] if "neighbors" in f and not limit else None
        metric = f.attrs.get("distance", "euclidean")
    name = os.path.basename(path).replace(".hdf5", "")
    if metric in ("angular", "cosine"):
        base, queries = _normalize(base), _normalize(queries)  # L2 on unit vectors == cosine
    return Dataset(name=name, base=base, queries=queries, shipped_gt=gt,
                   extra={"source_metric": str(metric)})


def make_synthetic(n: int, dim: int = 128, n_queries: int = 1000, clusters: int = 1000,
                   seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (clusters, dim)).astype("float32")
    assign = rng.integers(0, clusters, n)
    base = centers[assign] + rng.normal(0, 0.35, (n, dim)).astype("float32")
    qa = rng.integers(0, clusters, n_queries)
    queries = centers[qa] + rng.normal(0, 0.35, (n_queries, dim)).astype("float32")
    # give synthetic data a skewed "locale" mix so filtered paths get exercised too
    locs = np.array(["us", "jp", "es"])[rng.choice(3, n, p=[0.67, 0.19, 0.14])]
    cats = rng.integers(0, 8, n).astype("int16")
    qlocs = np.array(["us", "jp", "es"])[rng.choice(3, n_queries, p=[0.67, 0.19, 0.14])]
    return Dataset(name=f"synthetic{n // 1000}k", base=base, queries=queries,
                   payload=Payload(locale=locs, category=cats), query_locale=qlocs)


def load_esci(path: str, n_queries: int | None = None, limit: int | None = None) -> Dataset:
    """Load the artefacts written by `python -m bench.esci_prep`."""
    import pandas as pd
    base = np.load(os.path.join(path, "products.npy"), mmap_mode="r")
    meta = pd.read_parquet(os.path.join(path, "products_meta.parquet"))
    q = np.load(os.path.join(path, "queries.npy"))
    qmeta = pd.read_parquet(os.path.join(path, "queries_meta.parquet"))
    if limit:
        base, meta = base[:limit], meta.iloc[:limit]
    if n_queries and n_queries < len(qmeta):
        # stratified by locale so small locales keep enough queries to measure
        # equal queries per locale so small locales get tight confidence intervals
        per = max(1, n_queries // qmeta["locale"].nunique())
        qmeta = qmeta.sample(frac=1, random_state=0).groupby("locale").head(per).sort_values("qidx")
        q = q[qmeta["qidx"].to_numpy()]
    payload = Payload(locale=meta["locale"].to_numpy().astype(str),
                      category=meta["category"].to_numpy().astype("int16"))
    return Dataset(name="esci", base=np.ascontiguousarray(base, dtype="float32"),
                   queries=np.ascontiguousarray(q, dtype="float32"), payload=payload,
                   query_locale=qmeta["locale"].to_numpy().astype(str),
                   extra={"qmeta": qmeta.reset_index(drop=True), "meta_path": path})


def load_dataset(cfg: dict) -> Dataset:
    kind = cfg["name"]
    nq = cfg.get("n_queries")
    if kind == "synthetic":
        return make_synthetic(int(cfg.get("n", 100_000)), int(cfg.get("dim", 128)), nq or 1000)
    if kind == "esci":
        return load_esci(cfg["path"], nq, cfg.get("limit"))
    if kind == "ann":
        return load_ann_hdf5(cfg["path"], nq, cfg.get("limit"))
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# exact ground truth
# ---------------------------------------------------------------------------
def exact_knn(base: np.ndarray, ids: np.ndarray, queries: np.ndarray, k: int,
              allowed_mask: np.ndarray | None = None) -> np.ndarray:
    """Brute-force L2 top-k with FAISS IndexFlatL2. Returns ids (nq, k), -1 padded."""
    import faiss
    if allowed_mask is not None:
        base, ids = base[allowed_mask], ids[allowed_mask]
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(np.ascontiguousarray(base, dtype="float32"))
    _, I = index.search(np.ascontiguousarray(queries, dtype="float32"), k)
    out = np.where(I >= 0, ids[np.clip(I, 0, None)], -1)
    return out.astype("int64")


def cached_gt(cache_dir: str, tag: str, base, ids, queries, k, allowed_mask=None):
    h = hashlib.sha1()
    h.update(tag.encode()); h.update(ids[:1000].tobytes()); h.update(str(len(ids)).encode())
    h.update(queries[:50].tobytes()); h.update(str(k).encode())
    if allowed_mask is not None:
        h.update(np.packbits(allowed_mask).tobytes())
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"gt_{h.hexdigest()[:16]}.npy")
    if os.path.exists(p):
        return np.load(p)
    gt = exact_knn(base, ids, queries, k, allowed_mask)
    np.save(p, gt)
    return gt
