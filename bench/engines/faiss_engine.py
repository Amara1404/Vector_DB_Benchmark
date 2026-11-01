"""FAISS adapter (in-process library, no server).

Index types:
  hnsw  -> IndexHNSWFlat wrapped in IndexIDMap2. HNSW in FAISS has no delete;
           delete is emulated with a tombstone filter and the honest cost
           (a full rebuild) is reported by the runner.
  ivf   -> IndexIVFFlat with direct map; supports add + remove_ids.
  flat  -> exact search, used as the ground-truth / quality baseline.
"""
from __future__ import annotations

import time

import faiss
import numpy as np

from .base import Engine, EngineSpec, Payload


class FaissEngine(Engine):
    in_process = True

    def __init__(self, spec: EngineSpec, dim: int):
        super().__init__(spec, dim)
        self.index = None
        self.params = None
        self.tombstones: set[int] = set()
        self.locale_of: dict[int, str] = {}
        self._locale_sel_cache: dict[str, faiss.IDSelector] = {}
        threads = spec.build.get("threads")
        if threads:
            faiss.omp_set_num_threads(int(threads))
        if spec.index == "hnsw":
            self.supports_delete = False  # FAISS HNSW cannot remove nodes

    # ------------------------------------------------------------------
    def _new_index(self, train: np.ndarray | None):
        b = self.spec.build
        if self.spec.index == "hnsw":
            inner = faiss.IndexHNSWFlat(self.dim, int(b.get("m", 16)))
            inner.hnsw.efConstruction = int(b.get("ef_construction", 128))
            return faiss.IndexIDMap2(inner)
        if self.spec.index == "ivf":
            nlist = int(b.get("nlist", 4096))
            quant = faiss.IndexFlatL2(self.dim)
            idx = faiss.IndexIVFFlat(quant, self.dim, nlist, faiss.METRIC_L2)
            sample = train
            max_train = nlist * 64
            if sample is not None and len(sample) > max_train:
                rs = np.random.default_rng(0)
                sample = sample[rs.choice(len(sample), max_train, replace=False)]
            idx.train(sample)
            idx.make_direct_map(True)  # enables remove_ids/reconstruct
            idx.set_direct_map_type(faiss.DirectMap.Hashtable)
            return idx
        if self.spec.index == "flat":
            return faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))
        raise ValueError(f"unknown faiss index {self.spec.index}")

    def build(self, ids, vecs, payload):
        t0 = time.perf_counter()
        self.index = self._new_index(vecs)
        t1 = time.perf_counter()
        self.index.add_with_ids(np.ascontiguousarray(vecs, dtype="float32"), ids.astype("int64"))
        t2 = time.perf_counter()
        self._remember_payload(ids, payload)
        return {"train_s": t1 - t0, "load_s": 0.0, "index_s": t2 - t1}

    def _remember_payload(self, ids, payload: Payload | None):
        if payload is not None and payload.locale is not None:
            for i, loc in zip(ids.tolist(), payload.locale.tolist()):
                self.locale_of[i] = loc
            self._locale_sel_cache.clear()

    def drop(self):
        self.index = None
        self.locale_of.clear()
        self._locale_sel_cache.clear()

    # ------------------------------------------------------------------
    def set_search_params(self, **p):
        if self.spec.index == "hnsw":
            self.params = faiss.SearchParametersHNSW(efSearch=int(p.get("ef", 64)))
        elif self.spec.index == "ivf":
            self.params = faiss.SearchParametersIVF(nprobe=int(p.get("nprobe", 16)))
        else:
            self.params = None

    def _selector(self, locale: str | None):
        if locale is None and not self.tombstones:
            return None
        key = f"{locale}|{len(self.tombstones)}"
        if key not in self._locale_sel_cache:
            if locale is None:
                allowed = None
                sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(np.fromiter(self.tombstones, dtype="int64")))
            else:
                allowed = np.fromiter((i for i, l in self.locale_of.items()
                                       if l == locale and i not in self.tombstones), dtype="int64")
                sel = faiss.IDSelectorBatch(allowed)
            self._locale_sel_cache[key] = (sel, allowed)
        return self._locale_sel_cache[key][0]

    def search(self, q, k, locale=None):
        q = np.ascontiguousarray(q.reshape(1, -1), dtype="float32")
        sel = self._selector(locale)
        if self.spec.index == "hnsw":
            params = faiss.SearchParametersHNSW(efSearch=self.params.efSearch if self.params else 64, sel=sel) \
                if sel is not None else self.params
            # IndexIDMap2 forwards selector after id translation
            _, I = self.index.search(q, k, params=params)
        elif self.spec.index == "ivf":
            params = faiss.SearchParametersIVF(nprobe=self.params.nprobe if self.params else 16, sel=sel) \
                if sel is not None else self.params
            _, I = self.index.search(q, k, params=params)
        else:
            params = faiss.SearchParameters(sel=sel) if sel is not None else None
            _, I = self.index.search(q, k, params=params)
        r = I[0]
        return r[r >= 0]

    def search_batch(self, Q, k):
        """Batched search (used for exact ground truth and offline throughput)."""
        _, I = self.index.search(np.ascontiguousarray(Q, dtype="float32"), k, params=self.params)
        return I

    # ------------------------------------------------------------------
    def insert(self, ids, vecs, payload):
        self.index.add_with_ids(np.ascontiguousarray(vecs, dtype="float32"), ids.astype("int64"))
        self._remember_payload(ids, payload)

    def delete(self, ids):
        if self.spec.index == "hnsw":
            # no native delete: tombstone + filter at query time
            self.tombstones.update(int(i) for i in ids)
            self._locale_sel_cache.clear()
            return
        ids = np.ascontiguousarray(ids, dtype="int64")
        self.index.remove_ids(faiss.IDSelectorArray(len(ids), faiss.swig_ptr(ids)))
        for i in ids.tolist():
            self.locale_of.pop(i, None)
        self._locale_sel_cache.clear()

    def index_bytes(self):
        return int(faiss.serialize_index(self.index).nbytes) if self.index is not None else None
