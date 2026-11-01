"""Qdrant adapter (Rust vector DB, gRPC client).

Bulk load follows Qdrant's own guidance: upload with HNSW indexing deferred
(indexing_threshold=0), then switch indexing on and wait until the collection
is green and every vector is indexed. Build time = upload + index wait, so it
is directly comparable with the other engines.
"""
from __future__ import annotations

import time

import numpy as np
from qdrant_client import QdrantClient, models

from .base import Engine, EngineSpec, Payload


class QdrantEngine(Engine):

    def __init__(self, spec: EngineSpec, dim: int):
        super().__init__(spec, dim)
        c = spec.conn
        self.client = QdrantClient(host=c.get("host", "localhost"), port=int(c.get("port", 6333)),
                                   grpc_port=int(c.get("grpc_port", 6334)), prefer_grpc=True,
                                   timeout=int(c.get("timeout", 600)))
        self.coll = c.get("collection", f"bench_{spec.name}")
        self.sp = None
        self.pid = c.get("pid")

    # ------------------------------------------------------------------
    def build(self, ids, vecs, payload):
        b = self.spec.build
        if self.client.collection_exists(self.coll):
            self.client.delete_collection(self.coll)
        self.client.create_collection(
            self.coll,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.EUCLID,
                                               on_disk=bool(b.get("on_disk", False))),
            hnsw_config=models.HnswConfigDiff(m=int(b.get("m", 16)),
                                              ef_construct=int(b.get("ef_construction", 128)),
                                              on_disk=bool(b.get("on_disk", False))),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        if payload is not None and payload.locale is not None:
            self.client.create_payload_index(self.coll, "locale", models.PayloadSchemaType.KEYWORD)
        t0 = time.perf_counter()
        self._upload(ids, vecs, payload)
        t1 = time.perf_counter()
        self.client.update_collection(
            self.coll, optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))
        self.wait_searchable()
        t2 = time.perf_counter()
        return {"load_s": t1 - t0, "index_s": t2 - t1}

    def _upload(self, ids, vecs, payload):
        pl = None
        if payload is not None and payload.locale is not None:
            pl = [{"locale": l, "category": int(c)} for l, c in
                  zip(payload.locale.tolist(), payload.category.tolist())]
        self.client.upload_collection(self.coll, vectors=vecs, ids=ids.tolist(), payload=pl,
                                      batch_size=int(self.spec.build.get("batch_size", 1024)),
                                      parallel=int(self.spec.build.get("upload_parallel", 2)),
                                      wait=True)

    def wait_searchable(self, timeout=7200):
        t0 = time.time()
        while time.time() - t0 < timeout:
            info = self.client.get_collection(self.coll)
            n = self.client.count(self.coll, exact=True).count
            if info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= n * 0.99:
                return
            time.sleep(1.0)
        raise TimeoutError("qdrant indexing did not finish")

    def drop(self):
        try:
            self.client.delete_collection(self.coll)
        finally:
            self.client.close()

    # ------------------------------------------------------------------
    def set_search_params(self, **p):
        self.sp = models.SearchParams(hnsw_ef=int(p.get("ef", 64)), exact=False)

    def search(self, q, k, locale=None):
        flt = None
        if locale is not None:
            flt = models.Filter(must=[models.FieldCondition(key="locale",
                                                            match=models.MatchValue(value=locale))])
        res = self.client.query_points(self.coll, query=q.tolist(), limit=k, search_params=self.sp,
                                       query_filter=flt, with_payload=False).points
        return np.fromiter((p.id for p in res), dtype="int64", count=len(res))

    # ------------------------------------------------------------------
    def insert(self, ids, vecs, payload):
        # incremental writes go through a single upsert per batch, which is how an
        # application would stream catalogue changes (upload_collection spawns workers)
        pl = None
        if payload is not None and payload.locale is not None:
            pl = [{"locale": l, "category": int(c)} for l, c in
                  zip(payload.locale.tolist(), payload.category.tolist())]
        self.client.upsert(self.coll, points=models.Batch(ids=ids.tolist(), vectors=vecs.tolist(),
                                                          payloads=pl), wait=True)

    def delete(self, ids):
        self.client.delete(self.coll, points_selector=models.PointIdsList(points=ids.tolist()), wait=True)

    def index_bytes(self):
        return None  # Qdrant does not report index bytes; runner samples server RSS instead

    def server_pid(self):
        return self.pid
