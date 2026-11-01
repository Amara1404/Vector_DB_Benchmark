"""Milvus adapter (pymilvus).

Point `conn.uri` at a Milvus standalone server (http://localhost:19530, from
docker compose --profile milvus). A local file path (e.g. ./milvus.db) uses
Milvus Lite, which is only good for checking that the adapter works: Lite
supports a reduced index set and is not a fair performance stand-in.
"""
from __future__ import annotations

import time

import numpy as np
from pymilvus import DataType, MilvusClient

from .base import Engine, EngineSpec, Payload


class MilvusEngine(Engine):

    def __init__(self, spec: EngineSpec, dim: int):
        super().__init__(spec, dim)
        self.uri = spec.conn.get("uri", "http://localhost:19530")
        self.client = MilvusClient(uri=self.uri)
        self.coll = spec.conn.get("collection", f"bench_{spec.name}")
        self.search_params = {}
        self.has_payload = False

    def _index_params(self):
        b = self.spec.build
        ip = self.client.prepare_index_params()
        if self.spec.index == "hnsw":
            ip.add_index("vec", index_type="HNSW", metric_type="L2",
                         params={"M": int(b.get("m", 16)), "efConstruction": int(b.get("ef_construction", 128))})
        elif self.spec.index == "ivf":
            ip.add_index("vec", index_type="IVF_FLAT", metric_type="L2",
                         params={"nlist": int(b.get("nlist", 4096))})
        else:
            ip.add_index("vec", index_type="FLAT", metric_type="L2")
        if self.has_payload:
            ip.add_index("locale", index_type="INVERTED")
        return ip

    def build(self, ids, vecs, payload):
        if self.client.has_collection(self.coll):
            self.client.drop_collection(self.coll)
        self.has_payload = payload is not None and payload.locale is not None
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=self.dim)
        if self.has_payload:
            schema.add_field("locale", DataType.VARCHAR, max_length=8)
            schema.add_field("category", DataType.INT16)
        self.client.create_collection(self.coll, schema=schema)
        t0 = time.perf_counter()
        self._insert(ids, vecs, payload)
        self.client.flush(self.coll)
        t1 = time.perf_counter()
        self.client.create_index(self.coll, self._index_params())
        self._wait_index()
        self.client.load_collection(self.coll)
        t2 = time.perf_counter()
        return {"load_s": t1 - t0, "index_s": t2 - t1}

    def _wait_index(self, timeout=7200):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                d = self.client.describe_index(self.coll, "vec")
            except Exception:
                return  # Milvus Lite builds synchronously
            pending = int(d.get("pending_index_rows", 0) or 0)
            if d.get("state") in (None, "Finished") and pending == 0:
                return
            time.sleep(1.0)

    def _insert(self, ids, vecs, payload, batch=5000):
        for s in range(0, len(ids), batch):
            rows = []
            for j in range(s, min(s + batch, len(ids))):
                r = {"id": int(ids[j]), "vec": vecs[j].tolist()}
                if self.has_payload:
                    r["locale"] = str(payload.locale[j]); r["category"] = int(payload.category[j])
                rows.append(r)
            self.client.insert(self.coll, rows)

    def drop(self):
        try:
            self.client.drop_collection(self.coll)
        finally:
            self.client.close()

    def set_search_params(self, **p):
        if self.spec.index == "hnsw":
            self.search_params = {"metric_type": "L2", "params": {"ef": int(p.get("ef", 64))}}
        elif self.spec.index == "ivf":
            self.search_params = {"metric_type": "L2", "params": {"nprobe": int(p.get("nprobe", 16))}}
        else:
            self.search_params = {"metric_type": "L2", "params": {}}

    def search(self, q, k, locale=None):
        sp = dict(self.search_params)
        if self.spec.index == "hnsw":
            sp["params"] = {"ef": max(int(sp["params"]["ef"]), k)}
        res = self.client.search(self.coll, data=[q.tolist()], limit=k, search_params=sp,
                                 filter=f'locale == "{locale}"' if locale else "",
                                 output_fields=[])
        return np.fromiter((h["id"] for h in res[0]), dtype="int64", count=len(res[0]))

    def insert(self, ids, vecs, payload):
        self._insert(ids, vecs, payload)
        self.client.flush(self.coll)

    def delete(self, ids):
        self.client.delete(self.coll, ids=ids.tolist())

    def index_bytes(self):
        return None  # sampled from the container via `docker stats` by the runner
