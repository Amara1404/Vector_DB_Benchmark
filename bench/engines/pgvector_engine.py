"""pgvector adapter (PostgreSQL extension).

Vectors are bulk-loaded with binary COPY, then an HNSW (or IVFFlat) index is
built. Filtered search is a plain WHERE clause, which is how most teams use
pgvector in practice. Before pgvector 0.8 this is post-filtering over
hnsw.ef_search candidates, so selective filters can return fewer than k rows;
the runner measures exactly that.
"""
from __future__ import annotations

import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .base import Engine, EngineSpec, Payload


class PgvectorEngine(Engine):

    def __init__(self, spec: EngineSpec, dim: int):
        super().__init__(spec, dim)
        c = spec.conn
        self.dsn = c.get("dsn", "postgresql://bench:bench@localhost:5433/bench")
        self.table = c.get("table", f"items_{spec.name}")
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.has_payload = False
        self._iterative = None

    def _exec(self, sql, params=None):
        return self.conn.execute(sql, params)

    # ------------------------------------------------------------------
    def build(self, ids, vecs, payload):
        b = self.spec.build
        self._exec(f"DROP TABLE IF EXISTS {self.table}")
        self.has_payload = payload is not None and payload.locale is not None
        cols = "id bigint PRIMARY KEY, embedding vector(%d)" % self.dim
        if self.has_payload:
            cols += ", locale text, category smallint"
        self._exec(f"CREATE TABLE {self.table} ({cols})")
        # Load without the PK index cost being hidden: PK is part of real usage.
        t0 = time.perf_counter()
        self._copy(ids, vecs, payload)
        t1 = time.perf_counter()
        self._exec(f"SET maintenance_work_mem = '{b.get('maintenance_work_mem', '2GB')}'")
        self._exec(f"SET max_parallel_maintenance_workers = {int(b.get('parallel_workers', 2))}")
        if self.spec.index == "hnsw":
            self._exec(
                f"CREATE INDEX {self.table}_vidx ON {self.table} USING hnsw (embedding vector_l2_ops) "
                f"WITH (m = {int(b.get('m', 16))}, ef_construction = {int(b.get('ef_construction', 128))})")
        elif self.spec.index == "ivf":
            self._exec(
                f"CREATE INDEX {self.table}_vidx ON {self.table} USING ivfflat (embedding vector_l2_ops) "
                f"WITH (lists = {int(b.get('nlist', 1000))})")
        if self.has_payload:
            self._exec(f"CREATE INDEX ON {self.table} (locale)")
        self._exec(f"ANALYZE {self.table}")
        t2 = time.perf_counter()
        # warm the index into shared buffers when pg_prewarm exists
        try:
            self._exec("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
            self._exec(f"SELECT pg_prewarm('{self.table}_vidx')")
        except psycopg.Error:
            pass
        return {"load_s": t1 - t0, "index_s": t2 - t1}

    def _copy(self, ids, vecs, payload):
        types = ["int8", "vector"] + (["text", "int2"] if self.has_payload else [])
        colnames = "id, embedding" + (", locale, category" if self.has_payload else "")
        with self.conn.cursor() as cur:
            with cur.copy(f"COPY {self.table} ({colnames}) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(types)
                if self.has_payload:
                    for i, v, l, c in zip(ids.tolist(), vecs, payload.locale.tolist(),
                                          payload.category.tolist()):
                        copy.write_row((i, v, l, int(c)))
                else:
                    for i, v in zip(ids.tolist(), vecs):
                        copy.write_row((i, v))

    def drop(self):
        try:
            self._exec(f"DROP TABLE IF EXISTS {self.table}")
        finally:
            self.conn.close()

    # ------------------------------------------------------------------
    def set_search_params(self, **p):
        if self.spec.index == "hnsw":
            self._exec(f"SET hnsw.ef_search = {int(p.get('ef', 40))}")
            # pgvector >= 0.8 iterative scans fix post-filter under-fill; opt-in via config
            if self.spec.build.get("iterative_scan"):
                try:
                    self._exec(f"SET hnsw.iterative_scan = {self.spec.build['iterative_scan']}")
                except psycopg.Error:
                    pass
        elif self.spec.index == "ivf":
            self._exec(f"SET ivfflat.probes = {int(p.get('nprobe', 10))}")

    def search(self, q, k, locale=None):
        if locale is not None:
            rows = self._exec(
                f"SELECT id FROM {self.table} WHERE locale = %s ORDER BY embedding <-> %s LIMIT %s",
                (locale, q, k)).fetchall()
        else:
            rows = self._exec(
                f"SELECT id FROM {self.table} ORDER BY embedding <-> %s LIMIT %s", (q, k)).fetchall()
        return np.fromiter((r[0] for r in rows), dtype="int64", count=len(rows))

    # ------------------------------------------------------------------
    def insert(self, ids, vecs, payload):
        self._copy(ids, vecs, payload)

    def delete(self, ids):
        self._exec(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (ids.tolist(),))

    def vacuum(self):
        """Deleted HNSW entries are only reclaimed by VACUUM; this is part of
        pgvector's real delete cost, so the runner times it separately."""
        self._exec(f"VACUUM {self.table}")

    def index_bytes(self):
        r = self._exec(f"SELECT pg_relation_size('{self.table}_vidx'), "
                       f"pg_total_relation_size('{self.table}')").fetchone()
        return int(r[0]) if r else None

    def table_bytes(self):
        return int(self._exec(f"SELECT pg_total_relation_size('{self.table}')").fetchone()[0])

    def version(self):
        return self._exec("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
