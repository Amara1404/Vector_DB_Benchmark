"""Benchmark runner.

    python -m bench.runner --config configs/sift1m.yaml [--engines faiss_hnsw,qdrant_hnsw]

For each engine in the config, sequentially (one engine resident at a time):
  1. build the index on the base set minus a held-out slice   -> build time, memory
  2. sweep search parameters over the query set               -> recall@1/@k, latency, QPS
  3. pick an operating point (fastest setting with recall@k >= target)
  4. at that point: concurrent throughput, filtered search, top-100 lists
  5. update cost: insert the held-out slice, delete a random slice,
     time-to-searchable, and recall drift after churn
Every result row is written to results/<run>/*.csv alongside env.json.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml

from .datasets import Dataset, cached_gt, load_dataset
from .engines import EngineSpec, Payload, make_engine
from .metrics import latency_stats, recall_at_k
from .sysinfo import PeakSampler, docker_mem_bytes, dump_json, environment, find_pid, rss_bytes


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def _payload(ds: Dataset, idx: np.ndarray) -> Payload | None:
    if ds.payload is None or ds.payload.locale is None:
        return None
    return Payload(locale=ds.payload.locale[idx], category=ds.payload.category[idx])


class Bench:
    def __init__(self, cfg: dict, outdir: str):
        self.cfg = cfg
        self.out = outdir
        os.makedirs(os.path.join(outdir, "ids"), exist_ok=True)
        self.k = int(cfg.get("k", 10))
        self.k_eval = int(cfg.get("k_eval", 100))
        self.target = float(cfg.get("recall_target", 0.95))
        self.rows: dict[str, list] = {"build": [], "sweep": [], "filtered_sweep": [], "filtered": [],
                                      "update": [], "concurrency": []}

        log("loading dataset", cfg["dataset"])
        self.ds = ds = load_dataset(cfg["dataset"])
        log(f"dataset {ds.name}: n={ds.n:,} dim={ds.dim} queries={len(ds.queries):,}")
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        perm = rng.permutation(ds.n)
        n_hold = int(ds.n * float(cfg.get("holdout_frac", 0.01)))
        self.hold_idx = np.sort(perm[:n_hold])
        self.base_idx = np.sort(perm[n_hold:])
        n_del = int(ds.n * float(cfg.get("delete_frac", 0.01)))
        self.del_idx = np.sort(rng.choice(self.base_idx, n_del, replace=False))
        self.gt_dir = os.path.join(cfg.get("cache_dir", "data/cache"), ds.name)

        all_ids = ds.ids
        kg = max(self.k, self.k_eval)
        mask_base = np.zeros(ds.n, bool); mask_base[self.base_idx] = True
        mask_after_ins = mask_base.copy(); mask_after_ins[self.hold_idx] = True
        mask_final = mask_after_ins.copy(); mask_final[self.del_idx] = False
        log("computing exact ground truth (FAISS IndexFlatL2)")
        self.gt = {name: cached_gt(self.gt_dir, name, ds.base, all_ids, ds.queries, kg, m)
                   for name, m in (("base", mask_base), ("after_insert", mask_after_ins), ("final", mask_final))}
        if ds.shipped_gt is not None and n_hold == 0:
            agree = recall_at_k(list(self.gt["base"][:, :self.k]), ds.shipped_gt, self.k).mean()
            log(f"sanity: computed GT vs shipped GT recall@{self.k} = {agree:.4f}")
        self.filtered_gt = None
        if ds.query_locale is not None:
            self.filtered_gt = np.full((len(ds.queries), kg), -1, dtype="int64")
            for loc in np.unique(ds.query_locale):
                qm = ds.query_locale == loc
                m = mask_base & (ds.payload.locale == loc)
                self.filtered_gt[qm] = cached_gt(self.gt_dir, f"filt_{loc}", ds.base, all_ids,
                                                 ds.queries[qm], kg, m)
        # exact lists are the quality baseline for the per-slice evaluation
        np.save(os.path.join(outdir, "ids", f"exact_unfiltered_k{self.k_eval}.npy"), self.gt["base"][:, :self.k_eval])
        if self.filtered_gt is not None:
            np.save(os.path.join(outdir, "ids", f"exact_filtered_k{self.k_eval}.npy"), self.filtered_gt[:, :self.k_eval])
        if "qmeta" in ds.extra:
            ds.extra["qmeta"].to_parquet(os.path.join(outdir, "queries_used.parquet"), index=False)
        dump_json(os.path.join(outdir, "env.json"),
                  {"env": environment(), "config": cfg, "n_base": len(self.base_idx),
                   "n_holdout": n_hold, "n_delete": n_del})

    # ------------------------------------------------------------------
    def _mem_fn(self, eng, spec: EngineSpec):
        if eng.in_process:
            return lambda: rss_bytes()
        if spec.conn.get("container"):
            c = spec.conn["container"]
            return lambda: docker_mem_bytes(c)
        pid = eng.server_pid() or (find_pid(spec.conn["process"]) if spec.conn.get("process") else None)
        if pid:
            return lambda: rss_bytes(pid)
        return lambda: None

    def _timed_queries(self, eng, Q, k, locale=None, warmup=50):
        for i in range(min(warmup, len(Q))):
            eng.search(Q[i], k, None if locale is None else locale[i])
        found, lat = [], np.empty(len(Q))
        for i in range(len(Q)):
            t0 = time.perf_counter()
            r = eng.search(Q[i], k, None if locale is None else locale[i])
            lat[i] = time.perf_counter() - t0
            found.append(r)
        return found, lat

    def _pad(self, found, k):
        out = np.full((len(found), k), -1, dtype="int64")
        for i, f in enumerate(found):
            out[i, :min(k, len(f))] = f[:k]
        return out

    # ------------------------------------------------------------------
    def run_engine(self, spec: EngineSpec, repeat: int = 0):
        ds, k = self.ds, self.k
        log(f"== {spec.name} (repeat {repeat}) ==")
        eng = make_engine(spec, ds.dim)
        mem = self._mem_fn(eng, spec)
        rss0 = mem() or 0
        with PeakSampler(mem) as ps:
            t0 = time.perf_counter()
            parts = eng.build(ds.ids[self.base_idx], ds.base[self.base_idx], _payload(ds, self.base_idx))
            build_s = time.perf_counter() - t0
        time.sleep(1.0)
        rss1 = mem() or 0
        ib = eng.index_bytes()
        brow = {**eng.describe(), "repeat": repeat, "n": len(self.base_idx), "dim": ds.dim,
                "build_total_s": build_s,
                **parts, "index_mb": ib / 2 ** 20 if ib else np.nan,
                "rss_delta_mb": (rss1 - rss0) / 2 ** 20 if rss1 else np.nan,
                "rss_peak_delta_mb": (ps.peak - rss0) / 2 ** 20 if ps.peak else np.nan,
                "rss_after_mb": rss1 / 2 ** 20 if rss1 else np.nan,
                "bytes_per_vector": (ib / len(self.base_idx)) if ib else np.nan,
                "in_process": eng.in_process}
        if spec.kind == "pgvector":
            brow["table_mb"] = eng.table_bytes() / 2 ** 20
            brow["pgvector_version"] = eng.version()
        self.rows["build"].append(brow)
        log(f"built in {build_s:.1f}s  index={brow['index_mb']:.0f}MB  rss_delta={brow['rss_delta_mb']:.0f}MB")

        # ---- parameter sweep (unfiltered queries) ----------------------
        sweep = []
        for params in spec.search_sweep:
            eng.set_search_params(**params)
            found, lat = self._timed_queries(eng, ds.queries, k)
            rec = recall_at_k(found, self.gt["base"], k)
            row = {**eng.describe(), "repeat": repeat, **{f"p_{a}": b for a, b in params.items()},
                   "param": ",".join(f"{a}={b}" for a, b in params.items()), "k": k,
                   f"recall@{k}": rec.mean(), "recall@1": recall_at_k(found, self.gt["base"], 1).mean(),
                   **latency_stats(lat), "in_process": eng.in_process}
            sweep.append((params, row))
            self.rows["sweep"].append(row)
            log(f"  {row['param']:<14} recall@{k}={row[f'recall@{k}']:.4f}  p50={row['lat_p50_ms']:.2f}ms "
                f"p99={row['lat_p99_ms']:.2f}ms  qps={row['qps_1thread']:.0f}")

        ok = [(p, r) for p, r in sweep if r[f"recall@{k}"] >= self.target]
        op_params, op_row = (min(ok, key=lambda x: x[1]["lat_p50_ms"]) if ok
                             else max(sweep, key=lambda x: x[1][f"recall@{k}"]))
        log(f"  operating point: {op_row['param']} (target recall@{k} >= {self.target})")
        eng.set_search_params(**op_params)

        # ---- top-k_eval lists at the operating point (used by ESCI eval) ---
        eparams = dict(op_params)
        if "ef" in eparams:
            eparams["ef"] = max(eparams["ef"], self.k_eval)
        if repeat == 0:
            eng.set_search_params(**eparams)
            found, _ = self._timed_queries(eng, ds.queries, self.k_eval, warmup=0)
            np.save(os.path.join(self.out, "ids", f"{spec.name}_unfiltered_k{self.k_eval}.npy"),
                    self._pad(found, self.k_eval))
            eng.set_search_params(**op_params)

        # ---- concurrency ------------------------------------------------
        for c in self.cfg.get("concurrency", []):
            self.rows["concurrency"].append({**eng.describe(), "repeat": repeat, "clients": c,
                                             **self._concurrent(spec, eng, op_params, c)})

        # ---- filtered search ---------------------------------------------
        # Filtering changes the recall/latency trade-off (fewer effective candidates
        # per store), so the filtered operating point is swept separately rather
        # than reusing the unfiltered one.
        if self.filtered_gt is not None:
            fsweep = []
            for params in spec.search_sweep:
                eng.set_search_params(**params)
                found, lat = self._timed_queries(eng, ds.queries, k, ds.query_locale)
                rec = recall_at_k(found, self.filtered_gt, k)
                row = {**eng.describe(), "repeat": repeat, **{f"p_{a}": b for a, b in params.items()},
                       "param": ",".join(f"{a}={b}" for a, b in params.items()), "k": k,
                       f"recall@{k}": rec.mean(), **latency_stats(lat), "in_process": eng.in_process}
                fsweep.append((params, row))
                self.rows["filtered_sweep"].append(row)

            fok = [(p, r) for p, r in fsweep if r[f"recall@{k}"] >= self.target]
            fop_params, fop_row = (min(fok, key=lambda x: x[1]["lat_p50_ms"]) if fok
                                   else max(fsweep, key=lambda x: x[1][f"recall@{k}"]))
            log(f"  filtered operating point: {fop_row['param']} (target recall@{k} >= {self.target})")

            eng.set_search_params(**fop_params)
            found, lat = self._timed_queries(eng, ds.queries, k, ds.query_locale)
            rec = recall_at_k(found, self.filtered_gt, k)
            under = np.array([len(f) < k for f in found])
            for loc in ["ALL", *np.unique(ds.query_locale)]:
                m = np.ones(len(found), bool) if loc == "ALL" else ds.query_locale == loc
                share = 1.0 if loc == "ALL" else float((ds.payload.locale[self.base_idx] == loc).mean())
                self.rows["filtered"].append({
                    **eng.describe(), "repeat": repeat, "param": fop_row["param"], "locale": loc,
                    "locale_share": share, "n_queries": int(m.sum()), f"recall@{k}": rec[m].mean(),
                    "underfilled_rate": under[m].mean(), **latency_stats(lat[m])})
            log(f"  filtered: recall@{k}={rec.mean():.4f} underfilled={under.mean():.1%}")
            if repeat == 0:
                feparams = dict(fop_params)
                if "ef" in feparams:
                    feparams["ef"] = max(feparams["ef"], self.k_eval)
                eng.set_search_params(**feparams)
                found, _ = self._timed_queries(eng, ds.queries, self.k_eval, ds.query_locale, warmup=0)
                np.save(os.path.join(self.out, "ids", f"{spec.name}_filtered_k{self.k_eval}.npy"),
                        self._pad(found, self.k_eval))
            eng.set_search_params(**op_params)

        # ---- update cost -------------------------------------------------
        if self.cfg.get("updates", True):
            self.rows["update"].append({**self._updates(spec, eng, op_row), "repeat": repeat})

        eng.drop()
        log(f"== {spec.name} done ==")

    # ------------------------------------------------------------------
    def _concurrent(self, spec, eng, params, clients):
        ds, k = self.ds, self.k
        if eng.in_process:
            workers = [eng] * clients
        else:
            workers = [make_engine(copy.deepcopy(spec), ds.dim) for _ in range(clients)]
            for w in workers:
                if spec.kind == "pgvector":
                    w.table = eng.table
                w.set_search_params(**params)
        chunks = np.array_split(np.arange(len(ds.queries)), clients)

        def work(j):
            lat = []
            for i in chunks[j]:
                t0 = time.perf_counter(); workers[j].search(ds.queries[i], k); lat.append(time.perf_counter() - t0)
            return lat

        t0 = time.perf_counter()
        with ThreadPoolExecutor(clients) as ex:
            lats = sum(ex.map(work, range(clients)), [])
        wall = time.perf_counter() - t0
        if not eng.in_process:
            for w in workers:
                getattr(w, "conn", None) and w.conn.close()
        ms = np.array(lats) * 1000
        log(f"  clients={clients}: {len(ms) / wall:.0f} QPS  p99={np.percentile(ms, 99):.2f}ms")
        return {"qps": len(ms) / wall, "lat_p50_ms": float(np.percentile(ms, 50)),
                "lat_p99_ms": float(np.percentile(ms, 99))}

    def _updates(self, spec, eng, op_row):
        ds, k = self.ds, self.k
        batch = int(self.cfg.get("update_batch", 1000))
        row = {**eng.describe(), "param": op_row["param"], "recall_before": op_row[f"recall@{k}"]}
        # inserts
        t0 = time.perf_counter()
        for s in range(0, len(self.hold_idx), batch):
            idx = self.hold_idx[s:s + batch]
            eng.insert(ds.ids[idx], ds.base[idx], _payload(ds, idx))
        t1 = time.perf_counter()
        eng.wait_searchable()
        t2 = time.perf_counter()
        row.update({"n_insert": len(self.hold_idx), "insert_s": t1 - t0,
                    "insert_vps": len(self.hold_idx) / max(t1 - t0, 1e-9),
                    "insert_to_searchable_s": t2 - t0})
        found, _ = self._timed_queries(eng, ds.queries, k, warmup=0)
        row["recall_after_insert"] = recall_at_k(found, self.gt["after_insert"], k).mean()
        # deletes
        t0 = time.perf_counter()
        for s in range(0, len(self.del_idx), batch):
            eng.delete(ds.ids[self.del_idx[s:s + batch]])
        t1 = time.perf_counter()
        row.update({"n_delete": len(self.del_idx), "delete_s": t1 - t0,
                    "delete_vps": len(self.del_idx) / max(t1 - t0, 1e-9),
                    "delete_native": eng.supports_delete})
        if spec.kind == "pgvector":
            t0 = time.perf_counter(); eng.vacuum(); row["vacuum_s"] = time.perf_counter() - t0
        found, _ = self._timed_queries(eng, ds.queries, k, warmup=0)
        rec = recall_at_k(found, self.gt["final"], k)
        stale = np.mean([np.isin(f, ds.ids[self.del_idx]).any() for f in found])
        row.update({"recall_after_delete": rec.mean(), "deleted_returned_rate": stale})
        if not eng.supports_delete:
            row["note"] = "no native delete: tombstone filter; reclaiming space needs a full rebuild"
        log(f"  updates: insert {row['insert_vps']:.0f} vec/s (searchable after "
            f"{row['insert_to_searchable_s']:.1f}s), delete {row['delete_vps']:.0f} vec/s, "
            f"recall {row['recall_before']:.4f} -> {row['recall_after_delete']:.4f}")
        return row

    # ------------------------------------------------------------------
    def save(self):
        for name, rows in self.rows.items():
            if rows:
                p = os.path.join(self.out, f"{name}.csv")
                df = pd.DataFrame(rows)
                if os.path.exists(p):  # append across separate engine/repeat invocations
                    old = pd.read_csv(p)
                    if "repeat" in old.columns and "repeat" in df.columns:
                        new_key = set(zip(df.engine, df.repeat))
                        keep = [k not in new_key for k in zip(old.engine, old.repeat)]
                        df = pd.concat([old[keep], df], ignore_index=True)
                    else:
                        df = pd.concat([old[~old.engine.isin(df.engine.unique())], df], ignore_index=True)
                df.to_csv(p, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--engines", help="comma-separated subset of engine names")
    ap.add_argument("--out", help="results dir (default results/<config name>)")
    ap.add_argument("--repeat", type=int, default=1,
                     help="repeat each engine's run N times so numbers come with a spread (default 1)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    # "${VAR:-default}" values are read from the environment, so the same config
    # works natively and inside Docker Compose
    def expand(v):
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        m = re.fullmatch(r"\$\{(\w+)(?::-(.*))?\}", v) if isinstance(v, str) else None
        return os.environ.get(m.group(1), m.group(2) or "") if m else v
    cfg["engines"] = [expand(e) for e in cfg["engines"]]
    out = a.out or os.path.join("results", os.path.splitext(os.path.basename(a.config))[0])
    bench = Bench(cfg, out)
    wanted = set(a.engines.split(",")) if a.engines else None
    for rep in range(a.repeat):
        for e in cfg["engines"]:
            if wanted and e["name"] not in wanted:
                continue
            spec = EngineSpec(**{k: v for k, v in e.items() if k in EngineSpec.__dataclass_fields__})
            try:
                bench.run_engine(spec, repeat=rep)
            except Exception as ex:  # one failing engine must not lose the others' results
                log(f"!! {spec.name} failed: {type(ex).__name__}: {ex}")
            bench.save()
    log(f"results written to {out}")


if __name__ == "__main__":
    # Windows consoles default to a legacy code page (e.g. cp1252), which raises
    # UnicodeEncodeError on non-ASCII log output; UTF-8 is safe everywhere else.
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
