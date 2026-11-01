"""Memory sampling and environment capture (recorded with every run)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time

import psutil


def rss_bytes(pid: int | None = None) -> int:
    """RSS of a process plus its children (Postgres backends are children)."""
    try:
        p = psutil.Process(pid or os.getpid())
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return 0


def find_pid(name: str) -> int | None:
    """First process whose name matches (e.g. 'qdrant', 'postgres' postmaster)."""
    for p in psutil.process_iter(["pid", "name", "ppid"]):
        if p.info["name"] == name:
            try:
                parent = psutil.Process(p.info["ppid"])
                if parent.name() == name:
                    continue  # want the root process
            except psutil.Error:
                pass
            return p.info["pid"]
    return None


def docker_mem_bytes(container: str) -> int | None:
    """Container memory usage via `docker stats` (works for Milvus/Qdrant/Postgres in Compose)."""
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    used = out.split("/")[0].strip()
    units = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "kB": 1e3, "MB": 1e6, "GB": 1e9}
    for u in sorted(units, key=len, reverse=True):
        if used.endswith(u):
            return int(float(used[: -len(u)]) * units[u])
    return None


class PeakSampler:
    """Samples RSS in a background thread to capture the peak during a build."""

    def __init__(self, sample_fn, interval=0.5):
        self.fn, self.interval = sample_fn, interval
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            v = self.fn() or 0
            self.peak = max(self.peak, v)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.fn() or 0
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def environment() -> dict:
    import numpy
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1), "numpy": numpy.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for mod in ("faiss", "qdrant_client", "pymilvus", "psycopg", "pgvector"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    return info


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
