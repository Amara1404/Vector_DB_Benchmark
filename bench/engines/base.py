"""Common interface every engine adapter implements.

All engines index L2 distance. Datasets that are naturally cosine (ESCI text
embeddings) are L2-normalised first, which makes L2 ranking identical to
cosine ranking, so every engine is compared on the same metric.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Payload:
    """Per-vector metadata used for filtered search (ESCI: locale, category)."""
    locale: Optional[np.ndarray] = None    # array of str, len n
    category: Optional[np.ndarray] = None  # array of int, len n


@dataclass
class EngineSpec:
    name: str                       # e.g. "qdrant_hnsw"
    kind: str                       # adapter key: faiss | pgvector | qdrant | milvus
    index: str = "hnsw"             # hnsw | ivf | flat
    build: dict = field(default_factory=dict)
    search_sweep: list = field(default_factory=list)  # list of dicts of search params
    conn: dict = field(default_factory=dict)


class Engine(ABC):
    """One vector index inside one engine."""

    #: whether delete() is supported without a full rebuild
    supports_delete: bool = True
    #: whether insert() is supported without a full rebuild
    supports_insert: bool = True
    #: True when the engine runs in the harness process (no network hop)
    in_process: bool = False

    def __init__(self, spec: EngineSpec, dim: int):
        self.spec = spec
        self.dim = dim

    # ---- lifecycle -------------------------------------------------------
    @abstractmethod
    def build(self, ids: np.ndarray, vecs: np.ndarray, payload: Payload | None) -> dict:
        """Load vectors and build the index. Returns timing breakdown in seconds,
        e.g. {"load_s": .., "index_s": ..}. Must block until the index is fully
        built and searchable."""

    @abstractmethod
    def drop(self) -> None:
        """Remove the index/collection/table and free resources."""

    # ---- search ----------------------------------------------------------
    @abstractmethod
    def set_search_params(self, **params: Any) -> None:
        ...

    @abstractmethod
    def search(self, q: np.ndarray, k: int, locale: str | None = None) -> np.ndarray:
        """Search one query vector. Returns int64 ids (len <= k). If `locale` is
        given, only vectors with that locale may be returned (filtered search)."""

    # ---- updates ---------------------------------------------------------
    @abstractmethod
    def insert(self, ids: np.ndarray, vecs: np.ndarray, payload: Payload | None) -> None:
        ...

    @abstractmethod
    def delete(self, ids: np.ndarray) -> None:
        ...

    def wait_searchable(self) -> None:
        """Block until all inserted vectors are indexed (default: immediate)."""

    # ---- footprint -------------------------------------------------------
    @abstractmethod
    def index_bytes(self) -> int | None:
        """Engine-reported size of the index structure (+ vectors), if known."""

    def server_pid(self) -> int | None:
        """PID of the server process when it runs natively, for RSS sampling."""
        return None

    def describe(self) -> dict:
        return {"engine": self.spec.name, "kind": self.spec.kind,
                "index": self.spec.index, **{f"build_{k}": v for k, v in self.spec.build.items()}}
