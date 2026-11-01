from .base import Engine, EngineSpec, Payload


def make_engine(spec: EngineSpec, dim: int) -> Engine:
    """Import adapters lazily so a missing optional client never blocks the others."""
    if spec.kind == "faiss":
        from .faiss_engine import FaissEngine
        return FaissEngine(spec, dim)
    if spec.kind == "pgvector":
        from .pgvector_engine import PgvectorEngine
        return PgvectorEngine(spec, dim)
    if spec.kind == "qdrant":
        from .qdrant_engine import QdrantEngine
        return QdrantEngine(spec, dim)
    if spec.kind == "milvus":
        from .milvus_engine import MilvusEngine
        return MilvusEngine(spec, dim)
    raise ValueError(f"unknown engine kind: {spec.kind}")


__all__ = ["Engine", "EngineSpec", "Payload", "make_engine"]
