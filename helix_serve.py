# HELIX-Serve - Serve wave-lattice recall requests through a structured service
# Part of the Kapildhinwa2010-dev Toolkit.
"""ServeNet - model registry and request envelopes (service layer).

Lineage: the *-Serve toolkit repos. Models are registered with metadata and
version; requests travel as validated envelopes; responses carry latency and
payload size so the registry can be monitored without a web framework.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["Envelope", "ModelRecord", "ServeRegistry"]


class Envelope:
    """Validated request payload."""

    def __init__(self, request_id: str, payload: np.ndarray) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        payload = np.asarray(payload, dtype=np.float64)
        if payload.size == 0:
            raise ValueError("payload must be non-empty")
        self.request_id = request_id
        self.payload = payload

    def nbytes(self) -> int:
        return int(self.payload.nbytes)


class ModelRecord:
    """A registered model with version and callable."""

    def __init__(self, name: str, version: str, fn) -> None:
        if not name or not version:
            raise ValueError("name and version must be non-empty")
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.name = name
        self.version = version
        self.fn = fn
        self.calls = 0

    def invoke(self, payload: np.ndarray) -> np.ndarray:
        result = self.fn(payload)
        self.calls += 1
        return np.asarray(result)


class ServeRegistry:
    """Registry with versioned lookup and served-request statistics."""

    def __init__(self) -> None:
        self._models: dict = {}
        self.stats = {"requests": 0, "total_latency_ms": 0.0, "total_bytes": 0}

    def register(self, name: str, version: str, fn) -> ModelRecord:
        record = ModelRecord(name, version, fn)
        self._models[(name, version)] = record
        return record

    def get(self, name: str, version: str = "latest") -> ModelRecord:
        if version == "latest":
            versions = [v for ((n, v), _) in self._models.items() if n == name]
            if not versions:
                raise KeyError("unknown model: {}".format(name))
            version = max(versions)
        key = (name, version)
        if key not in self._models:
            raise KeyError("unknown model: {}@{}".format(name, version))
        return self._models[key]

    def serve(self, name: str, envelope: Envelope, version: str = "latest") -> dict:
        model = self.get(name, version)
        start = time.perf_counter()
        result = model.invoke(envelope.payload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.stats["requests"] += 1
        self.stats["total_latency_ms"] += latency_ms
        self.stats["total_bytes"] += envelope.nbytes()
        return {
            "request_id": envelope.request_id,
            "model": model.name,
            "version": model.version,
            "shape": tuple(int(d) for d in result.shape),
            "latency_ms": latency_ms,
        }

    def summary(self) -> dict:
        requests = self.stats["requests"]
        return {
            "models": len(self._models),
            "requests": requests,
            "avg_latency_ms": (self.stats["total_latency_ms"] / requests) if requests else 0.0,
            "total_bytes": self.stats["total_bytes"],
        }

__all__.append("RecallMiss")
__all__.append("MemoryService")


class RecallMiss(Exception):
    """No stored pattern is similar enough to the probe."""


class MemoryService:
    """Serve recall requests against a lattice of stored patterns.

    Patterns are unit-normalized vectors; recall picks the most similar
    stored pattern by dot product and rejects weak matches. Every serve
    call returns the matched name, similarity and latency, and updates
    hit/miss statistics.
    """

    def __init__(self, dim: int, threshold: float = 0.5) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.dim = int(dim)
        self.threshold = threshold
        self._names = []
        self._vectors = []
        self.stats = {"served": 0, "hits": 0, "misses": 0}

    def add(self, name: str, vector: np.ndarray) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        if name in self._names:
            raise ValueError("pattern {} already stored".format(name))
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.size != self.dim:
            raise ValueError("vector must have exactly {} entries".format(self.dim))
        norm = float(np.linalg.norm(v))
        if norm <= 0.0:
            raise ValueError("vector must be non-zero")
        self._names.append(name)
        self._vectors.append(v / norm)

    def recall(self, probe: np.ndarray) -> dict:
        """Most similar stored pattern; raises RecallMiss below threshold."""
        if not self._names:
            raise RecallMiss("no patterns stored")
        v = np.asarray(probe, dtype=np.float64).ravel()
        if v.size != self.dim:
            raise ValueError("probe must have exactly {} entries".format(self.dim))
        norm = float(np.linalg.norm(v))
        if norm <= 0.0:
            raise ValueError("probe must be non-zero")
        sims = np.stack(self._vectors) @ (v / norm)
        best = int(np.argmax(sims))
        if float(sims[best]) < self.threshold:
            raise RecallMiss("best similarity {:.3f} below threshold".format(sims[best]))
        return {"name": self._names[best], "similarity": float(sims[best])}

    def serve(self, request_id: str, probe: np.ndarray) -> dict:
        """Serve one recall request; returns match and latency or a miss."""
        if not request_id:
            raise ValueError("request_id must be non-empty")
        start = time.perf_counter()
        self.stats["served"] += 1
        try:
            match = self.recall(probe)
        except RecallMiss:
            self.stats["misses"] += 1
            return {"request_id": request_id, "hit": False,
                    "latency_ms": (time.perf_counter() - start) * 1000.0}
        self.stats["hits"] += 1
        return {"request_id": request_id, "hit": True,
                "name": match["name"], "similarity": match["similarity"],
                "latency_ms": (time.perf_counter() - start) * 1000.0}

