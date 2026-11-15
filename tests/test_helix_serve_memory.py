"""Tests for HELIX-Serve MemoryService: lattice recall serving."""

import numpy as np
import pytest

from helix_serve import MemoryService, RecallMiss, ServeRegistry


def test_add_and_recall_exact():
    service = MemoryService(dim=4)
    service.add("a", [1.0, 0.0, 0.0, 0.0])
    match = service.recall([1.0, 0.0, 0.0, 0.0])
    assert match["name"] == "a"
    assert match["similarity"] == pytest.approx(1.0)


def test_recall_miss_below_threshold():
    service = MemoryService(dim=4, threshold=0.9)
    service.add("a", [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(RecallMiss):
        service.recall([0.0, 1.0, 0.0, 0.0])


def test_serve_updates_hit_miss_stats():
    service = MemoryService(dim=4)
    service.add("a", [1.0, 0.0, 0.0, 0.0])
    hit = service.serve("r1", [1.0, 0.0, 0.0, 0.0])
    miss = service.serve("r2", [0.0, 0.0, 0.0, 1.0])
    assert hit["hit"] is True and miss["hit"] is False
    assert service.stats["hits"] == 1 and service.stats["misses"] == 1
    assert hit["latency_ms"] >= 0.0


def test_add_validates():
    service = MemoryService(dim=4)
    with pytest.raises(ValueError):
        service.add("", [1, 0, 0, 0])
    with pytest.raises(ValueError):
        service.add("bad", [1.0, 0.0])
    with pytest.raises(ValueError):
        service.add("zero", [0.0, 0.0, 0.0, 0.0])


def test_base_registry_still_works():
    registry = ServeRegistry()
    registry.register("m", "1.0", lambda x: x)
    envelope = __import__("helix_serve").Envelope("r1", np.ones(4))
    out = registry.serve("m", envelope)
    assert out["request_id"] == "r1"
