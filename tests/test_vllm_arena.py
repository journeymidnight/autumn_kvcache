"""The vLLM connector's transfers go through reused, registered host buffers.

Over UCX autumn copies a buffer outside any registered region through its own
pool, so fresh memory per transfer (`np.empty`, `.cpu()`) paid a copy each
time. No cluster: `autumn.register` and the store are recorded.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import autumn_kvcache.vllm_connector as vc  # noqa: E402


class Registered:
    def __init__(self, buf):
        a = np.asarray(buf)
        self.start = a.__array_interface__["data"][0]
        self.end = self.start + a.nbytes

    def holds(self, buf):
        a = np.asarray(buf)
        p = a.__array_interface__["data"][0]
        return self.start <= p and p + a.nbytes <= self.end


@pytest.fixture(autouse=True)
def _record_registrations(monkeypatch):
    made = []

    def register(buf):
        made.append(Registered(buf))
        return made[-1]

    monkeypatch.setattr(vc.autumn, "register", register)
    return made


def test_the_arena_is_registered_once_per_size(_record_registrations):
    arena = vc._Arena()
    a = arena.take([100, 50])
    assert [x.nbytes for x in a] == [100, 50] and len(_record_registrations) == 1
    assert all(arena.region.holds(x) for x in a)
    b = arena.take([120])  # fits: same buffer, same registration
    assert len(_record_registrations) == 1 and np.shares_memory(a[0], b[0])
    c = arena.take([400, 400])  # grows: a new buffer, registered
    assert len(_record_registrations) == 2 and all(arena.region.holds(x) for x in c)


class _Store:
    def __init__(self):
        self.saved = []

    def is_present(self, chash):
        return False

    def save_layers(self, chash, names, views, region=None):
        self.saved.append(([bytes(np.asarray(v)) for v in views], region, list(views)))
        return [True] * len(names)

    def mark_present(self, chash):
        return True


def test_a_save_sends_the_tensors_bytes_from_the_registered_arena():
    c = object.__new__(vc.AutumnKVConnector)
    c._store = _Store()
    c._inflight_lock = threading.Lock()
    c._inflight_saves = 1
    c._save_arena = vc._Arena()
    layers = [("l0", torch.arange(64, dtype=torch.bfloat16).reshape(4, 16)),
              ("l1", torch.full((3, 8), 7, dtype=torch.float16))]
    c._run_save_job("r", "h", layers, None)
    (data, region, views), = c._store.saved
    assert data == [t.contiguous().view(torch.uint8).numpy().tobytes() for _, t in layers]
    assert region is c._save_arena.region and all(region.holds(v) for v in views)
