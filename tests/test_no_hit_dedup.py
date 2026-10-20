"""BUG-KVC-NO-HIT regression: scheduler-side store-dedup + async durable save.

Drives `AutumnKVConnector`'s scheduler/worker methods against a FAKE
`_AutumnKVStore`, so no cluster / vLLM engine / GPU is needed. The connector is
built with `object.__new__` + the minimal attribute set these paths touch (its
real `__init__` connects to a cluster), which is exactly the surface to pin.

Pins:
  - a prefix ALREADY present in autumn is NOT re-saved (the write amplification
    the bug reported), even when the local prefix cache covered the prompt;
  - a cold prefix IS scheduled to save; a cold-local / warm-external prefix is
    LOADED (cross-instance hit intact);
  - identical cold prefixes in one batch are saved once, not N times;
  - the durable save runs on a background thread and publishes the marker only
    after every layer ACKs; over the in-flight cap the save is dropped.
"""
from __future__ import annotations

import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from autumn_kvcache.vllm_connector import AutumnKVConnector, align_to_block_size


def _torch():
    # Only the background-save tests build tensor pages; keep torch optional so
    # the pure scheduler-side tests still run on a torch-less CI.
    return pytest.importorskip("torch")

BLOCK = 16


class _FakeStore:
    """Records is_present / save_layers / mark_present; presence is caller-set."""

    def __init__(self, present: bool = False, save_gate: threading.Event | None = None):
        self._present = present
        self._save_gate = save_gate  # if set, save_layers blocks until it is set
        self.probes = 0
        self.saved: list = []   # [(content_hash, [layer_name...])]
        self.marked: list = []  # [content_hash...]

    def is_present(self, chash: str) -> bool:
        self.probes += 1
        return self._present

    def save_layers(self, chash: str, names, views) -> list:
        if self._save_gate is not None:
            self._save_gate.wait()
        self.saved.append((chash, list(names)))
        return [True] * len(names)

    def mark_present(self, chash: str) -> bool:
        self.marked.append(chash)
        return True

    @property
    def model_scope(self) -> str:
        return "fake"


def _mk(present=False, save_gate=None) -> AutumnKVConnector:
    c = object.__new__(AutumnKVConnector)  # bypass cluster-connecting __init__
    c._block_size = BLOCK
    c._is_mla = False
    c._store = _FakeStore(present, save_gate)
    c._reqs_need_load = {}
    c._alloc_blocks = {}
    c._kv_caches = {}
    c._meta = None
    c._pending_saves = {}
    c._save_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-save")
    c._inflight_lock = threading.Lock()
    c._inflight_saves = 0
    return c


def _req(rid: str, n_tokens: int):
    n_blocks = (n_tokens // BLOCK) + 2
    return types.SimpleNamespace(
        request_id=rid,
        prompt_token_ids=list(range(n_tokens)),
        block_ids=[list(range(n_blocks))],
    )


def _sched(*reqs):
    return types.SimpleNamespace(scheduled_new_reqs=list(reqs))


def _num_check(n_tokens):
    return align_to_block_size(max(0, n_tokens - 1), BLOCK)


# ── scheduler-side store-dedup ───────────────────────────────────────────────

def test_cold_prefix_is_stored():
    c = _mk(present=False)
    r = _req("a", 1333)
    assert c.get_num_new_matched_tokens(r, 0) == (0, False)  # not present → no load
    meta = c.build_connector_meta(_sched(r))
    assert len(meta.requests) == 1 and meta.requests[0].is_store is True


def test_repeat_does_not_probe_on_scheduler_path():
    # The reported bug's scenario: local prefix cache serves the repeat
    # (num_computed == num_check) so external is never LOADED. The scheduler now
    # does NO probe on this path (store-dedup moved to the background job).
    c = _mk(present=True)
    r = _req("a", 1333)
    assert c.get_num_new_matched_tokens(r, _num_check(1333)) == (0, False)
    assert c._store.probes == 0  # no remote probe on a same-instance repeat


def test_cross_instance_prefix_is_loaded():
    c = _mk(present=True)
    r = _req("a", 1333)
    n, is_async = c.get_num_new_matched_tokens(r, 0)  # cold local, warm external
    assert n == _num_check(1333) and is_async is False
    meta = c.build_connector_meta(_sched(r))
    assert len(meta.requests) == 1 and meta.requests[0].is_store is False


def test_same_batch_cold_prefix_is_saved_once():
    # Two DIFFERENT requests, SAME prompt (same chash), both cold, one batch →
    # only the first is scheduled to store (the marker isn't published yet, so
    # is_present() is False for both; the per-batch set de-dups the second).
    c = _mk(present=False)
    a, b = _req("a", 1333), _req("b", 1333)  # identical prompt_token_ids
    stores = [r for r in c.build_connector_meta(_sched(a, b)).requests if r.is_store]
    assert len(stores) == 1


# ── background save job (D2H copy + dedup + write, all off the critical path) ──

def test_background_job_dedups_then_writes_and_marks():
    # gather_event=None (CPU tensors, no CUDA needed). Not present → D2H copy +
    # durable write + marker, and the in-flight reservation is released.
    torch = _torch()
    c = _mk(present=False)
    c._inflight_saves = 1  # reserved by wait_for_save before submit
    layers = [("l0", torch.arange(16, dtype=torch.uint8)),
              ("l1", torch.arange(16, dtype=torch.uint8))]
    c._run_save_job("a", "hashA", layers, None)
    assert c._store.saved == [("hashA", ["l0", "l1"])]
    assert c._store.marked == ["hashA"]   # marker published ONLY after the write
    assert c._inflight_saves == 0


def test_background_job_skips_when_already_present():
    # Store-dedup now lives in the background: an already-durable prefix is
    # neither re-copied (D2H) nor re-written — the 20 GB amplification killer.
    torch = _torch()
    c = _mk(present=True)
    c._inflight_saves = 1
    layers = [("l0", torch.arange(16, dtype=torch.uint8))]
    c._run_save_job("a", "hashA", layers, None)
    assert c._store.saved == [] and c._store.marked == []  # deduped in background
    assert c._inflight_saves == 0                          # reservation still released


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
