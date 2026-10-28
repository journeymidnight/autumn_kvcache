"""BUG-KVC-LOAD-ATOMIC — `start_load_kv` must be all-or-nothing.

The live incident: the scheduler admits a request because the `__present__`
marker exists, so vLLM skips prefill for its prefix; the worker then loads the
per-layer KV. The OLD code injected layer-by-layer and `continue`d on a miss —
so a single missing layer left the request running on a MIX of loaded + still
uninitialised paged KV and silently emitted garbage
(`external KV load miss after positive presence`, layer 0..N).

Fix: if ANY layer fails to load, inject NOTHING for that request and report its
blocks via `get_block_ids_with_load_errors()` so vLLM re-runs normal prefill.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The fail-closed path does no tensor math (it bails before the inject loop);
# stub the heavy natives so the connector imports without a GPU/cluster. numpy
# stays REAL — start_load_kv sizes its staging buffers with it.
# torch is needed for real (the connector imports it); autumn is stubbed only
# where the native module is missing. A stub left in sys.modules is what every
# later test module imports too.
pytest.importorskip("torch")
try:
    import autumn  # noqa: F401
except ImportError:
    sys.modules["autumn"] = MagicMock()
np = pytest.importorskip("numpy")

import autumn_kvcache.vllm_connector as vc  # noqa: E402


class _FakeStore:
    def __init__(self, oks):
        self._oks = oks
        self.marker_ttl = 0
        self.model_scope = "qwen7b_deadbeef_0_1"
        self.unmarked = []

    def load_layers(self, content_hash, layer_names, dests):
        self.loads = getattr(self, "loads", 0) + 1
        # Mirror the real contract: one bool per layer (or a short list on a
        # backend fault) — start_load_kv must treat both as fail-closed.
        return list(self._oks)

    def unmark_present(self, content_hash):
        self.unmarked.append(content_hash)
        return True


def _mk_connector(oks, block_ids):
    conn = vc.AutumnKVConnector.__new__(vc.AutumnKVConnector)
    conn._is_mla = False
    conn._block_size = 16
    conn._kv_caches = {"model.layers.0.attn": 0, "model.layers.1.attn": 1, "model.layers.2.attn": 2}
    conn._load_failed_block_ids = set()
    conn._failed_hashes = set()
    conn._store = _FakeStore(oks)
    meta = vc.AutumnConnectorMetadata()
    meta.add("req-1", "hash-1", list(range(16)), is_store=False, block_ids=block_ids)
    conn._meta = meta
    return conn


@pytest.fixture(autouse=True)
def _stub_tensor_ops(monkeypatch):
    """`_extract_layer` returns a template we can size a staging buffer from;
    `_inject_layer` is a spy so a test can assert how many layers were written."""
    calls = []
    # The dtype of whichever torch is loaded (the real one, or the stub).
    tmpl = SimpleNamespace(numel=lambda: 4, element_size=lambda: 2,
                           dtype=sys.modules["torch"].float16, shape=(4,))
    monkeypatch.setattr(vc, "_extract_layer", lambda *a, **k: tmpl)
    monkeypatch.setattr(vc, "_inject_layer", lambda *a, **k: calls.append(a))
    return calls


def test_partial_load_injects_nothing_and_reports_blocks(_stub_tensor_ops):
    # [True, False, True] — the exact acceptance case: the middle layer misses.
    conn = _mk_connector(oks=[True, False, True], block_ids=[5, 6])
    conn.start_load_kv(forward_context=None)
    assert _stub_tensor_ops == [], "no layer may be injected when any layer misses"
    assert conn.get_block_ids_with_load_errors() == {5, 6}, "failed blocks must be reported for recompute"
    # drained: a second call reports nothing (return-and-clear)
    assert conn.get_block_ids_with_load_errors() == set()


def test_short_oks_list_is_also_fail_closed(_stub_tensor_ops):
    # A backend fault can return fewer bools than layers — must NOT inject.
    conn = _mk_connector(oks=[True], block_ids=[9])
    conn.start_load_kv(forward_context=None)
    assert _stub_tensor_ops == []
    assert conn.get_block_ids_with_load_errors() == {9}


def test_full_load_injects_every_layer_and_reports_no_errors(_stub_tensor_ops):
    conn = _mk_connector(oks=[True, True, True], block_ids=[5, 6])
    conn.start_load_kv(forward_context=None)
    assert len(_stub_tensor_ops) == 3, "all layers present -> inject all three"
    assert conn.get_block_ids_with_load_errors() == set()


def test_partial_load_deletes_the_marker(_stub_tensor_ops):
    # The memory tier evicts layers while the probed marker stays; a kept
    # marker would admit failing loads and stop the prefix being saved again.
    conn = _mk_connector(oks=[True, False, True], block_ids=[5, 6])
    conn.start_load_kv(forward_context=None)
    assert conn._store.unmarked == ["hash-1"]


def test_full_load_keeps_the_marker(_stub_tensor_ops):
    conn = _mk_connector(oks=[True, True, True], block_ids=[5, 6])
    conn.start_load_kv(forward_context=None)
    assert conn._store.unmarked == []


def test_a_hash_that_failed_this_step_is_not_fetched_again(_stub_tensor_ops):
    conn = _mk_connector(oks=[True, False, True], block_ids=[5, 6])
    conn._meta.add("req-2", "hash-1", list(range(16)), is_store=False, block_ids=[7])
    conn.start_load_kv(forward_context=None)
    assert conn._store.loads == 1
    assert conn._store.unmarked == ["hash-1"]
    assert conn.get_block_ids_with_load_errors() == {5, 6, 7}


def test_the_failed_hash_is_reported_to_the_scheduler_once(_stub_tensor_ops, monkeypatch):
    monkeypatch.setattr(vc, "_WORKER_META", True)
    conn = _mk_connector(oks=[True, False, True], block_ids=[5, 6])
    conn.start_load_kv(forward_context=None)
    assert conn.build_connector_worker_meta() == vc.AutumnWorkerMetadata(frozenset({"hash-1"}))
    assert conn.build_connector_worker_meta() is None


def test_ranks_reports_are_unioned():
    a = vc.AutumnWorkerMetadata(frozenset({"h1"}))
    b = vc.AutumnWorkerMetadata(frozenset({"h2", "h1"}))
    assert a.aggregate(b).failed_hashes == {"h1", "h2"}


def test_the_scheduler_deletes_the_markers_the_workers_report():
    # Under TP>1 the scheduler probes rank 0's marker; a failing worker of
    # another rank deletes only its own. The request may already be freed
    # (kv_load_failure_policy="fail"), so nothing per-request is consulted.
    conn = vc.AutumnKVConnector.__new__(vc.AutumnKVConnector)
    conn._store = _FakeStore([])
    conn.update_connector_output(SimpleNamespace(kv_connector_worker_meta=None, invalid_block_ids={5}))
    assert conn._store.unmarked == []
    meta = vc.AutumnWorkerMetadata(frozenset({"h1", "h2"}))
    conn.update_connector_output(SimpleNamespace(kv_connector_worker_meta=meta, invalid_block_ids={5}))
    assert sorted(conn._store.unmarked) == ["h1", "h2"]
