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
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("autumn", MagicMock())
np = pytest.importorskip("numpy")

import autumn_kvcache.vllm_connector as vc  # noqa: E402


class _FakeStore:
    def __init__(self, oks):
        self._oks = oks
        self.marker_ttl = 0
        self.model_scope = "qwen7b_deadbeef_0_1"

    def load_layers(self, content_hash, layer_names, dests):
        # Mirror the real contract: one bool per layer (or a short list on a
        # backend fault) — start_load_kv must treat both as fail-closed.
        return list(self._oks)


def _mk_connector(oks, block_ids):
    conn = vc.AutumnKVConnector.__new__(vc.AutumnKVConnector)
    conn._is_mla = False
    conn._block_size = 16
    conn._kv_caches = {"model.layers.0.attn": 0, "model.layers.1.attn": 1, "model.layers.2.attn": 2}
    conn._load_failed_block_ids = set()
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
    tmpl = SimpleNamespace(numel=lambda: 4, element_size=lambda: 2, dtype="f16", shape=(4,))
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
