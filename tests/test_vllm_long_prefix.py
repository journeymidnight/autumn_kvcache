"""Prompts longer than one prefill chunk, and layers larger than one value.

- Chunked prefill: vLLM computes a long prompt over several steps and hands the
  request its blocks as it goes. The prefix is saved at the step that computes
  its last token, over every block it has by then; saved at the first step it
  held only the first chunk's KV, under the whole prefix's hash.
- The partition server refuses a value over 64 MiB, so a layer above it is
  stored as pieces and read back the same way.

Scheduler and store are driven with fakes; no cluster, engine or GPU.
"""
from __future__ import annotations

import sys
import types
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
np = pytest.importorskip("numpy")
try:
    import autumn  # noqa: F401
except ImportError:
    sys.modules["autumn"] = MagicMock()

import autumn_kvcache.vllm_connector as vc  # noqa: E402

BLOCK = 16
CHUNK = 8192  # vLLM's default max_num_batched_tokens


class _Store:
    def __init__(self, present=False):
        self.present = present
        self.saved = []   # (content_hash, names, [bytes per layer])
        self.marked = []

    def is_present(self, chash):
        return self.present

    def save_layers(self, chash, names, views, region=None):
        self.saved.append((chash, list(names), [bytes(v) for v in views]))
        return [True] * len(names)

    def mark_present(self, chash):
        self.marked.append(chash)
        return True


class _Event:
    def record(self):
        pass

    def synchronize(self):
        pass


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "Event", _Event)


def _connector(kv_caches=None, present=False):
    c = object.__new__(vc.AutumnKVConnector)
    c._block_size = BLOCK
    c._is_mla = False
    c._store = _Store(present)
    c._reqs_need_load = {}
    c._alloc_blocks = {}
    c._pending_stores = {}
    c._kv_caches = kv_caches or {}
    c._meta = None
    c._pending_saves = {}
    c._save_jobs = {}
    c._save_pool = ThreadPoolExecutor(max_workers=1)
    c._free_arenas = [vc._Arena() for _ in range(vc._MAX_INFLIGHT_SAVES)]
    c._inflight_lock = threading.Lock()
    c._inflight_saves = 0
    return c


def _blocks(lo, hi):
    return [list(range(lo, hi))]


def _new(rid, n_tokens, blocks, computed=0, tokens=None):
    return types.SimpleNamespace(
        req_id=rid, prompt_token_ids=tokens or list(range(n_tokens)), block_ids=blocks,
        num_computed_tokens=computed, mm_features=[], lora_request=None,
    )


def _step(new=(), cached=None, scheduled=None):
    cached = cached or {}
    return types.SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=types.SimpleNamespace(
            req_ids=list(cached),
            resumed_req_ids={r for r, c in cached.items() if c.get("resumed")},
            new_block_ids=[c["blocks"] for c in cached.values()],
            num_computed_tokens=[c["computed"] for c in cached.values()],
        ),
        num_scheduled_tokens=scheduled or {},
    )


def _stores(meta):
    return [(r.start, r.start + len(r.slot_mapping), r.total) for r in meta.requests if r.is_store]


def _chunked_steps(n, rid="r"):
    """A prompt of n tokens prefilled CHUNK tokens a step: the steps' outputs."""
    per_chunk = CHUNK // BLOCK
    last_block = -(-n // BLOCK)
    steps = [_step(new=[_new(rid, n, _blocks(0, min(per_chunk, last_block)))], scheduled={rid: min(CHUNK, n)})]
    done = CHUNK
    while done < n:
        lo, hi = done // BLOCK, min((done + CHUNK) // BLOCK, last_block)
        steps.append(_step(cached={rid: {"blocks": _blocks(lo, hi), "computed": done}},
                           scheduled={rid: min(CHUNK, n - done)}))
        done += CHUNK
    return steps


def test_a_chunked_prefill_stores_each_step_s_tokens():
    n = 17_000
    prefix = vc.align_to_block_size(n - 1, BLOCK)
    c = _connector()
    got = [_stores(c.build_connector_meta(s)) for s in _chunked_steps(n)]
    assert got == [[(0, CHUNK, prefix)], [(CHUNK, 2 * CHUNK, prefix)], [(2 * CHUNK, prefix, prefix)]]
    assert c._pending_stores == {}


def test_slots_follow_the_blocks_given_step_by_step():
    n = 17_000
    c = _connector()
    metas = [c.build_connector_meta(s) for s in _chunked_steps(n)]
    slots = [t for m in metas for r in m.requests for t in r.slot_mapping.tolist()]
    assert slots == list(range(vc.align_to_block_size(n - 1, BLOCK)))


def test_a_prefix_ending_on_a_chunk_boundary_stores_at_that_step():
    n = 2 * CHUNK + 1  # prefix = 2 * CHUNK
    c = _connector()
    got = [_stores(c.build_connector_meta(s)) for s in _chunked_steps(n)]
    assert got == [[(0, CHUNK, 2 * CHUNK)], [(CHUNK, 2 * CHUNK, 2 * CHUNK)], []]


def test_a_local_hit_is_stored_with_the_first_step():
    n = 17_000
    c = _connector()
    meta = c.build_connector_meta(_step(new=[_new("r", n, _blocks(0, 1024), computed=4096)], scheduled={"r": CHUNK}))
    assert _stores(meta) == [(0, 4096 + CHUNK, vc.align_to_block_size(n - 1, BLOCK))]


def test_a_preempted_prefill_continues_from_what_it_stored():
    n = 17_000
    c = _connector()
    c.build_connector_meta(_chunked_steps(n)[0])
    # Preempted after its first chunk, then recomputed from scratch on other
    # blocks: nothing is stored again until it passes what it already stored.
    blocks = _blocks(5000, 5000 + -(-n // BLOCK))
    meta = c.build_connector_meta(_step(cached={"r": {"blocks": blocks, "computed": 0, "resumed": True}}, scheduled={"r": 4096}))
    assert _stores(meta) == []
    meta = c.build_connector_meta(_step(cached={"r": {"blocks": None, "computed": 4096}}, scheduled={"r": CHUNK}))
    [r] = meta.requests
    assert (r.start, r.start + len(r.slot_mapping)) == (CHUNK, 4096 + CHUNK)
    assert r.slot_mapping.tolist() == list(range(5000 * BLOCK + CHUNK, 5000 * BLOCK + 4096 + CHUNK))


def test_a_resumed_request_with_a_remote_hit_is_loaded():
    n = 10_000
    prefix = vc.align_to_block_size(n - 1, BLOCK)
    c = _connector(present=True)
    req = types.SimpleNamespace(request_id="r", prompt_token_ids=list(range(n)), mm_features=[], lora_request=None)
    c.build_connector_meta(_chunked_steps(n)[0])
    # Preempted; probed again on resume (num_computed_tokens is 0), a hit.
    assert c.get_num_new_matched_tokens(req, 0) == (prefix, False)
    blocks = _blocks(3000, 3000 + -(-n // BLOCK))
    meta = c.build_connector_meta(_step(cached={"r": {"blocks": blocks, "computed": prefix, "resumed": True}}, scheduled={"r": n - prefix}))
    [r] = meta.requests
    assert not r.is_store and r.slot_mapping.tolist() == list(range(3000 * BLOCK, 3000 * BLOCK + prefix))
    assert c._pending_stores == {}


def test_identical_prompts_in_one_step_store_once():
    c = _connector()
    meta = c.build_connector_meta(_step(new=[_new("a", 3000, _blocks(0, 200)), _new("b", 3000, _blocks(200, 400))],
                                        scheduled={"a": 3000, "b": 3000}))
    assert len(_stores(meta)) == 1


def test_finished_mid_prefill_forgets_the_pending_save():
    c = _connector()
    c.build_connector_meta(_chunked_steps(17_000)[0])
    c.request_finished(types.SimpleNamespace(request_id="r"), [])
    assert c._pending_stores == {}


# ── worker: the steps of one prefix land as one value per layer ──────────────

def _paged(num_blocks=1200):
    import torch
    g = torch.Generator().manual_seed(0)
    return {f"l{i}": torch.randint(0, 2**15, (num_blocks, 2, BLOCK, 2, 4), generator=g, dtype=torch.int16).view(torch.float16)
            for i in range(2)}


def _run_worker(c, meta):
    c.bind_connector_metadata(meta)
    for name, kv in c._kv_caches.items():
        c.save_kv_layer(name, kv, None)
    c.wait_for_save()
    c.clear_connector_metadata()


def _drain(c):
    c._save_pool.submit(lambda: None).result()


def test_a_chunked_prefix_is_written_once_as_the_whole_gather():
    n = 17_000
    prefix = vc.align_to_block_size(n - 1, BLOCK)
    c = _connector(_paged())
    for s in _chunked_steps(n):
        _run_worker(c, c.build_connector_meta(s))
    _drain(c)
    [(chash, names, data)] = c._store.saved
    assert c._store.marked == [chash] == [vc.prefix_hash(list(range(n)), prefix, [])]
    whole = vc._slot_mapping_for_blocks(list(range(-(-n // BLOCK))), BLOCK, prefix)
    for name, got in zip(names, data):
        want = vc._extract_layer(c._kv_caches[name], whole, False, BLOCK)
        assert got == want.contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes()
    assert c._inflight_saves == 0 and len(c._free_arenas) == vc._MAX_INFLIGHT_SAVES
    c.get_finished({"r"})  # the request ends after its save: nothing to release
    _drain(c)
    assert c._inflight_saves == 0 and len(c._free_arenas) == vc._MAX_INFLIGHT_SAVES


def test_a_prefix_found_present_releases_its_save_at_once():
    c = _connector(_paged(), present=True)
    steps = _chunked_steps(17_000)
    _run_worker(c, c.build_connector_meta(steps[0]))
    _drain(c)
    assert c._inflight_saves == 0, "held until the last step"
    _run_worker(c, c.build_connector_meta(steps[1]))
    assert c._save_jobs == {} and c._pending_saves == {}
    _run_worker(c, c.build_connector_meta(steps[2]))
    _drain(c)
    assert c._store.saved == [] and len(c._free_arenas) == vc._MAX_INFLIGHT_SAVES


def test_a_request_resumed_as_new_continues_its_save():
    # The v2 model runner sends a resumed request in scheduled_new_reqs.
    n = 17_000
    prefix = vc.align_to_block_size(n - 1, BLOCK)
    c = _connector(_paged())
    _run_worker(c, c.build_connector_meta(_chunked_steps(n)[0]))
    blocks = list(range(-(-n // BLOCK)))
    meta = c.build_connector_meta(_step(new=[_new("r", n, [blocks])], scheduled={"r": n}))
    assert _stores(meta) == [(CHUNK, prefix, prefix)]
    _run_worker(c, meta)
    _drain(c)
    [(_, names, data)] = c._store.saved
    whole = vc._slot_mapping_for_blocks(blocks, BLOCK, prefix)
    import torch
    for name, got in zip(names, data):
        assert got == vc._extract_layer(c._kv_caches[name], whole, False, BLOCK).view(-1).view(torch.uint8).numpy().tobytes()


def test_a_resumed_request_loaded_from_remote_ends_its_save(monkeypatch):
    c = _connector(_paged())
    _run_worker(c, c.build_connector_meta(_chunked_steps(17_000)[0]))
    assert c._inflight_saves == 1
    load = vc.AutumnConnectorMetadata()
    import torch
    load.add("r", "h", torch.tensor([0]), is_store=False, block_ids=[0])
    c._store.load_layers = lambda *a, **k: [False] * len(c._kv_caches)
    c._store.unmark_present = lambda h: None
    c._store.marker_ttl = 0
    c._store.model_scope = "m"
    c._load_arena = vc._Arena()
    c._load_failed_block_ids, c._failed_hashes = set(), set()
    c.bind_connector_metadata(load)
    c.start_load_kv(None)
    _drain(c)
    assert c._inflight_saves == 0 and c._save_jobs == {}


def test_a_request_ended_mid_prefill_releases_its_save():
    c = _connector(_paged())
    steps = _chunked_steps(17_000)
    _run_worker(c, c.build_connector_meta(steps[0]))
    assert c._inflight_saves == 1
    c.get_finished({"r"})
    _drain(c)
    assert c._store.saved == [] and c._inflight_saves == 0
    assert len(c._free_arenas) == vc._MAX_INFLIGHT_SAVES and c._save_jobs == {}


def test_over_the_cap_a_save_is_dropped_whole():
    c = _connector(_paged())
    gate = threading.Event()
    c._save_pool.submit(gate.wait)  # hold the save thread
    for rid, base in (("a", 0), ("b", 400), ("c", 800)):
        tokens = [base + t for t in range(5000)]
        _run_worker(c, c.build_connector_meta(_step(new=[_new(rid, 5000, _blocks(base, base + 313), tokens=tokens)], scheduled={rid: 5000})))
    gate.set()
    _drain(c)
    assert len(c._store.saved) == vc._MAX_INFLIGHT_SAVES and c._inflight_saves == 0


# ── layers over the value cap ────────────────────────────────────────────────

class _Batch:
    """Refuses values over the cap, as the partition server does."""

    def __init__(self):
        self.kv = {}

    def put_from(self, keys, views, ttl, region):
        oks = []
        for k, v in zip(keys, views):
            ok = len(v) <= vc._MAX_VALUE_BYTES
            if ok:
                self.kv[k] = bytes(v)
            oks.append(ok)
        return oks

    def get_into(self, keys, dests, region):
        oks = []
        for k, d in zip(keys, dests):
            v = self.kv.get(k)
            ok = v is not None and len(v) == len(d)
            if ok:
                d[:] = np.frombuffer(v, dtype=np.uint8)
            oks.append(ok)
        return oks


def _store(monkeypatch, cap):
    monkeypatch.setattr(vc, "_MAX_VALUE_BYTES", cap)
    s = object.__new__(vc._AutumnKVStore)
    s._model_scope = "m"
    s._layer_ttl = 0
    s._batch = _Batch()
    return s


def test_layer_over_the_cap_round_trips_in_pieces(monkeypatch):
    s = _store(monkeypatch, 1024)
    rng = np.random.default_rng(1)
    layers = [rng.integers(0, 256, n, dtype=np.uint8) for n in (1024, 2500, 300)]
    assert s.save_layers("h", ["l0", "l1", "l2"], layers) == [True, True, True]
    assert max(len(v) for v in s._batch.kv.values()) <= 1024
    dests = [np.zeros_like(x) for x in layers]
    assert s.load_layers("h", ["l0", "l1", "l2"], dests) == [True, True, True]
    for a, b in zip(layers, dests):
        assert np.array_equal(a, b)


def test_a_layer_within_the_cap_keeps_its_plain_key(monkeypatch):
    s = _store(monkeypatch, 1024)
    s.save_layers("h", ["l0"], [np.ones(1024, dtype=np.uint8)])
    assert list(s._batch.kv) == [s._key("h", "l0")]


def test_a_missing_piece_fails_its_layer(monkeypatch):
    s = _store(monkeypatch, 1024)
    s.save_layers("h", ["l0", "l1"], [np.ones(3000, dtype=np.uint8), np.ones(10, dtype=np.uint8)])
    del s._batch.kv[s._key("h", "l0/2")]
    assert s.load_layers("h", ["l0", "l1"], [np.zeros(3000, np.uint8), np.zeros(10, np.uint8)]) == [False, True]


def test_real_cap_splits_a_17k_token_layer():
    # Qwen3-VL-4B: 4 KiB per token per layer.
    pieces = vc._split("l", np.empty(17_280 * 4096, dtype=np.uint8))
    assert len(pieces) == 2
    assert all(len(v) <= 64 << 20 for _, v in pieces)
    assert sum(len(v) for _, v in pieces) == 17_280 * 4096


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
