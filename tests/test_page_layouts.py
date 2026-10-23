"""KV pages round-trip through the sglang backend into the host pool itself.

The pools below reproduce sglang's host pools' page accessors
(`sglang/srt/mem_cache/memory_pool_host.py`: MHA and MLA
`get_data_page` / `set_from_flat_data_page` for every layout, and a pool that
assembles its page with `torch.cat`, as `MambaPoolHost` does). A page is
generally NOT contiguous in the pool, so `get_data_page(flat=True)` is a copy:
a read received into it reported a hit and left the pool untouched.

No cluster: the storage clients are in-memory fakes.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autumn_kvcache.sglang_backend import AutumnKVCacheStorage, _PageCodec  # noqa: E402

L, T, H, D, P = 3, 256, 2, 8, 16  # layers, tokens, kv heads, head dim, page size


class MHAPool:
    def __init__(self, layout):
        self.layout, self.page_size = layout, P
        shape = {
            "layer_first": (2, L, T, H, D),
            "page_first": (2, T, L, H, D),
            "page_first_direct": (2, T // P, L, P, H, D),
        }[layout]
        self.kv_buffer = torch.zeros(shape, dtype=torch.bfloat16)

    def get_data_page(self, index, flat=True):
        if self.layout == "layer_first":
            page = self.kv_buffer[:, :, index : index + P]
        elif self.layout == "page_first":
            page = self.kv_buffer[:, index : index + P]
        else:
            page = self.kv_buffer[:, index // P : index // P + 1]
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index, data_page):
        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + P] = data_page.reshape(2, L, P, H, D)
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + P] = data_page.reshape(2, P, L, H, D)
        else:
            self.kv_buffer[:, index // P : index // P + 1] = data_page.reshape(2, 1, L, P, H, D)


class MLAPool:
    def __init__(self, layout):
        self.layout, self.page_size = layout, P
        shape = {
            "layer_first": (L, T, 1, D),
            "page_first": (T, L, 1, D),
            "page_first_direct": (T // P, L, P, 1, D),
        }[layout]
        self.kv_buffer = torch.zeros(shape, dtype=torch.bfloat16)

    def get_data_page(self, index, flat=True):
        if self.layout == "layer_first":
            page = self.kv_buffer[:, index : index + P]
        elif self.layout == "page_first":
            page = self.kv_buffer[index : index + P]
        else:
            page = self.kv_buffer[index // P : index // P + 1]
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index, data_page):
        if self.layout == "layer_first":
            self.kv_buffer[:, index : index + P] = data_page.reshape(L, P, 1, D)
        elif self.layout == "page_first":
            self.kv_buffer[index : index + P] = data_page.reshape(P, L, 1, D)
        else:
            self.kv_buffer[index // P : index // P + 1] = data_page.reshape(1, L, P, 1, D)


class CatPool:
    """A page gathered from several tensors with torch.cat (MambaPoolHost)."""

    def __init__(self):
        self.page_size = P
        self.conv = torch.zeros((T, 4), dtype=torch.float32)
        self.ssm = torch.zeros((T, 6), dtype=torch.float32)

    def get_data_page(self, index, flat=True):
        page = torch.cat([t[index : index + P].contiguous().view(torch.uint8).reshape(-1)
                          for t in (self.conv, self.ssm)])
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index, data_page):
        flat = data_page.contiguous().view(torch.uint8).reshape(-1)
        start = 0
        for t in (self.conv, self.ssm):
            part = t[index : index + P]
            n = part.numel() * part.element_size()
            t[index : index + P] = flat[start : start + n].view(t.dtype).reshape(part.shape)
            start += n

    def contents(self):
        return torch.cat([self.conv.view(torch.uint8).reshape(-1), self.ssm.view(torch.uint8).reshape(-1)])


def page_bytes(pool, start):
    # Bytes, not values: random bytes include NaNs, which never compare equal.
    return pool.get_data_page(start).view(torch.uint8)


def contents(pool):
    return pool.contents() if isinstance(pool, CatPool) else pool.kv_buffer.view(torch.uint8).reshape(-1)


def fill(pool, seed):
    g = torch.Generator().manual_seed(seed)
    for t in ([pool.conv, pool.ssm] if isinstance(pool, CatPool) else [pool.kv_buffer]):
        t.view(torch.uint8).copy_(torch.randint(0, 256, t.view(torch.uint8).shape, generator=g, dtype=torch.uint8))


class FakeStore:
    """put_from / get_into on bytes, batch_head on presence; records every buffer."""

    def __init__(self):
        self.data, self.put_bufs, self.get_bufs = {}, [], []
        self.exists_calls = self.head_calls = 0

    def put_from(self, keys, bufs, ttl_secs=0):
        self.put_bufs += bufs
        for k, b in zip(keys, bufs):
            self.data[bytes(k)] = bytes(np.asarray(b))
        return [True] * len(keys)

    def get_into(self, keys, bufs):
        self.get_bufs += bufs
        out = []
        for k, b in zip(keys, bufs):
            v = self.data.get(bytes(k))
            ok = v is not None and len(v) == b.nbytes
            if ok:
                np.asarray(b)[:] = np.frombuffer(v, dtype=np.uint8)
            out.append(ok)
        return out

    def exists(self, keys):
        self.exists_calls += 1
        return [bytes(k) in self.data for k in keys]

    async def batch_head(self, keys):
        self.head_calls += 1
        return [bytes(k) in self.data for k in keys]

    async def head(self, key):
        return bytes(key) in self.data


def backend(store, pool):
    b = object.__new__(AutumnKVCacheStorage)
    b._batch = b._client = store
    b._model_scope, b._ttl_secs = "m_0_1", 0
    b._stats = {k: 0 for k in ("get_hit", "get_miss", "get_error", "set_ok", "set_error", "set_skipped")}
    b._mem_pool_host = pool
    return b


POOLS = [
    pytest.param(lambda: MHAPool("layer_first"), _PageCodec.COPY, id="mha-layer_first"),
    pytest.param(lambda: MHAPool("page_first"), _PageCodec.HALVES, id="mha-page_first"),
    pytest.param(lambda: MHAPool("page_first_direct"), _PageCodec.HALVES, id="mha-page_first_direct"),
    pytest.param(lambda: MLAPool("layer_first"), _PageCodec.COPY, id="mla-layer_first"),
    pytest.param(lambda: MLAPool("page_first"), _PageCodec.WHOLE, id="mla-page_first"),
    pytest.param(lambda: MLAPool("page_first_direct"), _PageCodec.WHOLE, id="mla-page_first_direct"),
    pytest.param(CatPool, _PageCodec.COPY, id="cat"),
]
HASHES = [f"{i:064x}" for i in range(4)]
STARTS = [0, 3 * P, 5 * P, 9 * P]


def indices(starts):
    return [s + t for s in starts for t in range(P)]


@pytest.mark.parametrize("make,kind", POOLS)
def test_a_read_lands_in_the_pool(make, kind):
    store, src, dst = FakeStore(), make(), make()
    fill(src, 1)
    assert all(backend(store, src).batch_set_v1(HASHES, indices(STARTS)))
    reader = backend(store, dst)
    assert reader._codec(dst).kind == kind
    assert all(reader.batch_get_v1(HASHES, indices(STARTS)))
    for s in STARTS:
        assert torch.equal(page_bytes(dst, s), page_bytes(src, s)), f"page at {s} did not reach the pool"
    # Nothing outside those pages was touched.
    untouched = make()
    for s in STARTS:
        untouched.set_from_flat_data_page(s, src.get_data_page(s))
    assert torch.equal(contents(dst), contents(untouched))


@pytest.mark.parametrize("make,kind", POOLS)
def test_window_pages_move_without_a_copy(make, kind):
    store, pool = FakeStore(), make()
    fill(pool, 2)
    b = backend(store, pool)
    b.batch_set_v1(HASHES, indices(STARTS))
    store.data = {k: v for k, v in store.data.items()}
    b.batch_get_v1(HASHES, indices(STARTS))
    if kind == _PageCodec.COPY:
        return
    base = contents(pool).numpy()
    for buf in store.put_bufs + store.get_bufs:
        assert np.shares_memory(np.asarray(buf), base), "a page that is a window into the pool was copied"


def test_layouts_never_share_a_key():
    keys = set()
    for make, _ in [p.values for p in POOLS if p.id.startswith("mha")]:
        pool = make()
        b = backend(FakeStore(), pool)
        page_keys = b._codec(pool).keys(b._full_key(HASHES[0]))
        assert not keys & set(page_keys)
        keys |= set(page_keys)
    assert b"m_0_1/kv/" + HASHES[0].encode() in keys, "layer_first must keep the keys written before"


@pytest.mark.parametrize("make,kind", POOLS)
def test_a_stored_page_is_not_written_again(make, kind):
    store, pool = FakeStore(), make()
    fill(pool, 3)
    b = backend(store, pool)
    assert all(b.batch_set_v1(HASHES, indices(STARTS)))
    writes = len(store.put_bufs)
    assert all(b.batch_set_v1(HASHES, indices(STARTS)))
    assert len(store.put_bufs) == writes
    assert b._stats["set_skipped"] == len(HASHES)
    assert b.batch_exists(HASHES) == len(HASHES)


def test_a_half_stored_page_is_absent():
    store, pool = FakeStore(), MHAPool("page_first")
    fill(pool, 4)
    b = backend(store, pool)
    b.batch_set_v1(HASHES, indices(STARTS))
    del store.data[b._codec(pool).keys(b._full_key(HASHES[1]))[1]]  # page 1's V
    assert b.batch_exists(HASHES) == 1
    assert b.batch_get_v1(HASHES, indices(STARTS)) == [True, False, True, True]


def test_v2_transfers_land_in_the_named_pool():
    store, src, dst = FakeStore(), MHAPool("layer_first"), MHAPool("layer_first")
    fill(src, 5)
    writer, reader = backend(store, None), backend(store, None)
    writer.registered_pools, reader.registered_pools = {"kv": src}, {"kv": dst}
    t = types.SimpleNamespace(name="kv", keys=HASHES, host_indices=indices(STARTS))
    assert all(writer._batch_v2([t], writer._put_pages, "set")["kv"])
    assert all(reader._batch_v2([t], reader._get_pages, "get")["kv"])
    for s in STARTS:
        assert torch.equal(page_bytes(dst, s), page_bytes(src, s))


@pytest.mark.parametrize("make,kind", POOLS)
def test_a_batch_calls_no_get_data_page_per_page(make, kind):
    """Once the pool is classified, a batch takes its buffers from numpy
    slices: no per-page torch calls, each of which retakes the GIL behind
    sglang's busy scheduler thread."""
    store, pool = FakeStore(), make()
    fill(pool, 6)
    b = backend(store, pool)
    b._codec(pool)
    calls = []
    real = pool.get_data_page
    pool.get_data_page = lambda *a, **k: calls.append(a) or real(*a, **k)
    b.batch_set_v1(HASHES, indices(STARTS))
    store.data = dict(store.data)
    b.batch_get_v1(HASHES, indices(STARTS))
    if isinstance(pool, CatPool):
        assert calls, "a pool that assembles its pages has no page axis"
    else:
        assert calls == [], f"{len(calls)} per-page get_data_page calls"


@pytest.mark.parametrize("start", [P // 2, 10_000 * P])
def test_a_host_index_off_the_pool_fails_the_batch(start):
    store, pool = FakeStore(), MHAPool("layer_first")
    fill(pool, 7)
    b = backend(store, pool)
    assert b.batch_set_v1(HASHES[:1], indices([0])) == [True]
    assert b.batch_get_v1(HASHES[:1], indices([start])) == [False]
    assert b._stats["get_error"] == 1


def test_presence_is_probed_with_the_blocking_call():
    """BatchClient.exists, not the awaitable batch_head."""
    store, pool = FakeStore(), MHAPool("page_first")
    fill(pool, 8)
    b = backend(store, pool)
    b.batch_set_v1(HASHES, indices(STARTS))
    store.exists_calls = 0
    assert b.batch_exists(HASHES) == len(HASHES)
    assert (store.exists_calls, store.head_calls) == (2, 0)  # first page, then all


def test_presence_falls_back_to_batch_head_without_exists():
    class OldStore(FakeStore):
        exists = None  # a native module from before BatchClient.exists

    store, pool = OldStore(), MHAPool("page_first")
    fill(pool, 9)
    b = backend(store, pool)
    assert all(b.batch_set_v1(HASHES, indices(STARTS)))
    assert b.batch_exists(HASHES) == len(HASHES)
    assert store.head_calls >= 2
