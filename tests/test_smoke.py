"""autumn-kvcache smoke test (no sglang dependency).

Spins up against an externally-running 1-node autumn cluster (manager + EN
+ PS), exercises put/get/exists round-trip via the L3 backend API, and
validates:

  1. put_from / get_into zero-copy buffer round-trip
  2. batch_exists contiguous-prefix semantics
  3. model-scope wiring (different tp_rank → different partition key)

Setup before running:
    cd python && maturin develop --release         # builds the `autumn` module
    pip install -e python/autumn_kvcache           # installs this package
    ./cluster.sh reset 1                           # start a 1-node cluster
    AUTUMN_KVCACHE_ENDPOINT=127.0.0.1:9001 python -m autumn_kvcache.tests.test_smoke

Exit code: 0 on pass, non-zero on failure.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import numpy as np

from autumn_kvcache.sglang_backend import AutumnKVCacheStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,  # override any pre-existing root handler (Python 3.8+)
)
log = logging.getLogger("smoke")


@dataclass
class FakeStorageConfig:
    """Stand-in for sglang's HiCacheStorageConfig (we only need a few fields)."""

    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    is_mla_model: bool = False
    model_name: str = "smoke-model"


class FakeHostPool:
    """Stand-in for sglang's HostKVCache.

    Backs `get_data_page(idx, flat=True)` with a single contiguous numpy
    array carved into fixed-size pages. The returned object satisfies the
    Python buffer protocol — exactly what autumn's `put_from` / `get_into`
    need.
    """

    def __init__(self, n_pages: int, bytes_per_page: int):
        self.bytes_per_page = bytes_per_page
        self._arena = np.zeros(n_pages * bytes_per_page, dtype=np.uint8)

    def get_data_page(self, idx: int, flat: bool = True) -> np.ndarray:
        start = idx * self.bytes_per_page
        return self._arena[start : start + self.bytes_per_page]


def _make_key(i: int) -> str:
    """SHA-256 hex digest stand-in (sglang already provides these; we fabricate)."""
    return f"{i:064x}"


def main() -> int:
    endpoint = os.environ.get("AUTUMN_KVCACHE_ENDPOINT", "127.0.0.1:9001")
    log.info("connecting to autumn manager at %s", endpoint)

    cfg = FakeStorageConfig(tp_rank=0, tp_size=1, model_name="smoke-model")
    backend = AutumnKVCacheStorage(storage_config=cfg, extra_kwargs={"endpoint": endpoint})

    bytes_per_page = 4096
    n_pages = 8
    pool = FakeHostPool(n_pages=n_pages, bytes_per_page=bytes_per_page)
    backend.register_mem_pool_host(pool)

    # ── model-scope sanity: keys are namespaced ──────────────────────────
    sample_key = backend._full_key(_make_key(0))
    assert sample_key.startswith(b"kvc/smoke-model_0_1/kv/"), (
        f"unexpected key format: {sample_key!r}"
    )
    log.info("model-scope key format OK: %r", sample_key)

    # ── 1. write 4 pages with distinct fill bytes, then read back ────────
    keys = [_make_key(i) for i in range(4)]
    src_indices = list(range(4))

    # Fill source pages with recognisable data: page i = fill of byte 0xA0 + i.
    for i, idx in enumerate(src_indices):
        page = pool.get_data_page(idx, flat=True)
        page.fill(0xA0 + i)

    # Wipe any prior state from earlier test runs (idempotent).
    backend.clear()

    set_results = backend.batch_set_v1(keys, src_indices)
    assert all(set_results), f"batch_set_v1 had failures: {set_results}"
    log.info("batch_set_v1 OK: %d pages stored", sum(set_results))

    # Read into pages 4..7 (different slots) and verify byte content.
    dst_indices = list(range(4, 8))
    for idx in dst_indices:
        page = pool.get_data_page(idx, flat=True)
        page.fill(0)  # ensure starting from zero so a successful get is observable

    get_results = backend.batch_get_v1(keys, dst_indices)
    assert all(get_results), f"batch_get_v1 had misses: {get_results}"

    for i, idx in enumerate(dst_indices):
        page = pool.get_data_page(idx, flat=True)
        expected = 0xA0 + i
        if not np.all(page == expected):
            log.error("page %d content mismatch: expected %#x, got first=%#x last=%#x",
                      idx, expected, int(page[0]), int(page[-1]))
            return 1
    log.info("batch_get_v1 OK: %d pages round-tripped with byte-exact content", len(dst_indices))

    # ── 2. batch_exists contiguous-prefix semantics ──────────────────────
    # All 4 keys exist → expect 4.
    n = backend.batch_exists(keys)
    assert n == 4, f"batch_exists all-present: expected 4, got {n}"

    # First 2 exist, then a missing key → expect 2.
    mixed = keys[:2] + [_make_key(999)] + keys[2:]
    n = backend.batch_exists(mixed)
    assert n == 2, f"batch_exists mixed: expected 2, got {n}"

    # First key missing → expect 0.
    n = backend.batch_exists([_make_key(998)] + keys)
    assert n == 0, f"batch_exists head-missing: expected 0, got {n}"

    log.info("batch_exists contiguous-prefix semantics OK")

    # ── 3. v0 round-trip (off hot path, but must work) ───────────────────
    v0_key = _make_key(42)
    payload = b"hello-from-v0-path"
    assert backend.set(v0_key, payload) is True
    assert backend.exists(v0_key)
    got = backend.get(v0_key)
    assert got == payload, f"v0 get mismatch: {got!r} vs {payload!r}"
    log.info("v0 wrappers (set/get/exists) OK")

    # ── 4. stats sanity ──────────────────────────────────────────────────
    stats = backend.get_stats()
    log.info("stats: %s", stats)
    assert stats["set_ok"] >= 4
    assert stats["get_hit"] >= 4

    # ── cleanup ──────────────────────────────────────────────────────────
    backend.clear()
    log.info("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
