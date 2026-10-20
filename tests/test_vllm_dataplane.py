"""autumn-kvcache vLLM connector — data-plane smoke test (NO vLLM dependency).

Exercises the autumn-facing core the vLLM connector relies on — `_AutumnKVStore`
(key format + per-layer zero-copy save/load + existence) — against an
externally-running 1-node autumn cluster. This is the half that does
NOT need a model / vLLM runtime; the full e2e (cross-instance prefix hit) is
run in an isolated venv.

Setup before running:
    cd python && maturin develop --release         # builds the `autumn` module
    pip install -e python/autumn_kvcache           # installs this package
    ./cluster.sh reset 1                           # start a 1-node cluster
    AUTUMN_KVCACHE_ENDPOINT=127.0.0.1:9001 python -m autumn_kvcache.tests.test_vllm_dataplane

Exit code: 0 on pass, non-zero on failure.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

from autumn_kvcache._keys import KEY_NAMESPACE
from autumn_kvcache.vllm_connector import (
    _KV_STORAGE_FORMAT,
    VLLM_POOL_NAME,
    _AutumnKVStore,
    align_to_block_size,
    prefix_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("vllm-dataplane")


def _page(fill: int, nbytes: int) -> np.ndarray:
    a = np.empty(nbytes, dtype=np.uint8)
    a.fill(fill & 0xFF)
    return a


def main() -> int:
    endpoint = os.environ.get("AUTUMN_KVCACHE_ENDPOINT", "127.0.0.1:9001")
    transport = os.environ.get("AUTUMN_KVCACHE_TRANSPORT", "tcp")
    log.info("connecting to autumn manager at %s (transport=%s)", endpoint, transport)

    model_scope = "vllm-dp-model_0_1"
    store = _AutumnKVStore(endpoint, model_scope, transport=transport)

    # ── 1. pure-function sanity (no cluster) ─────────────────────────────────
    assert align_to_block_size(31, 16) == 16
    assert align_to_block_size(32, 16) == 32
    assert align_to_block_size(15, 16) == 0
    # deterministic + prefix-sensitive
    h_abc = prefix_hash([1, 2, 3, 4, 5], 3)
    assert h_abc == prefix_hash([1, 2, 3, 9, 9], 3), "hash must depend only on first num_tokens"
    assert h_abc != prefix_hash([1, 2, 4], 3), "different tokens → different hash"
    log.info("pure-function (align/prefix_hash) OK")

    # ── 2. key format ────────────────────────────────────────────────────────
    k = store._key("deadbeef", "layer.7")
    expected = (
        f"{KEY_NAMESPACE}/{model_scope}/{VLLM_POOL_NAME}/{_KV_STORAGE_FORMAT}/deadbeef/layer.7".encode()
    )
    assert k == expected, f"key mismatch: {k!r} != {expected!r}"
    log.info("key format OK: %r", k)

    # ── 3. per-layer zero-copy save/load round-trip ──────────────────────────
    chash = prefix_hash([10, 20, 30, 40], 4)
    layers = ["layer.0", "layer.1", "layer.2", "layer.3"]
    page_bytes = 64 * 1024  # 64 KiB — crosses the UCX bulk threshold
    src = [_page(0xC0 + i, page_bytes) for i in range(len(layers))]

    save_ok = store.save_layers(chash, layers, src)
    assert all(save_ok), f"save_layers failures: {save_ok}"
    log.info("save_layers OK: %d layers stored (%d B each)", len(layers), page_bytes)

    # load into fresh zeroed buffers, verify byte-exact
    dst = [np.zeros(page_bytes, dtype=np.uint8) for _ in layers]
    load_ok = store.load_layers(chash, layers, dst)
    assert all(load_ok), f"load_layers misses: {load_ok}"
    for i, (d, s) in enumerate(zip(dst, src)):
        if not np.array_equal(d, s):
            log.error("layer %d mismatch: first=%#x last=%#x expected=%#x",
                      i, int(d[0]), int(d[-1]), int(s[0]))
            return 1
    log.info("load_layers OK: %d layers round-tripped byte-exact", len(layers))

    # ── 4. existence semantics ───────────────────────────────────────────────
    assert store.exists(chash, "layer.0") is True
    assert store.exists(chash, "layer.999") is False
    assert store.exists(prefix_hash([7, 7, 7], 3), "layer.0") is False
    log.info("exists() semantics OK")

    # ── 4b. layer-name-independent presence marker (scheduler probe) ─────────
    # The scheduler probes is_present(), which must be False until the worker
    # publishes the marker in wait_for_save — decoupled from layer names.
    assert store.is_present(chash) is False, "marker must not exist before mark_present"
    assert store.mark_present(chash) is True
    assert store.is_present(chash) is True, "marker must exist after mark_present"
    assert store.is_present(prefix_hash([5, 5], 2)) is False
    log.info("presence-marker (mark_present/is_present) OK")

    # ── 5. partial-load miss is graceful (no crash, returns False) ───────────
    miss_dst = [np.zeros(page_bytes, dtype=np.uint8)]
    res = store.load_layers("0" * 64, ["layer.0"], miss_dst)
    assert res == [False], f"expected miss=[False], got {res}"
    log.info("missing-key load returns False (graceful) OK")

    # ── 6. TTL: marker expires before its layers; lazy expiry on read ────────
    # Exercises the new `ttl_secs` plumbing all the way through the native
    # binding (Client.put / BatchClient.put_from) into the partition layer's
    # expires_at + read-path lazy expiry. A ttl_secs=1 store writes the marker
    # with ttl=1 and the layers with ttl=1+grace, so after the marker expires
    # the layers must STILL be present (the "marker present ⇒ layers present"
    # invariant the scheduler relies on).
    ttl_store = _AutumnKVStore(endpoint, model_scope, transport=transport, ttl_secs=1)
    ttl_hash = prefix_hash([99, 98, 97, 96], 4)
    ttl_layers = ["layer.0", "layer.1"]
    ttl_src = [_page(0xE0 + i, page_bytes) for i in range(len(ttl_layers))]
    assert all(ttl_store.save_layers(ttl_hash, ttl_layers, ttl_src)), "ttl save failed"
    assert ttl_store.mark_present(ttl_hash) is True
    assert ttl_store.is_present(ttl_hash) is True, "marker must exist immediately"
    assert ttl_store.exists(ttl_hash, "layer.0") is True
    log.info("TTL save+marker OK (marker ttl=1s, layer ttl=1s+grace)")

    time.sleep(3)  # past the 1s marker TTL, well under the layer TTL (1+300s)
    assert ttl_store.is_present(ttl_hash) is False, "marker must lazily expire after its TTL"
    assert ttl_store.exists(ttl_hash, "layer.0") is True, (
        "layers must OUTLIVE the marker (grace) — else a positive is_present "
        "could race an expired layer into a silent partial load"
    )
    log.info("TTL lazy-expiry + marker-before-layer ordering OK")

    # ── 6b. batch path (BatchClient.put_from) TTL, no grace ──────────────────
    # §6 proves the marker (single Client.put) TTL, but the layer write goes via
    # the batched bulk path at ttl+grace (301s), so a regressed batch TTL would
    # still look alive at +3s. Drive BatchClient.put_from with a bare 1s TTL to
    # prove it actually threads expires_at (the path the connector uses most).
    bch = prefix_hash([1, 1, 1, 1], 4)
    bkey = store._key(bch, "batch.layer")
    assert ttl_store._batch.put_from([bkey], [_page(0xAB, page_bytes)], 1) == [True], (
        "batch TTL put failed"
    )
    assert ttl_store.exists(bch, "batch.layer") is True, "batch key must exist before its TTL"
    time.sleep(3)
    assert ttl_store.exists(bch, "batch.layer") is False, (
        "BatchClient.put_from must thread ttl_secs → expires_at (key must lazily expire)"
    )
    log.info("batch-path (BatchClient.put_from) TTL plumbing OK")

    log.info("ALL vLLM DATA-PLANE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
