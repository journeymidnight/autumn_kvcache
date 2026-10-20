"""kvcache data-path micro-bench (no sglang).

Drives autumn.BatchClient over a contiguous pinned-style host pool (numpy),
exactly the kvcache v1 hot path (per-page put_from / get_into into pinned
pages), and reports write/read throughput.

"ucx ⟹ zerocopy": there is no longer a `bulk` flag. The zero-copy data
path (MSG_PUT_BULK write + MSG_GET_BULK read for large pages) is the DEFAULT on the
UCX transport; the regular path runs on TCP. So the A/B is now done at the
TRANSPORT level — run this once per transport (separate processes, since the
transport is process-global, first-call-wins):

    # regular (TCP)
    AUTUMN_KVCACHE_ENDPOINT="127.0.0.1:9001" AUTUMN_TRANSPORT=tcp \
      AUTUMN_BENCH_PAGES=512 AUTUMN_BENCH_PAGE_KB=256 \
      /tmp/autumn-py-venv/bin/python -m autumn_kvcache.tests.bench_bulk
    # zero-copy (UCX/RoCE)
    AUTUMN_KVCACHE_ENDPOINT="[<roce-ip>]:9001" AUTUMN_TRANSPORT=ucx \
      AUTUMN_BENCH_PAGES=512 AUTUMN_BENCH_PAGE_KB=256 \
      /tmp/autumn-py-venv/bin/python -m autumn_kvcache.tests.bench_bulk
"""
from __future__ import annotations
import os, time
import numpy as np
import autumn

ENDPOINT = os.environ.get("AUTUMN_KVCACHE_ENDPOINT", "127.0.0.1:9001")
TRANSPORT = os.environ.get("AUTUMN_TRANSPORT", "ucx")
N_PAGES = int(os.environ.get("AUTUMN_BENCH_PAGES", "512"))
PAGE = int(os.environ.get("AUTUMN_BENCH_PAGE_KB", "256")) * 1024
N_WORKERS = int(os.environ.get("AUTUMN_BENCH_WORKERS", "8"))
CAP = int(os.environ.get("AUTUMN_BENCH_CAP", "8"))
ROUNDS = int(os.environ.get("AUTUMN_BENCH_ROUNDS", "5"))
# Every client must bind a key scope. The bench writes into its OWN `bench/`
# namespace so it never touches fs/kvc/mem.
NAMESPACE = os.environ.get("AUTUMN_BENCH_NAMESPACE", "bench")


def run():
    # bulk is auto-derived from the transport (set below). bc.bulk() reports it.
    bc = autumn.BatchClient(ENDPOINT, N_WORKERS, CAP, scope=NAMESPACE)
    # One contiguous pool; per-page views (zero-copy numpy slices).
    pool = np.zeros(N_PAGES * PAGE, dtype=np.uint8)
    for i in range(N_PAGES):
        pool[i * PAGE] = (i % 251) + 1  # make pages distinguishable
    views = [pool[i * PAGE:(i + 1) * PAGE] for i in range(N_PAGES)]
    keys = [f"zcbench/{i:08d}".encode() for i in range(N_PAGES)]
    total_mb = N_PAGES * PAGE / 1e6

    def time_put():
        t = time.perf_counter()
        ok = bc.put_from(keys, views)
        return sum(ok), time.perf_counter() - t

    def time_get():
        t = time.perf_counter()
        ok = bc.get_into(keys, views)
        return sum(ok), time.perf_counter() - t

    okw, _ = time_put()  # warm + first write
    best_w = min(time_put()[1] for _ in range(ROUNDS))
    okr = time_get()[0]
    best_r = min(time_get()[1] for _ in range(ROUNDS))
    print(f"  zerocopy={bc.bulk()!s:5}  write {total_mb/best_w:8.1f} MB/s ({N_PAGES/best_w:7.0f} pg/s, ok={okw}/{N_PAGES})"
          f"  |  read {total_mb/best_r:8.1f} MB/s ({N_PAGES/best_r:7.0f} pg/s, ok={okr}/{N_PAGES})")


def main():
    autumn.set_transport(TRANSPORT)
    print(f"endpoint={ENDPOINT} transport={TRANSPORT} pages={N_PAGES} page={PAGE//1024}KiB "
          f"workers={N_WORKERS} cap={CAP} rounds={ROUNDS}")
    run()
    print("  (A/B: re-run with AUTUMN_TRANSPORT=tcp vs =ucx)")


if __name__ == "__main__":
    main()
