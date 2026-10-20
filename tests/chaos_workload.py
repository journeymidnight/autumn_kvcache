"""kvcache-interface chaos workload (no sglang dependency).

Runs a continuous put → readback-verify loop through the L3 backend API
(`batch_set_v1` / `batch_get_v1`) against an externally-running cluster,
while the bash harness (`scripts/kvcache_chaos.sh`) kills PSes / the
manager underneath it.

Protocol (stdout, line-oriented — the harness greps these):
    OK <r>          round r stored AND a random prior round read back byte-exact
    ERR <r> <what>  transient failure (expected during failover windows)
    MISMATCH <r>    CORRUPTION — stored content came back wrong (fatal signal)
    DONE total=<n> verified=<m> mismatches=<k>

Page r content: every byte == r % 251 (+ r in the first 8 bytes LE) so a
fresh process can re-derive the expectation from the round number alone.

Stop: create the file given in $CHAOS_STOP_FILE.
Final-verify mode: `--verify <manifest>` re-reads EVERY round listed in
the manifest with a fresh backend and exits non-zero on any miss/mismatch.
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

from autumn_kvcache.sglang_backend import AutumnKVCacheStorage

PAGE = 4096


@dataclass
class FakeStorageConfig:
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    is_mla_model: bool = False
    model_name: str = "chaos-model"


class FakeHostPool:
    def __init__(self, n_pages: int, bytes_per_page: int):
        self.bytes_per_page = bytes_per_page
        self._arena = np.zeros(n_pages * bytes_per_page, dtype=np.uint8)

    def get_data_page(self, idx: int, flat: bool = True) -> np.ndarray:
        start = idx * self.bytes_per_page
        return self._arena[start : start + self.bytes_per_page]


def key_of(r: int) -> str:
    return f"{r:064x}"


def fill_page(page: np.ndarray, r: int) -> None:
    page.fill(r % 251)
    page[:8] = np.frombuffer(np.uint64(r).tobytes(), dtype=np.uint8)


def page_matches(page: np.ndarray, r: int) -> bool:
    if not np.all(page[8:] == r % 251):
        return False
    return page[:8].tobytes() == np.uint64(r).tobytes()


def make_backend() -> tuple[AutumnKVCacheStorage, FakeHostPool]:
    endpoint = os.environ.get("AUTUMN_KVCACHE_ENDPOINT", "127.0.0.1:9001")
    backend = AutumnKVCacheStorage(
        storage_config=FakeStorageConfig(), extra_kwargs={"endpoint": endpoint}
    )
    pool = FakeHostPool(n_pages=4, bytes_per_page=PAGE)
    backend.register_mem_pool_host(pool)
    return backend, pool


def run_verify(manifest: str) -> int:
    backend, pool = make_backend()
    rounds = [int(line) for line in open(manifest) if line.strip()]
    bad = 0
    for r in rounds:
        page = pool.get_data_page(1)
        page.fill(0)
        try:
            ok = backend.batch_get_v1([key_of(r)], [1])
        except Exception as e:  # noqa: BLE001
            print(f"MISMATCH {r} get-exception {e}")
            bad += 1
            continue
        if not all(ok) or not page_matches(pool.get_data_page(1), r):
            print(f"MISMATCH {r} miss-or-bytes")
            bad += 1
    print(f"VERIFY total={len(rounds)} mismatches={bad}")
    return 1 if bad else 0


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--verify":
        return run_verify(sys.argv[2])

    stop_file = os.environ["CHAOS_STOP_FILE"]
    manifest = os.environ["CHAOS_MANIFEST"]
    backend, pool = make_backend()
    stored: list[int] = []
    total = verified = mismatches = 0
    r = 0
    mf = open(manifest, "a", buffering=1)
    while not os.path.exists(stop_file):
        r += 1
        # put round r
        page = pool.get_data_page(0)
        fill_page(page, r)
        try:
            if not all(backend.batch_set_v1([key_of(r)], [0])):
                print(f"ERR {r} set-false", flush=True)
                time.sleep(0.5)
                continue
        except Exception as e:  # noqa: BLE001
            print(f"ERR {r} set-exception {type(e).__name__}", flush=True)
            time.sleep(0.5)
            continue
        stored.append(r)
        mf.write(f"{r}\n")
        total += 1
        # readback-verify a random prior round
        pick = random.choice(stored)
        dst = pool.get_data_page(1)
        dst.fill(0)
        try:
            ok = backend.batch_get_v1([key_of(pick)], [1])
        except Exception as e:  # noqa: BLE001
            print(f"ERR {pick} get-exception {type(e).__name__}", flush=True)
            time.sleep(0.5)
            continue
        if all(ok) and page_matches(pool.get_data_page(1), pick):
            verified += 1
            print(f"OK {r}", flush=True)
        else:
            # A stored (ACKed) page must NEVER come back wrong or missing.
            mismatches += 1
            print(f"MISMATCH {pick}", flush=True)
        time.sleep(0.1)
    print(f"DONE total={total} verified={verified} mismatches={mismatches}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
