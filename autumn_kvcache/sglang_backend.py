"""autumn-kvcache sglang HiCache L3 storage backend.

Plug in via:
  --enable-hierarchical-cache
  --hicache-storage-backend dynamic
  --hicache-storage-backend-extra-config '{
    "backend_name":"autumn",
    "module_path":"autumn_kvcache.sglang_backend",
    "class_name":"AutumnKVCacheStorage",
    "interface_v1":1,
    "endpoint":"manager:9001"
  }'

Architecture: thin Python adapter over the
existing `autumn` PyO3 client. No local DRAM cache, no sidecar daemon —
partition layer's memtable + block cache serves as the implicit DRAM tier.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import List, Optional

import autumn

from ._bridge import run, run_on, new_loop
from ._identity import fingerprint_from_sources, read_credential_pair as _read_credential_pair
from ._keys import build_model_scope, full_key, pool_prefix

# The interface this adapter implements is sglang's, but sglang is not the only
# engine that carries it: an engine may vendor the same module rather than
# depend on the whole runtime. FreeToken does exactly that — its only sglang
# requirement is `sglang-kernel` (which imports as `sgl_kernel`), so
# `sglang.srt.…` is absent there while `freetoken.kvcache.hicache.storage`
# holds the same symbols.
#
# Trying the vendored module second matters more than it looks. Without it the
# import below merely falls through to `object`, which is silent: the class
# still constructs, and the first v2 call fails with AttributeError on
# `register_mem_host_pool_v2` — inside an engine that treats a failed tier as
# "no tier". The result is a KV cache that is configured, reports no error, and
# stores nothing.
_IFACE_MODULES = (
    "sglang.srt.mem_cache.hicache_storage",
    "freetoken.kvcache.hicache.storage",
)


def _iface_candidates():
    """Interface modules to try, the host's own first.

    Already-imported beats installed. Every host imports its interface before it
    asks for a backend — sglang's `backend_factory` imports `hicache_storage` at
    module top, FreeToken's `attach` does `from .storage import
    HiCacheStorageConfig` before `import_module`ing this one — so what is in
    `sys.modules` names the HOST, while a fresh import in tuple order names only
    what happens to be installed.

    The difference bites in an image carrying both (a FreeToken image that also
    has sglang pulled in by something else). Both factories gate on
    `issubclass(backend_class, HiCacheStorage)` against their OWN class, so
    binding the wrong one is rejected by the host — as the same silent "no L3"
    this whole block exists to prevent.
    """
    out = []
    for mod_name in _IFACE_MODULES:
        mod = sys.modules.get(mod_name)
        # A `None` entry is the import system's "this failed"; treat it as absent.
        if mod is not None and mod not in out:
            out.append(mod)
    for mod_name in _IFACE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001
            # Broad, not just ImportError: a present-but-broken engine stack can
            # raise at import (e.g. flashinfer version-check RuntimeError) and
            # must not crash the adapter — the data-plane smoke test imports
            # this with no model stack at all.
            continue
        if mod not in out:
            out.append(mod)
    return out


def _names(mod, names):
    """Every `names` symbol from `mod`, or None if it does not carry them all."""
    try:
        return tuple(getattr(mod, n) for n in names)
    except AttributeError:
        return None


# ONE module supplies both halves. Taking v1 from one and v2 from another would
# build a class whose base is one engine's ABC while its pool types are
# another's — a combination no host produces, and one that would pass the
# adapter's own tests while failing a host factory's issubclass gate.
_iface = None
_v1 = None
for _cand in _iface_candidates():
    _v1 = _names(_cand, ("HiCacheStorage", "HiCacheStorageConfig"))
    if _v1 is not None:
        _iface = _cand
        break

if _v1 is not None:
    HiCacheStorage, HiCacheStorageConfig = _v1
    _SGLANG_AVAILABLE = True
else:
    # Fall back to `object` so the module is importable with no engine at all
    # (smoke tests + standalone usage).
    HiCacheStorage = object  # type: ignore[misc,assignment]
    HiCacheStorageConfig = None  # type: ignore[assignment]
    _SGLANG_AVAILABLE = False

# v2 multi-pool types, from the SAME module the ABC came from. Tolerated
# missing: they landed later than v1, so an engine without them must still be
# able to load this backend and run the v1 path. When they are absent the v2
# methods below degrade to NotImplementedError, which is exactly what a v1-only
# deployment expects.
_v2 = _names(_iface, ("PoolHitPolicy", "PoolName", "PoolTransfer", "PoolTransferResult")) \
    if _iface is not None else None
if _v2 is not None:
    PoolHitPolicy, PoolName, PoolTransfer, PoolTransferResult = _v2
    _SGLANG_V2_AVAILABLE = True
else:
    PoolHitPolicy = None  # type: ignore[assignment]
    PoolName = None  # type: ignore[assignment]
    PoolTransfer = None  # type: ignore[assignment]
    PoolTransferResult = None  # type: ignore[assignment]
    _SGLANG_V2_AVAILABLE = False

log = logging.getLogger(__name__)

# Reserved pool name. MVP only supports the "kv" pool; the slot is in the
# key format so future v2 multi-pool (mamba / swa) doesn't require a key
# migration. (The shared `kvc/` namespace + model scope live in `_keys`.)
DEFAULT_POOL_NAME = "kv"


def _normalize_indices(host_indices) -> List[int]:
    """Accept torch.Tensor, numpy array, list, or any iterable of ints."""
    if hasattr(host_indices, "tolist"):
        return host_indices.tolist()
    return list(host_indices)


def _page_start_indices(keys, host_indices) -> List[int]:
    """Resolve the starting host_index of each page.

    With page_size>1, sglang passes `page_size` host_indices PER page (the
    indices of every token in the page), so len(host_indices) ==
    len(keys) * page_size. `get_data_page` wants the page's first index and
    returns the whole flattened page. Mirror sglang's own
    `host_indices[i * page_size]` (cache_controller `_generic_page_set`).

    page_size is derived as len(indices) // len(keys) so we don't need to
    track it separately; for page_size==1 this is the identity.
    """
    indices = _normalize_indices(host_indices)
    n = len(keys)
    if n == 0:
        return []
    page_size = max(1, len(indices) // n)
    return [indices[i * page_size] for i in range(n)]


class AutumnKVCacheStorage(HiCacheStorage):  # type: ignore[misc]
    """sglang HiCache L3 backend.

    The constructor signature `(storage_config, extra_kwargs)` matches what
    `StorageBackendFactory` passes when sglang's `--hicache-storage-backend
    dynamic` plugs in a user-supplied class.
    """

    def __init__(self, storage_config=None, extra_kwargs: Optional[dict] = None):
        # `storage_config.extra_config` is the parsed JSON dict from
        # `--hicache-storage-backend-extra-config '{...}'`. sglang's
        # `_create_dynamic_backend` does NOT split out individual keys from
        # extra_config into kwargs; everything lives on the dataclass.
        # `extra_kwargs` is just whatever the factory's caller passed (today
        # always {}), kept for forward-compat.
        extra_kwargs = extra_kwargs or {}
        extra_config = getattr(storage_config, "extra_config", None) or {}
        endpoint = (
            extra_config.get("endpoint")
            or extra_config.get("manager")
            or extra_kwargs.get("endpoint")
            or extra_kwargs.get("manager")
        )
        if not endpoint:
            raise ValueError(
                "AutumnKVCacheStorage requires 'endpoint' in extra_config "
                "(comma-separated list of manager addr:port)"
            )

        self.storage_config = storage_config
        # On the sglang path `model_name` is the served
        # `--model-path` — normally a real identity (sglang has no equivalent
        # of a fixed config dir shared by several models), so the default
        # model-scope format is UNCHANGED (no key invalidation for existing sglang
        # deployments). `HiCacheStorageConfig` carries no architecture info to
        # fingerprint from, so the escape hatch for deployments where the path
        # is NOT unique (two finetunes at one path, containers mounting
        # different weights at the same mount point) is an explicit
        # extra_config["model_id"], folded in as a fingerprint.
        model_id = extra_config.get("model_id")
        fingerprint = (
            fingerprint_from_sources({"model_id": str(model_id)}) if model_id else None
        )
        self._model_scope = build_model_scope(storage_config, fingerprint)
        # Optional transport selection ("tcp" default, or "ucx" for RDMA).
        # Must be set before the first connect; idempotent process-global.
        transport = (extra_config.get("transport") or "tcp").lower()
        if transport != "tcp":
            try:
                autumn.set_transport(transport)
                log.info("autumn transport set to %s", transport)
            except Exception as e:  # noqa: BLE001
                log.warning("set_transport(%s) failed, falling back to tcp: %r", transport, e)
        # Hot path: the GIL-releasing Rust BatchClient. One PyO3 call extracts
        # all dest pointers under the GIL, then py.allow_threads while Rust
        # workers pipeline transfers + memcpy into the pinned pages. This both
        # lifts throughput (~600 MB/s vs ~480 on the asyncio path) and frees
        # the GIL during a batch so sglang's other threads run concurrently.
        #
        # per_worker_cap bounds in-flight depth per worker, keeping UCX under
        # its single-worker rendezvous cliff (≤16). client_workers is the
        # number of Rust worker threads; in-process scaling is currently
        # capped by the shared process-global ucp_context (workers serialize
        # on it), so default 1 — >1 is wired for when per-thread contexts land.
        default_cap = 16 if transport == "ucx" else 64
        self._max_inflight = int(extra_config.get("max_inflight", default_cap))
        n_workers = max(1, int(extra_config.get("client_workers", 1)))
        # `ttl_secs` (default 0 = no expiry): relative TTL applied to every L3
        # page write. Content-addressed keys never invalidate, so a TTL is the
        # only reclamation knob. sglang manages its own existence (its hash
        # table + L2), so a missing/expired L3 key is just a clean miss → no
        # marker-ordering concern (unlike the vLLM connector). A negative value
        # is a misconfiguration — fail fast rather than silently coercing to 0
        # (= never expire), which would mask the error.
        self._ttl_secs = int(extra_config.get("ttl_secs", 0) or 0)
        if self._ttl_secs < 0:
            raise ValueError(f"ttl_secs must be non-negative, got {self._ttl_secs}")
        # "ucx ⟹ zerocopy": the zero-copy data path (MSG_PUT_BULK write +
        # MSG_GET_BULK read for large pages) is now the DEFAULT whenever the
        # transport is UCX — no opt-in flag. BatchClient derives it from the
        # process transport set by set_transport() above. On TCP the regular
        # path is used. (The old extra_config["bulk"] opt-in was removed; KV-cache
        # pages are large so reads cross the bulk size threshold and writes are
        # always bulk on UCX — both win at this size; see BULK_MIN_BYTES.)
        # (D6-kvc): same authz wiring as
        # the vLLM connector — `auth_credential_file` in extra_config is the only
        # required key (Option 3's credential file names its own principal),
        # `auth_principal` overrides that name. The file read fails loudly at
        # startup. Threads to BOTH
        # clients (authz gates reads on protected prefixes too — a
        # credential-less probe client would silently turn every hit into a miss).
        auth_cred_file = extra_config.get("auth_credential_file")
        auth_principal = extra_config.get("auth_principal")
        # The client binds the `kvc` SCOPE and PREPENDS `kvc/` (`_keys.py`
        # emits the relative `{model_scope}/…`).
        auth: dict = {"scope": "kvc"}
        if auth_cred_file:
            file_principal, secret = _read_credential_pair(auth_cred_file)
            auth_principal = auth_principal or file_principal
            if not auth_principal:
                raise ValueError(
                    f"credential file {auth_cred_file!r} carries no principal "
                    f"name (expected a 'principal: <name>' line or "
                    f"'<name>\\n<hex>'); set `auth_principal` explicitly"
                )
            auth["principal"] = auth_principal
            auth["credential"] = secret
        elif auth_principal is not None:
            raise ValueError(
                "extra_config: auth_principal set without auth_credential_file"
            )
        self._batch = autumn.BatchClient(
            endpoint, n_workers, max(1, self._max_inflight), **auth
        )
        # Low-frequency v0 / batch_exists / clear paths use a regular async
        # Client on its own loop thread.
        self._loop0 = new_loop()
        self._client = run_on(self._loop0, lambda: autumn.Client.connect(endpoint, **auth))
        self._mem_pool_host = None
        self._stats = {
            "get_hit": 0,
            "get_miss": 0,
            "get_error": 0,
            "set_ok": 0,
            "set_error": 0,
        }
        log.info(
            "AutumnKVCacheStorage connected: endpoint=%s model_scope=%s sglang=%s",
            endpoint,
            self._model_scope,
            _SGLANG_AVAILABLE,
        )

    # ── lifecycle ──────────────────────────────────────────────────────────

    def register_mem_pool_host(self, mem_pool_host):
        """Cache the pinned-host KV pool reference.

        v1 batch_get/set use `mem_pool_host.get_data_page(idx, flat=True)` to
        resolve `host_indices` into a buffer view per docs:174.
        """
        self._mem_pool_host = mem_pool_host

    def _full_key(self, hash_str: str, pool_name: str = DEFAULT_POOL_NAME) -> bytes:
        return full_key(self._model_scope, hash_str, pool_name)

    def _page_view(self, idx: int):
        """Resolve a host_index to a buffer-protocol view of the pinned page.

        sglang's HostKVCache.get_data_page(idx, flat=True) returns a 1-D
        torch.Tensor view into the pinned host pool. Its dtype matches the
        model's KV cache dtype (commonly bfloat16 or float16).

        autumn's `put_from` / `get_into` use PyBuffer<u8>, so we reinterpret
        the tensor as uint8 before exposing it. `.view(torch.uint8)` is a
        zero-copy bit-reinterpret on the same storage; `.numpy()` then yields
        a numpy view of the same bytes (numpy has no native bfloat16 dtype,
        so calling `.numpy()` directly on a bf16 tensor raises TypeError).
        """
        if self._mem_pool_host is None:
            raise RuntimeError("register_mem_pool_host has not been called")
        page = self._mem_pool_host.get_data_page(int(idx), flat=True)
        # torch.Tensor branch: bf16/fp16/fp32 → uint8 byte view → numpy.
        if hasattr(page, "view") and hasattr(page, "dtype") and hasattr(page, "numpy"):
            try:
                import torch  # lazy; available wherever sglang runs
                return page.view(torch.uint8).numpy()
            except ImportError:
                # Fall through to direct .numpy() — works for already-uint8
                # tensors (numpy/torch agree on uint8).
                return page.numpy()
        # numpy / memoryview / bytes already.
        return page

    # ── v1 zero-copy hot path ──────────────────────────────────────────────

    def _batch_v1(self, keys, host_indices, transfer, verb, err_stat, stat_of):
        """Shared v1 transfer shape for batch_get_v1 / batch_set_v1: resolve
        page views, run `transfer(full_keys, views)`, account per-result via
        `stat_of(ok)`; any exception fails the whole batch under `err_stat`.
        """
        full_keys = [self._full_key(k) for k in keys]
        try:
            starts = _page_start_indices(keys, host_indices)
            views = [self._page_view(s) for s in starts]
            results = list(transfer(full_keys, views))
        except Exception as e:  # noqa: BLE001
            log.debug("batch %s error (n=%d): %r", verb, len(keys), e)
            self._stats[err_stat] += len(keys)
            return [False] * len(keys)
        for ok in results:
            self._stats[stat_of(ok)] += 1
        return results

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        return self._batch_v1(
            keys,
            host_indices,
            self._batch.get_into,
            "get_into",
            "get_error",
            lambda ok: "get_hit" if ok else "get_miss",
        )

    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        return self._batch_v1(
            keys,
            host_indices,
            lambda fk, views: self._batch.put_from(fk, views, self._ttl_secs),
            "put_from",
            "set_error",
            lambda ok: "set_ok" if ok else "set_error",
        )

    def batch_exists(self, keys, extra_info=None) -> int:
        """Return contiguous-prefix length (NOT per-key list).

        Matches sglang's HiCacheStorage `batch_exists` contract. Uses `head` (metadata-only) so
        admission probing doesn't transfer the value bytes.

        This runs on sglang's ADMISSION path (per request, before prefill), so
        it must be cheap. The answer is the present prefix FROM key[0], so the
        overwhelmingly common cold/no-hit case (key[0] absent ⇒ 0) is settled by
        a single `head` — probe that first and early-out, instead of fanning out
        one `head` RPC per key (page granularity ⇒ hundreds of keys per prompt;
        that fan-out measured ~4.8 ms/call and was the entire L3 TTFT overhead).
        Only when key[0] IS present do we fan out to count how far the prefix
        reaches (a hit — the case that saves a whole prefill, so worth the RPCs).
        """
        if not keys:
            return 0
        full_keys = [self._full_key(k) for k in keys]
        try:
            if not run(lambda: self._client.head(full_keys[0])):
                return 0  # cold prefix — one probe, no fan-out
            if len(full_keys) == 1:
                return 1
            founds = list(run(lambda: self._client.batch_head(full_keys)))
        except Exception as e:  # noqa: BLE001
            log.debug("batch_exists head error (n=%d): %r", len(keys), e)
            return 0
        # Contiguous-prefix length: stop at the first missing key.
        count = 0
        for ok in founds:
            if not ok:
                break
            count += 1
        log.debug("batch_exists n=%d hit=%d", len(keys), count)
        return count

    # ── v2 multi-pool ──────────────────────────────────────────────────────
    #
    # v2 exists for models whose prefix reuse needs MORE than the KV pool: a
    # sliding-window pool, a Mamba SSM state pool, a DSA indexer, DeepSeek-V4's
    # compressed regions, MTP draft KV. sglang calls these "sidecar" pools, and
    # a backend that only implements v1 makes those models unusable — the
    # controller calls batch_exists_v2 and gets NotImplementedError.
    #
    # The key schema already anticipated this: `full_key(model_scope, hash, pool)`
    # puts the pool in its own path segment, and the KV pool's segment is "kv",
    # which is what `PoolName.KV` stringifies to. So v2 keys for the KV pool are
    # byte-identical to v1's — this is additive, not a migration.

    def _pool_page_view(self, pool_name, idx: int):
        """Page view resolved against the pool that OWNS the index.

        v1 could assume one host pool. Under v2 each transfer names its pool and
        `host_indices` index into that pool's buffer, so resolving them against
        the KV pool would read the wrong bytes — silently, since the shapes
        match. `registered_pools` is filled by the ABC's default
        `register_mem_host_pool_v2`; the KV pool also arrives via the v1
        `register_mem_pool_host`, so fall back to it for KV.
        """
        pools = getattr(self, "registered_pools", None) or {}
        pool = pools.get(pool_name)
        if pool is None and str(pool_name) == DEFAULT_POOL_NAME:
            pool = self._mem_pool_host
        if pool is None:
            raise RuntimeError(
                f"no host pool registered for {pool_name!r}; "
                f"known={sorted(str(k) for k in pools)}"
            )
        page = pool.get_data_page(int(idx), flat=True)
        if hasattr(page, "view") and hasattr(page, "dtype") and hasattr(page, "numpy"):
            try:
                import torch

                return page.view(torch.uint8).numpy()
            except ImportError:
                return page.numpy()
        return page

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        """Longest usable prefix, folded across every pool.

        Fold semantics, matching the reference backends: start from the KV
        prefix, then narrow it by each sidecar pool's own boundary. A sidecar
        that is missing pages SHRINKS the answer — serving a prefix whose KV is
        present but whose window state is not would run attention over state
        that was never restored.

        Note which keys are probed: the KV `keys`, re-scoped into each pool's
        segment. `transfer.keys` is used only for its LENGTH, to size the
        trailing window. That is the contract, and it is easy to get wrong.
        """
        if not _SGLANG_V2_AVAILABLE:
            raise NotImplementedError("sglang v2 pool types unavailable")

        kv_pages = self.batch_exists(keys, extra_info)
        hit_count = {str(PoolName.KV): kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = str(transfer.name)
            try:
                present = self._pool_prefix_flags(name, keys[:kv_pages])
            except Exception as e:  # noqa: BLE001
                # A probe failure must not be read as "present": treat the pool
                # as a total miss and let the prefix shrink to zero.
                log.debug("batch_exists_v2 probe error pool=%s: %r", name, e)
                final_pages = 0
                break

            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not present[i]), kv_pages
                )
            else:  # TRAILING_PAGES — only the tail window has to be there
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    lo = max(0, prefix_len - trailing)
                    if all(present[i] for i in range(lo, prefix_len)):
                        boundary = prefix_len
                        break

            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        log.debug(
            "batch_exists_v2 kv=%d final=%d pools=%s", kv_pages, final_pages, hit_count
        )
        return PoolTransferResult(final_pages, hit_count)

    def _pool_prefix_flags(self, pool_name: str, keys) -> List[bool]:
        """Per-key presence within one pool's segment (NOT a prefix length).

        `batch_exists` answers a prefix question and early-outs on a cold key[0];
        the v2 fold needs the whole vector, because a TRAILING_PAGES pool can be
        absent at the head and present at the tail.
        """
        if not keys:
            return []
        full_keys = [self._full_key(k, pool_name) for k in keys]
        return list(run(lambda: self._client.batch_head(full_keys)))

    def _batch_v2(self, transfers, transfer_fn, verb: str):
        """Shared shape for batch_get_v2 / batch_set_v2.

        One pool's failure is reported for that pool only — the controller
        decides what a partial result means, and collapsing the whole call would
        hide which pool actually broke.
        """
        results = {}
        for t in transfers or []:
            name = str(t.name)
            t_keys = list(t.keys or [])
            if not t_keys:
                results[name] = []
                continue
            full_keys = [self._full_key(k, name) for k in t_keys]
            try:
                starts = _page_start_indices(t_keys, t.host_indices)
                views = [self._pool_page_view(t.name, i) for i in starts]
                results[name] = list(transfer_fn(full_keys, views))
            except Exception as e:  # noqa: BLE001
                log.debug("batch %s_v2 pool=%s error (n=%d): %r",
                          verb, name, len(t_keys), e)
                self._stats[f"{verb}_error"] += len(t_keys)
                results[name] = [False] * len(t_keys)
                continue
            for ok in results[name]:
                if verb == "get":
                    self._stats["get_hit" if ok else "get_miss"] += 1
                else:
                    self._stats["set_ok" if ok else "set_error"] += 1
        return results

    def batch_get_v2(self, transfers, extra_info=None):
        if not _SGLANG_V2_AVAILABLE:
            raise NotImplementedError("sglang v2 pool types unavailable")
        return self._batch_v2(transfers, self._batch.get_into, "get")

    def batch_set_v2(self, transfers, extra_info=None):
        if not _SGLANG_V2_AVAILABLE:
            raise NotImplementedError("sglang v2 pool types unavailable")
        return self._batch_v2(
            transfers,
            lambda fk, views: self._batch.put_from(fk, views, self._ttl_secs),
            "set",
        )

    # ── v0 thin wrappers (off hot path once interface_v1=1) ────────────────

    def get(self, key, target_location=None, target_sizes=None):
        full = self._full_key(key)
        try:
            return run(lambda: self._client.get(full))
        except Exception:  # noqa: BLE001
            return None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [self.get(k) for k in keys]

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        if value is None:
            return False
        full = self._full_key(key)
        try:
            buf = bytes(value) if not isinstance(value, (bytes, bytearray, memoryview)) else value
        except (TypeError, ValueError):
            try:
                buf = value.detach().cpu().numpy().tobytes()  # torch.Tensor path
            except Exception:  # noqa: BLE001
                return False
        try:
            payload = bytes(buf)
            run(lambda: self._client.put(full, payload, self._ttl_secs))
            return True
        except Exception:  # noqa: BLE001
            return False

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        if values is None or len(values) != len(keys):
            return False
        all_ok = True
        for k, v in zip(keys, values):
            all_ok = self.set(k, v) and all_ok
        return all_ok

    def exists(self, key) -> bool:
        full = self._full_key(key)
        try:
            return bool(run(lambda: self._client.head(full)))
        except Exception:  # noqa: BLE001
            return False

    # ── optional ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        # Pool-scoped: must NOT cross into the same model's vLLM (`vllm`) pool.
        prefix = pool_prefix(self._model_scope, DEFAULT_POOL_NAME)
        try:
            n = run(lambda: self._client.batch_delete(prefix))
            log.info("AutumnKVCacheStorage.clear deleted %d keys under %r", n, prefix)
        except Exception as e:  # noqa: BLE001
            log.warning("clear() failed: %r", e)

    def get_stats(self):
        return dict(self._stats)
