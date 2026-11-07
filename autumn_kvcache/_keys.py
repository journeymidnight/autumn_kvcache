"""Shared key-namespace + model-scope helpers for autumn-kvcache adapters.

Both the sglang HiCache L3 backend (`sglang_backend.py`) and the vLLM
`KVConnectorV1` adapter (`vllm_connector.py`) map an inference engine's KV
pages into the SAME autumn keyspace:

    kvc/{h}/{model_scope}/{pool}/{content_hash}

- `kvc/`           — reserved namespace, distinguishes kvcache keys from
                     autumn-fuse / autumn-client keys on the shared cluster.
- `h`              — the content hash's first hex char. The kvc namespace is
                     cut at `kvc/<hex>` when it is created (`autumn-op
                     namespace-create --name kvc --parts 16`), so putting it
                     ahead of the model spreads every model's pages over all
                     partitions with cuts made before any model runs.
- `model_scope`    — `(model, model_fingerprint, tp_rank, tp_size, pp_rank,
                     pp_size)` scoping so two models / parallel ranks never
                     collide. The fingerprint (`_identity.py`) exists because
                     the model *path* alone is NOT an identity: a deployment
                     may serve every model from the same fixed local config dir.
- `pool`           — `kv` for sglang, `vllm` for the vLLM connector (separate
                     keyspaces; also the v2 multi-pool reservation slot).
- `content_hash`   — the engine's own content-addressed hash (sglang chain hash
                     / vLLM block hash); adapters never re-hash.
"""

from __future__ import annotations

# Reserved namespace prefix shared by all autumn-kvcache adapters.
KEY_NAMESPACE = "kvc"

# The vLLM connector's OWN storage-format / KV-layout version. It is baked
# into every vLLM-pool key (`kvc/{h}/{model_scope}/vllm/{THIS}/{hash}/{layer}`, see
# `_AutumnKVStore._key`) AND folded into the model fingerprint
# (`_identity.vllm_identity_sources`), so bumping it makes every old entry
# unreachable twice over — old bytes can NEVER be loaded by an incompatible
# newer connector under the same content hash; they just stop matching and
# lazily expire. kvcache is a pure cache, so invalidation-by-versioning is the
# correct (zero-migration) upgrade story.
#
# INVARIANT — BUMP THIS whenever the connector's saved KV byte layout or key
# scheme changes. Concretely: any change to `vllm_connector._extract_layer` /
# `_inject_layer` (the serialize/deserialize pair — e.g. the 2026-06-22
# K/V-dim-order fix), to the save's byte copy (`_save_chunk`: the tensor's
# contiguous bytes), or to how keys are composed. Changing
# that layout WITHOUT bumping = same failure class as two models sharing one
# scope: shapes may still line up, nothing errors, output is silently garbage.
# `test_model_identity.py` pins the current value so a bump is always a
# deliberate, reviewed act (it cold-invalidates the whole vLLM pool).
#
# Lives here (not in `vllm_connector.py`, which imports the `autumn` native
# module at top level) so the pure offline unit tests can import it.
VLLM_KV_STORAGE_FORMAT = "v2"

# The hash buckets: sha256 hexdigest chars, the first char of every content hash.
HASH_BUCKETS = "0123456789abcdef"


def _rank_int(cfg, attr: str, default: int) -> int:
    """Read a rank/size attribute as an int, failing loudly on garbage.

    The values are f-string'd straight into the model scope, i.e. into every
    KV key. A duck-typed config sourced from JSON/env can hand us `"2"`, which
    used to blow up at the `pp_size > 1` comparison with a bare `TypeError`
    (`'>' not supported between 'str' and 'int'`) — from inside key
    construction, at engine start, with no hint of which field was at fault.
    Coercion keeps those configs working AND is byte-identical for every config
    that works today (`int(0) == 0`, `int("0") == 0` render the same); only
    genuinely non-numeric values now fail, and they fail with a message naming
    the field instead of producing a bogus rank in the keyspace.
    """
    v = getattr(cfg, attr, default)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(
            f"build_model_scope: {attr}={v!r} is not an integer — it would be "
            f"stamped into every KV key for this model"
        ) from None


def build_model_scope(cfg, model_fingerprint=None) -> str:
    """Build the per-model key scope from a model/parallel config.

    Mirrors sglang's `HiCacheFile._get_suffixed_key`. Accepts any object exposing the
    attributes below (an sglang `HiCacheStorageConfig`, a vLLM-derived shim, or
    a plain namespace), so both adapters share one implementation.

    Format: `{model}` then optionally `_{model_fingerprint}` then optionally
    `_{tp_rank}_{tp_size}` (skipped for MLA models, whose KV is
    rank-independent) then `_pp{pp_rank}_{pp_size}` (only if pp_size > 1).

    `model_fingerprint` (see `_identity.py`) carries the
    model's REAL identity (architecture shape + weights source). It is
    REQUIRED on the vLLM connector path, where `model_name` is just the local
    config-dir path, which is constant across models when one config dir is shared
    — without it two different models silently share one scope and cross-read
    KV (live incident: Qwen2.5-7B/32B both under `model-cfg_0_1`). When None
    (sglang default, or an unfingerprintable config) the format is exactly the
    pre-fingerprint one, so existing sglang model scopes are unchanged.
    """
    if cfg is None:
        return "default"
    model = getattr(cfg, "model_name", None) or "unknown"
    # `--model-path` is commonly a filesystem path like
    # `/data/models/Qwen3-VL-32B`; strip/normalise slashes so the key reads as
    # `kvc/Qwen3-VL-32B/...` rather than `kvc//data/models/...`.
    model = str(model).strip("/").replace("/", "_")
    is_mla = bool(getattr(cfg, "is_mla_model", False))
    tp_rank = _rank_int(cfg, "tp_rank", 0)
    tp_size = _rank_int(cfg, "tp_size", 1)
    pp_rank = _rank_int(cfg, "pp_rank", 0)
    pp_size = _rank_int(cfg, "pp_size", 1)
    parts = [str(model)]
    if model_fingerprint:
        parts.append(str(model_fingerprint))
    if not is_mla:
        parts.append(f"{tp_rank}_{tp_size}")
    if pp_size > 1:
        parts.append(f"pp{pp_rank}_{pp_size}")
    return "_".join(parts)


def full_key(model: str, content_hash: str, pool: str, tail: str = "") -> bytes:
    """Compose the stored partition key (utf-8 bytes), RELATIVE to the client's
    ``kvc`` binding scope: ``{h}/{model}/{pool}/{content_hash}{tail}``, ``h``
    the hash's first char.

    The ``BatchClient`` is bound to ``scope="kvc"`` and PREPENDS ``kvc/``
    itself, so the builder emits only the relative part. ``model`` is the
    per-model instance scope (what ``build_model_scope`` returns).
    """
    return f"{content_hash[0]}/{model}/{pool}/{content_hash}{tail}".encode("utf-8")


def pool_prefixes(model: str, pool: str) -> list:
    """Prefixes covering one model's keys within a single pool, one per hash
    bucket, RELATIVE to ``kvc`` (see `full_key`).

    `AutumnKVCacheStorage.clear()` (sglang) must use these so a debug clear of
    the `kv` pool does not also wipe the same model's `vllm` pool.
    """
    return [f"{h}/{model}/{pool}/".encode("utf-8") for h in HASH_BUCKETS]
