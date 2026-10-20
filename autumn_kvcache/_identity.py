"""Model-identity fingerprint for model-scope isolation.

Why this exists — the live incident: deployments that pin the
served path to a constant local dir (config/tokenizer in e.g. `/model-cfg`,
weights streamed from autumn), so vLLM's `model_config.model` is the SAME
string for every model served that way. `build_model_scope` inherited
sglang's assumption that `model_name` identifies a model → Qwen2.5-7B-AWQ and
Qwen2.5-32B-AWQ (same tokenizer ⇒ same token ids ⇒ same content hash) landed
in ONE model scope `model-cfg_0_1` and cross-read each other's KV. 28-vs-64-layer
mismatch made it visible; two same-shape models would have been a fully
SILENT correctness bug.

Fix: derive a deterministic fingerprint from what actually identifies the
KV bytes — architecture shape (layers / hidden / kv-heads / head-size /
vocab / dtype / quant / MLA) plus the weights source (`load_format` and, for
the autumn loader, the autumn weights `path`) plus an optional operator
`model_id`, plus the KV-CACHE dtype (`cache_config.cache_dtype`, which is
independent of the model dtype and decides the element type of the very
bytes we store), plus the LAYOUT versions (the running vLLM version and the
connector's own `VLLM_KV_STORAGE_FORMAT`): the page byte layout is an
internal detail of the vLLM build + this connector, so the same model on a
different vLLM must not read the old bytes — and fold it into the model
scope. Everything here is pure at import time (no vllm/torch/autumn module
imports; the vLLM version is a lazy, guarded lookup) so it is unit-testable
offline and, more importantly, deterministic across processes: the same
deployment MUST map to the same model scope or the cache never hits.

Residual collisions this does NOT close (documented, not hidden):
- same architecture + same weights identity source — e.g. two finetunes
  loaded from the SAME local dir path on different machines with the default
  loader (no `weights_path`), or weights overwritten IN PLACE at the same
  autumn path. Use `model_id` in `kv_connector_extra_config` /
  `extra_config` to disambiguate, or store new weights at a new path.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

from ._keys import VLLM_KV_STORAGE_FORMAT

# Length of the hex fingerprint folded into the model scope. 12 hex chars
# = 48 bits — collision-free for any realistic number of co-hosted models,
# short enough that keys stay readable in dumps.
FINGERPRINT_HEX_LEN = 12


def fingerprint_from_sources(sources: Optional[Dict[str, Any]]) -> Optional[str]:
    """Deterministic short hash over identity key/values; None if nothing usable.

    Canonicalisation: sorted keys, `k=v` pairs NUL-separated, str()-rendered
    values. str() of ints/strings/torch.dtype ("torch.bfloat16") is stable
    across processes and Python versions — never rely on repr() of exotic
    objects here; callers must pass primitives.
    """
    if not sources:
        return None
    h = hashlib.sha256()
    wrote = False
    for k in sorted(sources):
        v = sources[k]
        if v is None:
            continue
        h.update(str(k).encode("utf-8"))
        h.update(b"=")
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
        wrote = True
    if not wrote:
        return None
    return h.hexdigest()[:FINGERPRINT_HEX_LEN]


def vllm_runtime_version() -> Optional[str]:
    """Best-effort version of the vLLM actually running, None if undetectable.

    Lazy + guarded so this module stays importable (and offline-testable)
    without vLLM installed. `vllm.__version__` (re-exported from
    `vllm.version`) is the stable public spelling — verified present in the
    pinned 0.23.0; package metadata is the fallback for exotic installs.
    """
    try:
        import vllm  # noqa: PLC0415 — deliberate lazy import, vLLM path only

        v = getattr(vllm, "__version__", None)
        if v:
            return str(v)
    except Exception:  # noqa: BLE001 — broken/absent vllm must degrade
        pass
    try:
        from importlib import metadata

        return str(metadata.version("vllm"))
    except Exception:  # noqa: BLE001
        return None


def vllm_identity_sources(vllm_config: Any) -> Dict[str, Any]:
    """Extract model-identity sources from a vLLM `VllmConfig`, defensively.

    Field names verified against vLLM v0.23.0 (the pinned version, README):
    `ModelConfig.get_total_num_hidden_layers() / get_hidden_size() /
    get_total_num_kv_heads() / get_head_size() / get_vocab_size()`,
    `ModelConfig.dtype` (resolved torch.dtype post-init), `.quantization`,
    `.revision`, `.use_mla` (property), `.hf_text_config` / `.hf_config`;
    `LoadConfig.load_format` (str in 0.23; older enums handled via `.value`),
    `LoadConfig.model_loader_extra_config` (dict for the autumn loader —
    its `path` is the autumn weights dir, the strongest identity we have).
    Every access is getattr/try-guarded: a missing field degrades to a
    smaller fingerprint, never an exception.

    On top of the config-derived sources, folds in the two layout versions
    (`vllm` = running vLLM version, `kv_layout` = the connector's
    `VLLM_KV_STORAGE_FORMAT`) — see the inline comment for why and for the
    full-version granularity decision.
    """
    src: Dict[str, Any] = {}
    mc = getattr(vllm_config, "model_config", None)
    if mc is not None:
        hf = getattr(mc, "hf_text_config", None)
        if hf is None:
            hf = getattr(mc, "hf_config", None)
        # Architecture shape: prefer ModelConfig accessors (they normalise
        # exotic configs), fall back to raw HF-config attributes.
        for key, meth, attrs in (
            ("layers", "get_total_num_hidden_layers", ("num_hidden_layers",)),
            ("hidden", "get_hidden_size", ("hidden_size",)),
            ("kv_heads", "get_total_num_kv_heads", ("num_key_value_heads", "num_attention_heads")),
            ("head", "get_head_size", ("head_dim",)),
            ("vocab", "get_vocab_size", ("vocab_size",)),
        ):
            val: Optional[int] = None
            m = getattr(mc, meth, None)
            if callable(m):
                try:
                    val = int(m())
                except Exception:  # noqa: BLE001 — exotic models may raise
                    val = None
            if val is None and hf is not None:
                for a in attrs:
                    v = getattr(hf, a, None)
                    if v is not None:
                        try:
                            val = int(v)
                            break
                        except (TypeError, ValueError):
                            continue
            if val is not None:
                src[key] = val
        mt = getattr(hf, "model_type", None) if hf is not None else None
        if mt:
            src["model_type"] = str(mt)
        dt = getattr(mc, "dtype", None)
        if dt is not None:
            src["dtype"] = str(dt)  # "torch.bfloat16" — stable
        q = getattr(mc, "quantization", None)
        if q:
            src["quant"] = str(q)
        rev = getattr(mc, "revision", None)
        if rev:
            src["revision"] = str(rev)
        try:
            # MLA changes the saved KV layout for the SAME shape (and can be
            # toggled by VLLM_MLA_DISABLE), so it must split the model scope.
            if bool(getattr(mc, "use_mla", False)):
                src["mla"] = 1
        except Exception:  # noqa: BLE001 — property may touch vllm envs
            pass
    # ── KV-cache dtype (the "same model, different element type" axis) ──────
    # The bytes this connector stores ARE the KV tensors, and `_inject_layer`
    # reinterprets them with the CURRENT runtime dtype
    # (`from_numpy(staging).view(template.dtype)`), so a deployment that flips
    # --kv-cache-dtype without splitting the model scope reads old bytes under a new
    # type. The dangerous pair is SAME-itemsize (fp8_e4m3 ↔ fp8_e5m2, both one
    # byte): the reshape succeeds, nothing errors, the KV is silently wrong —
    # exactly the failure class this module exists to close, on the dtype axis.
    #
    # `cache_dtype` is INDEPENDENT of `ModelConfig.dtype` above ("auto" = follow
    # it, anything else overrides it), so the model dtype already in `src` does
    # NOT cover this. Recording the raw string is precise AND stable: the value
    # space is a fixed set of spellings ("auto"/"fp8"/"fp8_e4m3"/…), never a
    # resolved torch dtype, so "auto" cannot drift against an explicit spelling
    # of the same effective type.
    #
    # Field names verified against vLLM 0.23.0: `CacheConfig.cache_dtype`
    # (default "auto") and `.kv_cache_dtype_skip_layers` (default []); there is
    # NO `kv_cache_dtype` attribute despite the CLI flag being spelled that way.
    cc = getattr(vllm_config, "cache_config", None)
    if cc is not None:
        cdt = getattr(cc, "cache_dtype", None)
        if cdt is not None:
            src["kv_cache_dtype"] = str(cdt)
        # Per-layer exceptions to that dtype: two deployments both on fp8 but
        # skipping different layers store different bytes for those layers.
        skip = getattr(cc, "kv_cache_dtype_skip_layers", None)
        if skip:
            try:
                # sorted: a set/list from config must not make the fingerprint
                # depend on iteration order (see fingerprint_from_sources).
                src["kv_skip_layers"] = ",".join(sorted(str(x) for x in skip))
            except TypeError:  # not iterable — record it verbatim rather than drop
                src["kv_skip_layers"] = str(skip)
    # Deliberately NOT folded in: `block_size` — the saved bytes are gathered per
    # TOKEN via `slot_mapping`, so their layout is block-size independent, and the
    # content hash already varies with the block-aligned token count; and the
    # `mamba_*_cache_dtype` knobs — a model class this connector's
    # `_extract_layer`/`_inject_layer` pair does not handle at all.
    lc = getattr(vllm_config, "load_config", None)
    if lc is not None:
        fmt = getattr(lc, "load_format", None)
        if fmt is not None:
            fmt = getattr(fmt, "value", fmt)  # older vLLM used a LoadFormat enum
            src["load_format"] = str(fmt).lower()
        mle = getattr(lc, "model_loader_extra_config", None)
        if isinstance(mle, dict):
            # A weights-streaming loader may name its source dir here (`path`).
            # THE strongest identity on the deployment path that broke
            # model_name — two models must live at two paths. (The `manager`
            # endpoint is deliberately excluded: the same weights reached via
            # a different manager address must not split the model scope.)
            wpath = mle.get("path")
            if wpath:
                src["weights_path"] = str(wpath).strip("/")
    # Operator escape hatch: kv_connector_extra_config["model_id"] pins the
    # identity explicitly (e.g. two finetunes served from the same local dir).
    kt = getattr(vllm_config, "kv_transfer_config", None)
    extra = getattr(kt, "kv_connector_extra_config", None)
    if isinstance(extra, dict):
        mid = extra.get("model_id")
        if mid:
            src["model_id"] = str(mid)
    # ── layout versions (the "same model, incompatible bytes" axis) ─────────
    # The KV page byte layout is NOT part of the model: it is an internal
    # implementation detail of (a) the running vLLM build and (b) this
    # connector's extract/inject code. Either changing without splitting the
    # model scope is the same failure class as two models sharing one scope:
    # shapes may happen to line up, nothing errors, output is silently
    # garbage. So both versions are identity sources.
    #
    # FULL vLLM version (not major.minor): vLLM gives no stability contract
    # for its internal KV-cache tensor layout, and patch releases do touch
    # block shape / dtype packing / attention-backend selection. The asymmetry
    # decides it — over-splitting costs ONE predictable cache re-warm per
    # upgrade (pods restart and lose GPU cache anyway); under-splitting is
    # silent garbage nobody notices. NOTE the operational consequence: ANY
    # vLLM version change cold-invalidates the whole vLLM pool (README
    # upgrade note).
    #
    # Gated on `src` being non-empty: versions distinguish nothing BETWEEN
    # two models on the same stack, so a version-only fingerprint would only
    # defeat the "unfingerprintable config ⇒ None ⇒ loud connector warning"
    # contract.
    if src:
        ver = vllm_runtime_version()
        if ver:
            src["vllm"] = ver
        # else: the connector warns loudly at init (vllm_connector.__init__) —
        # this module is log-free/pure by design.
        src["kv_layout"] = VLLM_KV_STORAGE_FORMAT
    return src


def model_scope_cfg_from_vllm(vllm_config: Any):
    """Adapt a VllmConfig into the duck-typed object `build_model_scope` wants,
    now carrying `model_fingerprint` + `identity_sources` and a
    REAL `is_mla_model` (was hardcoded False — vLLM-side MLA was never detected).
    Pure + defensive so it is unit-testable without vllm installed.
    """
    import types

    model = None
    tp_rank = tp_size = pp_rank = pp_size = None
    try:
        model = getattr(getattr(vllm_config, "model_config", None), "model", None)
        # When weights come from a streaming loader, `model_config.model` can be a CONSTANT local
        # config-dir (e.g. /model-cfg) shared by every model served that way — it
        # identifies nothing and made the model scope `model-cfg_<fp>_...` (the
        # cross-model collision surface + noise). Prefer the autumn weights
        # path BASENAME (e.g. `qwen7b`): per-model unique, so the key is concise
        # AND model-distinguishing (`qwen7b_<fp>_...`) even if the fingerprint
        # hash ever degrades. The FULL path still feeds the fingerprint (see
        # vllm_identity_sources) for same-basename / different-parent safety.
        lc = getattr(vllm_config, "load_config", None)
        mle = getattr(lc, "model_loader_extra_config", None) if lc is not None else None
        if isinstance(mle, dict):
            wpath = mle.get("path")
            if wpath:
                model = str(wpath).rstrip("/").split("/")[-1] or model
    except Exception:  # noqa: BLE001
        pass
    try:
        par = getattr(vllm_config, "parallel_config", None)
        tp_size = getattr(par, "tensor_parallel_size", 1)
        pp_size = getattr(par, "pipeline_parallel_size", 1)
        # `ParallelConfig.rank` is the GLOBAL rank; vLLM lays ranks out with TP
        # as the fastest-varying dimension inside a pipeline stage. Both halves
        # must be derived from it: taking `rank` verbatim as the TP rank used to
        # yield an out-of-range tp_rank (rank 2 with tp_size 2) AND stamp every
        # stage as `pp0`, because pp_rank was initialised to None and never
        # assigned. Prefer explicit fields when a vLLM build exposes them.
        # With pp_size == 1 — every deployment today — the derivation is the
        # identity (`g % t == g`), so existing model scopes are byte-identical.
        tp_rank = getattr(par, "tensor_parallel_rank", None)
        pp_rank = getattr(par, "pipeline_parallel_rank", None)
        if tp_rank is None or pp_rank is None:
            g = int(getattr(par, "rank", 0) or 0)
            t = max(1, int(tp_size or 1))
            p = max(1, int(pp_size or 1))
            if tp_rank is None:
                tp_rank = g % t
            if pp_rank is None:
                pp_rank = (g // t) % p
    except Exception:  # noqa: BLE001
        pass
    is_mla = False
    try:
        is_mla = bool(getattr(getattr(vllm_config, "model_config", None), "use_mla", False))
    except Exception:  # noqa: BLE001
        pass
    sources = vllm_identity_sources(vllm_config)
    return types.SimpleNamespace(
        model_name=model,
        tp_rank=tp_rank or 0,
        tp_size=tp_size or 1,
        pp_rank=pp_rank or 0,
        pp_size=pp_size or 1,
        is_mla_model=is_mla,
        identity_sources=sources,
        model_fingerprint=fingerprint_from_sources(sources),
    )


def read_credential_pair(path: str) -> tuple[str, bytes]:
    """Read a principal credential file -> `(principal, RAW secret bytes)`.

    The credential file now carries the OWNER's name
    alongside the secret, because Option 3 made the principal a required half of
    the SDK's authz identity (`principal=` + `credential=` are both-or-neither).

    Mirrors `autumn_client::parse_credential_text` (crates/client/src/lib.rs) —
    keep the two in lockstep, this is one on-disk format with two readers:

      * a `credential: <hex>` line (+ optional `principal: <name>`) — what
        `autumn-op principal-create` / cluster.sh write;
      * two bare lines `<principal>\\n<hex>`;
      * a single bare hex line — anonymous, principal is `""` (the caller must
        reject it if it needs a name).

    The SDK/manager contract is RAW credential bytes (`mint-token` hex-decodes
    before sending; the manager stores the SHA-256 of the raw bytes), so passing
    the ASCII hex through would mint with a WRONG credential and every
    protected-prefix op would die with PermissionDenied once enforcement is on
    (coco P1 2026-07-17). Fail loudly here, at startup, instead.
    """
    # utf-8, NOT ascii: Rust's `read_to_string` accepts any valid UTF-8 and only
    # ASCII-checks the hex itself, so an ascii open() here would reject (with a
    # UnicodeDecodeError, not even a ValueError) a file Rust reads fine — e.g.
    # one carrying a non-ASCII comment line above the credential.
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    def _labeled(key: str) -> str | None:
        for ln in lines:
            if ln.lower().startswith(key):
                return ln.split(":", 1)[1].strip()
        return None

    hexs = _labeled("credential:")
    if hexs is not None:
        name = _labeled("principal:") or ""
    elif len(lines) == 2:
        name, hexs = lines[0], lines[1]
    elif len(lines) == 1:
        name, hexs = "", lines[0]
    else:
        raise ValueError(
            f"credential file {path!r}: expected '<name>\\n<hex>', a "
            f"'principal:'/'credential:' pair, or a single hex line, got "
            f"{len(lines)} non-empty lines"
        )
    # Mirror Rust's guard EXACTLY (crates/client/src/lib.rs) — `bytes.fromhex`
    # is laxer than `u8::from_str_radix`: it accepts an EMPTY string (a truncated
    # / empty-rendered Secret would sail through as a zero-length credential and
    # resurface later as a generic PermissionDenied) and it SKIPS embedded ASCII
    # whitespace (`"ab cd"`), which Rust rejects. Check before decoding so both
    # readers accept exactly the same files.
    if not hexs or not hexs.isascii() or len(hexs) % 2 != 0 or any(c.isspace() for c in hexs):
        raise ValueError(
            f"credential file {path!r}: not valid hex (empty, odd length, "
            f"non-ASCII, or contains whitespace)"
        )
    try:
        return name, bytes.fromhex(hexs)
    except ValueError as e:
        raise ValueError(f"credential file {path!r}: not valid hex: {e}") from None


def read_credential_file(path: str) -> bytes:
    """`read_credential_pair` keeping only the secret (back-compat shim)."""
    return read_credential_pair(path)[1]
