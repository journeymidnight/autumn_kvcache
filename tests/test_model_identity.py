"""Offline unit tests for model-scope / model-identity isolation.

No cluster, no vllm, no torch, no `autumn` native module required:

    cd python/autumn_kvcache && uv run --with pytest python -m pytest tests/test_model_identity.py -q

Covers the live incident class: Qwen2.5-7B-AWQ and Qwen2.5-32B-AWQ served via
a shared config dir (both with model path `/model-cfg`, same tokenizer ⇒ same
token ids ⇒ same content hash) MUST land in different model scopes; and the
stability requirement: the same deployment MUST fingerprint identically across
processes or the external cache never hits.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

from autumn_kvcache._identity import (
    FINGERPRINT_HEX_LEN,
    fingerprint_from_sources,
    read_credential_file,
    read_credential_pair,
    model_scope_cfg_from_vllm,
    vllm_identity_sources,
)
from autumn_kvcache._keys import build_model_scope, full_key


# ── fakes mimicking vLLM 0.23 config shapes ─────────────────────────────────

class FakeModelConfig:
    """Duck-typed vLLM 0.23 ModelConfig: accessor methods + dtype/quant attrs."""

    def __init__(
        self,
        model="/model-cfg",
        layers=28,
        hidden=3584,
        kv_heads=4,
        head=128,
        vocab=152064,
        model_type="qwen2",
        dtype="torch.float16",
        quantization="awq",
        revision=None,
        use_mla=False,
    ):
        self.model = model
        self.dtype = dtype
        self.quantization = quantization
        self.revision = revision
        self.use_mla = use_mla
        self.hf_text_config = SimpleNamespace(model_type=model_type)
        self.hf_config = self.hf_text_config
        self._layers, self._hidden = layers, hidden
        self._kv_heads, self._head, self._vocab = kv_heads, head, vocab

    def get_total_num_hidden_layers(self):
        return self._layers

    def get_hidden_size(self):
        return self._hidden

    def get_total_num_kv_heads(self):
        return self._kv_heads

    def get_head_size(self):
        return self._head

    def get_vocab_size(self):
        return self._vocab


def fake_vllm_config(model_config=None, weights_path=None, load_format="autumn",
                     model_id=None, tp_size=1, pp_size=1, rank=0,
                     cache_dtype="auto", kv_skip_layers=()):
    """Mirrors a real VllmConfig, INCLUDING `cache_config` — vLLM always builds
    one (`CacheConfig.cache_dtype` defaults to "auto", verified on 0.23.0), so a
    fake without it would only ever exercise the degraded path."""
    extra = {"endpoint": "mgr:9001"}
    if model_id:
        extra["model_id"] = model_id
    mle = {"manager": "mgr:9001"}
    if weights_path:
        mle["path"] = weights_path
    return SimpleNamespace(
        model_config=model_config or FakeModelConfig(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size, rank=rank
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype, kv_cache_dtype_skip_layers=list(kv_skip_layers)
        ),
        load_config=SimpleNamespace(load_format=load_format, model_loader_extra_config=mle),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
    )


# The two live-incident models: SAME served path, same tokenizer, different arch.
QWEN_7B = dict(layers=28, hidden=3584, kv_heads=4, head=128)
QWEN_32B = dict(layers=64, hidden=5120, kv_heads=8, head=128)


def _model_scope(vllm_config):
    cfg = model_scope_cfg_from_vllm(vllm_config)
    return build_model_scope(cfg, cfg.model_fingerprint)


# ── the incident: different models must get different scopes ───────────────

def test_live_incident_7b_vs_32b_same_path_different_model_scope():
    t7 = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/qwen7b"))
    t32 = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_32B), weights_path="models/qwen32b"))
    assert t7 != t32
    # The readable model segment now comes from the weights-path BASENAME, so it
    # ALONE distinguishes the two models — no longer the constant, useless
    # `model-cfg` (which was the collision surface); the fingerprint still rides
    # along after it.
    assert t7.startswith("qwen7b_") and t32.startswith("qwen32b_")
    assert "model-cfg" not in t7 and "model-cfg" not in t32


def test_vllm_model_scope_uses_weights_path_basename_not_config_dir():
    """Key must be concise AND effective: the model segment is the autumn
    weights-path BASENAME (per-model unique), never the constant `/model-cfg`
    config-dir several models can share (the old collision surface)."""
    t = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/qwen7b"))
    assert t.startswith("qwen7b_")
    assert "model-cfg" not in t
    # a nested / trailing-slash path collapses to just the basename (concise)
    t2 = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="/a/b/qwen7b/"))
    assert t2.startswith("qwen7b_")
    # no loader path (default loader) -> falls back to the config-dir model name
    t3 = _model_scope(fake_vllm_config(FakeModelConfig(model="/data/Llama-3-8B"), load_format="auto"))
    assert t3.startswith("data_Llama-3-8B_")


def test_same_arch_different_weights_path_different_model_scope():
    """Two finetunes of the same architecture stored at different autumn paths
    (the shared-config-dir case) must NOT share a model scope."""
    a = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/qwen7b-base"))
    b = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/qwen7b-sft"))
    assert a != b


def test_each_arch_dimension_splits_model_scope():
    base = fake_vllm_config(FakeModelConfig(), weights_path="models/m")
    t0 = _model_scope(base)
    for delta in (
        {"layers": 64},
        {"hidden": 5120},
        {"kv_heads": 8},
        {"head": 64},
        {"dtype": "torch.bfloat16"},
        {"quantization": None},
        {"model_type": "llama"},
    ):
        t = _model_scope(fake_vllm_config(FakeModelConfig(**delta), weights_path="models/m"))
        assert t != t0, f"model scope did not split on {delta}"


def test_model_id_override_splits_model_scope():
    a = _model_scope(fake_vllm_config(FakeModelConfig(), weights_path="models/m", model_id="v1"))
    b = _model_scope(fake_vllm_config(FakeModelConfig(), weights_path="models/m", model_id="v2"))
    none = _model_scope(fake_vllm_config(FakeModelConfig(), weights_path="models/m"))
    assert len({a, b, none}) == 3


# ── stability: same deployment ⇒ same model scope, across processes ──────────

def test_same_config_stable_model_scope():
    mk = lambda: fake_vllm_config(FakeModelConfig(**QWEN_32B), weights_path="models/qwen32b")
    assert _model_scope(mk()) == _model_scope(mk())


def test_fingerprint_is_pinned_canonical_sha256():
    """Pin the exact canonicalisation (sorted `k=v` NUL-joined, sha256, first
    12 hex). If this test breaks, the model scope of every deployed model changes
    ⇒ full cold-cache rebuild — bump deliberately, never accidentally."""
    sources = {"layers": 28, "dtype": "torch.float16"}
    expect = hashlib.sha256(b"dtype=torch.float16\x00layers=28\x00").hexdigest()[:FINGERPRINT_HEX_LEN]
    assert fingerprint_from_sources(sources) == expect
    # insertion order must not matter
    assert fingerprint_from_sources({"dtype": "torch.float16", "layers": 28}) == expect


def test_fingerprint_empty_and_all_none():
    assert fingerprint_from_sources(None) is None
    assert fingerprint_from_sources({}) is None
    assert fingerprint_from_sources({"a": None, "b": None}) is None


# ── source extraction: vLLM 0.23 accessors + degraded fallbacks ─────────────

def test_identity_sources_prefer_model_config_accessors():
    src = vllm_identity_sources(fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/q"))
    assert src["layers"] == 28 and src["hidden"] == 3584
    assert src["kv_heads"] == 4 and src["head"] == 128
    assert src["dtype"] == "torch.float16" and src["quant"] == "awq"
    assert src["model_type"] == "qwen2"
    assert src["weights_path"] == "models/q"
    assert src["load_format"] == "autumn"
    assert "mla" not in src


def test_identity_sources_hf_config_fallback():
    """A config exposing only raw HF attributes (no accessor methods) still
    yields the architecture fingerprint."""
    mc = SimpleNamespace(
        model="/model-cfg",
        dtype="torch.bfloat16",
        quantization=None,
        hf_text_config=SimpleNamespace(
            num_hidden_layers=64, hidden_size=5120, num_key_value_heads=8,
            head_dim=128, vocab_size=152064, model_type="qwen2",
        ),
    )
    src = vllm_identity_sources(SimpleNamespace(model_config=mc))
    assert src["layers"] == 64 and src["hidden"] == 5120
    assert src["kv_heads"] == 8 and src["head"] == 128 and src["vocab"] == 152064


def test_identity_sources_load_format_enum_like():
    lc = SimpleNamespace(
        load_format=SimpleNamespace(value="AUTO"), model_loader_extra_config={}
    )
    src = vllm_identity_sources(SimpleNamespace(load_config=lc))
    assert src["load_format"] == "auto"


def test_identity_sources_mla_flag():
    src = vllm_identity_sources(fake_vllm_config(FakeModelConfig(use_mla=True)))
    assert src["mla"] == 1


def test_unfingerprintable_config_degrades_to_none():
    cfg = model_scope_cfg_from_vllm(SimpleNamespace())
    assert cfg.model_fingerprint is None
    # model scope falls back to the legacy format (loudly warned at connector init)
    assert build_model_scope(cfg, cfg.model_fingerprint) == "unknown_0_1"


# ── legacy behavior must not regress (sglang default / MLA / tp / pp) ────────

def _sg_cfg(**kw):
    base = dict(model_name="smoke-model", tp_rank=0, tp_size=1, pp_rank=0,
                pp_size=1, is_mla_model=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_sglang_default_model_scope_unchanged():
    # exact strings from before this change — existing sglang caches keep hitting
    assert build_model_scope(_sg_cfg()) == "smoke-model_0_1"
    assert build_model_scope(_sg_cfg(tp_rank=1, tp_size=4)) == "smoke-model_1_4"
    assert build_model_scope(_sg_cfg(is_mla_model=True)) == "smoke-model"
    assert (
        build_model_scope(_sg_cfg(pp_rank=1, pp_size=2)) == "smoke-model_0_1_pp1_2"
    )
    assert build_model_scope(_sg_cfg(model_name="/data/m/x")) == "data_m_x_0_1"
    assert build_model_scope(None) == "default"


def test_fingerprint_slots_between_model_and_parallel_suffix():
    assert build_model_scope(_sg_cfg(), "abc123") == "smoke-model_abc123_0_1"
    assert build_model_scope(_sg_cfg(is_mla_model=True), "abc123") == "smoke-model_abc123"
    assert (
        build_model_scope(_sg_cfg(pp_rank=1, pp_size=2), "abc123")
        == "smoke-model_abc123_0_1_pp1_2"
    )


def test_vllm_model_scope_keeps_tp_pp_semantics():
    t = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/q",
                                      tp_size=4, rank=2))
    assert t.endswith("_2_4")
    # MLA drops the tp suffix (rank-independent KV) but keeps the fingerprint
    tm = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_7B, use_mla=True),
                                       weights_path="models/q", tp_size=4, rank=2))
    assert not tm.endswith("_2_4")
    assert tm.startswith("q_") and tm != "q"  # weights-path basename + fingerprint


def test_full_key_layout_with_fingerprint():
    cfg = model_scope_cfg_from_vllm(fake_vllm_config(FakeModelConfig(**QWEN_32B),
                                                     weights_path="models/qwen32b"))
    model = build_model_scope(cfg, cfg.model_fingerprint)
    # (Prepend-only): full_key is RELATIVE to the
    # client's `kvc` binding scope — the builder emits {model}/{pool}/{hash} and
    # the ClusterClient prepends `kvc/` (scope locked by construction).
    key = full_key(model, "v1/deadbeef/model.layers.0", "vllm")
    assert key.startswith(b"qwen32b_")  # relative: model segment = weights-path basename
    assert key.endswith(b"/vllm/v1/deadbeef/model.layers.0")
    # No namespace prefix in the relative key — the binding owns it.
    assert not key.startswith(b"kvc/")


def test_mla_detection_wired_from_use_mla():
    """vLLM-side is_mla_model used to be hardcoded False; it now follows
    ModelConfig.use_mla."""
    assert model_scope_cfg_from_vllm(fake_vllm_config(FakeModelConfig(use_mla=True))).is_mla_model
    assert not model_scope_cfg_from_vllm(fake_vllm_config(FakeModelConfig())).is_mla_model


# ── layout versions: vLLM version + connector storage format split scopes ──
# The KV page byte layout is an internal detail of the vLLM build and of the
# connector's own extract/inject code — the same model on a different layout
# must not share a model scope (same silent-garbage class as two models
# sharing one scope).

from autumn_kvcache import _identity  # noqa: E402 — for monkeypatching
from autumn_kvcache._keys import VLLM_KV_STORAGE_FORMAT  # noqa: E402


def _mk7b():
    return fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/q")


def test_vllm_version_splits_model_scope(monkeypatch):
    monkeypatch.setattr(_identity, "vllm_runtime_version", lambda: "0.23.0")
    a = _model_scope(_mk7b())
    assert _model_scope(_mk7b()) == a  # same version + same config ⇒ stable
    monkeypatch.setattr(_identity, "vllm_runtime_version", lambda: "0.24.0")
    assert _model_scope(_mk7b()) != a


def test_vllm_patch_version_splits_model_scope(monkeypatch):
    """FULL-version granularity: patch releases can change internal KV layout
    (no stability contract), so 0.23.0 vs 0.23.1 must not share a model scope.
    Over-splitting = one predictable re-warm per upgrade; under-splitting =
    silent garbage."""
    monkeypatch.setattr(_identity, "vllm_runtime_version", lambda: "0.23.0")
    a = _model_scope(_mk7b())
    monkeypatch.setattr(_identity, "vllm_runtime_version", lambda: "0.23.1")
    assert _model_scope(_mk7b()) != a


def test_vllm_version_in_sources_and_undetectable_degrades(monkeypatch):
    monkeypatch.setattr(_identity, "vllm_runtime_version", lambda: "0.23.0")
    src = vllm_identity_sources(_mk7b())
    assert src["vllm"] == "0.23.0"
    # undetectable version ⇒ key absent (connector warns loudly at init), but
    # the model-identity fingerprint itself still derives
    monkeypatch.setattr(_identity, "vllm_runtime_version", lambda: None)
    src2 = vllm_identity_sources(_mk7b())
    assert "vllm" not in src2
    assert fingerprint_from_sources(src2) is not None


def test_version_only_config_still_unfingerprintable(monkeypatch):
    """Versions must not rescue an unfingerprintable config: they distinguish
    nothing BETWEEN models, and a version-only fingerprint would suppress the
    connector's loud model-scope collision warning."""
    monkeypatch.setattr(_identity, "vllm_runtime_version", lambda: "0.23.0")
    cfg = model_scope_cfg_from_vllm(SimpleNamespace())
    assert cfg.model_fingerprint is None
    assert cfg.identity_sources == {}


def test_kv_layout_version_splits_model_scope(monkeypatch):
    a = _model_scope(_mk7b())
    monkeypatch.setattr(_identity, "VLLM_KV_STORAGE_FORMAT", "v2")
    assert _model_scope(_mk7b()) != a


def test_kv_layout_version_pinned_and_in_sources():
    """Pin the connector storage-format version: it is baked into every
    vLLM-pool key path AND the model fingerprint. Bump it DELIBERATELY —
    together with an `_extract_layer`/`_inject_layer` layout change — never
    accidentally: any change cold-invalidates the whole vLLM pool."""
    assert VLLM_KV_STORAGE_FORMAT == "v1"
    src = vllm_identity_sources(_mk7b())
    assert src["kv_layout"] == "v1"


# ── KV-cache dtype: the "same model, different element type" axis ───────────
# `_inject_layer` reinterprets the stored raw bytes with the CURRENT runtime
# dtype (`from_numpy(staging).view(template.dtype)`), so --kv-cache-dtype must
# split the model scope. The fp8_e4m3 ↔ fp8_e5m2 pair is the dangerous one: same
# itemsize ⇒ the reshape succeeds and the KV is silently wrong.


def test_kv_cache_dtype_splits_model_scope():
    same = lambda dt: _model_scope(  # noqa: E731
        fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/q", cache_dtype=dt)
    )
    auto, e4m3, e5m2 = same("auto"), same("fp8_e4m3"), same("fp8_e5m2")
    # the silent-corruption pair: identical itemsize, so nothing would have
    # errored at load time had they shared a model scope
    assert e4m3 != e5m2
    assert len({auto, e4m3, e5m2}) == 3
    # and it is still stable for a fixed dtype
    assert same("fp8_e4m3") == e4m3


def test_kv_cache_dtype_is_independent_of_model_dtype():
    """`ModelConfig.dtype` (already fingerprinted) does NOT cover this: the same
    model dtype with two different cache dtypes must still split."""
    mk = lambda dt: vllm_identity_sources(  # noqa: E731
        fake_vllm_config(FakeModelConfig(dtype="torch.bfloat16"), weights_path="m", cache_dtype=dt)
    )
    a, b = mk("auto"), mk("fp8")
    assert a["dtype"] == b["dtype"] == "torch.bfloat16"
    assert a["kv_cache_dtype"] == "auto" and b["kv_cache_dtype"] == "fp8"
    assert fingerprint_from_sources(a) != fingerprint_from_sources(b)


def test_kv_skip_layers_splits_model_scope_and_is_order_independent():
    mk = lambda skip: vllm_identity_sources(  # noqa: E731
        fake_vllm_config(weights_path="m", cache_dtype="fp8", kv_skip_layers=skip)
    )
    none, a, b = mk(()), mk(("model.layers.0",)), mk(("model.layers.1",))
    assert "kv_skip_layers" not in none  # empty list ⇒ absent, not ""
    assert fingerprint_from_sources(a) != fingerprint_from_sources(b)
    assert fingerprint_from_sources(none) != fingerprint_from_sources(a)
    # iteration order of the config value must not move the fingerprint
    ab = mk(("model.layers.0", "model.layers.1"))
    ba = mk(("model.layers.1", "model.layers.0"))
    assert fingerprint_from_sources(ab) == fingerprint_from_sources(ba)


def test_missing_cache_config_degrades_not_raises():
    """A config shape without `cache_config` still fingerprints (one source
    fewer) — never an exception."""
    cfg = SimpleNamespace(model_config=FakeModelConfig(**QWEN_7B))
    src = vllm_identity_sources(cfg)
    assert "kv_cache_dtype" not in src
    assert fingerprint_from_sources(src) is not None


# ── TP/PP rank derivation from the global rank ──────────────────────────────
# `ParallelConfig.rank` is the GLOBAL rank (TP is the fastest-varying dimension
# inside a pipeline stage). pp_rank used to be initialised to None and never
# assigned, and `rank` was taken verbatim as the TP rank.


def test_tp_pp_ranks_derived_from_global_rank():
    scopes = set()
    for g in range(4):  # tp_size=2, pp_size=2 ⇒ global ranks 0..3
        cfg = model_scope_cfg_from_vllm(fake_vllm_config(tp_size=2, pp_size=2, rank=g))
        assert 0 <= cfg.tp_rank < 2, f"tp_rank {cfg.tp_rank} out of range for tp_size=2"
        assert 0 <= cfg.pp_rank < 2, f"pp_rank {cfg.pp_rank} out of range for pp_size=2"
        assert (cfg.tp_rank, cfg.pp_rank) == (g % 2, g // 2)
        scopes.add(build_model_scope(cfg, cfg.model_fingerprint))
    assert len(scopes) == 4  # every rank still gets its own model scope


def test_explicit_rank_fields_win_over_derivation():
    par = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=2, rank=3,
                          tensor_parallel_rank=0, pipeline_parallel_rank=1)
    cfg = model_scope_cfg_from_vllm(SimpleNamespace(parallel_config=par))
    assert (cfg.tp_rank, cfg.pp_rank) == (0, 1)


def test_pp1_rank_derivation_is_byte_identical():
    """Regression guard on the ONLY shape deployed today (pp_size == 1): the
    derivation must be the identity, or this fix silently cold-invalidates
    every live model scope a second time."""
    t = _model_scope(fake_vllm_config(FakeModelConfig(**QWEN_7B), weights_path="models/q",
                                      tp_size=4, rank=2))
    assert t.endswith("_2_4")  # exactly what the pre-fix code produced
    assert "_pp" not in t


# ── rank/size numeric normalisation (duck-typed configs from JSON/env) ──────


def test_build_model_scope_coerces_numeric_strings_byte_identically():
    assert build_model_scope(_sg_cfg(tp_rank="1", tp_size="4")) == "smoke-model_1_4"
    # the case that used to raise a bare TypeError from `pp_size > 1`
    assert (
        build_model_scope(_sg_cfg(pp_rank="1", pp_size="2")) == "smoke-model_0_1_pp1_2"
    )


def test_build_model_scope_rejects_non_numeric_rank():
    import pytest
    with pytest.raises(ValueError, match="tp_size"):
        build_model_scope(_sg_cfg(tp_size="four"))


# ── credential file must hex-DECODE to raw bytes ────────────
# The SDK/manager contract is RAW credential bytes; `autumn-op principal-create`
# prints lowercase hex. Passing the ASCII hex through would mint with a wrong
# credential → PermissionDenied once enforcement is on (coco P1 2026-07-17).

def _write(tmp_path, content: str):
    p = tmp_path / "cred"
    p.write_text(content)
    return str(p)


def test_credential_file_bare_hex_decodes_to_raw(tmp_path):
    raw = bytes(range(32))
    assert read_credential_file(_write(tmp_path, raw.hex())) == raw
    # trailing newline tolerated
    assert read_credential_file(_write(tmp_path, raw.hex() + "\n")) == raw


def test_credential_file_accepts_principal_create_stdout(tmp_path):
    raw = bytes(range(1, 33))
    stdout = f"principal 'hermes' created\ncredential: {raw.hex()}\n"
    assert read_credential_file(_write(tmp_path, stdout)) == raw


def test_credential_file_rejects_non_hex(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        read_credential_file(_write(tmp_path, "not-a-hex-string-zz"))


# ── the file now names its principal ────────────────
# Mirrors `autumn_client::parse_credential_text` — keep the two in lockstep.

def test_credential_pair_labeled_form(tmp_path):
    """What `autumn-op principal-create` / cluster.sh write."""
    raw = bytes(range(32))
    text = f"principal: fs\ncredential: {raw.hex()}\n"
    assert read_credential_pair(_write(tmp_path, text)) == ("fs", raw)


def test_credential_pair_two_bare_lines(tmp_path):
    """`<name>\\n<hex>` — accepted, and NO LONGER the ambiguous-multiline error
    it was pre-Option-3 (the first line is the principal, not a second secret)."""
    raw = bytes(range(32))
    assert read_credential_pair(_write(tmp_path, f"kvc\n{raw.hex()}\n")) == ("kvc", raw)


def test_credential_pair_bare_hex_is_anonymous(tmp_path):
    """A lone hex line still parses, with an EMPTY principal — callers that
    need a name (loader, kvcache) must reject it themselves."""
    raw = bytes(range(32))
    assert read_credential_pair(_write(tmp_path, raw.hex())) == ("", raw)


def test_credential_pair_rejects_three_bare_lines(tmp_path):
    import pytest
    raw = bytes(range(32)).hex()
    with pytest.raises(ValueError):
        read_credential_pair(_write(tmp_path, f"{raw}\n{raw}\n{raw}"))


# ── parser parity with Rust `parse_credential_text` (coco P3) ───────────────
# `bytes.fromhex` is laxer than `u8::from_str_radix`; these pin the gap shut so
# the two readers accept exactly the same set of files.

def test_credential_pair_rejects_empty_secret(tmp_path):
    """A truncated / empty-rendered Secret must fail HERE as a format error, not
    later as a generic PermissionDenied with a zero-length credential."""
    import pytest
    with pytest.raises(ValueError):
        read_credential_pair(_write(tmp_path, "principal: loader\ncredential:\n"))


def test_credential_pair_rejects_embedded_whitespace(tmp_path):
    """`bytes.fromhex` silently skips inner spaces; Rust rejects them."""
    import pytest
    spaced = " ".join(bytes(range(32)).hex()[i:i + 2] for i in range(0, 64, 2))
    with pytest.raises(ValueError):
        read_credential_pair(_write(tmp_path, f"credential: {spaced}\n"))


def test_credential_pair_rejects_odd_length(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        read_credential_pair(_write(tmp_path, "abc"))


def test_credential_pair_tolerates_non_ascii_comment(tmp_path):
    """Rust ASCII-checks only the HEX, so a non-ASCII line elsewhere is fine —
    an `encoding="ascii"` open() would have raised UnicodeDecodeError."""
    raw = bytes(range(32))
    text = f"# 凭据文件\nprincipal: fs\ncredential: {raw.hex()}\n"
    assert read_credential_pair(_write(tmp_path, text)) == ("fs", raw)
