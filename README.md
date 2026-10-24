# autumn-kvcache

Daemon-less KV-cache **L3 storage backends** built on the autumn-rs partition
layer. Two adapters, one data plane (`autumn.BatchClient` + the sync/async
`_bridge`):

| Adapter | Engine | Contract | Module |
|---------|--------|----------|--------|
| `AutumnKVCacheStorage` | **sglang** | HiCache `HiCacheStorage` (L3) | `autumn_kvcache.sglang_backend` |
| `AutumnKVConnector` | **vLLM** | `KVConnectorBase_V1` (external KV transfer) | `autumn_kvcache.vllm_connector` |

## Which autumn-rs this tracks

`AUTUMN_REV` names the autumn-rs commit this tree was last verified against.
It matters more than a normal version pin: the adapters ride autumn's rkyv
client, and a client/cluster mismatch does **not** fail the handshake — it can
decode garbage. FreeToken's `docker/Dockerfile.freetoken-l3` carries the same
pin as `AUTUMN_IMAGE_PIN` for exactly this reason.

The PyO3 extension (`import autumn`) is built from autumn-rs and is NOT
installable as a portable wheel (native ext, bound to the builder's glibc and
exact CPython — no abi3), so every consumer compiles it from that commit.

There is **no daemon and no local DRAM cache** — the autumn partition layer is
the only persistence path. KV pages are content-addressed
(`kvc/{model_scope}/{pool}/{hash}/{layer}`), so there is no invalidation.

## Install

The native `autumn` PyO3 extension must be on the same interpreter:

```bash
# 1) build + install the native autumn client into your venv
cd python
maturin build --release --interpreter /path/to/venv/bin/python
#    for UCX: add `--features ucx --auditwheel skip` — the wheel must link the
#    system UCX; auditwheel's bundled copy has no mlx5 transport modules, and
#    the client then cannot connect over RDMA
/path/to/venv/bin/pip install --force-reinstall target/wheels/autumn-*.whl

# 2) install this package (pure-python) — still in python/ from step 1
/path/to/venv/bin/pip install -e autumn_kvcache
```

Both adapter modules import **without** the engine (sglang / vLLM) installed —
the engine import is defensive — so `python -c "from
autumn_kvcache.vllm_connector import AutumnKVConnector"` works for tooling and
the data-plane smoke test regardless.

## Faster reads: the partition server's memory tier

Start the partition servers with `--mem-tier-bytes <budget> --mem-tier-prefix kvc/`
and KV pages are held in their DRAM and acknowledged without a log write; reads
come from memory. Pages held there are lost on a PS restart (a cache miss, then
recompute). Measured with SGLang + Qwen3-VL-32B: per-request p50 1344 → 980 ms,
2 GiB fetch 999 → 699 ms. See autumn-rs `docs/ops.md` ("Memory tier for KV cache").

## Using autumn-kvcache as a vLLM L3 (cross-instance prefix cache)

`AutumnKVConnector` is a native vLLM V1 KV connector (Phase 3a — CPU-offload
path). It offloads each request's block-aligned prompt-prefix KV into autumn on
the producing instance, and loads it back on **any** instance that later sees
the same prefix — a cross-instance / cross-restart prefix cache backed by the
partition layer, with no extra service to run.

> **Pinned vLLM:** verified against **vLLM 0.23.0** (torch 2.11). The
> `KVConnectorBase_V1` signatures drift across vLLM releases (e.g. the external
> connector constructor gained a 3rd `kv_cache_config` argument); the adapter
> forwards it when present and falls back to the 2-arg form, but pin a known
> version in production.

Point both engines at a running autumn cluster's **manager** endpoint and give
each `vllm serve` the connector config:

```bash
KVCFG='{
  "kv_connector":"AutumnKVConnector",
  "kv_connector_module_path":"autumn_kvcache.vllm_connector",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{"endpoint":"127.0.0.1:9001","transport":"tcp"}
}'

# instance 1 (e.g. GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve /models/Qwen3-8B --served-model-name qwen3 \
    --port 8101 --enforce-eager --kv-transfer-config "$KVCFG"

# instance 2 (e.g. GPU 1) — a SEPARATE engine, empty local prefix cache
CUDA_VISIBLE_DEVICES=1 vllm serve /models/Qwen3-8B --served-model-name qwen3 \
    --port 8102 --enforce-eager --kv-transfer-config "$KVCFG"
```

`kv_connector_extra_config` keys: `endpoint` (autumn manager `host:port`,
required), `transport` (`tcp` default, or `ucx`), `max_inflight`, `ttl_secs`,
`model_id` (optional explicit model identity — see below). The client reads
with one Rust worker: ~5 GB/s per rank on TCP, ~42 GB/s on UCX (rendezvous
into the page); `client_workers` is no longer read.

### What the external cache does (and does not) speed up

The connector is an **L3** behind vLLM's own local prefix cache (GPU + host
RAM). vLLM matches the local cache first and asks the connector only for the
tokens *beyond* the local match, because remote storage is slower than a local
hit — you would never want to prefer autumn over GPU cache. Concretely:

- **Same engine, repeated prompt** → the local prefix cache serves it, so the
  external cache is (correctly) never *loaded* and its hit rate is ~0%. This is
  expected, not a failure.
- **A *different* engine / a *restarted* engine / after local eviction** → the
  local cache is cold, so the connector loads the prefix from autumn and skips
  prefill. This is the whole point (measured: a 1.3 k-token prefix loads in
  ~0.3 s vs. ~2 s of recompute — a ~3–4× TTFT win on the cold-local path).

So judge the connector by **cross-instance / post-restart** hit rate, not by
same-instance repeats. It also **never re-saves a prefix that is already durable
in autumn** (BUG-KVC-NO-HIT): a repeat whose KV the local cache already served
writes nothing, so it costs neither storage nor prefill time. (It issues one
lightweight presence probe per new request to make that de-dup decision.)

### Saves are asynchronous (almost nothing on the prefill critical path)

On the forward pass, `save_kv_layer` does **only** the cheap GPU-side gather
(`_extract_layer` → a *standalone* tensor, decoupled from the paged blocks — no
CPU sync). Everything expensive runs on a background thread: the D2H `.cpu()`
copy, the store-dedup probe, the durable batched `put_from`, and the marker. A
CUDA event lets the background thread wait for the gathers before the D2H, and
the `__present__` marker is published only *after* every layer ACKs (so a reader
that sees the marker always finds a complete, correct prefix — verified: an
external-load reproduces vLLM's local-cache output token-for-token).

Measured no-hit overhead (Qwen3-VL-32B, 1 k-token prompt, H200): **TTFT +≈6 ms,
TPOT ≈0** vs. no connector — i.e. a prefill that never gets reused costs almost
nothing. (Before moving the D2H off the path it was +≈148 ms.) When the prefix
*is* reused cross-instance, the connector skips the whole prefill instead.

At most `_MAX_INFLIGHT_SAVES` saves are in flight (bounding the held GPU/CPU
staging); over that a save is dropped rather than blocking prefill — it is a pure
cache, so a later request re-saves. The **store-dedup also lives in the
background**: a prefix already durable in autumn is neither re-copied nor
re-written, so a repeated prompt (served by the local GPU cache) costs no storage
and no extra latency.

### Model-scope isolation: the model's real identity is part of the key

The model-scope segment of every key is
`{model}_{fingerprint}_{tp_rank}_{tp_size}[...]`, where `fingerprint` is a
short hash of the model's **identity**: architecture shape
(layers / hidden / kv-heads / head-size / vocab / dtype / quantization / MLA)
plus the weights source (`load_format`, and for `--load-format autumn` the
autumn weights `path` from `model_loader_extra_config`), plus `model_id` if
you set one, plus the **layout versions** — the running **vLLM version**
(full `x.y.z`: the KV page layout is a vLLM-internal detail with no stability
contract, patch releases included) and the connector's own storage-format
version (`VLLM_KV_STORAGE_FORMAT`, bumped whenever the extract/inject byte
layout changes).

This exists because the model *path* is not an identity: with
a shared config dir, every model is served from the same fixed local config
dir, so two different models used to share one model scope and cross-read each
other's KV (observed live: Qwen2.5-7B and 32B under one model scope; a same-shape
pair would have been silently wrong). Set
`kv_connector_extra_config["model_id"]` when even the fingerprint can't
distinguish your deployments — e.g. two finetunes with identical architecture
loaded from the **same** path, or weights overwritten in place at one autumn
path (don't do that — store new weights at a new path).

> **Upgrade note:** any change to the fingerprint inputs — **including every
> vLLM version change, patch releases too** — moves the model scope, so the whole
> vLLM pool goes cold and rebuilds on first use (pure cache, content-addressed
> — no migration needed). This is deliberate: an upgrade already restarts the
> pods (GPU cache is lost anyway), so the re-warm is a one-time, predictable
> cost, whereas carrying KV across a layout-incompatible vLLM would be silent
> garbage. Keys written by older connectors stay behind under the old model scope;
> with `ttl_secs=0` they never expire, so reclaim them manually if you care
> about the space: `client.batch_delete(b"kvc/<old-model-scope>/vllm/")` (the
> old model scope is in the previous deployment's startup log).

**`ttl_secs`** (default `0` = no expiry) is the relative TTL after which an
offloaded prefix stops being *served*. Content-addressed keys never invalidate,
so a TTL is the only reclamation knob for a long-running cluster. Two caveats on
what exactly it bounds:

- The connector writes the per-prefix completion marker with `ttl_secs` and the
  layer pages with `ttl_secs + grace` (grace = 300 s), so the marker always
  expires *before* its layers. The scheduler admits a load on the marker; a
  marker that outlived its layers would let the worker silently load
  uninitialised KV (a correctness bug). So `ttl_secs` bounds **admissibility**
  (when the prefix can still be hit); the layer **data** lingers ~`grace`
  longer before its own lazy expiry.
- Reclamation is **lazy** (keys are dropped on read after they expire, plus
  background compaction), not a hard wall-clock free — size capacity planning
  accordingly.

### Verifying the cross-instance hit

Send a long prompt (longer than the cache block size) to **instance 1**, then
the **same** prompt to **instance 2**. Instance 2 has never seen it locally, so
any external-cache hit is served from autumn:

```bash
curl -s localhost:8101/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","prompt":"<long prompt>","max_tokens":1}' >/dev/null
curl -s localhost:8102/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","prompt":"<same long prompt>","max_tokens":1}' >/dev/null

# instance 2's connector metrics now show the cross-instance hit:
curl -s localhost:8102/metrics | grep -E \
  'vllm:external_prefix_cache_hits_total|external_kv_transfer'
# vllm:external_prefix_cache_hits_total{...}                 272.0
# vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} 272.0
```

(vLLM 0.23's OpenAI `usage` does **not** surface `cached_tokens` for the
connector path — use the `vllm:external_prefix_cache_*` /
`prompt_tokens_by_source{source="external_kv_transfer"}` metrics, which are the
authoritative external-hit signal.)

A scripted version of this check is `tests/` + the e2e in the repo.

## Data-plane smoke test (no engine required)

Exercises the autumn-facing core (`_AutumnKVStore`) against a real 1-node
cluster — key format, byte-exact page store/load round-trip, existence probe:

```bash
AUTUMN_KVCACHE_ENDPOINT=127.0.0.1:9001 AUTUMN_KVCACHE_TRANSPORT=tcp \
  python -m tests.test_vllm_dataplane    # run from this repo root
```

## Architecture notes

- **Stateless adapter / content-addressed keys** — no invalidation, partition
  is the only persistence path.
- **TTL reclamation (`ttl_secs`)** — both adapters accept `ttl_secs` in their
  extra-config (vLLM `kv_connector_extra_config`, sglang
  `hicache-storage-backend-extra-config`); default `0` = no expiry. The
  partition layer expires keys lazily on read. The sglang backend just stamps
  every page (its own L1/L2 + hash manage existence, so an expired L3 key is a
  clean miss); the vLLM connector additionally keeps the completion marker's TTL
  shorter than its layers' (see above) to preserve load correctness.
- **sglang host-pool layouts** — a page is read into, and written from, the
  host pool itself. Where the page (MLA `page_first*`) or each of its K/V
  halves (MHA `page_first*`) is a window into the pool, it moves with no copy;
  otherwise (`layer_first`, the default, and pools that assemble a page, such
  as Mamba) it is copied out to write, and a read lands in a scratch buffer
  that `set_from_flat_data_page` puts into the pool. Stored bytes follow the
  pool's own order, so every layout but `layer_first` carries its name in the
  key: instances with different layouts never read each other's pages. For the
  copy-free path, run sglang with `--hicache-mem-layout page_first_direct`; the
  copy path keeps a scratch buffer the size of one batch (128 pages) on each of
  sglang's backup and prefetch threads.
  These layouts need the v1 interface (`"interface_v1": 1`); the v0 wrappers
  use the untagged key.
- **No duplicate writes** — a page already stored (all of its values present)
  is not written again; one existence probe per batch. Its `ttl_secs`
  therefore counts from the first write: backing a hot page up again does not
  extend it.
- **Return fast** — never block past the engine's step budget; the load path is
  synchronous in Phase 3a (per-layer overlap is Phase 3b).
- **Model-scope isolation** — keys carry a model scope derived from model +
  model-identity fingerprint (`_identity.py`) + TP/PP, so different models /
  parallel layouts never alias. The sglang model-scope format is unchanged
  (`model_name` is a real identity there); its escape hatch is
  `extra_config["model_id"]`.

## Offline unit tests (no cluster, no engine, no native module)

```bash
uv run --with pytest python -m pytest tests/test_model_identity.py -q   # from this repo root
python -m pytest tests/test_page_layouts.py -q   # needs torch; every sglang host-pool layout
```
