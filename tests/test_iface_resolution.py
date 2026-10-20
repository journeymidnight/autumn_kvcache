"""Which module supplies the L3 interface this adapter binds to.

`AutumnKVCacheStorage` implements sglang's HiCacheStorage interface, but sglang
is not the only engine that carries it. FreeToken vendors the same module rather
than depending on the runtime — its only sglang requirement is `sglang-kernel`,
which imports as `sgl_kernel` — so inside a FreeToken image `sglang.srt.…` does
not exist while `freetoken.kvcache.hicache.storage` holds the identical symbols.

The failure this guards against is silent, which is why it gets a test rather
than a comment. Resolving only against sglang leaves the base class as `object`
and `_SGLANG_V2_AVAILABLE` False. The backend still constructs; the tier that
owns it then calls `register_mem_host_pool_v2`, gets AttributeError, and the
engine — written so that a failed tier costs nothing — logs one line and serves
with no L3. A KV cache that is configured, reports no error, and stores nothing.

Pure unit tests: they stub `autumn` and the interface modules in `sys.modules`
and re-import the adapter, so no cluster and no built PyO3 module is needed.
"""

from __future__ import annotations

import importlib
import sys
import types

import autumn_kvcache

BACKEND = "autumn_kvcache.sglang_backend"
SGLANG_MOD = "sglang.srt.mem_cache.hicache_storage"
FREETOKEN_MOD = "freetoken.kvcache.hicache.storage"

V2_NAMES = ("PoolHitPolicy", "PoolName", "PoolTransfer", "PoolTransferResult")


def _iface_module(tag: str, *, v2: bool = True) -> types.ModuleType:
    """A stand-in carrying the symbols the adapter resolves, tagged by origin.

    `register_mem_host_pool_v2` is the ABC default the real interface supplies,
    and the one whose absence produced the silent failure.
    """
    mod = types.ModuleType(tag)

    class HiCacheStorage:
        origin = tag

        def register_mem_host_pool_v2(self, host_pool, host_pool_name):
            if not hasattr(self, "registered_pools"):
                self.registered_pools = {}
            self.registered_pools[host_pool_name] = host_pool

    mod.HiCacheStorage = HiCacheStorage
    mod.HiCacheStorageConfig = type(f"HiCacheStorageConfig_{tag}", (), {})
    if v2:
        for n in V2_NAMES:
            setattr(mod, n, type(f"{n}_{tag}", (), {"origin": tag}))
    return mod


def _reimport(monkeypatch):
    """Re-import the adapter under the current `sys.modules`, reversibly.

    Both the module entry AND the package attribute must be handed to
    monkeypatch. `import pkg.sub` sets `pkg.sub` as an attribute on the package
    object, and a later `from pkg import sub` takes that attribute WITHOUT
    re-importing — so clearing only `sys.modules` leaves the stub-bound module
    reachable by every test that runs after this file.

    And it must be `setitem`/`setattr`, not `delitem`/`delattr`. With
    `raising=False` the delete forms are a NO-OP when the name is already
    absent — which is exactly the state before the first test here, since this
    file imports the package but not the submodule — so they record nothing and
    restore nothing, while the re-import below installs the stub-bound module
    for good. The set forms record "was absent" and undo by deleting.
    """
    monkeypatch.setitem(sys.modules, BACKEND, None)
    monkeypatch.setattr(autumn_kvcache, "sglang_backend", None, raising=False)
    del sys.modules[BACKEND]                      # absent, so this re-imports
    return importlib.import_module(BACKEND)


def _present(monkeypatch, leaf: str, mod: types.ModuleType | None):
    """Make `leaf` resolve to `mod`, or (with None) fail to import.

    A `None` entry in `sys.modules` makes `import_module` raise
    ModuleNotFoundError — the leaf entry is checked before any parent walk, so
    the parent packages need no stubbing either way.
    """
    monkeypatch.setitem(sys.modules, leaf, mod)


def _install(monkeypatch, *, sglang, freetoken):
    monkeypatch.setitem(sys.modules, "autumn", types.ModuleType("autumn"))
    _present(monkeypatch, SGLANG_MOD, _iface_module(SGLANG_MOD) if sglang else None)
    _present(monkeypatch, FREETOKEN_MOD,
             _iface_module(FREETOKEN_MOD) if freetoken else None)
    return _reimport(monkeypatch)


# --------------------------------------------------------------------------- #
# which module wins
# --------------------------------------------------------------------------- #
def test_freetoken_supplies_the_interface_when_sglang_is_absent(monkeypatch):
    """The case that was broken, and why the image would have shipped a dead
    tier."""
    b = _install(monkeypatch, sglang=False, freetoken=True)

    assert b._SGLANG_AVAILABLE is True
    assert b._SGLANG_V2_AVAILABLE is True, (
        "v2 unresolved without sglang: batch_get_v2/batch_set_v2 would raise "
        "NotImplementedError on every call"
    )
    assert b.HiCacheStorage.origin == FREETOKEN_MOD
    # `register_mem_host_pool_v2` is the ABC default; with the base fallen back
    # to `object` it is simply missing, and that AttributeError is what killed
    # the tier at construction. (Reached here through inheritance from the stub
    # base — this pins the binding, not the real adapter's own behaviour.)
    assert hasattr(b.AutumnKVCacheStorage, "register_mem_host_pool_v2")
    assert issubclass(b.AutumnKVCacheStorage, b.HiCacheStorage)


def test_sglang_wins_when_both_are_already_imported(monkeypatch):
    """Under a real sglang the adapter must subclass sglang's own class:
    sglang's `backend_factory._load_backend_class` gates the dynamic path on
    `issubclass(backend_class, HiCacheStorage)` against it. With both hosts
    equally present, tuple order breaks the tie."""
    b = _install(monkeypatch, sglang=True, freetoken=True)

    assert b.HiCacheStorage.origin == SGLANG_MOD
    assert b.PoolTransfer.origin == SGLANG_MOD


def test_the_host_that_already_imported_its_interface_wins_over_tuple_order(
    monkeypatch,
):
    """The rule that makes the binding follow the HOST rather than the image.

    A FreeToken image that also has sglang installed for some unrelated reason
    would, under plain tuple order, bind sglang's ABC — and FreeToken's own
    factory has the identical `issubclass(..., HiCacheStorage)` gate against ITS
    class, so it would reject the backend. Same silent "no L3".

    Modelled exactly: freetoken is in `sys.modules` (its attach imports
    `.storage` before loading a backend), sglang is importable but NOT yet
    imported.
    """
    monkeypatch.setitem(sys.modules, "autumn", types.ModuleType("autumn"))
    monkeypatch.setitem(sys.modules, FREETOKEN_MOD, _iface_module(FREETOKEN_MOD))
    monkeypatch.delitem(sys.modules, SGLANG_MOD, raising=False)

    real_import = importlib.import_module

    def fake_import(name, *a, **kw):
        if name == SGLANG_MOD:
            return _iface_module(SGLANG_MOD)      # installed, importable
        return real_import(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setitem(sys.modules, BACKEND, None)
    monkeypatch.setattr(autumn_kvcache, "sglang_backend", None, raising=False)
    del sys.modules[BACKEND]
    b = real_import(BACKEND)

    assert b.HiCacheStorage.origin == FREETOKEN_MOD, (
        "bound the merely-installed engine over the one the host is running"
    )
    assert b.PoolTransfer.origin == FREETOKEN_MOD


def test_no_engine_at_all_still_imports(monkeypatch):
    """The property the original single import was written for, and which the
    fallback must not cost: the data-plane smoke test imports this adapter with
    no model stack present."""
    b = _install(monkeypatch, sglang=False, freetoken=False)

    assert b._SGLANG_AVAILABLE is False
    assert b._SGLANG_V2_AVAILABLE is False
    assert b.HiCacheStorage is object


# --------------------------------------------------------------------------- #
# v1 / v2 must come from the same module
# --------------------------------------------------------------------------- #
def test_a_v1_only_provider_degrades_to_the_v1_path(monkeypatch):
    """v2 landed after v1, so an engine can supply the ABC and none of the pool
    types. Binding v2 anyway would raise AttributeError at import instead of
    degrading, which is what a v1-only deployment is entitled to."""
    monkeypatch.setitem(sys.modules, "autumn", types.ModuleType("autumn"))
    _present(monkeypatch, SGLANG_MOD, _iface_module(SGLANG_MOD, v2=False))
    _present(monkeypatch, FREETOKEN_MOD, None)
    b = _reimport(monkeypatch)

    assert b._SGLANG_AVAILABLE is True, "v1 is there and must still be used"
    assert b._SGLANG_V2_AVAILABLE is False
    assert b.PoolTransfer is None


def test_v2_is_never_taken_from_a_different_module_than_v1(monkeypatch):
    """A v1-only host alongside a complete vendored copy must NOT be spliced.

    Cross-binding builds a class whose base is one engine's ABC while its pool
    types are another's. No host produces that combination: if the host is the
    old sglang it never calls v2 (and its ABC has no
    `register_mem_host_pool_v2`), so a True flag would be a lie; if the host is
    FreeToken, taking v1 from sglang already fails its factory's issubclass
    gate. Degrading to v1 is the only honest answer.
    """
    monkeypatch.setitem(sys.modules, "autumn", types.ModuleType("autumn"))
    _present(monkeypatch, SGLANG_MOD, _iface_module(SGLANG_MOD, v2=False))
    _present(monkeypatch, FREETOKEN_MOD, _iface_module(FREETOKEN_MOD))
    b = _reimport(monkeypatch)

    assert b.HiCacheStorage.origin == SGLANG_MOD, "v1 came from the host"
    assert b._SGLANG_V2_AVAILABLE is False, (
        "v2 was spliced in from a different engine than the ABC"
    )
    assert b.PoolTransfer is None
