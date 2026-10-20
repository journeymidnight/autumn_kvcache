"""Sync ↔ async bridge for the autumn PyO3 client.

The `autumn` PyO3 module exposes async methods (each returns an awaitable that
resolves when the compio worker thread completes the RPC). sglang's
HiCacheController calls L3 backend methods **synchronously** from its
prefetch/backup daemon threads, so we
own a dedicated asyncio event loop thread and dispatch coroutines onto it via
`run_coroutine_threadsafe`.

One loop per process is sufficient — sglang's worker threads share it. The
loop is created lazily on first call and lives for the process lifetime.
"""

from __future__ import annotations

import asyncio
import threading

_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()
_LOOP_THREAD: threading.Thread | None = None


def get_loop() -> asyncio.AbstractEventLoop:
    global _LOOP, _LOOP_THREAD
    with _LOOP_LOCK:
        if _LOOP is None or not _LOOP.is_running():
            loop = asyncio.new_event_loop()
            t = threading.Thread(
                target=loop.run_forever,
                name="autumn-kvcache-loop",
                daemon=True,
            )
            t.start()
            _LOOP = loop
            _LOOP_THREAD = t
        return _LOOP


def run(thunk):
    """Submit a `thunk` (zero-arg callable that returns an awaitable when
    invoked) to the worker loop and block until the result/exception.

    Why a thunk and not a coroutine: autumn's PyO3 methods call
    `asyncio.get_running_loop()` at the moment they're invoked (to bind a
    future to the loop that will receive the resolve callback). Calling them
    on the caller's thread (no loop) raises `RuntimeError: no running event
    loop`. By taking a thunk and invoking it inside `_wrap`, the await
    machinery runs on the worker loop and `get_running_loop()` sees it.
    """

    async def _wrap():
        return await thunk()

    fut = asyncio.run_coroutine_threadsafe(_wrap(), get_loop())
    return fut.result()


def new_loop() -> asyncio.AbstractEventLoop:
    """Create and start a fresh asyncio loop on its own daemon thread.

    Used to give each autumn Client its own loop so result marshaling
    (`set_result` on completion) doesn't serialize through one loop thread —
    the bottleneck that made in-process multi-client fan-out not scale.
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="autumn-kvcache-loop", daemon=True).start()
    return loop


def run_on(loop, thunk):
    """Submit `thunk` to a specific loop and block until result/exception."""
    return asyncio.run_coroutine_threadsafe(_wrap_thunk(thunk), loop).result()


async def _wrap_thunk(thunk):
    return await thunk()


