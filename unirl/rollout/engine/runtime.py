"""Local coroutine-session runtime for synchronous rollout engines.

The rollout worker boundary is synchronous, while single-turn engines expose an
async ``agenerate`` core.  ``LocalAsyncRuntime`` owns the persistent event loop
used to bridge that boundary.  Callers provide a *factory* so a coroutine is
created only after the runtime has established that it can run it.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Coroutine, TypeAlias, TypeVar

T = TypeVar("T")

CoroutineFactory: TypeAlias = Callable[[], Coroutine[Any, Any, T]]


class LocalAsyncRuntime:
    """Own and serially drive one local :class:`asyncio.Runner`.

    ``run`` and ``close`` share one lock: only one synchronous caller can drive
    the runner, and closing waits for an active session to finish.  ``close`` is
    idempotent; a session attempted after close fails before its coroutine
    factory is invoked.
    """

    def __init__(self) -> None:
        self._runner = asyncio.Runner()
        self._lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether this runtime has been closed."""
        return self._closed

    def run(self, factory: CoroutineFactory[T]) -> T:
        """Create and run one coroutine session on the owned event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("LocalAsyncRuntime.run cannot be called from a running event loop")
        with self._lock:
            if self._closed:
                raise RuntimeError("LocalAsyncRuntime is closed")
            coroutine = factory()
            try:
                return self._runner.run(coroutine)
            except BaseException:
                # ``Runner.run`` can reject before taking ownership (for example,
                # if called from an already-running loop). Closing is harmless
                # when the coroutine did run and has already completed.
                coroutine.close()
                raise

    def close(self) -> None:
        """Close the runner and its event loop once, waiting for any active run."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("LocalAsyncRuntime.close cannot be called from a running event loop")
        with self._lock:
            if self._closed:
                return
            self._runner.close()
            self._closed = True


__all__ = ["CoroutineFactory", "LocalAsyncRuntime"]
