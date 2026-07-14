"""The backend seam contract — the ``Backend`` protocol + the wire types.

Every ``sglang`` collaborator reaches the SGLang SRT runtime through this
protocol; the real implementation lives beside it (``http.py`` — SRT server
subprocess + HTTP). This module holds no runtime code at all, so it is trivially
CPU-importable.

**No RL types cross this seam.** ``generate`` takes ready-to-POST ``/generate``
payload dicts (one per prompt) and returns ``list[RawResult]`` (a structural view
of one parsed ``/generate`` candidate); the adapters do the
``Sample``↔wire translation. The impl absorbs its transport
asymmetries (async fan-out, retries, SGLang's dict-vs-list response shape for
``n``) behind these signatures.

Deliberate divergences from the ``sglang_diffusion`` seam:

- No ``target_modules`` on the update verbs — the diffusion-side default
  ``["transformer"]`` doesn't match LLM module naming; omitting the field lets
  the SRT server accept all incoming weights correctly.
- No ``weights_checksum`` — the checksum/verify path is vLLM-Omni-only.
- ``flush_cache`` is a first-class verb so the engine can orchestrate
  flush-before-sleep as a visible line.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from unirl.rollout.engine.runtime import CoroutineFactory

logger = logging.getLogger(__name__)
T = TypeVar("T")


class SessionRunner:
    """Drive one backend loop session and admit controls only while it is active.

    The coroutine *factory* is load-bearing: an idle/closed backend rejects work
    before a coroutine object exists, so best-effort controls cannot leak an
    unawaited coroutine. Controls accepted at the end of a session are tracked and
    drained before ``run`` lets the loop stop, closing the
    ``is_running()``/``run_coroutine_threadsafe`` race at that boundary.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        label: str,
        control_timeout_s: float = 10.0,
    ) -> None:
        self._loop = loop
        self._label = label
        self._control_timeout_s = float(control_timeout_s)
        self._drive_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._accept_controls = False
        self._controls: set[Future[Any]] = set()

    def _reject_async_caller(self, operation: str) -> None:
        """Fail before lock acquisition when called from any running loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(f"{self._label} {operation} cannot be called from a running event loop")

    def run(self, operation: CoroutineFactory[T]) -> T:
        """Run one factory-created awaitable on the owned loop, serialized."""
        self._reject_async_caller("run_session")
        with self._drive_lock:
            if self._closed or self._loop.is_closed():
                raise RuntimeError(f"{self._label} async event loop is not available.")
            if self._loop.is_running():
                raise RuntimeError(f"{self._label} async event loop is already running")
            with self._state_lock:
                self._accept_controls = True
            session = self._run_operation(operation)
            try:
                return self._loop.run_until_complete(session)
            except BaseException:
                # ``run_until_complete`` can reject before taking ownership.
                # Closing the wrapper is harmless once it has already completed.
                session.close()
                raise
            finally:
                with self._state_lock:
                    self._accept_controls = False
                    self._controls.clear()

    def run_sync(self, operation: Callable[[], T]) -> T:
        """Serialize a synchronous loop-driving backend verb with sessions."""
        self._reject_async_caller("run_sync")
        with self._drive_lock:
            if self._closed or self._loop.is_closed():
                raise RuntimeError(f"{self._label} async event loop is not available.")
            return operation()

    def assert_active_loop(self) -> None:
        """Fail fast unless called on this runner's currently active loop."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(f"{self._label} async operation requires run_session") from exc
        with self._state_lock:
            active = not self._closed and self._accept_controls
        if running_loop is not self._loop or not active:
            raise RuntimeError(f"{self._label} async operation requires its active run_session loop")

    def close(
        self,
        *,
        async_cleanup: Optional[CoroutineFactory[Any]] = None,
        finalizer: Optional[Callable[[], None]] = None,
    ) -> None:
        """Close once under the session lock, optionally cleaning up on the loop."""
        self._reject_async_caller("close")
        with self._drive_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                self._accept_controls = False
            try:
                if async_cleanup is not None and not self._loop.is_closed():
                    cleanup = async_cleanup()
                    try:
                        self._loop.run_until_complete(cleanup)
                    except BaseException:
                        cleanup.close()
                        raise
            finally:
                if finalizer is not None:
                    finalizer()

    async def _run_operation(self, operation: CoroutineFactory[T]) -> T:
        try:
            return await operation()
        finally:
            # Stop admission before taking the snapshot. A control that won the
            # state lock is already registered and must finish before this session
            # lets run_until_complete stop the loop.
            with self._state_lock:
                self._accept_controls = False
                pending = tuple(future for future in self._controls if not future.done())
            if pending:
                await asyncio.gather(
                    *(asyncio.wrap_future(future, loop=self._loop) for future in pending),
                    return_exceptions=True,
                )

    def run_control(self, operation: CoroutineFactory[Any]) -> Any:
        """Run a control on the active session, or no-op without creating it."""
        try:
            on_session_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_session_loop = False
        if on_session_loop:
            # This synchronous method cannot wait on its own loop. Controls enter
            # through a concurrent Worker thread; a same-loop call is best-effort.
            return None

        with self._state_lock:
            if self._closed or not self._accept_controls or not self._loop.is_running():
                return None
            coroutine = self._run_control(operation)
            try:
                future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            except Exception:
                close = getattr(coroutine, "close", None)
                if close is not None:
                    close()
                return None
            self._controls.add(future)

        try:
            return future.result()
        finally:
            with self._state_lock:
                self._controls.discard(future)

    async def _run_control(self, operation: CoroutineFactory[Any]) -> Any:
        try:
            return await asyncio.wait_for(operation(), timeout=self._control_timeout_s)
        except TimeoutError:
            logger.warning(
                "%s control timed out after %.1fs (best-effort)",
                self._label,
                self._control_timeout_s,
            )
            return None


class RawResult(Protocol):
    """Structural view of one parsed SRT ``/generate`` candidate — the wire
    fields this engine consumes. The HTTP impl deserializes responses into this
    shape (``n>1`` returns a list of candidates per prompt; the impl flattens
    them prompt-major: candidate ``k`` of prompt ``i`` at index ``i*n + k``);
    test fakes stand in structurally.

    Population: ``text`` and ``finish_reason`` are always set. ``token_ids`` /
    ``logprobs`` both come from the ``meta_info['output_token_logprobs']``
    items — the runtime's only source of generated token ids (there is no
    separate ``output_token_ids`` field) — so they are length-aligned by
    construction, and both empty when the request didn't ask for logprobs.
    """

    #: The raw generated text (``<think>`` tags intact — stripping is a
    #: driver-side concern, applied by the adapter at decode time).
    text: str
    #: Generated token ids, always length-aligned with ``logprobs``.
    token_ids: List[int]
    #: Per-token log-probs; both lists empty when ``return_logprob`` was off.
    logprobs: List[float]
    #: Normalized finish reason (SRT returns a dict or a bare string).
    finish_reason: str


@runtime_checkable
class Backend(Protocol):
    """The seam every ``sglang`` collaborator reaches the runtime through."""

    # generation — synchronous and safe for CONCURRENT callers: the agentic
    # drain calls it from one thread per trajectory, and the impl must keep the
    # in-flight requests batching together on the runtime (never serialize them).
    def generate(self, requests: List[Dict[str, Any]]) -> List[RawResult]: ...
    # best-effort controls
    def abort(self, *, abort_all: bool = True, rid: Optional[str] = None) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    # memory / lifecycle / health
    def flush_cache(self) -> None: ...
    def release_memory(self, *, tags: Optional[Sequence[str]] = None) -> None: ...
    def resume_memory(self, *, tags: Optional[Sequence[str]] = None) -> None: ...
    def shutdown(self) -> None: ...
    def ping(self) -> bool: ...
    # weight-sync verbs (serialization stays inside the impl)
    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None: ...
    def init_weights_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None: ...
    def update_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        flush_cache: bool,
    ) -> None: ...
    def destroy_weights_group(self, *, group_name: str) -> None: ...
    def set_lora(
        self,
        *,
        lora_name: str,
        lora_tensors: Dict[str, Any],
        config_dict: Optional[dict] = None,
    ) -> None: ...

    # update_from_ipc is intentionally absent — SGLang has no IPC receiver.


__all__ = ["Backend", "RawResult", "SessionRunner"]
