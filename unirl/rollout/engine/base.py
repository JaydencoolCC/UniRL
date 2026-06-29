"""Rollout engine base class for the ``Sample`` → ``Sample`` path.

Subclasses (``VLLMOmniRolloutEngine``, ``SGLangDiffusionRolloutEngine``,
``SGLangRolloutEngine``, ``ComposedRolloutEngine``) take all runtime deps as ``__init__`` kwargs
and complete construction in one shot — no separate ``initialize(device)``
step. After ``__init__`` returns the engine is fully usable: model loaded,
worker subprocesses spawned, dist groups brought up. This matches the
actor flow where ``_setup_distributed_env`` runs before the engine is
built.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.sample import Sample


class BaseEngineConfig(ABC):
    """Marker base for all rollout engine config dataclasses.

    Used as the type annotation / base class for the engine config dataclasses.
    Each concrete engine config maps itself to its runtime engine class via
    :meth:`make_engine`.
    """

    def make_engine(self, **deps: Any) -> "BaseRolloutEngine":
        """Construct the runtime engine declared by this config.

        ``deps`` carry the runtime injections (``device``, ``strategy``,
        ``rank``, ``model_config``); the engine ctor contract is uniformly
        ``Engine(config=self, **deps)``. Subclasses override to import (lazily,
        so config modules stay importable without the engine's heavy optional
        deps) and return their engine class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement make_engine()")


class BaseRolloutEngine(Remote, ABC):
    """Rollout engine ABC. One-shot construction; new types only."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def shutdown(self) -> None:
        """Release worker subprocesses and any other engine-owned resources."""

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        """Best-effort runtime offload. Default no-op.

        Decorated so the driver-side ``Handle.sleep()`` dispatches to every
        worker. Subclasses that override should re-apply ``@distributed``
        on their override (Handle's method-binding sees the subclass's
        attribute and won't pick up a base-class decorator alone).
        """

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        """Restore runtime resources after ``sleep``. Default no-op.

        Same dispatch contract as :meth:`sleep`; see its docstring.
        """

    def onload_weights(self, *, track_prefix: str = "") -> None:
        """Restore the resources needed to receive a weight update."""
        del track_prefix
        self.wake_up()

    @property
    def is_offloaded(self) -> bool:
        """Whether the engine has released its runtime resources."""
        return False

    def health_check(self) -> bool:
        """Return True iff the engine is ready to serve a generate call."""
        return True

    def get_memory_info(self) -> Dict[str, float]:
        """Per-engine GPU memory snapshot. Default reads CUDA totals."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "cached_gb": torch.cuda.memory_reserved() / 1e9,
        }

    # ------------------------------------------------------------------
    # Generation — async per-group core + sync batch façade
    # ------------------------------------------------------------------
    #
    # The contract is async and per-GROUP: ``agenerate(group) -> group`` (one
    # prompt-group in, the filled group out — direct return, no iterator).
    # ``generate`` is the unchanged synchronous batch entry the trainers call; it
    # fans the batch over its groups (``Sample.split``) and reassembles
    # (``Sample.concat``), running the per-request work concurrently on the
    # engine's own event loop, so it stays byte-identical to the pre-async path.
    # Streaming is the *consumer* composing many ``agenerate`` coroutines
    # as-completed (the deferred driver); stopping is the ``abort``/``pause``
    # control plane below.

    #: The asyncio loop this engine's coroutines run on. Installed by
    #: :meth:`_init_async_loop` in the engine ctor — SGLang adopts the rollout
    #: backend's ``engine.loop`` (the loop its awaitables are bound to); loop-less
    #: engines get a fresh on-demand loop. ``generate`` drives it with
    #: ``run_until_complete`` (serialized by ``_loop_lock``); control RPCs schedule
    #: onto it with ``run_coroutine_threadsafe`` while it is being driven.
    _loop: Optional[asyncio.AbstractEventLoop] = None
    #: Policy weight version the current weights correspond to (bumped on each
    #: weight sync; stamped onto generated Parts by :meth:`_stamp_weight_version`).
    _weight_version: int = 0

    def _init_async_loop(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Install the coroutine loop + the lock that serializes driving it.

        Call once from the engine ctor. Pass the rollout backend's own loop
        (SGLang: ``self._backend.loop``) so generation coroutines run on the loop
        the backend awaitables are bound to; pass ``None`` for a loop-less engine
        (trainside / vllm_omni / composed) to create a fresh on-demand loop.
        """
        self._loop = loop if loop is not None else asyncio.new_event_loop()
        self._loop_lock = threading.Lock()

    def _run_coro(self, coro: Any) -> Any:
        """Drive ``coro`` to completion on the engine loop (one driver at a time).

        ``_loop_lock`` serializes ``run_until_complete`` (two concurrent drives of
        one loop would raise). A control RPC (``abort``/``pause``) does NOT take
        the lock — it schedules onto the same loop with
        :meth:`_run_coro_threadsafe`, which runs because a generation is driving
        the loop. Engines whose backend owns the lock (SGLang) override this.
        """
        loop, lock = getattr(self, "_loop", None), getattr(self, "_loop_lock", None)
        if loop is None or lock is None:
            raise RuntimeError(f"{type(self).__name__}: call _init_async_loop() in __init__ before generation")
        with lock:
            return loop.run_until_complete(coro)

    def _run_coro_threadsafe(self, coro: Any) -> Any:
        """Schedule a control coro onto the engine loop from another thread, wait.

        Returns ``None`` if the loop is not currently being driven (nothing in
        flight to act on). Used by :meth:`abort`/:meth:`pause`, which run on a
        different (threaded-Worker) actor thread than the in-flight ``generate``.
        """
        loop = getattr(self, "_loop", None)
        if loop is None or not loop.is_running():
            return None
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    async def agenerate(self, sample: Sample) -> Sample:
        """Run ONE prompt-group: fill its gen Part(s) and return the group.

        The async per-group core every concrete engine implements. The default
        raises so a not-yet-migrated engine fails loudly only if the async path
        is actually exercised.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement async agenerate()")

    async def _agenerate_batch(self, sample: Sample) -> Sample:
        """Fan a batch over its groups and reassemble — the body of :meth:`generate`."""
        groups = await asyncio.gather(*(self.agenerate(g) for g in sample.split()))
        return Sample.concat(groups)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Synchronous batch façade — the entry trainers call (unchanged contract).

        Splits the batch into prompt-groups, runs them concurrently on the engine
        loop via :meth:`agenerate`, and concatenates — byte-identical to the
        pre-async whole-batch path. Engines needing bespoke batch handling may
        override this (re-applying ``@distributed``); none do today.
        """
        return self._run_coro(self._agenerate_batch(sample))

    # ------------------------------------------------------------------
    # Control plane — sync methods reached via the raw ``Worker.call`` RPC (the
    # un-decorated weight-sync pattern), so they interleave with an in-flight
    # ``generate`` on a threaded Worker (``worker_max_concurrency>1``).
    # ------------------------------------------------------------------

    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        """Best-effort cancel of in-flight generation; return any partials.

        Default no-op (``[]``). Engines whose backend supports it (SGLang) cancel
        running requests; sync/batch backends can only drop not-yet-started work.
        """
        del ids
        return []

    def pause(self) -> None:
        """Stop admitting new generation (best-effort). Default no-op."""

    def resume(self) -> None:
        """Resume generation after :meth:`pause`. Default no-op."""

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _stamp_weight_version(self, sample: Sample) -> Sample:
        """Stamp ``self._weight_version`` onto the frontier (last) gen Part."""
        v = getattr(self, "_weight_version", None)
        if v is None or not sample.parts:
            return sample
        gen = sample.parts[-1].fill(weight_version=int(v))
        return sample.with_parts([*sample.parts[:-1], gen])

    # ------------------------------------------------------------------
    # Weight sync — bucketed CUDA-IPC (verl-omni pattern)
    # ------------------------------------------------------------------

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        track_prefix: str = "",
    ) -> None:
        """Receive a state dict over a per-rank ZMQ + CUDA-IPC channel."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — NCCL broadcast
    # ------------------------------------------------------------------

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None:
        """Bring up a trainer-rollout NCCL process group on the engine side."""
        raise NotImplementedError

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Receive a bucket of weights via the established NCCL group."""
        raise NotImplementedError

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        """Tear down a previously-initialized NCCL update group."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — LoRA tensor bag
    # ------------------------------------------------------------------

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Load a LoRA adapter directly from in-memory tensors."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — SGLang-shape one-bag tensor payload
    # ------------------------------------------------------------------

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Receive a state-dict packed as a single SGLang-shape payload per TP rank."""
        del track_prefix
        raise NotImplementedError


__all__ = ["BaseRolloutEngine"]
