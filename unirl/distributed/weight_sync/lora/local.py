"""Colocate LoRA weight-sync: push the trained adapter into a same-Worker sibling
engine, in-process.

Constructed inside the trainer's ``placement(...)`` block with the ``backend`` and
``rollout`` siblings; they arrive as the LOCAL ``Remote`` instances (``HandleRef``
resolved by ``Worker.add_remote``), so method calls run in-process on this Worker.
The engine owns the Worker→Omni-subprocess transfer (serialize + ``collective_rpc``),
so there is no separate ZMQ pump and no sender/receiver overlap to orchestrate.

This deliberately does NOT reuse the full-weight handler family (``full/``): for
LoRA the engine's in-process ``set_lora_from_tensors`` already owns the transfer.
"""

from __future__ import annotations

import logging
from typing import Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.lora.base import LoraWeightSyncBase

logger = logging.getLogger(__name__)


class LocalLoraWeightSync(LoraWeightSyncBase):
    """Push one track's trained FSDP LoRA adapter into a co-located rollout engine.

    ``rollout`` is the same-Worker sibling engine for the ``sync()`` push and is
    REQUIRED. (Cross-process engines that are not siblings are handled by
    :class:`~unirl.distributed.weight_sync.lora.remote.RemoteLoraWeightSync`.)
    """

    def __init__(
        self,
        *,
        backend,
        rollout,
        param_prefix: str = "",
        adapter_name: Optional[str] = None,
        verify: bool = False,
        track_prefix: str = "",
    ) -> None:
        super().__init__(
            backend=backend,
            param_prefix=param_prefix,
            adapter_name=adapter_name,
            verify=verify,
            track_prefix=track_prefix,
        )
        self._rollout = rollout
        self._cached = None

    def _is_engine_launcher(self) -> bool:
        return self.rank_info is None or int(self.rank_info.tp_rank) == 0

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def extract(self) -> None:
        """Collectively extract LoRA; cache it on each rollout replica launcher."""

        self._extract_to_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def push(self) -> None:
        """Push the cached LoRA after the trainer has made room for rollout."""

        self._push_from_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Extract LoRA from the local FSDP model and load it into the engine.

        Runs on every Worker (``BROADCAST``); the extract is a train-mesh
        collective that lines up because every rank runs together. Each TP group's
        launcher keeps one CPU payload and pushes it to its sibling engine; shell
        ranks discard theirs. The engine must be awake before the push.
        """
        self._extract_to_cache()
        self._push_from_cache()

    def _extract_to_cache(self) -> None:
        """Run the train-mesh collective and retain one CPU copy per TP group."""

        lora_tensors, peft_config = self._extract()
        if self._is_engine_launcher():
            self._cached = (lora_tensors, peft_config)

    def _push_from_cache(self) -> None:
        """Load the cached adapter into this launcher's sibling engine."""

        if not self._is_engine_launcher():
            return
        if self._cached is None:
            raise RuntimeError("LocalLoraWeightSync.push: call extract() (or sync()) first")
        lora_tensors, peft_config = self._cached
        self._cached = None
        self._rollout.set_lora_from_tensors(self._adapter_name, lora_tensors, peft_config=peft_config)
        rank = self.rank_info.rank if self.rank_info is not None else 0
        logger.info(
            "[LoRA-SYNC] rank %s: pushed %d LoRA tensors to rollout (adapter=%s, track=%s)",
            rank,
            len(lora_tensors),
            self._adapter_name,
            self._track_prefix or "<single>",
        )
        if self._verify:
            self._verify_loaded(lora_tensors, peft_config)

    def _verify_loaded(self, lora_tensors, peft_config) -> None:
        """Assert the sibling engine's loaded LoRA matches what we just pushed."""
        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
        )

        exp_a, exp_b = self._expected_checksums(lora_tensors, peft_config)
        topology = self._rollout.tp_per_stage()
        loaded = self._rollout.loaded_lora_checksums(adapter_id=int(DIFFRL_LORA_INT_ID))
        rank = self.rank_info.rank if self.rank_info is not None else 0
        self._assert_loaded(
            exp_a,
            exp_b,
            loaded,
            topology=topology,
            label=f"train-rank {rank} rollout",
        )
        logger.info(
            "[LoRA-SYNC] rank %s: verify OK (%d lora_A / %d lora_B layers match)",
            rank,
            len(exp_a),
            len(exp_b),
        )


__all__ = ["LocalLoraWeightSync"]
