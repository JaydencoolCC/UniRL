"""v2 full-weight checkpoint-path sync (COLOCATE, single-node).

Simplest full-weight transport: the train slab serializes the freshly-trained
full base weights to a file on a shared/local path, and each co-located rollout
engine loads it via ``update_weights_from_path``. Full-weight analogue of the
LoRA / tensor / nccl / ipc handlers, used to bring up the FastVideo engine
before the faster zero-copy transports are wired.

Mirrors ``TensorWeightSync`` (colocate, ``rollout`` is a LOCAL sibling) but the
handoff is a torch.save file instead of a serialized tensor bag:

  rank0: ``_iter_full_tensors`` (FSDP all-gather; all ranks in lockstep) →
         ``torch.save`` to ``{sync_dir}/weights_v{N}`` + ``.ready`` marker
  every rank: wait for marker → ``self._rollout.update_weights_from_path(path)``

The weight walk yields the bare trainable-module (transformer) keys, which is
exactly what the FastVideo engine's ``transformer.load_state_dict`` expects — so
no ``name_remap`` is needed for the default FastVideo case. All torch imports
are deferred so the driver can import this module for ``remote(...)``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync


class CheckpointWeightSync(FullWeightSync):
    """Colocate full-weight sync via a torch.save checkpoint file."""

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        sync_dir: str = "/tmp/unirl_fastvideo_weight_sync",
        wait_timeout_s: float = 1200.0,
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        super().__init__(
            backend=backend,
            bucket_size_mb=bucket_size_mb,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            adapter_name=adapter_name,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        self._rollout = rollout  # local engine sibling (colocate)
        self._dir = str(sync_dir)
        self._wait_timeout_s = float(wait_timeout_s)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Publish the full weights to a file and load it into the local engine.

        Runs on every train rank (``BROADCAST``). ``_iter_full_tensors`` all-gathers
        each FSDP shard on every rank in lockstep, so all ranks must iterate; only
        rank-0 keeps the materialized tensors and writes the file. The path is
        deterministic from ``weight_version`` (incremented in lockstep on every
        rank), so all ranks agree on it without a broadcast.
        """
        import torch

        state_dict: Dict[str, torch.Tensor] = {}
        for name, tensor in self._iter_full_tensors():
            if self._my_rank == 0:
                state_dict[name] = tensor.detach().to("cpu", copy=True)

        version = int(self.weight_version)
        path = os.path.join(self._dir, f"weights_v{version}")
        marker = path + ".ready"

        if self._my_rank == 0:
            os.makedirs(self._dir, exist_ok=True)
            tmp = path + ".tmp"
            torch.save(state_dict, tmp)
            os.replace(tmp, path)  # atomic publish
            with open(marker, "w") as fh:
                fh.write("ok")
            del state_dict

        self._wait_for_marker(marker)
        self._rollout.update_weights_from_path(path, track_prefix=self._track_prefix)

        self.weight_version += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _wait_for_marker(self, marker: str) -> None:
        t0 = time.time()
        while not os.path.exists(marker):
            if time.time() - t0 > self._wait_timeout_s:
                raise TimeoutError(f"CheckpointWeightSync: ready marker not found after {self._wait_timeout_s}s: {marker}")
            time.sleep(0.2)


__all__ = ["CheckpointWeightSync"]
