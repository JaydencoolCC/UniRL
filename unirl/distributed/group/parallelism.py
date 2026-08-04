"""Backend-neutral placement context for distributed rollout engines.

``Handle`` owns the logical DP/TP/PP/EP layout and the physical GPU tokens.
An engine that launches a multi-process runtime consumes this immutable value;
it must not rediscover placement from global ranks or hard-coded device ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RolloutParallelContext:
    """One worker's coordinates and the device group owned by its engine.

    Only ``tp_rank == 0`` launches the backend runtime for a TP group. The
    remaining workers are SPMD shells: they still participate in trainer-side
    collectives, while rollout dispatch/collect keeps the launcher's result.

    ``visible_devices`` contains scheduler-issued CUDA tokens (ordinals, UUIDs,
    or MIG ids), ordered by TP rank. It is intentionally empty for TP=1 so the
    worker's existing Ray visibility remains authoritative.
    """

    dp_rank: int = 0
    dp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1
    visible_devices: Tuple[str, ...] = ()

    @property
    def is_engine_launcher(self) -> bool:
        """Whether this worker owns the backend process tree for its TP group."""

        return self.tp_rank == 0


__all__ = ["RolloutParallelContext"]
