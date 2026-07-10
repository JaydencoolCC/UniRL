"""AsyncAgenticEnvTrainer — fully-async agentic RL with ENV-SOURCED rewards (LIN-531).

The disaggregated producer/consumer sibling of
:class:`~unirl.trainer.agentic_env.AgenticEnvTrainer` — the same relation
:class:`~unirl.trainer.agentic_async.AsyncAgenticTrainer` has to
:class:`~unirl.trainer.agentic.AgenticTrainer`. Used for interactive ENVIRONMENTS
(ALFWorld, Sokoban, WebShop, …) where the reward is the simulator's per-trajectory
return, already attached to each trajectory's last generated Part by the
:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine` — not a grade of a
final ``<answer>``.

Only the reward SOURCE differs from ``AsyncAgenticTrainer``: this reads the env return
off the frontier instead of building a scoring Sample. The fully-async machinery — the
staleness-bounded producer/consumer, disaggregated slabs, NCCL cross-slab sync, and the
turn-boundary partial-rollout quiesce — is inherited unchanged, so the variable-depth
ALFWorld trajectories are exactly the workload the partial-rollout overlap targets.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from unirl.distributed.tensor import hydrate
from unirl.trainer.agentic_async import AsyncAgenticTrainer
from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class AsyncAgenticEnvTrainer(AsyncAgenticTrainer):
    """Fully-async agentic trainer whose reward is the environment's per-trajectory return.

    The recipe still carries a (built-but-unused) ``reward`` backend so the shared
    construction path is happy; this trainer never calls it. The reconstructed
    answer-request that ``AsyncAgenticTrainer._train_on_groups`` passes to
    ``_rewards_and_groups`` is ignored here (env reward rides the trajectories).
    """

    def _rewards_and_groups(
        self, sample: Sample, trajs: List[Sample], rollout_id: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """Read the env-sourced scalar reward the engine attached to each trajectory's
        last generated Part; group by the shared root id. No reward backend, no scoring
        Sample (mirrors :meth:`AgenticEnvTrainer._rewards_and_groups`)."""
        del sample, rollout_id  # env reward is already on the trajectories
        values: List[float] = []
        group_ids: List[str] = []
        missing = 0
        for tr in trajs:
            gens = tr.gen_parts()
            reward: Optional[torch.Tensor] = gens[-1].rewards if gens else None
            if reward is not None:
                values.append(float(hydrate(reward).to(torch.float32).flatten()[0].item()))
            else:
                values.append(0.0)  # gen-less / failed trajectory stays a legit group member
                missing += 1
            group_ids.append(tr.parts[0].sample_ids[0])
        if missing:
            logger.warning(
                "AsyncAgenticEnvTrainer: %d/%d trajectories had no env reward (scored 0.0).",
                missing,
                len(trajs),
            )
        return torch.tensor(values, dtype=torch.float32), group_ids


__all__ = ["AsyncAgenticEnvTrainer"]
