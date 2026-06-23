"""AgenticBuffer — multi-track, group-keyed rollout buffer with dynamic filter.

A sibling of :class:`unirl.trainer.async_ar._RolloutBuffer` that stores whole
**multi-track** prompt groups (a ``RolloutResp`` shard holding one prompt's
think + image subtree) instead of a single ``RolloutTrack``. Two extra P0
features beyond the AR buffer:

- **Dynamic filter** (slime ``check_reward_nonzero_std``): a prompt group whose
  scored-track rewards have ~zero variance gives GRPO no gradient signal — every
  sibling shares the group mean, so every advantage is 0. :meth:`drain_freshest`
  evicts those groups before counting, so a training batch is always made of
  groups that actually move the policy. This is what makes over-sampling pay off:
  generate more groups than needed, keep only the informative ones.
- **Freshness + staleness** (AReaL capacity / AsyncARTrainer parity): groups are
  stamped with the ``weight_version`` they were generated under and a monotonic
  ``gen_id``; ``drain_freshest`` evicts groups older than
  ``current_version - max_staleness`` and returns the freshest ``n``.

Single-threaded (driver-owned), so no lock — same contract as the AR buffer.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from unirl.distributed.tensor.ref import hydrate
from unirl.types.rollout_resp import RolloutResp

logger = logging.getLogger(__name__)


class AgenticBuffer:
    """Group-keyed buffer of multi-track ``RolloutResp`` shards (one per prompt)."""

    def __init__(self, *, scored_track: str = "image", reward_std_eps: float = 1e-6) -> None:
        # Each item: (resp_shard, weight_version, gen_id).
        self._items: List[Tuple[RolloutResp, int, int]] = []
        self._scored_track = scored_track
        self._reward_std_eps = float(reward_std_eps)

    def put(self, resp: RolloutResp, *, weight_version: int, gen_id: int) -> None:
        self._items.append((resp, int(weight_version), int(gen_id)))

    def size(self) -> int:
        return len(self._items)

    # ---- dynamic filter ----------------------------------------------------

    def _group_has_signal(self, resp: RolloutResp) -> bool:
        """True iff the scored track's rewards in this group have nonzero variance.

        A group with all-equal rewards (std ~ 0) is zero-signal for GRPO and is
        dropped. Groups missing the scored track or its rewards are treated as
        no-signal (defensive — they cannot contribute a gradient).
        """
        track = resp.tracks.get(self._scored_track)
        if track is None or track.rewards is None:
            return False
        rewards = hydrate(track.rewards).to(torch.float32).flatten()
        if rewards.numel() <= 1:
            return False
        return float(rewards.std(unbiased=False).item()) > self._reward_std_eps

    def num_with_signal(self) -> int:
        return sum(1 for it in self._items if self._group_has_signal(it[0]))

    # ---- drain -------------------------------------------------------------

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
        apply_filter: bool = True,
    ) -> Optional[List[Tuple[RolloutResp, int, int]]]:
        """Pop the ``n`` freshest informative groups, carrying leftovers forward.

        Order of operations: staleness eviction → dynamic reward-variance filter
        → freshness sort → take ``n``. Returns ``None`` (and keeps every kept
        item buffered) when fewer than ``n`` informative groups remain, so the
        caller can generate more before retrying.
        """
        if max_staleness is not None and current_version is not None:
            self._items = [it for it in self._items if current_version - it[1] <= max_staleness]
        if apply_filter:
            kept = [it for it in self._items if self._group_has_signal(it[0])]
            dropped = len(self._items) - len(kept)
            if dropped:
                logger.info("AgenticBuffer: dropped %d zero-variance group(s)", dropped)
            self._items = kept
        if len(self._items) < n:
            return None
        self._items.sort(key=lambda it: it[2], reverse=True)  # freshest gen_id first
        picked, self._items = self._items[:n], self._items[n:]
        return picked


__all__ = ["AgenticBuffer"]
