"""AgenticBuffer unit tests (verification plan #2 + dynamic filter).

Covers freshness ordering, staleness eviction, and the zero-variance reward
filter that makes over-sampling worthwhile.
"""

from __future__ import annotations

import torch

from unirl.agentic.buffer import AgenticBuffer
from unirl.types.rollout_resp import RolloutResp, RolloutTrack


def _group(rewards, gid_prefix="p"):
    """One prompt-group resp shard with an 'image' track carrying ``rewards``."""
    n = len(rewards)
    return RolloutResp(
        tracks={
            "image": RolloutTrack(
                sample_ids=[f"{gid_prefix}/i{j}" for j in range(n)],
                parent_ids=[gid_prefix] * n,
                rewards=torch.tensor(rewards, dtype=torch.float32),
            )
        }
    )


def test_put_size_and_drain_freshest_order():
    buf = AgenticBuffer(scored_track="image")
    buf.put(_group([0.1, 0.9], "a"), weight_version=0, gen_id=0)
    buf.put(_group([0.2, 0.8], "b"), weight_version=0, gen_id=1)
    buf.put(_group([0.3, 0.7], "c"), weight_version=0, gen_id=2)
    assert buf.size() == 3

    picked = buf.drain_freshest(2, current_version=0, max_staleness=0)
    assert picked is not None
    # Freshest two by gen_id: c (2), b (1).
    assert [it[2] for it in picked] == [2, 1]
    assert buf.size() == 1  # leftover 'a' carried forward


def test_drain_returns_none_when_insufficient():
    buf = AgenticBuffer(scored_track="image")
    buf.put(_group([0.1, 0.9]), weight_version=0, gen_id=0)
    assert buf.drain_freshest(2, current_version=0, max_staleness=0) is None
    assert buf.size() == 1  # nothing consumed


def test_reward_variance_filter_drops_zero_signal_groups():
    buf = AgenticBuffer(scored_track="image", reward_std_eps=1e-6)
    buf.put(_group([0.5, 0.5, 0.5], "flat"), weight_version=0, gen_id=0)  # zero variance
    buf.put(_group([0.1, 0.9, 0.5], "varied"), weight_version=0, gen_id=1)
    assert buf.num_with_signal() == 1

    picked = buf.drain_freshest(1, current_version=0, max_staleness=0)
    assert picked is not None
    assert picked[0][0].tracks["image"].parent_ids[0] == "varied"
    # The flat (zero-variance) group was evicted, not just skipped.
    assert buf.size() == 0


def test_staleness_eviction():
    buf = AgenticBuffer(scored_track="image")
    buf.put(_group([0.1, 0.9], "old"), weight_version=0, gen_id=0)
    buf.put(_group([0.2, 0.8], "new"), weight_version=2, gen_id=1)
    # current_version=3, max_staleness=1 -> evict anything older than version 2.
    picked = buf.drain_freshest(1, current_version=3, max_staleness=1)
    assert picked is not None
    assert picked[0][0].tracks["image"].parent_ids[0] == "new"


def test_filter_can_be_disabled():
    buf = AgenticBuffer(scored_track="image")
    buf.put(_group([0.5, 0.5], "flat"), weight_version=0, gen_id=0)
    picked = buf.drain_freshest(1, current_version=0, max_staleness=0, apply_filter=False)
    assert picked is not None  # zero-variance group kept when filter is off
