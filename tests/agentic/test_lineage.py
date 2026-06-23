"""Lineage + reward/advantage flow on an agentic-shaped 2-track resp.

Validates the credit-assignment + GRPO grouping the trainer relies on:
- ``propagate_rewards('mean')`` lifts each image reward to its 1:1 think parent.
- think advantages group by the prompt (its N*T turns/sessions).
- image advantages group by the ROOT prompt (all N*T images of a prompt).
"""

from __future__ import annotations

import torch

from unirl.types.rollout_resp import RolloutResp, RolloutTrack


def _two_track_resp(group_ids, image_rewards):
    """Build {ar (root), image (1:1 child)} with the given per-image rewards.

    ``group_ids[k]`` is the prompt group of row k (think + image share row k).
    """
    n = len(group_ids)
    think_sids = [f"t{k}#think" for k in range(n)]
    think = RolloutTrack(sample_ids=think_sids, parent_ids=list(group_ids), parent_track=None)
    image = RolloutTrack(
        sample_ids=[f"t{k}" for k in range(n)],
        parent_ids=list(think_sids),  # 1:1 child of think
        parent_track="ar",
        rewards=torch.tensor(image_rewards, dtype=torch.float32),
    )
    return RolloutResp(tracks={"ar": think, "image": image})


def test_propagate_lifts_image_reward_to_think_one_to_one():
    resp = _two_track_resp(["g0"] * 4, [0.1, 0.2, 0.3, 0.4])
    out = resp.propagate_rewards(op="mean")
    # 1:1 branch -> think reward == image reward.
    assert torch.allclose(out.tracks["ar"].rewards, torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_think_advantages_group_by_prompt_are_mean_zero():
    # 2 prompts x (N*T = 2 rows) each.
    resp = _two_track_resp(["g0", "g0", "g1", "g1"], [0.1, 0.9, 0.4, 0.6])
    resp = resp.propagate_rewards(op="mean")
    think = resp.tracks["ar"].compute_advantages(normalize=False, scope="group")
    adv = think.advantages
    # each group's advantages are mean-centered.
    assert abs(float(adv[:2].mean())) < 1e-6
    assert abs(float(adv[2:].mean())) < 1e-6
    # g0 has spread -> nonzero advantages.
    assert float(adv[:2].abs().sum()) > 0


def test_image_advantages_group_by_root_prompt():
    # 1 prompt, N*T = 4 images -> one root group of 4.
    resp = _two_track_resp(["g0"] * 4, [0.1, 0.2, 0.3, 0.4])
    resp = resp.propagate_rewards(op="mean")
    image = resp.compute_track_advantages("image", group_key="root", normalize=True)
    adv = image.advantages
    assert adv.shape == (4,)
    assert abs(float(adv.mean())) < 1e-5  # mean-centered across the root group
    # monotonic rewards -> monotonic advantages.
    assert float(adv[0]) < float(adv[-1])


def test_zero_variance_group_yields_zero_advantage():
    resp = _two_track_resp(["g0"] * 4, [0.5, 0.5, 0.5, 0.5])
    resp = resp.propagate_rewards(op="mean")
    think = resp.tracks["ar"].compute_advantages(normalize=True, scope="group")
    # all-equal rewards -> zero advantage (no GRPO signal; the dynamic filter drops these).
    assert torch.allclose(think.advantages, torch.zeros(4), atol=1e-5)
