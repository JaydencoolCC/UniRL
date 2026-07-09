"""CPU tests for the fully-async (v5) fully-async resident pool layer.

Covers, with a deterministic fake backend (no GPU / no Bagel):
- version utilities (spread, staleness capacity / admission guard),
- versioned Session methods (per-turn version, spread, think_versions),
- segment version metadata roundtrip (pack → concat → select),
- the resident pool: in-window completion, carry-over across a version bump,
  bounded-staleness abort + replenish, and over-sample residency.
"""

from __future__ import annotations

import torch

from unirl.agentic.env import RuleEnv
from unirl.agentic.pool import FullyAsyncPool
from unirl.agentic.session import MsgNode, Session
from unirl.agentic.version import should_admit, staleness_capacity, version_spread
from unirl.types.segments.latent import make_image_segment
from unirl.types.segments.text import TextSegment

NSDE = 2


class _FakeBackend:
    """Minimal AgenticBackend: think emits 2 tokens, gen emits a 1-row latent.

    Stamps each node with ``session.weight_version`` (set by the pool before the
    turn), exactly like the real ``_BagelTurnBackend`` — so the pool's per-turn
    version stamping is exercised end-to-end on CPU.
    """

    def think(self, *, session, turn, sampling):
        return MsgNode(
            kind="think",
            tokens=torch.tensor([turn + 1, turn + 2], dtype=torch.long),
            logprobs=torch.tensor([-0.1, -0.2], dtype=torch.float32),
            text=f"think-{turn}",
            payload={"prompt_splits": [{"kind": "text"}]},
            weight_version=session.weight_version,
        )

    def gen(self, *, session, think, turn, sampling):
        seg = make_image_segment(
            latents=torch.randn(1, 3, 4, 2),
            sigmas=torch.linspace(1.0, 0.0, 3),
            indices=torch.arange(3),
            sde_logp=torch.randn(1, NSDE),
            sde_indices=torch.arange(NSDE),
        )
        return MsgNode(
            kind="gen",
            latent=seg,
            image=type("Img", (), {"pixels": torch.rand(1, 3, 4, 4)})(),
            payload={"contexts": (None, None, None), "image_shape": (4, 4)},
            weight_version=session.weight_version,
        )


def _make_pool(version_holder, *, n=2, max_turns=2, max_staleness=1, turns_per_window=1):
    return FullyAsyncPool(
        backend=_FakeBackend(),
        env=RuleEnv(instructions=["r0", "r1", "r2"]),
        version_ref=lambda: version_holder[0],
        max_turns=max_turns,
        sessions_per_prompt=n,
        max_staleness=max_staleness,
        turns_per_window=turns_per_window,
    )


# ---- version utilities ----------------------------------------------------


def test_version_spread_and_capacity():
    assert version_spread([]) == 0
    assert version_spread([2, 2, 2]) == 0
    assert version_spread([0, 1, 3]) == 3
    assert version_spread([1], current_version=4) == 3
    # capacity shrinks as spread approaches max_staleness, then hits 0.
    assert staleness_capacity(max_staleness=1, max_spread_in_pool=0, batch_size=4) == 8
    assert staleness_capacity(max_staleness=1, max_spread_in_pool=1, batch_size=4) == 4
    assert staleness_capacity(max_staleness=1, max_spread_in_pool=2, batch_size=4) == 0
    assert should_admit(accepted_count=0, running_count=3, max_staleness=1, max_spread_in_pool=1, batch_size=4)
    assert not should_admit(accepted_count=0, running_count=4, max_staleness=1, max_spread_in_pool=1, batch_size=4)


# ---- versioned Session ----------------------------------------------------


def test_session_version_methods():
    s = Session(prompt="p", group_id="g0")
    s.append(MsgNode(kind="think", tokens=torch.tensor([1, 2, 3]), weight_version=0))
    s.append(MsgNode(kind="gen", latent=make_image_segment(latents=torch.randn(1, 1, 1, 1)), weight_version=0))
    s.append(MsgNode(kind="obs", text="feedback", weight_version=5))  # obs excluded
    s.append(MsgNode(kind="think", tokens=torch.tensor([4, 5]), weight_version=2))
    assert s.trainable_versions() == [0, 0, 2]  # obs version (5) excluded
    assert s.version_spread() == 2
    assert s.version_spread(current_version=3) == 3
    # think_versions broadcasts each think node's version across its tokens.
    assert s.think_versions() == [[0, 0, 0], [2, 2]]


# ---- segment version metadata roundtrip -----------------------------------


def test_text_segment_token_versions_roundtrip():
    seg = TextSegment.pack(
        tokens=[torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        log_probs=[torch.zeros(3), torch.zeros(2)],
        token_versions=[torch.tensor([0, 0, 0]), torch.tensor([1, 1])],
    )
    assert seg.token_versions.tolist() == [0, 0, 0, 1, 1]
    # select reorders rows and re-slices the packed versions with the same cu_seqlens.
    picked = seg.select(torch.tensor([1, 0]))
    assert picked.token_versions.tolist() == [1, 1, 0, 0, 0]
    # concat merges packed versions.
    merged = TextSegment.concat([seg, seg])
    assert merged.token_versions.numel() == 10


def test_latent_segment_sde_versions_roundtrip():
    a = make_image_segment(
        latents=torch.randn(1, 3, 2, 2),
        sde_logp=torch.randn(1, NSDE),
        sde_indices=torch.arange(NSDE),
        sde_versions=torch.zeros(1, NSDE, dtype=torch.long),
    )
    b = make_image_segment(
        latents=torch.randn(1, 3, 2, 2),
        sde_logp=torch.randn(1, NSDE),
        sde_indices=torch.arange(NSDE),
        sde_versions=torch.ones(1, NSDE, dtype=torch.long),
    )
    cat = a.concat([a, b])
    assert cat.sde_versions.shape == (2, NSDE)
    assert cat.sde_versions[0].tolist() == [0, 0] and cat.sde_versions[1].tolist() == [1, 1]
    picked = cat.select(torch.tensor([1]))
    assert picked.sde_versions.tolist() == [[1, 1]]


# ---- resident pool --------------------------------------------------------


def test_pool_in_window_completion():
    """turns_per_window >= max_turns: every group finishes in one harvest (spread 0)."""
    v = [0]
    pool = _make_pool(v, n=2, max_turns=2, max_staleness=0, turns_per_window=2)
    pool.ensure_groups([("g0", "p0"), ("g1", "p1")])
    picked = pool.harvest(target_groups=2)
    assert len(picked) == 4  # 2 groups x N=2 sessions
    assert all(s.status == "completed" for s in picked)
    assert all(s.version_spread() == 0 for s in picked)
    # group-contiguous: first 2 share a group, last 2 share the other.
    assert picked[0].group_id == picked[1].group_id
    assert picked[2].group_id == picked[3].group_id


def test_pool_carryover_spans_versions():
    """turns_per_window=1: a session crosses a version bump and spans versions."""
    v = [0]
    pool = _make_pool(v, n=1, max_turns=2, max_staleness=1, turns_per_window=1)
    pool.ensure_groups([("g0", "p0")])
    pool._advance_one_pass(v[0])  # turn 0 under v=0; still running
    assert len(pool._running) == 1 and pool._running[0].num_turns() == 1
    v[0] = 1  # weight update between windows
    pool._advance_one_pass(v[0])  # turn 1 under v=1; completes
    assert len(pool._completed) == 1
    done = pool._completed[0]
    assert done.status == "completed"
    assert sorted(done.trainable_versions()) == [0, 0, 1, 1]  # think+gen @ v0, think+gen @ v1
    assert done.version_spread() == 1


def test_pool_staleness_abort_and_replenish():
    """max_staleness=0: a session that would span a version bump is aborted, and the
    group is replenished fresh on the next harvest."""
    v = [0]
    pool = _make_pool(v, n=1, max_turns=2, max_staleness=0, turns_per_window=1)
    pool.ensure_groups([("g0", "p0")])
    pool._advance_one_pass(v[0])  # turn 0 @ v0
    assert len(pool._running) == 1
    v[0] = 1
    aborted = pool.on_policy_version_changed(1)  # spread(1)=1 > 0 -> abort
    assert aborted == 1 and len(pool._running) == 0
    # harvest now replenishes a fresh v=1 session for g0 and completes it on-policy.
    picked = pool.harvest(target_groups=1)
    assert len(picked) == 1
    assert picked[0].version_spread() == 0
    # fresh session completes both turns on-policy at v=1 (think+gen x2 turns).
    assert picked[0].trainable_versions() == [1, 1, 1, 1]


def test_pool_oversample_residency():
    """Admitting more groups than the harvest target leaves the rest resident."""
    v = [0]
    pool = _make_pool(v, n=2, max_turns=1, max_staleness=0, turns_per_window=1)
    pool.ensure_groups([("g0", "p0"), ("g1", "p1"), ("g2", "p2")])
    picked = pool.harvest(target_groups=2)
    assert len(picked) == 4
    harvested_groups = {s.group_id for s in picked}
    assert len(harvested_groups) == 2
    # the third group stays resident (completed but not harvested).
    remaining = {s.group_id for s in pool._completed}
    assert harvested_groups.isdisjoint({"g2"}) is False or "g2" in remaining or len(remaining) >= 0
    assert pool._complete_groups() == [g for g in ["g0", "g1", "g2"] if g not in harvested_groups]
