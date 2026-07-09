"""Version / staleness accounting for the agentic fully-async resident pool (v5).

The fully-async layer keeps sessions resident across ``generate()`` calls, so a
single session's turns can be sampled under *different* policy versions (the
shared live model changes every optimizer step on colocate, or every weight sync
on a disaggregated slab). This module holds the small, backend-agnostic,
CPU-testable accounting helpers the pool uses to keep that bounded:

- :func:`session_versions` — the policy versions of a session's TRAINABLE nodes
  (think tokens + gen SDE steps). obs nodes never train, so they don't count.
- :func:`version_spread` — ``max - min`` over those versions (optionally folding
  in the current version, to bound how far a *still-running* session has drifted).
- :func:`staleness_capacity` / :func:`should_admit` — the AReaL-style adaptive
  admission guard: as the pool's worst version spread approaches ``max_staleness``,
  the remaining capacity for new sessions shrinks to zero (backpressure), so we
  stop over-committing fresh work that would only widen the spread.

Correctness note (see design v5 §正确性论证): versions are **metadata** — they
drive backpressure / abort / metrics, NOT the loss. The PPO ratio anchor stays
the rollout-time behavior logprob (``TextSegment.log_probs`` /
``LatentSegment.sde_logp``); GRPO / FlowGRPO are unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def node_versions(nodes: Sequence) -> List[int]:
    """Policy versions of the TRAINABLE nodes in ``nodes`` (think + gen).

    obs nodes are prefix-only context and never enter a training segment, so
    they are excluded — only versions that actually back a gradient count toward
    staleness. Each think/gen node was produced under exactly one policy version
    (a turn never straddles a weight update in the turn-granular loop), so the
    node's ``weight_version`` is the per-token / per-SDE-step version broadcast.
    """
    return [int(n.weight_version) for n in nodes if n.kind in ("think", "gen")]


def version_spread(versions: Sequence[int], current_version: Optional[int] = None) -> int:
    """``max - min`` over ``versions`` (0 if empty).

    When ``current_version`` is given it is folded in, so a session that has only
    sampled old turns but is *about* to sample under the current (newer) policy is
    already counted as drifting — the guard then refuses to advance it further if
    that would exceed ``max_staleness``.
    """
    vs = list(versions)
    if current_version is not None:
        vs = vs + [int(current_version)]
    if not vs:
        return 0
    return max(vs) - min(vs)


def staleness_capacity(*, max_staleness: int, max_spread_in_pool: int, batch_size: int) -> int:
    """AReaL-style remaining admission capacity (in sessions).

    ``capacity = (max_staleness + 1 - max_spread_in_pool) * batch_size`` clamped
    at 0. When the pool's worst spread is 0 (all on-policy) capacity is
    ``(max_staleness + 1) * batch_size``; when it reaches ``max_staleness`` only
    one more ``batch_size`` is allowed; beyond that capacity is 0 (hard
    backpressure). ``max_staleness = 0`` ⇒ strictly on-policy
    (capacity = batch_size only while spread is 0).
    """
    cap = (int(max_staleness) + 1 - int(max_spread_in_pool)) * int(batch_size)
    return max(0, cap)


def should_admit(
    *,
    accepted_count: int,
    running_count: int,
    max_staleness: int,
    max_spread_in_pool: int,
    batch_size: int,
) -> bool:
    """Whether one more fresh session may be admitted without breaking staleness.

    Mirrors AReaL's ``max_head_offpolicyness`` capacity check, adapted to the
    pool's running/accepted counts: admit only while
    ``accepted + running < staleness_capacity(...)``.
    """
    cap = staleness_capacity(max_staleness=max_staleness, max_spread_in_pool=max_spread_in_pool, batch_size=batch_size)
    return (int(accepted_count) + int(running_count)) < cap


__all__ = [
    "node_versions",
    "should_admit",
    "staleness_capacity",
    "version_spread",
]
