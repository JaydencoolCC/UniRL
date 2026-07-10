"""CPU contract tests for AsyncAgenticTrainer's producer-side logic (LIN-531).

Exercises the pure, error-prone pieces of the fully-async producer/consumer in
isolation — bucketing polled trajectories into complete GRPO groups
(``_GroupAssembler``) and the staleness-bounded freshest-first drain
(``_GroupBuffer``) — without a GPU / Ray / the full trainer. The end-to-end
disaggregated run is a GPU recipe.
"""

from __future__ import annotations

from typing import List

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from unirl.trainer.agentic_async import _GroupAssembler, _GroupBuffer  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402


def _traj(root: str) -> Sample:
    """A minimal trajectory Sample carrying just its root id (all the producer logic
    reads: ``parts[0].sample_ids[0]``)."""
    return Sample.request(Part.input([root]))


def _roots(group: List[Sample]) -> List[str]:
    return [_GroupAssembler.root_of(t) for t in group]


# --------------------------------------------------------------------------- #
# _GroupAssembler — complete-group bucketing
# --------------------------------------------------------------------------- #


def test_assembler_emits_only_when_all_siblings_terminal():
    """A group of ``n`` is emitted only once all ``n`` siblings are terminal; a partial
    root is held (its done siblings kept, not re-run) until the last one arrives."""
    asm = _GroupAssembler(n=3)
    asm.add_completed([_traj("r0"), _traj("r0")])  # 2 of 3 → incomplete
    assert asm.pop_complete_groups() == []
    assert asm.pending_roots() == {"r0"}

    asm.add_completed([_traj("r0")])  # the 3rd sibling completes the group
    groups = asm.pop_complete_groups()
    assert len(groups) == 1
    assert _roots(groups[0]) == ["r0", "r0", "r0"]
    assert asm.pending_roots() == set()  # nothing held after emit
    assert asm.pop_complete_groups() == []  # already drained


def test_assembler_buckets_by_root_across_interleaved_polls():
    """Siblings arrive interleaved across polls (completion order); grouping is by root
    id, and only the roots that reach ``n`` emit — others stay pending."""
    asm = _GroupAssembler(n=2)
    asm.add_completed([_traj("a"), _traj("b")])  # 1 each
    assert asm.pop_complete_groups() == []
    asm.add_completed([_traj("a"), _traj("c")])  # a → 2 (complete), b still 1, c 1
    groups = asm.pop_complete_groups()
    assert len(groups) == 1 and _roots(groups[0]) == ["a", "a"]
    assert asm.pending_roots() == {"b", "c"}


def test_assembler_extra_sibling_does_not_over_pack_a_group():
    """A group is exactly ``n`` even if more than ``n`` siblings of a root complete
    (defensive — the engine fans exactly ``n``, but the group is capped either way)."""
    asm = _GroupAssembler(n=2)
    asm.add_completed([_traj("a"), _traj("a"), _traj("a")])
    groups = asm.pop_complete_groups()
    assert len(groups) == 1 and len(groups[0]) == 2


# --------------------------------------------------------------------------- #
# _GroupBuffer — freshest-first drain with staleness eviction
# --------------------------------------------------------------------------- #


def test_buffer_needs_a_full_batch_then_drains_freshest_first():
    """``drain_freshest`` returns None below ``n``; at/above ``n`` it pops the freshest
    (highest gen_id) groups and carries the rest forward."""
    buf = _GroupBuffer()
    assert buf.drain_freshest(2) is None  # empty
    buf.put([_traj("a")], weight_version=0, gen_id=0)
    assert buf.drain_freshest(2) is None  # 1 < 2
    buf.put([_traj("b")], weight_version=0, gen_id=1)
    buf.put([_traj("c")], weight_version=0, gen_id=2)

    picked = buf.drain_freshest(2)
    assert picked is not None and len(picked) == 2
    assert _roots(picked[0]) == ["c"] and _roots(picked[1]) == ["b"]  # freshest first (gen_id 2, 1)
    assert buf.size() == 1  # 'a' (gen_id 0) carried forward


def test_buffer_evicts_over_stale_groups_before_draining():
    """Groups older than ``current_version - max_staleness`` are evicted first; a fresh
    group within the window survives and is drained."""
    buf = _GroupBuffer()
    buf.put([_traj("old")], weight_version=0, gen_id=0)  # staleness 3-0 = 3
    buf.put([_traj("new")], weight_version=3, gen_id=1)  # staleness 3-3 = 0

    picked = buf.drain_freshest(1, current_version=3, max_staleness=1)
    assert picked is not None and len(picked) == 1
    assert _roots(picked[0]) == ["new"]  # 'old' was evicted (3 > 1)
    assert buf.size() == 0


def test_buffer_underflow_after_eviction_returns_none():
    """If eviction drops the buffer below ``n``, drain returns None (consumer waits),
    and the evicted groups are gone (not silently trained on)."""
    buf = _GroupBuffer()
    buf.put([_traj("stale")], weight_version=0, gen_id=0)
    assert buf.drain_freshest(2, current_version=5, max_staleness=1) is None
    assert buf.size() == 0  # the stale group was evicted


def test_buffer_no_staleness_bound_keeps_everything():
    """max_staleness=None (on-policy default off) never evicts — old groups still drain."""
    buf = _GroupBuffer()
    buf.put([_traj("g0")], weight_version=0, gen_id=0)
    buf.put([_traj("g1")], weight_version=9, gen_id=1)
    picked = buf.drain_freshest(2, current_version=100, max_staleness=None)
    assert picked is not None and len(picked) == 2
