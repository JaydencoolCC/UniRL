"""Contract/parity tests for the async rollout-engine façade (LIN-499).

The sync ``generate`` façade inherited from ``BaseRolloutEngine`` must stay
byte-identical to running each prompt-group's ``agenerate`` and concatenating:
split -> ``gather(agenerate)`` -> concat. These exercise that round-trip, the
per-prompt wire order the backend sees, the shared-semaphore concurrency bound,
and ``Sample`` split/concat identity — all CPU-only against a fake backend.
"""

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from tests.rollout.engine._fakes import (  # noqa: E402
    FakeEngine,
    build_request_batch,
    raw_text_for,
)
from unirl.types.sample import Sample  # noqa: E402


def test_facade_matches_per_group_reference_in_group_by_parent_order():
    """generate(batch) fills every P*n gen row, in group-by-parent order, exactly
    matching a reference built by running each group's agenerate and concatenating."""
    P, n = 3, 2
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=P, n=n)

    out = engine.generate(batch)

    # Frontier gen Part filled for all P*n rows.
    gen = out.parts[-1]
    assert gen.primitive is not None
    assert len(gen.primitive.texts) == P * n

    # Reference: run each prompt-group's agenerate (sequentially) and concat.
    ref_engine = FakeEngine(concurrency=8)
    groups = [ref_engine._run_coro(ref_engine.agenerate(g)) for g in batch.split()]
    reference = Sample.concat(groups)
    assert out == reference

    # Explicit group-by-parent expected order: prompt-major, sibling-contiguous.
    prompts = list(batch.parts[0].primitive.texts)
    expected = [raw_text_for(p, k) for p in prompts for k in range(n)]
    assert gen.primitive.texts == expected

    engine.shutdown()
    ref_engine.shutdown()


def test_backend_sees_per_prompt_wire_in_batch_order():
    """The backend receives one generate_one per prompt, in the whole batch's
    per-prompt wire order."""
    P, n = 3, 2
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=P, n=n)

    engine.generate(batch)

    prompts = list(batch.parts[0].primitive.texts)
    assert [c["text"] for c in engine._backend.calls] == prompts
    assert len(engine._backend.calls) == P  # one payload per group/prompt
    assert all(c["sampling_params"]["n"] == n for c in engine._backend.calls)

    engine.shutdown()


def test_shared_semaphore_bounds_concurrency_across_groups():
    """All groups of one generate share a single semaphore, so peak in-flight is
    bounded by the configured concurrency C — not P (or P×C) for P groups."""
    P, n, C = 4, 2, 2
    engine = FakeEngine(concurrency=C)
    batch = build_request_batch(P=P, n=n)

    engine.generate(batch)

    peak = engine._backend.peak
    assert peak <= C  # the shared bound holds across all P groups
    assert peak > 1  # but generation genuinely overlapped (not serialized)

    engine.shutdown()


def test_split_concat_round_trip_identity():
    """Sample.concat(sample.split()) reconstructs the batch exactly — the
    invariant the DP_SCATTER façade and per-group fan-out both rely on."""
    batch = build_request_batch(P=3, n=2)
    assert Sample.concat(batch.split()) == batch
