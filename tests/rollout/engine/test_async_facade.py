"""Contract/parity tests for independent sync/async rollout paths.

Concrete ``generate`` and ``agenerate`` implementations must preserve the same
whole-Sample semantics without either path calling the other. These exercise parity, the
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


def test_native_sync_and_async_paths_match_in_group_by_parent_order():
    """Both native paths fill every P*n row in group-by-parent order."""
    P, n = 3, 2
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=P, n=n)

    out = engine.generate(sample=batch)

    # Frontier gen Part filled for all P*n rows.
    gen = out.parts[-1]
    assert gen.primitive is not None
    assert len(gen.primitive.texts) == P * n

    # Independent async reference over the same whole Sample.
    ref_engine = FakeEngine(concurrency=8)
    reference = ref_engine.run_session(factory=lambda: ref_engine.agenerate(sample=batch))
    assert out == reference

    # Explicit group-by-parent expected order: prompt-major, sibling-contiguous.
    prompts = list(batch.parts[0].primitive.texts)
    expected = [raw_text_for(p, k) for p in prompts for k in range(n)]
    assert gen.primitive.texts == expected

    engine.shutdown()
    ref_engine.shutdown()


def test_generate_does_not_call_engine_agenerate():
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=2, n=2)

    async def forbidden(_sample):
        raise AssertionError("sync generate must not call engine.agenerate")

    engine.agenerate = forbidden  # type: ignore[method-assign]
    out = engine.generate(batch)
    assert out.parts[-1].primitive is not None
    engine.shutdown()


def test_agenerate_does_not_call_engine_generate():
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=2, n=2)

    def forbidden(_sample):
        raise AssertionError("agenerate must not call engine.generate")

    engine.generate = forbidden  # type: ignore[method-assign]
    out = engine.run_session(lambda: engine.agenerate(batch))
    assert out.parts[-1].primitive is not None
    engine.shutdown()


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
