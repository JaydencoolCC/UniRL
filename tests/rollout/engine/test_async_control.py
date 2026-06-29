"""Control-plane + provenance tests for the async rollout engine (LIN-499).

The control verbs (``abort``/``pause``/``resume``) are sync methods reached via
the raw ``Worker.call`` RPC on a *different* thread than the in-flight
``generate``; they schedule their backend coroutine onto the running engine loop
with ``run_coroutine_threadsafe`` (``_run_coro_threadsafe``). Weight-version
provenance is stamped onto the frontier gen ``Part`` by ``_stamp_weight_version``
(bumped per weight sync). Streaming is the consumer composing ``agenerate``
coroutines as-completed (the deferred driver), overlapping a finished group's
scoring with the others' generation. These exercise all three, CPU-only against a
fake backend.
"""

import asyncio
import threading

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from tests.rollout.engine._fakes import FakeEngine, build_request_batch  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402


# --------------------------------------------------------------------------- #
# abort / pause / resume
# --------------------------------------------------------------------------- #


def test_abort_sets_flag_and_inflight_groups_still_complete():
    """A driver coroutine sharing the loop with two in-flight agenerates triggers
    aabort; the abort flag is set and both groups still return filled (partials)."""
    engine = FakeEngine(concurrency=8)
    engine._backend.block_until_released = True  # park generate_one until released
    g0, g1 = build_request_batch(P=2, n=2).split()

    async def scenario():
        async def driver():
            # Let g0/g1 reach their in-flight wait, then abort to release them.
            for _ in range(8):
                await asyncio.sleep(0)
            await engine._backend.aabort(abort_all=True)

        return await asyncio.gather(engine.agenerate(g0), engine.agenerate(g1), driver())

    r0, r1, _ = engine._run_coro(scenario())

    assert engine._backend.aborted is True
    # Both groups completed despite the abort (best-effort cancel returns partials).
    assert r0.parts[-1].primitive is not None and len(r0.parts[-1].primitive.texts) == 2
    assert r1.parts[-1].primitive is not None and len(r1.parts[-1].primitive.texts) == 2

    engine.shutdown()


def test_control_verbs_scheduled_threadsafe_while_generate_drives_loop():
    """The worker_max_concurrency>1 overlap: a control RPC interleaves with an
    in-flight generate. While generate drives the loop on one (worker) thread,
    pause/resume/abort called from a second thread schedule their backend coroutine
    onto that running loop via _run_coro_threadsafe (which takes NO loop-driving
    lock) — flags flip and the parked generate is released."""
    engine = FakeEngine(concurrency=8)
    engine._backend.block_until_released = True
    batch = build_request_batch(P=1, n=2)

    result: dict = {}

    def drive():
        result["out"] = engine.generate(batch)  # drives the loop; parks in generate_one

    worker = threading.Thread(target=drive)
    worker.start()
    try:
        # Wait until the loop is actually driving a generate_one body.
        assert engine._backend.entered.wait(timeout=5.0)

        # pause()/resume() ride run_coroutine_threadsafe onto the running loop.
        engine.pause()
        assert engine._backend.paused is True
        engine.resume()
        assert engine._backend.paused is False

        # abort() schedules aabort, which releases the parked generate_one.
        engine.abort()
        assert engine._backend.aborted is True
    finally:
        worker.join(timeout=5.0)

    assert not worker.is_alive()  # the generate completed after the abort released it
    out = result["out"]
    assert out.parts[-1].primitive is not None
    assert len(out.parts[-1].primitive.texts) == 2

    engine.shutdown()


# When the loop is idle, the inherited _run_coro_threadsafe returns early and the
# unscheduled backend coroutine is dropped unawaited (the faithful real no-op path
# in sglang/engine.py) — that "coroutine was never awaited" RuntimeWarning is
# incidental to what this test asserts (flags untouched), so scope it out here.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_control_verbs_noop_when_loop_idle():
    """_run_coro_threadsafe returns without scheduling when nothing drives the loop
    — pause/resume/abort are safe no-ops with no generate in flight."""
    engine = FakeEngine(concurrency=8)

    engine.pause()
    engine.resume()
    assert engine.abort() == []
    # No coroutine ran, so the backend flags are untouched.
    assert engine._backend.paused is False
    assert engine._backend.aborted is False

    engine.shutdown()


# --------------------------------------------------------------------------- #
# streaming — the deferred driver consumes agenerate as-completed (overlap)
# --------------------------------------------------------------------------- #


def test_streaming_consumes_agenerate_as_completed_with_score_overlap():
    """The streaming contract (base.py): a consumer composes per-group agenerate
    coroutines and processes each as it completes (asyncio.as_completed), so a
    finished group is scored while the others are still generating."""
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=3, n=2)
    prompts = list(batch.parts[0].primitive.texts)  # prompt-0, prompt-1, prompt-2
    groups = batch.split()

    # Stagger completion: group 0 is submitted first but finishes LAST, so
    # as_completed must yield by completion time (streaming), not submission order.
    engine._backend.delay_for = {prompts[0]: 16, prompts[1]: 8, prompts[2]: 0}

    scored = []  # the prompt of each group, in the order it was scored
    inflight_at_first_score = []

    async def consume():
        tasks = [asyncio.ensure_future(engine.agenerate(g)) for g in groups]
        for fut in asyncio.as_completed(tasks):
            result = await fut
            if not scored:  # first group finished — the rest are still generating
                inflight_at_first_score.append(engine._backend._inflight)
            await asyncio.sleep(0)  # a trivial async "score" step over the finished group
            scored.append(result.parts[0].primitive.texts[0])

    engine._run_coro(consume())

    # Consumed in completion order (fast group first), NOT submission order.
    assert scored == [prompts[2], prompts[1], prompts[0]]
    assert set(scored) == set(prompts)  # every group streamed through exactly once
    assert inflight_at_first_score[0] >= 1  # scored while other groups were generating
    assert engine._backend.peak == 3  # all three generations overlapped in flight

    engine.shutdown()


# --------------------------------------------------------------------------- #
# weight-version provenance
# --------------------------------------------------------------------------- #


def test_part_weight_version_defaults_to_none():
    """A freshly built input Part and a forked gen shell carry no weight version."""
    head = Part.input(["p0"], primitive=Texts(texts=["hello"]))
    assert head.weight_version is None
    shell = head.fork(2, sampling_params=ARSamplingParams(samples_per_prompt=2))
    assert shell.weight_version is None


def test_stamp_marks_frontier_and_survives_split_then_concat():
    """_stamp_weight_version stamps the engine's version onto the frontier gen Part
    only, and that stamp survives a split()/concat() round-trip."""
    engine = FakeEngine(concurrency=8)
    engine._weight_version = 5
    batch = build_request_batch(P=2, n=2)

    stamped = engine._stamp_weight_version(batch)
    assert stamped.parts[-1].weight_version == 5
    assert stamped.parts[0].weight_version is None  # the input Part is untouched

    round_tripped = Sample.concat(stamped.split())
    assert round_tripped.parts[-1].weight_version == 5
    assert round_tripped == stamped

    engine.shutdown()


def test_weight_version_bump_stamps_later_gens():
    """A weight sync bumps _weight_version (the real engine does this in its
    update_weights_* verbs); generations after the bump carry the new version, so
    the two cohorts keep distinct provenance."""
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=2, n=2)

    before = engine.generate(batch)
    assert before.parts[-1].weight_version == 0  # default starting version

    engine._weight_version += 1  # a weight update bumps the counter
    after = engine.generate(batch)
    assert after.parts[-1].weight_version == 1
    assert before.parts[-1].weight_version != after.parts[-1].weight_version

    engine.shutdown()
