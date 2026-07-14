"""LoopThread + NativeBackend concurrency mechanics (CPU, no sglang).

The native backend's serve/park lifecycle is the load-bearing piece of the
thread-pool rollout design: N trajectory threads submit onto SGLang's one
``engine.loop`` and must stay in flight TOGETHER (continuous batching), while
weight/memory verbs require quiesced generation and run with the loop parked.
``NativeBackend`` is constructed directly over a fake engine object (``boot``
is the only sglang-importing path), so all of this runs on CPU.
"""

import asyncio
import threading
import time

import pytest

pytest.importorskip("torch")

from unirl.rollout.engine.sglang.backends.native import NativeBackend  # noqa: E402


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached in time")


class _FakeSRTEngine:
    """The slice of ``sglang.Engine`` the backend touches: ``loop`` +
    ``async_generate`` (gated so tests control overlap deterministically)."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.release = asyncio.Event()  # loop-bound on first await (3.10+)
        self.inflight = 0
        self.peak = 0
        self.calls = 0

    async def async_generate(self, **kwargs):
        self.calls += 1
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            await self.release.wait()
        finally:
            self.inflight -= 1
        return {
            "text": str(kwargs.get("prompt", "")),
            "meta_info": {"output_token_logprobs": [(-0.5, 3)], "finish_reason": "stop"},
        }

    def open_gate(self) -> None:
        self.loop.call_soon_threadsafe(self.release.set)


def _backend(concurrency: int = 4):
    fake = _FakeSRTEngine()
    return fake, NativeBackend(fake, concurrency=concurrency, runtime={})


# ---------------------------------------------------------------------------
# Concurrent generation — the continuous-batching regression
# ---------------------------------------------------------------------------


def test_concurrent_callers_stay_in_flight_together():
    fake, backend = _backend(concurrency=4)
    results = [None] * 4
    threads = [
        threading.Thread(target=lambda i=i: results.__setitem__(i, backend.generate([{"text": f"p{i}"}])))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    _wait_until(lambda: fake.inflight == 4)  # all four callers in flight at once
    fake.open_gate()
    for t in threads:
        t.join(timeout=5)
    assert fake.peak == 4
    assert sorted(r[0].text for r in results) == ["p0", "p1", "p2", "p3"]
    backend._lt.close()


def test_shared_semaphore_bounds_inflight_across_threads():
    fake, backend = _backend(concurrency=2)
    threads = [threading.Thread(target=lambda i=i: backend.generate([{"text": f"p{i}"}])) for i in range(5)]
    for t in threads:
        t.start()
    _wait_until(lambda: fake.inflight == 2)
    fake.open_gate()  # stays set: the queued three flow straight through
    for t in threads:
        t.join(timeout=5)
    assert fake.peak == 2
    assert fake.calls == 5
    backend._lt.close()


def test_batch_generate_flattens_prompt_major():
    fake, backend = _backend(concurrency=4)
    fake.release.set()  # no gating: batch path just runs
    out = backend.generate([{"text": "a"}, {"text": "b"}, {"text": "c"}])
    assert [r.text for r in out] == ["a", "b", "c"]
    backend._lt.close()


# ---------------------------------------------------------------------------
# Serve/park lifecycle
# ---------------------------------------------------------------------------


def test_run_parked_requires_quiesced_generation():
    fake, backend = _backend(concurrency=4)
    t = threading.Thread(target=lambda: backend.generate([{"text": "p"}]))
    t.start()
    _wait_until(lambda: fake.inflight == 1)
    with pytest.raises(RuntimeError, match="quiesced generation"):
        backend._lt.run_parked(lambda: None)
    fake.open_gate()
    t.join(timeout=5)
    backend._lt.close()


def test_run_parked_leaves_loop_idle_and_reserves_after():
    fake, backend = _backend(concurrency=4)
    fake.release.set()
    backend.generate([{"text": "warm"}])  # loop thread now serving
    assert backend._lt.serving

    async def _tm_coroutine():
        return "tm-result"

    # The tokenizer-manager pattern: drive a coroutine via run_until_complete
    # on the PARKED loop (would raise "already running" if serving).
    observed = backend._lt.run_parked(lambda: (backend._lt.serving, fake.loop.run_until_complete(_tm_coroutine())))
    assert observed == (False, "tm-result")

    out = backend.generate([{"text": "again"}])  # lazily re-serves
    assert out[0].text == "again"
    assert backend._lt.serving
    backend._lt.close()


def test_close_waits_for_inflight_then_finalizes():
    fake, backend = _backend(concurrency=4)
    t = threading.Thread(target=lambda: backend.generate([{"text": "p"}]))
    t.start()
    _wait_until(lambda: fake.inflight == 1)

    finalized = threading.Event()
    closer = threading.Thread(target=lambda: backend._lt.close(finalizer=finalized.set))
    closer.start()
    closer.join(timeout=0.3)
    assert closer.is_alive() and not finalized.is_set()  # blocked on the in-flight generate

    fake.open_gate()
    closer.join(timeout=5)
    assert finalized.is_set() and not backend._lt.serving
    t.join(timeout=5)


def test_run_after_close_raises_and_controls_noop():
    fake, backend = _backend(concurrency=4)
    lt = backend._lt

    async def _control():
        return "ran"

    assert lt.run_control(_control()) is None  # parked: no-op, coroutine closed
    fake.release.set()
    backend.generate([{"text": "p"}])
    assert lt.run_control(_control()) == "ran"  # serving: rides the loop

    lt.close()
    with pytest.raises(RuntimeError, match="closed"):
        lt.run(_control())
    assert lt.run_control(_control()) is None
