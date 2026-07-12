"""CPU-only fakes for exercising native sync/async rollout contracts.

No GPU, no sglang/vllm. ``FakeBackend`` mimics the native sglang ``Backend``
shape (its own asyncio loop driven under a lock; one bounding ``Semaphore``
hoisted across all groups; ``generate_one(payload) -> list`` plus the
``aabort``/``apause``/``aresume`` control coroutines), and
``FakeEngine(BaseSingleTurnRolloutEngine)`` implements both generation paths like the real
``SGLangRolloutEngine`` (build per-prompt wire -> ``gather(generate_one)`` ->
flatten -> fill the frontier gen ``Part`` -> ``_stamp_weight_version``). That
lets the tests exercise independent engine-owned sync and async paths, the shared
semaphore concurrency bound, and the ``abort``/``pause``/``resume`` +
weight-version control plane — without any real generation.

To keep ``Sample``/``Part`` equality (dataclass ``__eq__``) usable in assertions,
the fake fills only the gen ``Part``'s ``primitive`` (a tensor-free ``Texts``)
and leaves the tensor-bearing ``segment`` ``None`` — a raw ``torch.Tensor`` field
would make ``==`` raise "ambiguous truth value".
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseSingleTurnRolloutEngine
from unirl.rollout.engine.sglang.backends.base import SessionRunner
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

# --------------------------------------------------------------------------- #
# Deterministic "generation"
# --------------------------------------------------------------------------- #


def raw_text_for(prompt_text: str, k: int) -> str:
    """The decoded text for candidate ``k`` of ``prompt_text`` — a pure function,
    so a reference can be reconstructed independently of the engine run."""
    return f"{prompt_text}::cand{k}"


@dataclass
class FakeRaw:
    """Structural stand-in for the seam's ``RawResult`` (the wire fields the
    adapter consumes). Only ``text`` is read by the fake's response builder; the
    aligned ``token_ids``/``logprobs`` are carried to mirror the real shape."""

    text: str
    token_ids: List[int]
    logprobs: List[float]
    finish_reason: str = "stop"


def _raw_for(prompt_text: str, k: int) -> FakeRaw:
    return FakeRaw(
        text=raw_text_for(prompt_text, k),
        token_ids=[len(prompt_text), k],
        logprobs=[-0.1 * k, -0.2 * k],
    )


# --------------------------------------------------------------------------- #
# Fake backend — faithful to NativeBackend's loop/lock/semaphore + generate_one
# --------------------------------------------------------------------------- #


class FakeBackend:
    """The native ``Backend`` shape, in-memory: own loop, loop-driving lock, one
    shared semaphore, ``generate_one`` + the abort/pause/resume coroutines."""

    def __init__(self, *, concurrency: int, yields: int = 4) -> None:
        self._loop = asyncio.new_event_loop()
        self._session = SessionRunner(self._loop, label="FakeBackend")
        # One bound across ALL groups of a generate (binds to self._loop on first
        # use), like NativeBackend._sem — the load-bearing "shared, not per-group".
        self._sem = asyncio.Semaphore(int(concurrency))
        self.concurrency = int(concurrency)
        self._yields = int(yields)

        # Observability for the parity/overlap assertions.
        self.calls: List[Dict[str, Any]] = []  # payloads, recorded at entry order
        self._inflight = 0
        self.peak = 0
        # Optional per-prompt extra yields, keyed by payload["text"], to stagger
        # completion order for the streaming / score-as-complete overlap test
        # (default empty: no stagger, every request finishes after the same yields).
        self.delay_for: Dict[str, int] = {}

        # Control-plane flags the abort/pause/resume coroutines set.
        self.aborted = False
        self.paused = False

        # In-flight gating for the control tests: when set, generate_one parks
        # until a control coroutine releases it (the "abort returns partials" path).
        self.block_until_released = False
        self.entered = threading.Event()  # set once a generate_one body is running
        self._proceed: Optional[asyncio.Event] = None  # created on self._loop

    # ---- the single loop seam (mirrors NativeBackend) ----
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def run_session(self, factory: Any) -> Any:
        """Drive a caller-provided async session on the backend-owned loop."""
        return self._session.run(factory)

    def _proceed_event(self) -> asyncio.Event:
        if self._proceed is None:
            self._proceed = asyncio.Event()
        return self._proceed

    # ---- generation: one payload -> list of n candidates, bounded by the sem ----
    async def generate_one(self, payload: Dict[str, Any]) -> List[FakeRaw]:
        # Record BEFORE any await: with the façade's gather-of-gathers the first
        # execution of each task is in creation (= per-prompt wire) order, so this
        # is the deterministic order the whole batch's prompts were issued in.
        self.calls.append(payload)
        proceed = self._proceed_event() if self.block_until_released else None
        self.entered.set()
        n = int(payload["sampling_params"]["n"])
        async with self._sem:
            self._inflight += 1
            self.peak = max(self.peak, self._inflight)
            try:
                # Yield so sibling requests can build concurrency up to the bound;
                # the optional per-prompt extra staggers completion order.
                extra = self.delay_for.get(payload.get("text"), 0)
                for _ in range(self._yields + extra):
                    await asyncio.sleep(0)
                if proceed is not None:
                    await asyncio.wait_for(proceed.wait(), timeout=5.0)
            finally:
                self._inflight -= 1
        return [_raw_for(payload["text"], k) for k in range(n)]

    async def agenerate(self, requests: List[Dict[str, Any]]) -> List[FakeRaw]:
        nested = await asyncio.gather(*(self.generate_one(p) for p in requests))
        return [item for sublist in nested for item in sublist]

    def generate(self, requests: List[Dict[str, Any]]) -> List[FakeRaw]:
        return self.run_session(lambda: self.agenerate(requests))

    # ---- control plane: coroutines the engine schedules onto the driven loop ----
    async def aabort(self, *, abort_all: bool = True, rid: Optional[str] = None) -> None:
        self.aborted = True
        if self._proceed is not None:
            self._proceed.set()  # release any in-flight generate_one (partial return)

    async def apause(self) -> None:
        self.paused = True

    async def aresume(self) -> None:
        self.paused = False

    def abort(self, *, abort_all: bool = True, rid: Optional[str] = None) -> None:
        self._session.run_control(lambda: self.aabort(abort_all=abort_all, rid=rid))

    def pause(self) -> None:
        self._session.run_control(self.apause)

    def resume(self) -> None:
        self._session.run_control(self.aresume)

    def shutdown(self) -> None:
        self._session.close(finalizer=self._loop.close)

    def __del__(self) -> None:
        # Tests intentionally construct short-lived engines directly rather than
        # through the worker lifecycle. Mirror the production backend teardown so
        # forgotten explicit shutdowns do not leak selector sockets/event loops.
        try:
            self.shutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Fake engine — independent sync and async paths like the real SGLang engine
# --------------------------------------------------------------------------- #


class FakeEngine(BaseSingleTurnRolloutEngine):
    """Minimal single-turn engine over a backend-owned async session."""

    def __init__(self, *, concurrency: int = 8, yields: int = 4) -> None:
        self._backend = FakeBackend(concurrency=concurrency, yields=yields)
        self._weight_version = 0

    def run_session(self, factory: Any) -> Any:
        return self._backend.run_session(factory)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Native synchronous whole-Sample path through ``backend.generate``."""
        wire, _ = self._build_inputs(sample)
        raw = self._backend.generate(wire)
        return self._stamp_weight_version(self._build_response(sample, raw))

    async def agenerate(self, sample: Sample) -> Sample:
        """Run ONE prompt-group: per-prompt wire -> gather(generate_one) ->
        flatten -> fill the frontier gen Part -> stamp. Mirrors the real engine,
        awaiting ``generate_one`` directly so all groups share the one semaphore."""
        wire, _ = self._build_inputs(sample)
        raw = await self._backend.agenerate(wire)
        out = self._build_response(sample, raw)
        return self._stamp_weight_version(out)

    @staticmethod
    def _build_inputs(sample: Sample) -> Tuple[List[Dict[str, Any]], int]:
        """Adapter-like build_inputs: one ``/generate``-shaped payload per prompt,
        carrying ``n`` = the per-prompt fan-out (gen rows / #prompts)."""
        prompts = list(sample.parts[0].primitive.texts)
        n = sample.parts[-1].batch_size // len(prompts)
        wire = [{"text": p, "sampling_params": {"n": n}} for p in prompts]
        return wire, n

    @staticmethod
    def _build_response(sample: Sample, raw: List[FakeRaw]) -> Sample:
        """Adapter-like build_response: row ``j`` of the gen Part <- ``raw[j]``
        (prompt-major / group-by-parent), filling only the tensor-free primitive."""
        gen_part = sample.parts[-1]
        filled = gen_part.fill(primitive=Texts(texts=[r.text for r in raw]))
        return Sample(parts=[*sample.parts[:-1], filled])

    # ---- control plane: sync verbs scheduled onto the driven loop ----
    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        del ids
        self._backend.abort(abort_all=True)
        return []

    def pause(self) -> None:
        self._backend.pause()

    def resume(self) -> None:
        self._backend.resume()

    def shutdown(self) -> None:
        self._backend.shutdown()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def build_request_batch(*, P: int, n: int) -> Sample:
    """A multi-prompt request forked ``n`` ways: parts = [input(P), gen-shell(P*n)].

    ``Part.input`` + ``Sample.fork`` lay the lineage down group-by-parent, the
    shape ``generate``'s split -> gather -> concat round-trips over.
    """
    ids = [f"p{i}" for i in range(P)]
    prompts = [f"prompt-{i}" for i in range(P)]
    head = Part.input(ids, primitive=Texts(texts=prompts))
    request = Sample.request(head)
    return request.fork(n, sampling_params=ARSamplingParams(samples_per_prompt=n))


__all__ = [
    "FakeBackend",
    "FakeEngine",
    "FakeRaw",
    "build_request_batch",
    "raw_text_for",
]
