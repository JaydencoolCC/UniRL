"""Tests for ToolEnvironment.astep + re-entrancy (LIN-522).

``astep`` is the non-blocking tool boundary the agentic per-worker loop awaits. It
must (1) match ``step`` (parity), (2) be re-entrant — one env instance serves many
concurrent trajectories, so the turn count must be derived per-sample, not held on
the instance — and (3) yield the event loop while a blocking tool runs, so a slow
tool doesn't stall sibling trajectories sharing the worker's loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from unirl.rollout.loop import ToolEnvironment  # noqa: E402
from unirl.rollout.loop.tools.calculator import CalculatorTool  # noqa: E402
from unirl.rollout.loop.tools.tool import Tool  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402

TOOLCALL = '<tool_call>{"name": "calculator", "arguments": {"expression": "1234 * 5678"}}</tool_call>'
ANSWER = "7006652"


def _sample_at_turn(root: str, n_turns: int, body: str = TOOLCALL) -> Sample:
    """A trajectory with ``n_turns`` filled gen Parts (frontier carries ``body``)."""
    s = Sample.request(Part.input([root], primitive=Texts(texts=[f"prompt-{root}"])))
    for i in range(n_turns):
        s = s.fork(1, sampling_params=ARSamplingParams(samples_per_prompt=1))
        s = s.with_filled_frontier(primitive=Texts(texts=[body]))
        if i < n_turns - 1:
            s = s.observe(Texts(texts=[f"obs{i}"]))
    return s


def test_astep_matches_step():
    """``astep`` is observationally identical to ``step``."""
    env = ToolEnvironment([CalculatorTool()])
    sample = _sample_at_turn("p0", 1)

    obs_s, done_s, info_s = env.step(sample)
    obs_a, done_a, info_a = asyncio.run(env.astep(sample))

    assert done_a == done_s is False
    assert isinstance(obs_a, Texts) and obs_a.texts == obs_s.texts == [ANSWER]
    assert info_a["turn"] == info_s["turn"] == 1
    assert info_a["results"] == info_s["results"] == [ANSWER]


def test_astep_is_reentrant_across_concurrent_trajectories():
    """One env instance, two concurrent trajectories at DIFFERENT depths: each
    derives its own turn (no shared-state clobbering). A stateful ``self._turn``
    would race here."""
    env = ToolEnvironment([CalculatorTool()])
    a = _sample_at_turn("p0", 1)  # turn 1
    b = _sample_at_turn("p1", 2)  # turn 2

    async def _run():
        return await asyncio.gather(env.astep(a), env.astep(b))

    (obs_a, done_a, info_a), (obs_b, done_b, info_b) = asyncio.run(_run())

    assert info_a["turn"] == 1
    assert info_b["turn"] == 2  # not clobbered by a's concurrent step
    assert obs_a.texts == obs_b.texts == [ANSWER]
    assert done_a is False and done_b is False


class _SlowTool(Tool):
    """A tool whose ``execute`` blocks (synchronously) for a beat — stands in for a
    browser/sandbox/search tool."""

    name = "slow"

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "parameters": {"type": "object", "properties": {}}},
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        del arguments
        time.sleep(0.1)
        return "slow-done"


def test_slow_astep_yields_the_loop():
    """While a blocking tool runs (off-loop via ``run_in_executor``), a sibling
    coroutine makes progress — proving ``astep`` does not stall the shared loop."""
    env = ToolEnvironment([_SlowTool()])
    slow_call = '<tool_call>{"name": "slow", "arguments": {}}</tool_call>'
    sample = _sample_at_turn("p0", 1, body=slow_call)

    order = []

    async def _run():
        task = asyncio.ensure_future(env.astep(sample))
        await asyncio.sleep(0)  # let astep dispatch its tool to the executor
        order.append("sibling")  # runs while the tool blocks in the executor thread
        result = await task
        order.append("astep-done")
        return result

    (obs, done, info) = asyncio.run(_run())

    assert order == ["sibling", "astep-done"]  # the sibling progressed during the slow tool
    assert obs.texts == ["slow-done"]
    assert done is False and info["turn"] == 1
