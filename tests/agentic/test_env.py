"""RuleEnv unit tests — deterministic per-turn feedback + termination flag."""

from __future__ import annotations

import pytest

from unirl.agentic.env import AgenticEnv, Observation, RuleEnv


def test_ruleenv_is_agenticenv():
    assert isinstance(RuleEnv(), AgenticEnv)


def test_ruleenv_emits_feedback_per_turn():
    env = RuleEnv(instructions=["fix A", "fix B"])
    o0 = env.step(image="i0", think_text="t0", turn=0)
    o1 = env.step(image="i1", think_text="t1", turn=1)
    assert o0.feedback_text == "fix A"
    assert o1.feedback_text == "fix B"
    assert not o0.done and not o1.done


def test_ruleenv_cycles_instructions():
    env = RuleEnv(instructions=["only"])
    assert env.step(image=None, think_text="", turn=0).feedback_text == "only"
    assert env.step(image=None, think_text="", turn=5).feedback_text == "only"


def test_ruleenv_done_on_last_turn():
    env = RuleEnv(instructions=["x", "y", "z"], max_turns=3)
    assert not env.step(image=None, think_text="", turn=0).done
    assert not env.step(image=None, think_text="", turn=1).done
    last = env.step(image=None, think_text="", turn=2)
    assert last.done
    assert last.feedback_text is None  # no feedback consumed after the last turn


def test_ruleenv_rejects_empty_instructions():
    with pytest.raises(ValueError):
        RuleEnv(instructions=[])


def test_observation_defaults():
    o = Observation()
    assert o.feedback_text is None
    assert o.done is False
