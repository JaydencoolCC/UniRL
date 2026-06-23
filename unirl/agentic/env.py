"""AgenticEnv — pluggable environment that turns a turn's output into feedback.

After each think→gen turn the workflow calls ``env.step(...)`` with the turn's
decoded image and think text; the env returns an :class:`Observation` whose
``feedback_text`` is woven into the *next* turn's AR context (as an ``obs`` node)
and whose ``done`` flag can request early termination.

P0 ships :class:`RuleEnv` — a deterministic, model-free env that emits a fixed
refinement instruction each turn. It needs no GPU, no external service, and no
randomness, so it makes the multi-turn loop fully reproducible and unit-testable.
Richer envs (a VLM critic over HTTP) are a P1 concern and only need to implement
the same one-method protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, runtime_checkable


@dataclass
class Observation:
    """Environment feedback for one turn.

    ``feedback_text`` (when set) becomes an ``obs`` node — prefix-only context
    for the next turn's think, never a training segment. ``done`` requests early
    termination (honored by the workflow only when variable turns are enabled;
    P0 runs a fixed number of turns and merely records ``done``).
    """

    feedback_text: Optional[str] = None
    done: bool = False


@runtime_checkable
class AgenticEnv(Protocol):
    """One-method environment protocol.

    ``turn`` is the 0-based index of the turn that just finished. ``image`` is
    the decoded image for that turn (modality-dependent; may be ``None`` for a
    text-only env). ``think_text`` is the planner's text for that turn.
    """

    def step(self, *, image: Any, think_text: str, turn: int) -> Observation: ...


class RuleEnv:
    """Deterministic, model-free refinement env.

    Emits a fixed instruction per turn from ``instructions`` (cycling if there
    are more turns than instructions) so the planner is asked to revise its plan
    each turn. ``done`` stays ``False`` until the configured ``max_turns`` (or
    never, when ``max_turns`` is ``None``) — P0 termination is driven by the
    workflow's fixed turn count, not the env.
    """

    DEFAULT_INSTRUCTIONS: tuple[str, ...] = (
        "Reflect on the previous attempt and revise the plan to improve visual "
        "fidelity, composition, and prompt alignment. Then describe the refined image.",
        "Critique the latest image for missing or distorted details and rewrite the "
        "plan to fix them while staying faithful to the original request.",
    )

    def __init__(
        self,
        *,
        instructions: Optional[List[str]] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        self.instructions: tuple[str, ...] = (
            tuple(instructions) if instructions is not None else self.DEFAULT_INSTRUCTIONS
        )
        if not self.instructions:
            raise ValueError("RuleEnv: instructions must be non-empty (pass None to use defaults).")
        self.max_turns = max_turns

    def step(self, *, image: Any, think_text: str, turn: int) -> Observation:
        done = self.max_turns is not None and (turn + 1) >= self.max_turns
        if done:
            # Last turn: no further feedback is consumed, so emit none.
            return Observation(feedback_text=None, done=True)
        instruction = self.instructions[turn % len(self.instructions)]
        return Observation(feedback_text=instruction, done=False)


__all__ = ["AgenticEnv", "Observation", "RuleEnv"]
