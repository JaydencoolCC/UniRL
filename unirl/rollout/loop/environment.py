"""Environment — the world side of an agent-loop turn (LIN-492).

See ``docs/agent-loop-design.md``. A structural ``Protocol`` seam only; concrete
environments (tool / critic) are designed separately. The loop treats it as optional.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

from unirl.types.sample import Primitive, Sample


class Environment(Protocol):
    """The world side of a turn. Optional for the loop; concrete environments are separate."""

    def reset(self, request: Sample) -> Sample:
        """Optional per-episode setup; return the (possibly augmented) request Sample."""
        ...

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        """Consume the latest action; return ``(observation, done, info)``.

        ``observation is None`` appends nothing this turn; ``done`` ends the episode.
        """
        ...

    async def astep(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        """Async :meth:`step` — the non-blocking tool boundary (LIN-522).

        Runs genuinely blocking tool I/O off the event loop (e.g. via
        ``loop.run_in_executor``) so a slow tool yields the worker's shared loop to
        sibling trajectories instead of stalling them. Must be re-entrant: one env
        instance serves many concurrent trajectories on a worker.
        """
        ...


__all__ = ["Environment"]
