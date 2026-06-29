"""RolloutEnginePort — the generation seam the agent loop calls (LIN-492).

See ``docs/agent-loop-design.md``. A structural ``Protocol``; the existing synchronous
``BaseRolloutEngine`` already satisfies it, so the loop needs no engine changes.
"""

from __future__ import annotations

from typing import Protocol

from unirl.types.sample import Sample


class RolloutEnginePort(Protocol):
    """What the loop calls to generate one turn. ``BaseRolloutEngine`` already satisfies it."""

    def generate(self, sample: Sample) -> Sample:
        """Fill the request Sample's frontier gen Part and return it."""
        ...


__all__ = ["RolloutEnginePort"]
