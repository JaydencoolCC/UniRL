"""RolloutEnginePort — the generation seam the agent loop calls (LIN-492).

See ``docs/agent-loop-design.md``. A structural ``Protocol`` for a single-turn
engine; ``BaseSingleTurnRolloutEngine`` is its nominal runtime counterpart.
"""

from __future__ import annotations

from typing import Protocol

from unirl.types.sample import Sample


class RolloutEnginePort(Protocol):
    """What the loop calls to generate one model turn."""

    def generate(self, sample: Sample) -> Sample:
        """Fill the request Sample's frontier gen Part and return it."""
        ...

    async def agenerate(self, sample: Sample) -> Sample:
        """Async per-turn core: fill the frontier gen Part and return it.

        Loop-bound engines must be awaited inside their engine-owned
        ``run_session(lambda: ...)`` context; the agentic coordinator uses that
        session for its whole drain. One turn is always one ``Sample`` (never a
        trajectory list) — that list belongs to the coordinator contract.
        """
        ...


__all__ = ["RolloutEnginePort"]
