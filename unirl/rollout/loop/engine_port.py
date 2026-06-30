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

    async def agenerate(self, sample: Sample) -> Sample:
        """Async per-turn core: fill the frontier gen Part and return it.

        The seam the agentic per-trajectory loop (LIN-522) awaits each turn so the
        whole trajectory runs as a coroutine on the inner engine's event loop.
        ``BaseRolloutEngine.agenerate`` already satisfies it. One turn is always one
        ``Sample`` (never a list) — the list contract is batch-level only.
        """
        ...


__all__ = ["RolloutEnginePort"]
