"""Agentic (multi-turn, tool-use) rollout engine (LIN-522).

A rank-0 coordinator over a DP-replicated slab of per-worker pull-drain loops;
``generate`` returns a flat ``List[Sample]`` of variable-depth trajectories. See
``docs/async-rollout-service-design.md``.
"""

from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig
from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine

__all__ = ["AgenticRolloutEngine", "AgenticRolloutEngineConfig"]
