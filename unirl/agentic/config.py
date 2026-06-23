"""Config for the half-async agentic RL stack.

Two dataclasses, both plain ``@dataclass`` referenced from recipes by ``_target_``:

- :class:`AgenticWorkflowConfig` — the per-rollout multi-turn shape (how many
  turns, the system prompt, track names). Consumed by the engine + workflow.
- :class:`AgenticAsyncConfig` — the half-async knobs (over-sampling, dynamic
  filter, in-flight depth, staleness). Consumed by the trainer.

These are intentionally small and additive: every field has a default that
reproduces the simplest single-driver, on-policy, fixed-turn behavior, so a
recipe only sets what it deviates from.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical track names. "ar" (the think/plan text) is the GRPO-trained root
# track; "image" is its 1:1 diffusion child. These names match
# ``UnifiedModelTrainStack``'s hardcoded {"ar", "image"} algorithm keys so the
# dual-algorithm step consumes the agentic resp unchanged.
THINK_TRACK = "ar"
IMAGE_TRACK = "image"


@dataclass
class AgenticWorkflowConfig:
    """Per-rollout multi-turn workflow shape.

    ``max_turns`` is the FIXED number of think→gen→obs turns every session runs
    in P0 (no early termination) so each prompt's GRPO group has a uniform
    sample count — ``compute_advantages`` requires uniform group sizes. Variable
    termination is a P1 upgrade (it needs global-scope advantages or padding).
    """

    # Number of think->gen->obs turns per session (fixed in P0).
    max_turns: int = 2
    # System prompt prepended to every session's AR context (the planner role).
    # ``None`` -> the engine falls back to the model's native think system prompt.
    system_prompt: str | None = None
    # Track names (kept configurable, default to the UnifiedModelTrainStack keys).
    think_track: str = THINK_TRACK
    image_track: str = IMAGE_TRACK
    # Score every turn's image (True) or only the final turn's image (False).
    # Per-turn scoring gives a denser signal; final-only rewards pure outcome.
    score_every_turn: bool = True


@dataclass
class AgenticAsyncConfig:
    """Half-async + over-sampling knobs (slime/AReaL-style, simplified).

    ``over_sample_ratio`` generates ``ceil(ratio * batch_size)`` prompt groups
    per rollout, drops the ones with near-zero reward variance (no GRPO
    gradient signal), and keeps the freshest ``batch_size``. ``max_inflight`` /
    ``buffer_max_staleness`` mirror :class:`AsyncARTrainer`'s two knobs; on the
    colocate trainside path ``max_inflight=1`` (generation can't overlap a step
    on the same cards) and they exist for the disaggregated upgrade.
    """

    # Over-sample factor: generate this many x batch_size prompt groups, filter
    # by reward variance, keep batch_size. 1.0 = no over-sampling.
    over_sample_ratio: float = 1.0
    # Drop a prompt group whose image-reward std is <= this (zero-signal group).
    reward_std_eps: float = 1e-6
    # Concurrent in-flight generations (overlap depth). Colocate trainside: 1.
    max_inflight: int = 1
    # Weight-syncs a buffered group may cross before eviction. 0 = on-policy.
    buffer_max_staleness: int = 0


__all__ = [
    "AgenticAsyncConfig",
    "AgenticWorkflowConfig",
    "IMAGE_TRACK",
    "THINK_TRACK",
]
