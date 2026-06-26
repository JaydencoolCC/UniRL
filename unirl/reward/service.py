"""Reward service: score a response :class:`~unirl.types.sample.Sample` in place.

Holds exactly one :class:`~unirl.reward.base.RewardBackend` — a local in-process
scorer or the remote RewardService HTTP client. Builds a :class:`RewardRequest`
from the Sample's frontier Part (the generated output) plus its conditioning (the
input context), scores it, and returns a copy of the Sample with the rewards
attached to the frontier Part, under DP-sharded distributed dispatch.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.primitives import primitive_modality_key
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.types.sample import Primitive, Sample, _part_with_field
from unirl.types.sampling import ARSamplingParams

from .base import RewardBackend

logger = logging.getLogger(__name__)


def _build_reward_request(sample: Sample, preferred_input_kind: str) -> RewardRequest:
    """Assemble a :class:`RewardRequest` from a response ``Sample``.

    The frontier (last) Part is the generated output being scored; the input
    context is :meth:`Sample.conditioning` — each ancestor primitive keyed by its
    modality slot with the NEAREST ancestor winning, so a PE/recaption image
    scores against the rewrite and an it2i edit against the instruction (plain T2I
    has a single ancestor, so the choice is moot). Prompt metadata is the root's,
    aligned to the frontier (:meth:`Sample.root_metadata`). Everything is already
    row-aligned to the frontier, so there is no request/track expansion to
    reconcile.
    """
    frontier = sample.parts[-1]
    primitives: Dict[str, Primitive] = {}
    for prim in sample.conditioning():
        primitives[primitive_modality_key(prim)] = prim  # nearest ancestor wins (last)

    generated_kind = primitive_modality_key(frontier.primitive)
    if generated_kind != preferred_input_kind:
        raise ValueError(
            f"Reward backend consumes {preferred_input_kind!r} but the frontier Part generated "
            f"{generated_kind!r}; check the recipe's reward/model pairing."
        )

    metadata = sample.root_metadata(-1)
    return RewardRequest(
        primitives=primitives,
        generated={preferred_input_kind: frontier.primitive},
        prompt_ids=[str(sid) for sid in frontier.sample_ids],
        sample_ids=list(frontier.sample_ids),
        group_ids=list(frontier.group_ids),
        metadata=(metadata if any(m is not None for m in metadata) else None),
    )


class RewardService(Remote):
    """Actor-side reward entry: one backend, scores a Sample's frontier Part in place."""

    def __init__(
        self,
        backend: RewardBackend,
        truncated_reward: str = "zero",
        overlong_buffer_len: int = 4096,
        overlong_penalty_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.backend = backend
        # How to score AR generations that hit max_new_tokens (sglang finish=="length"):
        #   "zero" — force reward 0 on truncated traces (anti-ramble; the default).
        #   "keep" — keep the raw score on the partial text (= verl dapo reward manager
        #            with overlong_buffer.enable=False: no zeroing, no penalty).
        #   "soft" — verl DAPO overlong reward shaping (overlong_buffer.enable=True): a
        #            graded NEGATIVE penalty over the last `overlong_buffer_len` tokens
        #            before max_new_tokens — never a hard zero. Mirrors
        #            verl.workers.reward_manager.dapo: reward += min(-exceed/buf*factor, 0).
        self.truncated_reward = str(truncated_reward)
        self.overlong_buffer_len = int(overlong_buffer_len)
        self.overlong_penalty_factor = float(overlong_penalty_factor)
        if self.truncated_reward not in ("zero", "keep", "soft"):
            raise ValueError(f"truncated_reward must be zero|keep|soft, got {self.truncated_reward!r}")
        logger.info(
            "RewardService initialized with backend=%s, truncated_reward=%s",
            backend.get_model_name() or type(backend).__name__,
            self.truncated_reward,
        )

    @property
    def preferred_input_kind(self) -> str:
        """The decoded media kind the backend consumes (image/video/text)."""
        kind = str(getattr(self.backend, "preferred_input_kind", "") or "").strip().lower()
        if kind not in {"image", "video", "text"}:
            raise ValueError(
                f"Reward backend must expose preferred_input_kind as 'image', 'video', or 'text'. Got {kind!r}."
            )
        return kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        return self.backend.compute_rewards(request)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_and_attach(self, sample: Sample) -> Sample:
        """Score the frontier (last) Part's generated media; return the Sample with
        rewards attached on that Part.

        The frontier is the generated output; its :meth:`Sample.conditioning` is the
        input context and :meth:`Sample.root_metadata` the per-sample spec — both
        already row-aligned to the frontier, so there is no request/track expansion
        to reconcile. DP_SCATTER shards the whole Sample by prompt-tree
        (:meth:`Sample.slice`), keeping each shard's conditioning and frontier
        co-resident.

        Returns a new :class:`~unirl.types.sample.Sample` with ``rewards`` and
        ``component_rewards`` on the frontier Part; the other parts are untouched
        (the trainer credit-assigns upward via :meth:`Sample.propagate_rewards`).
        Fail-fast on per-sample failure flags so partial/corrupt rewards cannot
        silently enter advantage computation.
        """
        frontier = sample.parts[-1]
        if frontier.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the frontier Part.")
        if frontier.primitive is None:
            raise ValueError("RewardService.score_and_attach: frontier Part has no generated primitive to score.")

        request = _build_reward_request(sample, self.preferred_input_kind)
        reward_response = self.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        if failed:
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of {len(reward_response.successes)} "
                f"sample(s) as failure. First few: {failed[:3]}"
            )

        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)

        # Length-based reward shaping for AR generations that hit max_new_tokens
        # (sglang finish == "length"). A non-terminating trace whose text happens to
        # contain a matching answer (e.g. a mid-reasoning \boxed{}) can teach the
        # model to ramble up to the token cap — a real failure mode at long
        # max_new_tokens. `truncated_reward` (see __init__) picks the policy:
        #   "zero" — force reward 0 on truncated traces (anti-ramble).
        #   "keep" — leave the raw score (= verl dapo, overlong disabled). No-op here.
        #   "soft" — verl DAPO graded overlong penalty (never a hard zero).
        # Only applies when the SCORED frontier is itself an AR generation, where
        # its segment lengths are 1:1 with the rewards.
        sp = frontier.sampling_params
        if self.truncated_reward != "keep" and isinstance(sp, ARSamplingParams) and frontier.segment is not None:
            seg_lengths = getattr(frontier.segment, "lengths", None)
            if seg_lengths is not None and seg_lengths.numel() == rewards.numel():
                seg_lengths = seg_lengths.to(rewards.device).float()
                max_len = float(int(sp.max_new_tokens))
                if self.truncated_reward == "zero":
                    truncated = seg_lengths >= max_len
                    rewards = torch.where(truncated, torch.zeros_like(rewards), rewards)
                else:  # "soft": verl overlong shaping — graded negative penalty over the
                    # last overlong_buffer_len tokens before max_len, clamped to <= 0.
                    buf = float(self.overlong_buffer_len)
                    exceed = seg_lengths - (max_len - buf)
                    penalty = torch.clamp(-exceed / buf * self.overlong_penalty_factor, max=0.0)
                    rewards = rewards + penalty

        component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(reward_response.component_rewards or {}).items()
        }
        scored = _part_with_field(frontier, "rewards", rewards)
        scored = _part_with_field(scored, "component_rewards", component_rewards)
        return sample.with_parts([*sample.parts[:-1], scored])

    def is_available(self) -> bool:
        return self.backend.is_available()

    def offload(self) -> None:
        self.backend.offload()

    def onload(self) -> None:
        self.backend.onload()

    def dispose(self) -> None:
        self.backend.dispose()


__all__ = [
    "RewardService",
]
