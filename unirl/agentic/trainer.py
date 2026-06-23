"""AsyncAgenticTrainer — colocate half-async agentic trainer for Bagel.

A multi-turn generalization of :class:`~unirl.trainer.unified_model.UnifiedModelTrainer`
for ONE shared Bagel backbone: the und path plans (``think``) and the gen experts
render (``image``), both trained jointly by GRPO (think) + FlowGRPO (image) into a
single :class:`~unirl.train.unified_model_stack.UnifiedModelTrainStack` step.

Colocate / trainside (fraction=1.0): the live FSDP-wrapped pipeline IS the
sampler, driven by :class:`~unirl.agentic.engine.BagelAgenticEngine` over the
shared modules — so there is no separate inference engine and no weight sync,
the constraint that makes Bagel half-async *colocate*. The disaggregated upgrade
(resident engine on a separate slab + NCCL sync + true rollout/train overlap) is
the documented P3 path; the half-async **machinery that pays off on colocate** —
over-sampling + the zero-variance dynamic filter + the multi-track buffer — is
implemented here.

One ``train_step``::

    over-sample P' = ceil(ratio*B) prompts -> engine.generate (T-turn sessions)
    reward.score_and_attach(image track)            # PickScore on each image
    resp.propagate_rewards("mean")                  # image reward -> think (1:1)
    buffer per-prompt shards; drop zero-variance groups; drain freshest B
    think.compute_advantages (by prompt) / image (by root prompt)
    UnifiedModelTrainStack.train_track(think, image)   # 2 backward -> 1 step

Pairs with ``examples/agentic/bagel_thinkgen_rulenv.yaml`` and
``unirl/train_agentic.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.agentic.buffer import AgenticBuffer
from unirl.agentic.config import IMAGE_TRACK, THINK_TRACK
from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class AsyncAgenticTrainer(BaseTrainer):
    """Colocate multi-turn agentic trainer (Bagel think→gen, dual-algorithm)."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        ar_algorithm_cfg: DictConfig,
        image_algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        logging_cfg: Optional[DictConfig] = None,
        sessions_per_prompt: int = 2,
        max_turns: int = 2,
        over_sample_ratio: float = 1.0,
        reward_std_eps: float = 1e-6,
        buffer_max_staleness: int = 0,
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        dump_dir: Optional[str] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = int(batch_size)  # P prompts per rollout
        self.sessions_per_prompt = int(sessions_per_prompt)  # N (GRPO group size)
        self.max_turns = int(max_turns)  # T turns per session
        self.over_sample_ratio = float(over_sample_ratio)
        self.reward_std_eps = float(reward_std_eps)
        self.buffer_max_staleness = int(buffer_max_staleness)
        self.adv_normalization_scope = str(adv_normalization_scope)
        self.normalize_adv_by_std = bool(normalize_adv_by_std)
        self.dump_dir = str(dump_dir) if dump_dir else None

        if self.sessions_per_prompt < 1 or self.max_turns < 1 or self.batch_size < 1:
            raise ValueError("AsyncAgenticTrainer: batch_size / sessions_per_prompt / max_turns must be >= 1.")

        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        if "ar" not in self.sampling_params or "diffusion" not in self.sampling_params:
            raise ValueError(
                "AsyncAgenticTrainer: sampling must define both 'ar' and 'diffusion' entries "
                f"(got {sorted(self.sampling_params)})."
            )
        # Driver-tracked policy version (# weight syncs). Colocate trainside never
        # syncs, so this stays 0 (on-policy) — kept for buffer staleness parity.
        self._weight_version = 0
        self._gen_id = 0

        # ONE shared slab: backbone + both algorithms + the trainside agentic
        # engine + reward, all colocate siblings (UnifiedModelTrainer pattern,
        # minus the memory dance — the engine IS the live FSDP model).
        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            self.ar_algorithm = remote_hydra(ar_algorithm_cfg, pipeline=self.pipeline)
            self.image_algorithm = remote_hydra(image_algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(
                stack_cfg,
                fsdp_backend=self.backend,
                ar_algorithm=self.ar_algorithm,
                image_algorithm=self.image_algorithm,
            )
            # Trainside agentic engine shares the live pipeline (samples the FSDP
            # modules under eval/no_grad). Like ARTrainer's trainside wiring.
            self.rollout = remote(**parse_hydra_cfg(rollout_cfg), pipeline=self.pipeline)

    # ------------------------------------------------------------------
    # Request construction (expand P prompts -> P*N session rows)
    # ------------------------------------------------------------------

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Expand ``n_prompts`` prompts into ``n_prompts*N`` session rows.

        Each prompt's ``N`` sessions share its group id (so all ``N*T`` think rows
        of a prompt form one GRPO group). The diffusion SDE-step schedule is
        resolved per rollout and stamped onto a per-request copy (mirrors
        :meth:`UnifiedModelTrainer._build_req`).
        """
        diff_params = self.sampling_params.get("diffusion")
        sde_indices = diff_params.resolve_sde_indices(rollout_id)
        diffusion = dataclasses.replace(diff_params, sde_indices=sde_indices, scheduler=None)
        sampling_params = {**self.sampling_params, "diffusion": diffusion}

        texts = inputs.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError("AsyncAgenticTrainer._build_req: inputs.primitives['text'] must be Texts.")
        prompts = list(texts.texts)
        n = self.sessions_per_prompt
        exp_prompts = [p for p in prompts for _ in range(n)]
        exp_sample_ids = [f"{sid}/s{s}" for sid in inputs.sample_ids for s in range(n)]
        exp_group_ids = [gid for gid in inputs.group_ids for _ in range(n)]
        return RolloutReq(
            sample_ids=exp_sample_ids,
            group_ids=exp_group_ids,
            primitives={"text": Texts(texts=exp_prompts)},
            request_conditions={},
            sampling_params=sampling_params,
        )

    # ------------------------------------------------------------------
    # Reward + credit assignment
    # ------------------------------------------------------------------

    def _score(self, req: RolloutReq, resp: RolloutResp) -> RolloutResp:
        """Score the image track (PickScore) then credit-assign up to think.

        The image track is ``P'*N*T`` rows ordered (session, turn); image row ``k``
        descends from session ``k // T`` whose prompt is ``req.primitives['text']
        [k // T]`` (the expanded per-session prompt). A 1:1 reward req shards with
        the track under ``score_and_attach``'s DP_SCATTER (PETrainer pattern).
        """
        image_track = resp.tracks[IMAGE_TRACK]
        n_img = len(image_track.sample_ids)
        exp_prompts = list(req.primitives["text"].texts)
        t = self.max_turns
        reward_texts = Texts(texts=[exp_prompts[k // t] for k in range(n_img)])
        reward_req = RolloutReq(
            sample_ids=list(image_track.sample_ids),
            group_ids=list(image_track.parent_ids) if image_track.parent_ids else list(image_track.sample_ids),
            primitives={"text": reward_texts},
            request_conditions={},
            sampling_params=req.sampling_params,
        )
        scored = self.reward.score_and_attach(req=reward_req, track=image_track)
        if scored.rewards is not None:
            scored.rewards = hydrate(scored.rewards)
        resp.tracks[IMAGE_TRACK] = scored
        # Credit-assign image reward up to the think track (1:1, mean over 1).
        return resp.propagate_rewards(op="mean")

    # ------------------------------------------------------------------
    # Over-sample + dynamic filter -> a full batch of informative groups
    # ------------------------------------------------------------------

    def _collect_batch(self, rollout_id: int) -> List[RolloutResp]:
        """Generate (over-sampled), score, filter, and return ``batch_size`` shards.

        Generates ``ceil(ratio*B)`` prompts, scores + credit-assigns, splits into
        per-prompt shards, drops zero-variance groups, and drains the freshest
        ``B``. Falls back to keeping zero-variance groups only if too few
        informative ones survive (a training batch must be full).

        ``get_samples(over)`` is respected by data sources that honor the arg
        (e.g. ``DefaultDataSource``); ``MultimodalRLDataSource`` ignores it and
        yields its configured ``prompts_per_rollout`` batch — so to over-sample
        with that source, set ``prompts_per_rollout = ceil(ratio*batch_size)``.
        The filter is applied whenever more groups than ``batch_size`` arrive,
        independent of how many prompts the source actually returned.
        """
        over = max(self.batch_size, math.ceil(self.over_sample_ratio * self.batch_size))
        inputs = self.data_source.get_samples(over)
        req = self._build_req(inputs, rollout_id)
        resp = self.rollout.generate(req)
        scored = self._score(req, resp)

        buffer = AgenticBuffer(scored_track=IMAGE_TRACK, reward_std_eps=self.reward_std_eps)
        for shard in scored.split():
            buffer.put(shard, weight_version=self._weight_version, gen_id=self._gen_id)
            self._gen_id += 1

        apply_filter = buffer.size() > self.batch_size
        picked = buffer.drain_freshest(
            self.batch_size,
            current_version=self._weight_version,
            max_staleness=self.buffer_max_staleness,
            apply_filter=apply_filter,
        )
        if picked is None and apply_filter:
            logger.warning(
                "AsyncAgenticTrainer: < %d informative groups after filtering; "
                "falling back to unfiltered freshest groups.",
                self.batch_size,
            )
            picked = buffer.drain_freshest(self.batch_size, apply_filter=False)
        if picked is None:
            raise RuntimeError(
                f"AsyncAgenticTrainer: collected {buffer.size()} group(s) but need {self.batch_size}; "
                "increase over_sample_ratio or batch_size."
            )
        return [p[0] for p in picked]

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def train_step(self, rollout_id: int, training_progress: float) -> Tuple[Dict[str, TrainStepResult], float]:
        t0 = time.perf_counter()
        shards = self._collect_batch(rollout_id)
        resp = RolloutResp(
            tracks={
                THINK_TRACK: RolloutTrack.concat([s.tracks[THINK_TRACK] for s in shards]),
                IMAGE_TRACK: RolloutTrack.concat([s.tracks[IMAGE_TRACK] for s in shards]),
            }
        )

        # Mean image reward for the log line (before advantages mutate tracks).
        mean_reward = 0.0
        img_rewards = resp.tracks[IMAGE_TRACK].rewards
        if img_rewards is not None:
            mean_reward = float(hydrate(img_rewards).to(torch.float32).mean().item())

        # Per-track GRPO advantages: think groups by prompt (its N*T turns/sessions);
        # image groups by the ROOT prompt (all N*T images of a prompt) so a session
        # that beats the prompt-wide mean earns non-zero advantage.
        resp.tracks[THINK_TRACK] = resp.tracks[THINK_TRACK].compute_advantages(
            normalize=self.normalize_adv_by_std, scope=self.adv_normalization_scope
        )
        resp.tracks[IMAGE_TRACK] = resp.compute_track_advantages(
            IMAGE_TRACK, group_key="root", normalize=self.normalize_adv_by_std
        )

        # Per-image captions for the media preview (image track order = session,turn).
        image_track = resp.tracks[IMAGE_TRACK]
        n_img = len(image_track.sample_ids)
        media_prompts = {IMAGE_TRACK: [f"img-{i}" for i in range(n_img)]}
        self._drop_decoded(RolloutReq(), resp, rollout_id=rollout_id, media_prompts=media_prompts)

        results: Dict[str, TrainStepResult] = self.stack.train_track(
            resp.tracks[THINK_TRACK],
            resp.tracks[IMAGE_TRACK],
            training_progress=float(training_progress),
        )
        self.wandb_logger.log_rollout_step(rollout_id, results, resp, step_time_s=time.perf_counter() - t0)
        return results, mean_reward

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        num_rollouts: int,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "sessions_per_prompt": self.sessions_per_prompt,
                "max_turns": self.max_turns,
                "over_sample_ratio": self.over_sample_ratio,
                "buffer_max_staleness": self.buffer_max_staleness,
            },
        )
        try:
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                results, mean_reward = self.train_step(rollout_id, training_progress)
                self.wandb_logger.log_progress(rollout_id, num_rollouts, results, mean_reward, logger=logger)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()


__all__ = ["AsyncAgenticTrainer"]
