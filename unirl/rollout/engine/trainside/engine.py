"""Trainside (in-process) rollout engine adapter.

Wraps a materialized ``models`` :class:`Pipeline` plus the trainable
stage, and exposes them as a :class:`BaseRolloutEngine`.  Used in
direct-sampling mode where the training model IS the sampler (on-policy
RL) and rollout runs in the same Python process as training — so no
worker subprocess and no weight sync are needed.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence, Union

import torch

from unirl.models.types.ar import ARStage
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.pipeline import Pipeline
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.sample import Part, Sample

Stage = Union[DiffusionStage, ARStage]


class TrainsideRolloutEngine(BaseRolloutEngine):
    """In-process rollout engine: the train actor's Pipeline IS the sampler.

    Args:
        pipeline: A materialized ``models`` pipeline whose
            ``generate(sample)`` fills the request ``Sample``'s gen Parts.
        stage: Optional pre-resolved trainable stage whose
            ``trainable_module()`` is the FSDP-wrapped model (the v1 train
            actor passes one). Takes precedence over ``stage_attrs``.
        stage_attrs: Stage attribute(s) to read off ``pipeline`` and
            eval-scope around ``generate``. A list so composed pipelines can
            drive more than one trainable module (e.g. PE's
            ``["diffusion", "ar"]``); defaults to ``("diffusion",)`` for the
            common single-diffusion engine.
        forward_batch_size: Optional intra-call chunk size for the
            ``pipeline.generate`` forward path. When set and the gen frontier
            exceeds this, ``generate`` slices the frontier Part via
            :meth:`Part.slice`, runs ``pipeline.generate`` per chunk, and
            concatenates the filled gen parts via :meth:`Part.concat`. Bounds
            stage peak memory (e.g. SD3 VAE decode) when there is no external
            inference runtime to chunk for us.
    """

    _component_name = "trainside"

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        stage: Optional[Stage] = None,
        stage_attrs: Sequence[str] = ("diffusion",),
        forward_batch_size: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        # Resolve the trainable module(s) to eval-scope around generate().
        # A pre-resolved ``stage`` (the v1 train actor passes one) wins;
        # otherwise resolve ``stage_attrs`` off the pipeline. ``stage_attrs``
        # is a list so composed pipelines eval-scope more than one trainable
        # module (e.g. PE's ["diffusion", "ar"]); the ("diffusion",) default
        # keeps the common single-diffusion case.
        if stage is not None:
            stages = [stage]
        else:
            stages = [getattr(pipeline, a) for a in stage_attrs]
        self._models = [s.trainable_module() for s in stages]
        if forward_batch_size is not None and forward_batch_size < 1:
            raise ValueError(
                f"TrainsideRolloutEngine.forward_batch_size must be >= 1 when set; got {forward_batch_size!r}"
            )
        self.forward_batch_size = forward_batch_size
        # Build a σ-schedule only when a diffusion stage is present (PE wraps
        # both diffusion + ar, so check the resolved list, not the lone `stage`
        # param which is None on the stage_attrs path); AR-only needs none.
        if any(isinstance(s, DiffusionStage) for s in stages):
            if hasattr(pipeline, "build_schedule_policy"):
                self.schedule_policy = pipeline.build_schedule_policy()
            else:
                self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
                    getattr(pipeline.bundle, "pretrained_path", None),
                    shift=float(pipeline.shift),
                )
        else:
            # AR stage — no diffusion schedule needed
            self.schedule_policy = None

        # Async per-group path: a fresh on-demand loop (no rollout backend) plus a
        # 1-permit semaphore. The in-process FSDP pipeline shares one GPU context,
        # so overlapping forwards would only contend — serialize them at 1. There
        # is no engine-side weight sync here, so the version counter stays at 0.
        self._weight_version = 0
        self._init_async_loop()
        self._sem = asyncio.Semaphore(1)

    # ------------------------------------------------------------------ #
    # Generation — async per-group core (``generate`` façade inherited from base)
    # ------------------------------------------------------------------ #

    async def agenerate(self, sample: Sample) -> Sample:
        """Run ONE prompt-group and return it with its gen Part filled.

        The per-group async core. ``pipeline.generate`` is a blocking in-process
        forward, so it runs in a worker thread (``to_thread``) to keep the engine
        loop free for the sibling ``agenerate`` coroutines the base ``generate``
        façade fans out. ``self._sem`` caps that to one forward at a time: the
        FSDP pipeline shares a single GPU context, so overlap would only contend.
        """
        async with self._sem:
            out = await asyncio.to_thread(self._generate_core, sample)
        return self._stamp_weight_version(out)

    async def _agenerate_batch(self, sample: Sample) -> Sample:
        # Sync/batch backend, not a streaming target: run the whole shard through
        # one ``_generate_core`` (the v1 whole-batch path — GPU-efficient) rather
        # than the base split→gather, which would do many small per-group forwards
        # serialized under sem=1. ``agenerate`` stays the per-group unit the
        # deferred streaming driver consumes.
        return await self.agenerate(sample)

    def _generate_core(self, sample: Sample) -> Sample:
        """Synchronous pipeline forward for one group (the former ``generate`` body)."""
        if self.schedule_policy is not None:
            self._ensure_sample_sigmas(sample)
        prev_modes = [m.training for m in self._models]
        for m in self._models:
            m.eval()
        try:
            with torch.no_grad():
                fbs = self.forward_batch_size
                gen = sample.parts[-1]
                bs = int(gen.batch_size)
                if fbs is None or bs <= fbs:
                    return self.pipeline.generate(sample)
                # Keep the (small, shared) input part(s) whole; slice the gen
                # frontier into <= fbs-row chunks, generate each, concat the filled
                # gen parts back. Mirrors SGLangDiffusionRolloutEngine.generate.
                input_parts = sample.parts[:-1]
                gen_chunks: List[Part] = []
                for start in range(0, bs, fbs):
                    end = min(start + fbs, bs)
                    chunk = self.pipeline.generate(Sample(parts=[*input_parts, gen.slice(start, end)]))
                    gen_chunks.append(chunk.parts[-1])
                    # LIN-387: no per-chunk empty_cache() — it forced allocator
                    # re-warm on the next chunk (decode 0.87s -> 2.76s spikes).
                    # Chunking alone bounds the live-tensor peak; cached blocks
                    # are reused, not leaked.
                return Sample(parts=[*input_parts, Part.concat(gen_chunks)])
        finally:
            for m, mode in zip(self._models, prev_modes):
                m.train(mode)

    def _ensure_sample_sigmas(self, sample: Sample) -> None:
        """Pin the σ schedule onto the gen part's ``DiffusionSamplingParams.sigmas``.

        Sample-shaped analogue of ``ensure_req_sigmas``: σ is the single source of
        truth, computed from the model-owned schedule policy applied to the gen
        part's (T, H, W). Shared across the part's samples (one params object). Only
        reached when a diffusion stage is present (``schedule_policy is not None``).
        """
        diffusion = sample.parts[-1].sampling_params
        if diffusion is None or diffusion.sigmas is not None:
            return
        diffusion.sigmas = self.schedule_policy.compute_sigma(
            num_inference_steps=int(diffusion.num_inference_steps),
            height=int(diffusion.height),
            width=int(diffusion.width),
        )

    def shutdown(self) -> None:
        pass

    # sleep / wake_up inherit BaseRolloutEngine's @distributed no-op default.

    def health_check(self) -> bool:
        return self.pipeline is not None and all(m is not None for m in self._models)


__all__ = ["TrainsideRolloutEngine"]
