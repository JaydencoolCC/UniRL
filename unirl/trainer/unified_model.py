"""UniRL v2 HunyuanImage3 unified-backbone trainer.

One shared HunyuanImage3 backbone (a single MoE transformer that operates in
``mode="gen_text"`` for AR and ``mode="gen_image"`` for DiT) trained jointly by
two algorithms — ``GRPO`` over the AR ``TextSegment`` and ``FlowGRPO``
over the DiT ``LatentSegment`` — both backward-accumulating into ONE LoRA
adapter with a single optimizer step (see :class:`UnifiedModelTrainStack`).

Two-engine design (mirrors :class:`~unirl.models.pe.pipeline.PEPipeline`'s
two-level fan-out but with the backbone shared). PE composes two in-process
child pipelines (SD3 + Qwen3, two LoRAs); HI3 instead drives TWO standalone
vLLM-Omni engine Remotes that share ONE backbone / ONE LoRA:

- ``ar_rollout`` (modality ``hi3_ar_recaption``, GPUs 0-3): original prompt → ``N``
  think/recaption texts (group-by-prompt → AR GRPO).
- ``dit_rollout`` (modality ``hi3_dit_recaption``, GPUs 4-7): each recaption → ``M``
  images of distinct noise (group-by-recaption → FlowGRPO).

The trainer assembles the lineage itself (pre-forks ``[input, ar_shell,
image_shell]`` then re-roots a flat 1:1 sub-request per engine and fills the
shells, exactly like ``ComposedRolloutEngine.generate``) because the two engines
are independent Remotes, not a composed pipeline. Reward routing then matches
:class:`~unirl.trainer.pe.PETrainer`: score the image Part, credit-assign
the mean image reward up to the AR Part, per-Part GRPO advantages, then ONE
:class:`UnifiedModelTrainStack` step (ar.loss + image.loss → one optimizer step on the
single shared LoRA).

GPU partition: each engine is ONE multi-GPU actor anchored on a distinct worker
via ``pool.create_remote(device_ids=[0])`` / ``[4]`` (NOT plain ``remote()``,
which would bind it to the whole fraction=1.0 scope and collide both engines'
device-env in one process). Each engine clears ``CUDA_VISIBLE_DEVICES`` for its
multi-GPU HI3 modality (see ``engine._HI3_MULTI_GPU_MODALITIES``) and its stage
YAML's ``runtime.devices`` pins AR→0-3 / DiT→4-7 — disjoint physical cards. The
boot-smoke anchor was unsafe only because nothing time-shared the cards; here
the colocate dance (base offloaded during rollout, engines asleep during train)
makes anchoring correct — see ``train_step`` and ``_wire_engine``.

One ``train_step``::

    wake ar+dit; [sync → both]; sample = run_rollout(sample)  # → [input, ar, image]
    sleep ar+dit
    reward.score_and_attach(sample)              # only the frontier image Part is scorable
    sample.propagate_rewards("mean")             # image reward → ar Part
    part.compute_advantages() per Part           # ar groups by prompt, image by recaption
    unified_model_stack.train_track(sample)      # tree-shard lineage → 2 backward → 1 step

Pairs with ``examples/unified_model/hi3_vllmomni.yaml`` and ``unirl/train_unified_model.py``.
Deferred (same as the reference trainers): multi-epoch replay, checkpoint /
eval cadence, structured logging.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement
from unirl.distributed.tensor import TensorRef, hydrate
from unirl.distributed.tensor.batch import Batch
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, BaseSamplingParams, DiffusionSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


def deep_hydrate(obj: Any) -> Any:
    """Materialize every ``TensorRef`` leaf in ``obj`` to a real tensor, in place.

    The anchored single-actor engines return each track as ONE transport handle
    (a single ref spanning all samples), but the train side is num_devices-way DP and
    slices each track into per-rank shards — a single ref can't be intra-handle
    sliced. Hydrating on the driver fixes the mismatch (the DP dispatch then
    re-shards real tensors), but the driver has no ``TensorTransportRuntime``
    installed, so the runtime-backed ``TensorTransport.hydrate`` is
    unavailable here. ``hydrate`` instead pulls each leaf through
    its ref's ``.materialize(backend=None)`` (a plain ``ray.get`` from the owning worker's store),
    which works from the driver — we walk the nested Batch/dict/list/TUPLE
    structure and apply it to every ``TensorRef``.

    NB: this walks TUPLES too (rebuilding them), unlike ``_collect_leaves``
    which skips them. HunyuanImage3's fused condition stores ``rope_cache`` as a
    ``tuple`` of two TensorRef; the DP scatter's driver-side
    ``Part.concat`` pads that rope (``conditions.concat`` → ``_pad_seq``
    → ``t.ndim``), so the rope MUST be real tensors here. (dp=1 never concats on
    the driver, so it never tripped on this.)
    """
    if isinstance(obj, TensorRef):
        return hydrate(obj)
    if isinstance(obj, Batch):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if v is not None:
                new = deep_hydrate(v)
                if new is not v:
                    setattr(obj, f.name, new)
        return obj
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = deep_hydrate(obj[k])
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = deep_hydrate(obj[i])
        return obj
    if isinstance(obj, tuple):
        return tuple(deep_hydrate(x) for x in obj)
    return obj


class UnifiedModelTrainer(BaseTrainer):
    """HunyuanImage3 unified-backbone joint trainer (AR + DiT, one LoRA)."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        ar_rollout_cfg: DictConfig,
        dit_rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        ar_algorithm_cfg: DictConfig,
        image_algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        dump_dir: Optional[str] = None,
        logging_cfg: Optional[DictConfig] = None,
        enable_fsdp_offload: bool = True,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        # Colocate memory dance: offload the FSDP train state (base + grads +
        # optimizer) to CPU during rollout so the awake engines fit, onload
        # before the train backward. HI3's ~150GB base needs this → default True.
        self._enable_fsdp_offload = bool(enable_fsdp_offload)

        # W&B logging (logging_cfg, wandb_logger, optimizer-step counter) is owned
        # by BaseTrainer + UniRLWandBLogger now — see super().__init__ above.

        # Intrusive debug dump: per rollout, write original prompt + AR output
        # text (= the think/recaption that conditions DiT) + decoded images +
        # rewards under ``dump_dir/rollout_<id>/``. None disables. Best-effort —
        # never breaks training (see :meth:`_dump_rollout`).
        self.dump_dir = str(dump_dir) if dump_dir else None
        self._dump_rollout_id = 0
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        # Set below from the `sync` block; None means no sync (e.g. trainside).
        self.weight_sync = None

        # Single shared slab: train backbone + both algorithms + rollout +
        # reward are siblings on one Worker (colocate; mirrors DiffusionTrainer's
        # non-separate branch).
        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)

            # Two algorithms over the SAME shared pipeline (each resolves its
            # own stage via ``stage_attr``: ar→pipeline.ar, image→pipeline.diffusion).
            self.ar_algorithm = remote_hydra(ar_algorithm_cfg, pipeline=self.pipeline)
            self.image_algorithm = remote_hydra(image_algorithm_cfg, pipeline=self.pipeline)

            # One stack owns the single backend + both algorithms → one step.
            self.stack = remote_hydra(
                stack_cfg,
                fsdp_backend=self.backend,
                ar_algorithm=self.ar_algorithm,
                image_algorithm=self.image_algorithm,
            )

            # COLOCATE MEMORY: offload the ~150GB frozen base to CPU BEFORE
            # booting the engines. Each engine grabs ~70GB (AR) / ~45GB (DiT) on
            # its 4 cards at boot; with the FSDP base still resident (~19GB/card)
            # that overlaps to >78GB and OOMs. With the base on CPU the engines
            # boot on their disjoint cards (AR 0-3, DiT 4-7) with room to spare.
            if self._enable_fsdp_offload:
                self.backend.offload()

            # Two standalone vLLM-Omni engines, each ONE multi-GPU actor anchored
            # on a DISTINCT worker (AR→device 0, DiT→device 4). The anchor is
            # load-bearing: plain remote() binds the engine to the whole
            # fraction=1.0 scope (all 8 devices, shared base worker), so BOTH
            # engines land in the same worker process and their device-env setup
            # collides — vllm-omni's set_stage_devices then remaps DiT's yaml
            # "4,5,6,7" back onto physical 0-3, overlapping AR → OOM. Anchoring on
            # separate workers keeps them in separate processes: each pops
            # CUDA_VISIBLE_DEVICES and its stage YAML's runtime.devices pins the
            # TP group to disjoint physical cards (AR 0-3, DiT 4-7) — the layout
            # boot smoke gotcha C verified. Colocate-safe because the train base is
            # offloaded during rollout and the engines sleep during train (the
            # memory dance in train_step time-shares the cards — so this is NOT
            # the boot-smoke landmine of engine+FSDP residing simultaneously).
            # DP over engine REPLICAS, one (AR, DiT) pair per node. dp = nodes
            # (16 devices / 8 per node → dp=2; single node → dp=1, fully
            # backward-compatible: range(1), anchors 0/4 = the original path).
            # Replica r is anchored on node r (DevicePool is node-aware,
            # node = device_id // devices_per_node): AR host-worker on device
            # r*8+1, DiT on r*8+4; each engine still spans cards r*8..r*8+3 /
            # r*8+4..r*8+7 via its stage YAML. AR is +1 (not r*8) to keep its host
            # worker off the train rank-0 worker (device 0) — see the push
            # self-deadlock note at the _wire_engine call below.
            per_node = self.pool.devices_per_node
            # Each replica pins ONE (AR 0-3, DiT 4-7) engine pair to a single
            # node, anchored at base+1 / base+4 with base = r*per_node. That
            # layout needs >= 8 cards on the node; with fewer, base+4 spills onto
            # the next node and silently splits the pair cross-node. Fail loud.
            if per_node < 8:
                raise ValueError(
                    "UnifiedModelTrainer: HI3 needs >= 8 devices/node for one "
                    "(AR 0-3, DiT 4-7) engine pair per node; got "
                    f"devices_per_node={per_node}."
                )
            self.dp = max(1, self.pool.num_devices // per_node)
            self.ar_rollouts = []
            self.dit_rollouts = []
            for r in range(self.dp):
                base = r * per_node
                # SERIALIZE engine boot: build one engine, then immediately
                # .sleep() it before building the next. Every @distributed Handle
                # call is synchronous (ray.get) and the heavy boot is Omni(...) in
                # the engine's __init__, so .sleep() blocks until THIS engine has
                # finished booting. Booting all dp*2 engines concurrently deadlocks
                # in the DiT warmup's kv_transfer_manager handshake (the 4-way-boot
                # blocker), so the per-engine quiesce is load-bearing — and it also
                # leaves every engine asleep, the steady state train_step expects.
                # AR anchor is base+1, NOT base: weight_sync rank 0 lives on the
                # train DP rank-0 worker = global device 0. If the AR engine were
                # anchored there too (base==0 for replica 0), it shares that one
                # worker PROCESS, and RemoteLoraWeightSync.push() — which runs on
                # rank 0 and does ray.get([... set_lora on the AR engine ...]) —
                # would block-call its own actor (the set_lora task queues behind
                # the in-flight push) → self-deadlock (push never returns, AR
                # set_lora never runs; DiT on device 4 is a separate process so it
                # loads fine). base+1 keeps the AR host worker off device 0 while
                # the engine still uses cards 0-3 via its stage YAML's runtime.devices.
                ar = self._wire_engine(ar_rollout_cfg, anchor_device=base + 1)
                ar.sleep()
                self.ar_rollouts.append(ar)
                dit = self._wire_engine(dit_rollout_cfg, anchor_device=base + 4)
                dit.sleep()
                self.dit_rollouts.append(dit)
            # Back-compat aliases for replica 0 (single-node code paths, dump,
            # debug, and any single-engine references still use these).
            self.ar_rollout = self.ar_rollouts[0]
            self.dit_rollout = self.dit_rollouts[0]

            if sync_cfg is not None:
                # LoRA sync gets ONLY the backend (a same-worker sibling); the
                # engines are cross-slab. RemoteLoraWeightSync.sync() extracts on
                # the train workers and pushes from rank 0 to EACH engine via a
                # plain Ray RPC, so hand it every replica's (role, workers) here.
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
                self.weight_sync.set_rollout_targets(
                    [(eng.role_name, eng.workers) for eng in self.ar_rollouts + self.dit_rollouts]
                )

    def _wire_engine(self, cfg: DictConfig, *, anchor_device: int) -> Any:
        """Build ONE multi-GPU vLLM-Omni engine actor anchored on one worker.

        ``device_ids=[anchor_device]`` pins the actor to a SINGLE worker (one
        process), not the whole placement scope — the engine is one TP-parallel
        Omni server, not a per-device DP replica. Inside the Omni subprocess the
        engine clears ``CUDA_VISIBLE_DEVICES`` and its stage YAML's
        ``runtime.devices`` spreads the TP group across its physical cards; using
        a distinct anchor per engine keeps the two engines' device-env setup in
        separate processes so they pin to disjoint cards (see the call site).
        The standalone HI3 engines take no ``pipeline`` (they boot their own
        Omni), so nothing sibling-handle-resolved is forwarded.
        """
        parsed = parse_hydra_cfg(cfg)
        role_cls = parsed.pop("role_cls")
        return self.pool.create_remote(role_cls, device_ids=[anchor_device], init_kwargs=parsed)

    def _build_request_sample(self, inputs: RolloutInputs, rollout_id: int) -> Sample:
        """Turn a data-source batch of ``P`` prompts into the unified request ``Sample``.

        Pre-forks the unified lineage shells ``[input, ar_shell(P*N),
        image_shell(P*N*M)]`` (located by sampling-params type); ``run_rollout``
        drives the two engines and fills these shells. ``rollout_id`` keys the
        diffusion SDE-step schedule (``scheduler`` nulled so only the concrete
        ``sde_indices`` ride) and salts the root ids. The AR sub-block has no SDE
        machinery and is left untouched.
        """
        diff_params = self.sampling_params.get("diffusion")
        ar_params = self.sampling_params.get("ar")
        sde_indices = diff_params.resolve_sde_indices(rollout_id)
        # Driver-x_T opt-out: env DISABLE_DRIVER_XT (parity with DiffusionTrainer) OR a
        # recipe-set params flag. The hi3 DiT adapter skips init_noise_group_ids when
        # set, so every engine falls back to its own RNG (the debug escape hatch).
        disable_xt = bool(os.environ.get("DISABLE_DRIVER_XT")) or bool(getattr(diff_params, "disable_driver_xt", False))
        diffusion = dataclasses.replace(
            diff_params, sde_indices=sde_indices, scheduler=None, disable_driver_xt=disable_xt
        )
        root_ids = [f"r{rollout_id}:{sid}" for sid in inputs.sample_ids]
        input_part = Part.input(
            root_ids,
            primitive=inputs.primitives["text"],
            metadata=list(inputs.metadata) if inputs.metadata else None,
        )
        return (
            Sample.request(input_part)
            .fork(ar_params.samples_per_prompt, sampling_params=ar_params)
            .fork(diffusion.samples_per_prompt, sampling_params=diffusion)
        )

    def run_rollout(self, sample: Sample) -> Sample:
        """DP rollout: scatter the ``P`` prompt-trees of the request ``Sample``
        across the ``dp`` engine replicas (one (AR, DiT) pair per node), run each
        on its replica, then ``Sample.concat`` the per-replica filled Samples.
        ``dp<=1`` or ``P<=1`` falls back to the single-replica path.

        v1 runs the replicas SEQUENTIALLY — this validates placement + the
        scatter/concat correctness; issuing the per-replica ``generate()`` as Ray
        futures for true concurrent throughput is the follow-up (handoff §8).

        CAVEAT — the fused condition's rope_cache is a ``shared_field``, so
        ``Sample.concat`` (→ ``Part.concat``) keeps replica-0's tensor verbatim:
        the merged condition carries a rope_cache whose batch dim is replica-0's
        sample count, NOT the global P*N*M. Harmless TODAY because HI3 replay
        rebuilds rope from gen_image_mask + the real latent shape and never reads
        the part's rope_cache — it only rides along in the KV-propagation kwargs.
        If a future change makes replay consume ``fused.rope_cache``, dp>1 would
        SILENTLY feed replica-0 rope to every sample (wrong gradient, no crash);
        make rope_cache a tuple-aware CONCAT field before relying on it.
        """
        n = sample.parts[0].batch_size
        if self.dp <= 1 or n <= 1:
            return self._run_rollout_one(self.ar_rollouts[0], self.dit_rollouts[0], sample)

        # Split into P per-prompt trees, regroup into dp contiguous shards.
        groups = sample.split()
        bounds = [(n * r) // self.dp for r in range(self.dp + 1)]
        shards: list[Sample] = []
        for r in range(self.dp):
            lo, hi = bounds[r], bounds[r + 1]
            if lo >= hi:
                continue
            sub = Sample.concat(groups[lo:hi])
            shards.append(self._run_rollout_one(self.ar_rollouts[r], self.dit_rollouts[r], sub))
        return Sample.concat(shards)

    def _run_rollout_one(self, ar_engine: Any, dit_engine: Any, sample: Sample) -> Sample:
        """One (AR, DiT) engine pair: fill the unified ``[input, ar, image]`` lineage.

        Drives the given ``ar_engine`` / ``dit_engine`` pair for this replica's
        prompt-trees::

            P prompts ──AR engine──▶ P*N recaptions  (root "ar", groups by prompt)
                      ──DiT engine─▶ P*N*M images     ("image", groups by recaption)

        Each engine runs FLAT (re-rooted, 1:1) — the vLLM-Omni adapters require the
        input primitive 1:1 with the gen samples — so the AR engine sees ``P*N``
        pre-expanded prompts and the DiT engine sees ``P*N*M`` (the original prompt
        plus the recaption chained as a ``cot_text`` input Part via
        :meth:`Part.input_child`). Their per-sample outputs are mapped back, by row
        order, onto the unified lineage shells (:meth:`Part.fill`) — both sides are
        group-by-parent in the same order, so the rows line up. Each image's
        ``r{rollout_id}:d{k}`` root makes its x_T per-rollout-VARYING (the engine
        derives the noise key from the gen Part ids).
        """
        input_part = sample.parts[0]
        ar_shell = sample.gen_part(ARSamplingParams)
        image_shell = sample.gen_part(DiffusionSamplingParams)
        prompts = input_part.primitive
        if not isinstance(prompts, Texts):
            raise TypeError("UnifiedModelTrainer.run_rollout: input Part.primitive must be a Texts primitive.")
        n_rec = int(ar_shell.sampling_params.samples_per_prompt)
        n_img = int(image_shell.sampling_params.samples_per_prompt)
        rid = int(self._dump_rollout_id)

        # ── Level 1: AR. P*N pre-expanded prompts, re-rooted flat (1:1). The
        # hi3_ar_recaption adapter reads its AR slice for sampling AND the diffusion
        # slice's height/width, so ship the AR shell's params whole.
        ar_texts = Texts(texts=[t for t in prompts.texts for _ in range(n_rec)])
        n_ar = len(ar_texts.texts)
        ar_input = Part.input([f"r{rid}:a{k}" for k in range(n_ar)], primitive=ar_texts, control=dict(input_part.control))
        # The hi3_ar_recaption adapter (carries_target_size) reads the diffusion gen
        # Part's height/width for the recaption prompt, so the AR request carries a
        # params-only diffusion shell ahead of the AR frontier. Only the AR stage
        # runs (stages=("ar",)), so that shell is never generated/filled — it just
        # supplies the canvas size; the "ar" output still fills the AR frontier
        # (gen Parts are located by sampling_params type, not position).
        ar_request = (
            Sample.request(ar_input)
            .fork(1, sampling_params=image_shell.sampling_params)
            .fork(1, sampling_params=ar_shell.sampling_params)
        )
        ar_out = ar_engine.generate(ar_request)
        ar_gen = ar_out.parts[-1]
        recaptions = ar_gen.primitive
        if not isinstance(recaptions, Texts) or len(recaptions.texts) != n_ar:
            got = len(recaptions.texts) if isinstance(recaptions, Texts) else type(recaptions).__name__
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: AR engine must return {n_ar} decoded Texts (= P*N); got {got}."
            )
        # Fill the unified ar shell (carries the N-grouping lineage) by row order.
        ar_part = ar_shell.fill(segment=ar_gen.segment, primitive=recaptions, conditions=dict(ar_gen.conditions))

        # ── Level 2: DiT. P*N*M pre-expanded (original prompt + recaption cot_text),
        # re-rooted flat (1:1). The recaption rides as a chained cot_text input Part.
        dit_prompts = Texts(texts=[prompts.texts[i // n_rec] for i in range(n_ar) for _ in range(n_img)])
        dit_cot = Texts(texts=[recaptions.texts[i] for i in range(n_ar) for _ in range(n_img)])
        n_total = len(dit_prompts.texts)
        # Re-root from the globally-unique image-shell lineage (flatten the path into a
        # legal root id) rather than replica-local ``d{k}``: the DiT engine derives the
        # x_T noise key from these ids, and ``d{k}`` restarts at 0 per dp>1 replica so
        # images on different replicas would collide on identical noise. The shell ids
        # are row-aligned with ``dit_prompts`` and the map-back is positional, so only
        # the noise key changes — restoring the pre-migration lineage-based key.
        dit_input = Part.input([sid.replace("/", "_") for sid in image_shell.sample_ids], primitive=dit_prompts)
        cot_input = dit_input.input_child(dit_cot)
        dit_out = dit_engine.generate(
            Sample.request(dit_input, cot_input).fork(1, sampling_params=image_shell.sampling_params)
        )
        img_gen = dit_out.parts[-1]
        if len(img_gen.sample_ids) != len(image_shell.sample_ids):
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: DiT engine returned {len(img_gen.sample_ids)} image(s) "
                f"but the image shell expects {len(image_shell.sample_ids)} (= P*N*M). The DiT engine must be 1:1."
            )
        image_part = image_shell.fill(
            segment=img_gen.segment,
            primitive=img_gen.primitive,
            conditions=dict(img_gen.conditions),
            media_preview=img_gen.media_preview,
        )

        # Each anchored engine returns its part as ONE transport handle (a single
        # ref spanning all samples); the train side is num_devices-way DP and can't
        # intra-handle-slice a single ref ("does not align to ref boundaries").
        # Materialize to real tensors on the driver here; the reward / advantage /
        # train DP dispatch then re-shards real tensors. (DiffusionTrainer dodges
        # this because its per-worker DP engine already emits one ref per rank.)
        deep_hydrate(ar_part)
        deep_hydrate(image_part)
        return Sample(parts=[input_part, ar_part, image_part])

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)`` — ``mean_reward`` is the
        mean unnormalized image reward (for the log line). ``rollout_id`` keys
        the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        # Colocate memory dance (150GB base can't coexist with an awake engine on
        # the same card). Steady state on entry: base offloaded, engines asleep.
        #   1. EXTRACT while engines ASLEEP + base ONLOADED — extract() runs a
        #      train-mesh collective whose state_dict() gathers the full FSDP model
        #      to GPU; an awake engine (~85GB) alongside the onloaded base
        #      (~19GB/card) would OOM. extract() caches the adapter on rank 0
        #      (nothing returned); offload the base again right after.
        if sync_weights and self.weight_sync is not None:
            if self._enable_fsdp_offload:
                self.backend.onload()
            self.weight_sync.extract()
            if self._enable_fsdp_offload:
                self.backend.offload()
        #   2. Wake both engines (base on CPU → room; AR 0-3, DiT 4-7 disjoint),
        #      then PUSH the cached adapter from rank 0 into each engine's
        #      set_lora_from_tensors_copy (cross-process; engines are not siblings).
        for _eng in self.ar_rollouts + self.dit_rollouts:
            _eng.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.push()
        #   3. Rollout (base offloaded), then sleep engines and onload the base
        #      for the train backward.
        sample = self.run_rollout(sample)
        for _eng in self.ar_rollouts + self.dit_rollouts:
            _eng.sleep()
        if self._enable_fsdp_offload:
            self.backend.onload()

        # Locate the two gen Parts by sampling-params type (the image Part is the
        # frontier of the unified [input, ar, image] lineage).
        ar_idx = sample.gen_part_index(ARSamplingParams)
        img_idx = sample.gen_part_index(DiffusionSamplingParams)

        # 1. Score the frontier (image) Part only — the AR TextSegment is not
        #    directly scorable; its reward is credit-assigned below. The reward
        #    derives each image's prompt context from the lineage (conditioning),
        #    so no manual req expansion is needed.
        sample = self.reward.score_and_attach(sample)
        # propagate_rewards reshapes child rewards directly (no hydration), so
        # realize the worker-returned TensorRef first.
        img_part = sample.parts[img_idx]
        if img_part.rewards is not None:
            img_part.rewards = hydrate(img_part.rewards)

        # 2. Credit-assign image reward up the lineage → fills the "ar" Part.
        sample = sample.propagate_rewards(op="mean")

        # 3. Mean image reward for the log line.
        mean_reward = 0.0
        di_rewards = sample.parts[img_idx].rewards
        if di_rewards is not None:
            mean_reward = float(hydrate(di_rewards).to(torch.float32).mean().item())

        # 3b. Intrusive debug dump (best-effort) — observe what AR generated and
        #     what DiT rendered before advantages/training mutate the Parts.
        if self.dump_dir:
            self._dump_rollout(self._dump_rollout_id, sample)

        # 4. Per-Part GRPO advantages (ar groups by prompt, image by recaption).
        new_parts = list(sample.parts)
        for idx in (ar_idx, img_idx):
            new_parts[idx] = new_parts[idx].compute_advantages(normalize=True)
        sample = sample.with_parts(new_parts)

        # Captions for the image previews fall back to the frontier-aligned prompt
        # texts (``Sample.conditioning``), so no per-track caption override is needed.
        self._drop_decoded(sample, rollout_id=rollout_id)
        # 5. Two backward (shared backbone) → one optimizer step. Pass the whole
        #    [input, ar, image] lineage so the stack DP-scatters it as a unit
        #    (Sample.chunk tree-shards both stages at the SAME prompt boundaries);
        #    passing the two Parts separately replicates the P*N*M image Part at dp>1.
        results: Dict[str, TrainStepResult] = self.stack.train_track(
            sample,
            training_progress=float(training_progress),
        )
        self.wandb_logger.log_rollout_step(
            rollout_id,
            results,
            sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics={"sync_weights": float(bool(sync_weights))},
        )

        # 6. Back to steady state (base on CPU) so the next rollout's engines
        #    have room to wake.
        if self._enable_fsdp_offload:
            self.backend.offload()
        return results, mean_reward

    def _dump_rollout(self, rollout_id: int, sample: Any) -> None:
        """Best-effort intrusive dump of one rollout to ``self.dump_dir``.

        Writes ``rollout_<id>/`` with:

        - ``samples.jsonl`` — one line per sample: original prompt, AR output
          text (the ``<think>``/``<recaption>`` that conditions DiT in
          think_recaption mode), image reward, sample/parent ids.
        - ``img_<k>.png`` — the decoded DiT image for sample ``k``.

        Wrapped so a dump failure never aborts training — observation only.
        """
        try:
            out_dir = os.path.join(self.dump_dir, f"rollout_{rollout_id}")
            os.makedirs(out_dir, exist_ok=True)

            prompts_obj = sample.parts[0].primitive
            prompts = list(prompts_obj.texts) if isinstance(prompts_obj, Texts) else []

            ar_part = next((p for p in sample.parts[1:] if isinstance(p.sampling_params, ARSamplingParams)), None)
            ar_decoded = getattr(ar_part, "primitive", None) if ar_part is not None else None
            ar_texts = list(ar_decoded.texts) if isinstance(ar_decoded, Texts) else []

            image_part = next((p for p in sample.parts[1:] if isinstance(p.sampling_params, DiffusionSamplingParams)), None)
            img_decoded = getattr(image_part, "primitive", None) if image_part is not None else None
            sample_ids = list(image_part.sample_ids) if image_part is not None else []
            parent_ids = list(image_part.group_ids) if image_part is not None else []

            rewards = None
            if image_part is not None and image_part.rewards is not None:
                rewards = hydrate(image_part.rewards).to(torch.float32).tolist()

            # Save images (best-effort): hydrate pixels and write per-sample PNGs.
            n_imgs = 0
            if img_decoded is not None and getattr(img_decoded, "pixels", None) is not None:
                from torchvision.utils import save_image

                pixels = hydrate(img_decoded.pixels).detach().to(torch.float32).clamp(0, 1).cpu()
                n_imgs = int(pixels.shape[0])
                for k in range(n_imgs):
                    save_image(pixels[k], os.path.join(out_dir, f"img_{k}.png"))

            # Two-level lineage: image sample k (0..P*N*M-1) descends from AR
            # sample k // M and original prompt k // (N*M). Index the smaller
            # prompt / recaption lists through those factors.
            ar_params = self.sampling_params.get("ar")
            diff_params = self.sampling_params.get("diffusion")
            n_rec = int(ar_params.samples_per_prompt) if ar_params is not None else 1
            n_img = max(1, int(diff_params.samples_per_prompt))
            n = max(len(sample_ids), n_imgs)
            with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
                for k in range(n):
                    p_idx = k // (n_rec * n_img)
                    a_idx = k // n_img
                    f.write(
                        json.dumps(
                            {
                                "sample_id": sample_ids[k] if k < len(sample_ids) else None,
                                "parent_id": parent_ids[k] if k < len(parent_ids) else None,
                                "prompt": prompts[p_idx] if p_idx < len(prompts) else None,
                                # In think_recaption the AR output IS the text fed
                                # into DiT (the recaption conditions the DiT stage).
                                "ar_text_fed_to_dit": ar_texts[a_idx] if a_idx < len(ar_texts) else None,
                                "image_reward": rewards[k] if (rewards is not None and k < len(rewards)) else None,
                                "image_file": f"img_{k}.png" if k < n_imgs else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            logger.info("[HI3-DUMP] rollout %d → %s (%d samples, %d images)", rollout_id, out_dir, n, n_imgs)
        except Exception as exc:  # noqa: BLE001 — dump must never break training
            logger.warning("[HI3-DUMP] rollout %d dump failed (non-fatal): %s", rollout_id, exc)

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one); ``0`` disables it. ``save_dir`` defaults to ``./checkpoints``;
        ``save_mode="auto"`` writes LoRA-only checkpoints when LoRA is active
        and full checkpoints otherwise. ``load_dir``: restore from a checkpoint
        directory and RESUME from its saved step — ``num_rollouts`` is the TOTAL
        budget.
        """
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                self._dump_rollout_id = rollout_id  # picked up by train_step's dump
                inputs = self.data_source.get_samples(self.batch_size)
                sample = self._build_request_sample(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate. The
                # HI3_SYNC_FIRST env forces a sync on rollout 0 too — a debug knob
                # to exercise the LoRA-sync path early (cheaply) without a full
                # extra rollout; the rollout-0 adapter is ~0 but that's fine for
                # testing the register→activate mechanism.
                force_sync = (resumed and rollout_id == start_rollout) or (
                    rollout_id == 0 and bool(os.environ.get("HI3_SYNC_FIRST"))
                )
                sync_weights = force_sync or (rollout_id > 0 and rollout_id % interval == 0)
                results, mean_reward = self.train_step(
                    sample,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                # Per-track console line (ar / image) with the step-0 ratio probe
                # (π_old vs π_θ alignment): on rollout 0 the LoRA is ~0 so a correct
                # replay should give ratio≈1, std≈0; a systematic offset means the
                # logp convention (temperature / top-k-p filtering / full-vs-renorm
                # softmax) doesn't match vLLM's sampler.
                self.wandb_logger.log_progress(rollout_id, num_rollouts, results, mean_reward, logger=logger)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()


__all__ = ["UnifiedModelTrainer"]
