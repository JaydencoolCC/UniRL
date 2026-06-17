"""Bagel diffusion: typed params + per-step kernel + rollout-level stage.

Bagel (BAGEL-7B-MoT) is a unified MoT flow-matching T2I model. Unlike SD3 /
HunyuanImage3 it does not expose a dense ``predict_noise(sample, sigma)``: its
forward (``_forward_flow``) consumes a *packed* (navit) sequence plus three KV-cache
contexts (gen / cfg_text / cfg_img) and returns the CFG-combined velocity ``v_t``.

This stage reads **exactly like** :class:`unirl.models.sd3.diffusion.SD3DiffusionStage`
— it rides UniRL's shared diffusion runtime and only swaps in Bagel's velocity call:

- **σ / timestep schedule** comes from ``req.sigmas`` (pinned by the engine via
  :func:`unirl.sde.runtime.ensure_req_sigmas` from the pipeline's
  ``build_schedule_policy``), passed in as ``schedule`` — NOT computed here.
- **which steps run SDE** comes from ``params.sde_indices`` (the driver resolved it
  via :meth:`DiffusionSamplingParams.resolve_sde_indices` → the recipe's indices
  scheduler, ``unirl.utils.scheduler_utils.AllSDEScheduler``) — NOT drawn here.
- **the SDE transition + log-prob** is :class:`unirl.sde.kernels.FlowSDEStrategy`
  (``strategy.denoise``) — NOT a flow_grpo port.
- **the initial noise x_T** is the driver-authored :class:`NoiseRecipe` value passed
  as ``initial_latents`` — NOT drawn here.

The only Bagel-specific machinery left is the navit adapter: building the three
packed KV contexts (``_build_generation_inputs`` / ``_forward_kwargs``), the per-step
CFG gate (``_gated_cfg_scales``), and the velocity call (``rl_ops.forward_flow`` over
the pristine ``_forward_flow``, grad-capable via ``__wrapped__``). Bagel runs navit
``bs=1`` so packed latents are ``[seq, C]`` (no batch dim); the kernel call adds a
unit batch dim so ``FlowSDEStrategy``'s per-sample log-prob reduction matches.

The on-policy invariant: under identical weights, replay's ``new_logp`` matches the
rollout's emitted ``old_logp`` so the PPO ratio ``exp(new-old) ≈ 1`` — exactly as for
SD3, because rollout and replay use the SAME ``FlowSDEStrategy`` over the same stored
fp32 trajectory.

This module deliberately avoids importing the vendored modeling (and its hard
``flash_attn`` dependency) at module load — it reaches the model methods through the
bundle instance at call time — so ``BagelDiffusionParams`` stays CPU-importable.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch

from unirl.config.require import require
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment
from unirl.types.trajectory_store import compute_trajectory_positions
from unirl.utils.dtypes import parse_torch_dtype

from . import rl_ops
from .conditions import BagelDiffusionConditions

if TYPE_CHECKING:
    from .bundle import BagelBundle

CFG_RENORM_TYPES = ("global", "channel", "text_channel")


@dataclass
class BagelDiffusionParams(DiffusionSamplingParams):
    """Bagel diffusion knobs — one object for BOTH the trainer and the stage.

    Subclasses :class:`DiffusionSamplingParams` so a single ``sampling`` config
    object satisfies the two consumers that share it: the ``DiffusionTrainer``
    reads inherited fields (``samples_per_prompt`` for the GRPO group fan-out,
    ``height`` / ``width`` / ``seed`` / ``num_inference_steps`` / ``eta`` /
    ``init_same_noise`` / ``scheduler`` / ``sde_indices``), while
    :class:`BagelDiffusionStage` reads the Bagel-specific CFG knobs added here.

    Bagel runs **one prompt at a time** (``bs=1`` packed): ``samples_per_prompt``
    fan-out is materialized by the trainer (``RolloutInputs.expand``) into separate
    samples, not batched inside one ``_forward_flow``.

    The SDE machinery is now **all inherited / central**: ``eta`` is the SDE noise
    scale (flow_grpo's ``noise_level``); ``scheduler`` (the recipe's
    :class:`~unirl.utils.scheduler_utils.AllSDEScheduler`) picks the SDE steps via
    :meth:`resolve_sde_indices`; the σ schedule rides on ``req.sigmas``. No
    Bagel-specific window / schedule fields remain.
    """

    # Override base defaults for Bagel. ``num_inference_steps`` is the number of
    # STEPS (the σ schedule has steps+1 points); BAGEL's flow_grpo setup uses a
    # 15-point schedule → 14 steps.
    num_inference_steps: int = 14
    guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    # SDE noise scale (== flow_grpo's ``noise_level``); consumed by FlowSDEStrategy
    # as ``eta``. Inherited field, surfaced here for the BAGEL default.
    eta: float = 1.0

    # Bagel-specific CFG knobs (consumed by the navit ``_forward_flow``).
    cfg_text_scale: float = 1.0
    cfg_img_scale: float = 1.0
    cfg_interval: Tuple[float, float] = (0.0, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
    cfg_type: str = "parallel"

    def __post_init__(self) -> None:
        super().__post_init__()
        require(
            int(self.num_inference_steps) >= 2,
            f"BagelDiffusionParams.num_inference_steps must be >= 2; got {self.num_inference_steps}",
        )
        require(
            self.cfg_renorm_type in CFG_RENORM_TYPES,
            f"BagelDiffusionParams.cfg_renorm_type must be one of {CFG_RENORM_TYPES}; got {self.cfg_renorm_type!r}",
        )


def _to_device(d: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move every tensor value in a ``prepare_vae_latent*`` dict onto ``device``.

    The vendored ``prepare_vae_latent`` / ``prepare_vae_latent_cfg`` build their
    packed index tensors on CPU; the MoT forward needs them on the model device.
    Non-tensor values pass through untouched.
    """
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


class BagelDiffusionStep:
    """Per-step Bagel kernel — stateless navit adapter over the shared SDE strategy.

    Splits the step into the two Bagel-specific halves SD3 also has:
    :meth:`predict_velocity` (the navit ``_forward_flow`` call, with CFG done inside
    the model) and :meth:`denoise` (the shared :class:`StepStrategy.denoise`, with a
    unit batch dim added so the packed ``[seq, C]`` latent gets a per-sample log-prob).
    :meth:`step_with_logp` runs both, mirroring ``SD3DiffusionStep.step_with_logp``.
    """

    def predict_velocity(
        self,
        bagel: Any,
        *,
        x_t: torch.Tensor,
        t_cur: torch.Tensor,
        cfg_text_scale: float,
        cfg_img_scale: float,
        forward_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """CFG-combined velocity ``v_t`` for packed ``x_t`` ``[seq, C]`` at time ``t_cur``.

        ``_forward_flow`` takes a per-token ``timestep`` ``[seq]`` (all equal to the
        scalar ``t_cur``) and does the CFG combine internally (gen / cfg_text /
        cfg_img contexts in ``forward_kwargs`` + the gated scales).
        """
        # The pristine ``_forward_flow`` reads ``language_model.model.enable_taylorseer``
        # (the official ``generate_image`` sets it; the RL path calls ``_forward_flow``
        # directly). Set it here — the single chokepoint before every velocity call —
        # so the TaylorSeer cache is off (per-step determinism for replay). Idempotent.
        rl_ops.disable_inference_cache(bagel)
        seq = int(x_t.shape[0])
        timestep = torch.full((seq,), float(t_cur), device=x_t.device)
        return rl_ops.forward_flow(
            bagel,
            x_t=x_t,
            timestep=timestep,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            **forward_kwargs,
        )

    def denoise(
        self,
        strategy: StepStrategy,
        *,
        v_t: torch.Tensor,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_max: torch.Tensor,
        eta: float,
        prev_sample: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """One SDE transition via the shared ``strategy.denoise`` over packed latents.

        ``n_samples == 1`` (legacy, navit bs=1): packed Bagel latents are ``[seq, C]``
        (no batch dim), but ``FlowSDEStrategy`` reduces the per-element log-prob over
        dims ``>=1`` to get ONE scalar per sample. So we add a unit batch dim
        (``[1, seq, C]``) and squeeze it back: log-prob = mean over all ``seq*C``
        elements (== flow_grpo's ``log_prob.mean()``). Returns ``(prev_sample
        [seq, C], log_prob scalar, prev_sample_mean [seq, C])``.

        ``n_samples == B > 1`` (forward_batch_size pack-B): the packed ``[B*seq, C]``
        (B same-shape images) reshapes to ``[B, seq, C]``; the SAME per-element
        reduction then yields ``[B]`` per-image log-probs and ``[B, seq, C]`` means,
        and the kernel draws independent per-image SDE noise. ``prev_sample`` is
        returned re-packed as ``[B*seq, C]`` for the next forward.

        ``log_prob`` / ``prev_sample_mean`` are ``None`` for deterministic
        (``eta < 1e-7``) steps.
        """
        if n_samples == 1:
            prev, log_prob, prev_mean = strategy.denoise(
                noise_pred=v_t.unsqueeze(0),
                sample=x_t.unsqueeze(0),
                sigma=sigma,
                sigma_next=sigma_next,
                eta=float(eta),
                prev_sample=None if prev_sample is None else prev_sample.unsqueeze(0),
                sigma_max=float(sigma_max),
            )
            return (
                prev.squeeze(0),
                None if log_prob is None else log_prob.reshape(()),
                None if prev_mean is None else prev_mean.squeeze(0),
            )
        seq = int(x_t.shape[0]) // n_samples
        c = int(x_t.shape[-1])
        prev, log_prob, prev_mean = strategy.denoise(
            noise_pred=v_t.reshape(n_samples, seq, c),
            sample=x_t.reshape(n_samples, seq, c),
            sigma=sigma,
            sigma_next=sigma_next,
            eta=float(eta),
            prev_sample=None if prev_sample is None else prev_sample.reshape(n_samples, seq, c),
            sigma_max=float(sigma_max),
        )
        return (
            prev.reshape(-1, c),
            log_prob,  # [B]
            prev_mean,  # [B, seq, C]
        )

    def step_with_logp(
        self,
        bagel: Any,
        strategy: StepStrategy,
        *,
        x_t: torch.Tensor,
        prev_sample: Optional[torch.Tensor],
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        sigma_max: torch.Tensor,
        eta: float,
        cfg_text_scale: float,
        cfg_img_scale: float,
        forward_kwargs: Dict[str, Any],
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run ``predict_velocity`` then ``denoise`` for one step.

        ``prev_sample=None`` ⇒ sampling (draws a fresh next sample); ``prev_sample``
        set ⇒ replay (log-prob of the stored transition). ``n_samples`` packs B
        same-shape images per forward (forward_batch_size>1); 1 = legacy navit bs=1.
        Returns ``(prev_sample, log_prob, prev_sample_mean)``.
        """
        v_t = self.predict_velocity(
            bagel,
            x_t=x_t,
            t_cur=t_cur,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            forward_kwargs=forward_kwargs,
        )
        return self.denoise(
            strategy,
            v_t=v_t,
            x_t=x_t,
            sigma=t_cur,
            sigma_next=t_next,
            sigma_max=sigma_max,
            eta=eta,
            prev_sample=prev_sample,
            n_samples=n_samples,
        )


class BagelDiffusionStage(DiffusionStage[BagelDiffusionConditions]):
    """Bagel rollout-level diffusion stage (trainside A1) — central-runtime, SD3-shaped.

    Owns the bundle, the per-step navit kernel, the SDE ``strategy``, and the
    precision policy. ``diffuse`` runs the full sampling loop over the engine-pinned
    ``schedule`` (``req.sigmas``), recording SDE log-probs at ``params.sde_indices``;
    ``replay`` recomputes those log-probs for GRPO. Reuses
    ``unirl.algorithms.flowgrpo.FlowGRPO`` unchanged.

    Implements the ``DiffusionStage`` protocol (``diffuse`` / ``replay`` /
    ``predict_noise_at_step``) so the trainside engine's ``isinstance(stage,
    DiffusionStage)`` check passes → it builds the σ-schedule policy from
    :meth:`BagelPipeline.build_schedule_policy` and pins ``req.sigmas`` via
    ``ensure_req_sigmas`` before ``generate`` (same path as SD3 / Wan / Qwen-Image).
    """

    def __init__(
        self,
        *,
        model: "BagelBundle",
        step: Optional[BagelDiffusionStep] = None,
        strategy: Optional[StepStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp32",
        logprob_precision: str = "fp32",
    ) -> None:
        # ``model`` is the bundle (kept name-compatible with the other stages so
        # the pipeline / FSDPPolicy treat it uniformly). The Bagel nn.Module is
        # ``model.model``; the trainable MoT is ``model.transformer``.
        self.model = model
        self.step = step if step is not None else BagelDiffusionStep()
        self.strategy = strategy if strategy is not None else FlowSDEStrategy()
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    # ------------------------------------------------------------------
    # Helpers (navit adapter)
    # ------------------------------------------------------------------

    def _autocast_ctx(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    def _build_contexts(self, text: str) -> Tuple[Any, Any, Any]:
        """Rebuild the (gen, cfg_text, cfg_img) KV contexts from conditioning ``text``.

        Mirrors the former ``BagelPipeline._build_contexts`` byte-for-byte: ``cfg_text``
        is the init snapshot taken *before* the text (unconditional); ``gen`` and
        ``cfg_img`` both ingest the text (``cfg_img == gen`` for pure T2I). Runs the
        und path under ``torch.no_grad`` + autocast, so the contexts are detached
        constants the image forward consumes (image loss never flows into the und
        experts) and the build is deterministic under fixed weights — rollout
        (``diffuse``) and replay reproduce byte-identical contexts → ratio ≈ 1.

        Rebuilding from text ON THE WORKER (instead of carrying the prebuilt KV caches
        on the track) is what keeps the large, opaque ``NaiveCache`` objects off the
        DP round-trip and off the driver's cuda:0. See ``BagelDiffusionConditions``.
        """
        inf = self.model.inferencer
        device = torch.device(self.model.device)
        # The prefill below goes through inferencer.update_context_text ->
        # forward_cache_update_text -> language_model.forward_inference, but
        # Qwen2Model.forward dispatches train-vs-inference on ``self.training``.
        # During replay/MSE the MoT is in train() mode (the AR teacher-force set
        # it), so without forcing eval() the inference prefill is routed to
        # forward_train -> "unexpected keyword argument 'packed_query_sequence'".
        # Mirror rl_ops.forward_flow: force eval() for the inference-path prefill,
        # restore the prior mode after (this build is always under no_grad, so no
        # backward recompute depends on the mode persisting). See rl_ops.py.
        # ``self.model`` is the BagelBundle; ``.transformer`` IS ``model.language_model``
        # (bundle.py), the same Qwen2ForCausalLM rl_ops toggles. .eval()/.train()
        # recurse into the inner Qwen2Model whose forward() dispatches on .training.
        lm = self.model.transformer
        was_training = lm.training
        if was_training:
            lm.eval()
        try:
            gen = inf.init_gen_context()
            cfg_img = deepcopy(gen)
            with torch.no_grad(), self._autocast_ctx(device):
                cfg_text = deepcopy(gen)  # snapshot before the text → unconditional
                gen = inf.update_context_text(text, gen)
                cfg_img = inf.update_context_text(text, cfg_img)
        finally:
            if was_training:
                lm.train()
        return gen, cfg_text, cfg_img

    def _build_contexts_batch(self, texts: List[str]) -> Tuple[Any, Any, Any]:
        """B-image analog of :meth:`_build_contexts` (forward_batch_size pack-B).

        Builds B conditioning contexts in ONE packed und prefill: ``prepare_prompts``
        zips ``prompts`` with per-context ``curr_kvlens`` / ``curr_rope`` (each starts
        from 0) and ``forward_cache_update_text`` populates one shared NaiveCache
        holding all B prompt KVs (block-diagonal; kv_lens = [L₁..L_B]). ``gen`` ==
        ``cfg_img`` (pure T2I); ``cfg_text`` is B empty (unconditional) contexts. Same
        eval()/no_grad/autocast contract as the single-text path so contexts are
        deterministic detached constants (rollout/replay reproduce them). Inputs stay
        on CPU; the bundle's accelerate hooks move them to device (as update_context_text
        does).
        """
        inf = self.model.inferencer
        bagel = self.model.model
        device = torch.device(self.model.device)
        n = len(texts)
        lm = self.model.transformer
        was_training = lm.training
        if was_training:
            lm.eval()
        try:
            with torch.no_grad(), self._autocast_ctx(device):
                gi, kv_lens, ropes = bagel.prepare_prompts(
                    curr_kvlens=[0] * n,
                    curr_rope=[0] * n,
                    prompts=list(texts),
                    tokenizer=inf.tokenizer,
                    new_token_ids=self.model.new_token_ids,
                )
                pkv = bagel.forward_cache_update_text(inf.init_gen_context()["past_key_values"], **gi)
                gen = {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": pkv}
                cfg_img = gen  # cfg_img == gen for pure T2I
                cfg_text = {
                    "kv_lens": [0] * n,
                    "ropes": [0] * n,
                    "past_key_values": inf.init_gen_context()["past_key_values"],
                }
        finally:
            if was_training:
                lm.train()
        return gen, cfg_text, cfg_img

    def _build_generation_inputs(
        self,
        gen: Any,
        cfg_text: Any,
        cfg_img: Any,
        image_shapes: List[Tuple[int, int]],
        *,
        device: torch.device,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Reconstruct the packed gen / cfg_text / cfg_img inputs from the contexts.

        Deterministic given (context kv_lens / ropes, image_shapes) — the same packed
        index tensors ``_forward_flow`` consumes. ``image_shapes`` carries ONE (H, W) per
        packed image (length 1 = the legacy per-sample path; >1 = the forward_batch_size
        pack-B path): the vendored ``prepare_vae_latent`` zips it with the per-image
        ``kv_lens`` / ``ropes`` → one block-diagonal packed sequence for all images.
        ``gi["packed_init_noises"]`` is the vendored default x_T draw; ``diffuse``
        overrides it with the driver-authored :class:`NoiseRecipe` value
        (``initial_latents``) when present, otherwise uses it as the fallback.
        """
        bagel = self.model.model
        gi = bagel.prepare_vae_latent(
            curr_kvlens=gen["kv_lens"],
            curr_rope=gen["ropes"],
            image_sizes=image_shapes,
            new_token_ids=self.model.new_token_ids,
        )
        gi_cfg_text = bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text["kv_lens"], curr_rope=cfg_text["ropes"], image_sizes=image_shapes
        )
        gi_cfg_img = bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img["kv_lens"], curr_rope=cfg_img["ropes"], image_sizes=image_shapes
        )
        return _to_device(gi, device), _to_device(gi_cfg_text, device), _to_device(gi_cfg_img, device)

    def _forward_kwargs(
        self,
        gen: Any,
        cfg_text: Any,
        cfg_img: Any,
        gi: Dict[str, Any],
        gi_cfg_text: Dict[str, Any],
        gi_cfg_img: Dict[str, Any],
        params: BagelDiffusionParams,
    ) -> Dict[str, Any]:
        """Static (step-invariant) kwargs for ``_forward_flow``.

        Everything except ``x_t`` / ``timestep`` / the per-step CFG scales. The three
        ``past_key_values`` come from the conditions' contexts; the packed index
        tensors come from the (device-pinned) generation inputs.
        """
        return dict(
            packed_vae_token_indexes=gi["packed_vae_token_indexes"],
            packed_vae_position_ids=gi["packed_vae_position_ids"],
            packed_text_ids=gi["packed_text_ids"],
            packed_text_indexes=gi["packed_text_indexes"],
            packed_position_ids=gi["packed_position_ids"],
            packed_indexes=gi["packed_indexes"],
            packed_seqlens=gi["packed_seqlens"],
            key_values_lens=gi["key_values_lens"],
            past_key_values=gen["past_key_values"],
            packed_key_value_indexes=gi["packed_key_value_indexes"],
            cfg_renorm_min=params.cfg_renorm_min,
            cfg_renorm_type=params.cfg_renorm_type,
            cfg_text_packed_position_ids=gi_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=gi_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=gi_cfg_text["cfg_key_values_lens"],
            cfg_text_past_key_values=cfg_text["past_key_values"],
            cfg_text_packed_key_value_indexes=gi_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=gi_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=gi_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=gi_cfg_img["cfg_key_values_lens"],
            cfg_img_past_key_values=cfg_img["past_key_values"],
            cfg_img_packed_key_value_indexes=gi_cfg_img["cfg_packed_key_value_indexes"],
            cfg_type=params.cfg_type,
        )

    def _resolve_contexts(
        self,
        conditions: BagelDiffusionConditions,
        *,
        params: BagelDiffusionParams,
        device: torch.device,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Turn (lightweight) ``conditions`` into the inputs ``_forward_flow`` consumes.

        The single chokepoint: ``conditions.single()`` → text → rebuild the 3 KV
        contexts (:meth:`_build_contexts`, the und-path prefill) → pack the per-sample
        index tensors (:meth:`_build_generation_inputs`) → assemble the step-invariant
        ``forward_kwargs``. Returns ``(gi, forward_kwargs)``; ``gi`` is surfaced only so
        :meth:`diffuse` can read ``gi["packed_init_noises"]`` for the fallback x_T.
        Called once per :meth:`diffuse` / :meth:`replay`; the velocity-MSE caller uses
        :meth:`build_forward_kwargs` to build ``forward_kwargs`` once and reuse it.
        """
        text, image_shape = conditions.single()
        gen, cfg_text, cfg_img = self._build_contexts(text)
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs(gen, cfg_text, cfg_img, [image_shape], device=device)
        forward_kwargs = self._forward_kwargs(gen, cfg_text, cfg_img, gi, gi_cfg_text, gi_cfg_img, params)
        return gi, forward_kwargs

    @staticmethod
    def _gated_cfg_scales(t_value: float, params: BagelDiffusionParams) -> Tuple[float, float]:
        """CFG scales after the per-step ``cfg_interval`` gate (matches generate_image)."""
        lo, hi = float(params.cfg_interval[0]), float(params.cfg_interval[1])
        if lo < t_value <= hi:
            return float(params.cfg_text_scale), float(params.cfg_img_scale)
        return 1.0, 1.0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: BagelDiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: BagelDiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run Bagel sampling over the engine-pinned ``schedule``; return a segment.

        Mirrors ``SD3DiffusionStage.diffuse``: loops ``T = len(schedule) - 1`` steps,
        records an SDE log-prob (``eta`` noise) at every ``i in params.sde_indices``
        and runs deterministic Euler (``eta = 0``) elsewhere. ``initial_latents`` is
        the driver-authored x_T (:class:`NoiseRecipe`); when ``None`` the vendored
        ``prepare_vae_latent`` draw is used (tests / no driver recipe).

        The returned ``LatentSegment`` (``S = len(sde_indices)``, ``seq`` = packed
        image tokens, ``C`` = packed latent channels):

            latents      : [1, K, seq, C]   stored trajectory frames (window + final)
            sde_logp     : [1, S]           old per-step log-probs
            sde_indices  : [S]              SDE step indices
            indices      : [K]              stored frame step indices
            sigmas       : [T+1]            the full schedule

        When ``conditions`` carries >1 image (forward_batch_size pack-B), the loop is
        navit-packed across the B images in :meth:`_diffuse_batched`, returning a
        ``[B, ...]`` segment (same shape :meth:`BagelPipeline._batch_segments` builds
        from B per-sample segments).
        """
        if conditions.batch_size > 1:
            return self._diffuse_batched(
                conditions, schedule=schedule, params=params, initial_latents=initial_latents
            )
        bagel = self.model.model
        device = torch.device(self.model.device)
        schedule = schedule.to(device)
        T = int(schedule.shape[0]) - 1
        require(
            T == int(params.num_inference_steps),
            f"BagelDiffusionStage.diffuse: schedule length {schedule.shape[0]} != "
            f"num_inference_steps+1 ({int(params.num_inference_steps) + 1})",
        )
        sigma_max = schedule[1] if int(schedule.shape[0]) > 1 else schedule[0]

        sde_set: Set[int] = set(int(i) for i in (params.sde_indices or []))
        sde_sorted: List[int] = sorted(sde_set)

        # Rebuild the KV contexts from the conditioning text (the conditions now carry
        # only text + shape, not the opaque caches — see BagelDiffusionConditions).
        gi, forward_kwargs = self._resolve_contexts(conditions, params=params, device=device)

        if initial_latents is not None:
            x_t = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            x_t = gi["packed_init_noises"].to(device=device, dtype=self.trajectory_dtype)

        self.strategy.init_schedule(schedule)

        # Store SDE step boundaries (x_t before AND after each SDE step) so replay can
        # re-score them, plus the final clean latent (T) for VAE decode.
        needed: Set[int] = set(compute_trajectory_positions(sde_set, T))
        needed.add(T)
        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, x_t.detach().clone()))
        sde_logp_list: List[torch.Tensor] = []
        # μ_old per SDE step (the SDE prev_sample_mean). Only GRPO-Guard RatioNorm
        # (BagelFlowUniGRPO ratio_norm=True) consumes it, via segment.sde_means;
        # cheap and stored unconditionally, mirroring how FlowDPPO carries sde_means.
        sde_means_list: List[torch.Tensor] = []

        with torch.no_grad(), self._autocast_ctx(device):
            for i in range(T):
                t_cur = schedule[i]
                t_next = schedule[i + 1]
                cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(float(t_cur.item()), params)
                step_eta = float(params.eta) if i in sde_set else 0.0
                x_t, log_prob, prev_mean = self.step.step_with_logp(
                    bagel,
                    self.strategy,
                    x_t=x_t,
                    prev_sample=None,
                    t_cur=t_cur,
                    t_next=t_next,
                    sigma_max=sigma_max,
                    eta=step_eta,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    forward_kwargs=forward_kwargs,
                )
                x_t = x_t.to(dtype=self.trajectory_dtype)
                if (i + 1) in needed:
                    stored_pairs.append((i + 1, x_t.detach().clone()))
                if log_prob is not None:
                    sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))
                    if prev_mean is not None:
                        sde_means_list.append(prev_mean.detach().to(dtype=self.trajectory_dtype))

        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=0).unsqueeze(0)  # [1, K, seq, C]
        sde_logp = torch.stack(sde_logp_list, dim=0).unsqueeze(0) if sde_logp_list else None  # [1, S]
        # μ_old per SDE step [1, S, seq, C] (None when no SDE steps); RatioNorm reads it.
        sde_means = torch.stack(sde_means_list, dim=0).unsqueeze(0) if sde_means_list else None
        sde_indices = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None

        indices = torch.tensor(positions_collected, dtype=torch.long, device=device)

        return LatentSegment(
            latents=latents_stacked,
            sigmas=schedule,
            indices=indices,
            sde_logp=sde_logp,
            sde_means=sde_means,
            sde_indices=sde_indices,
        )

    def _diffuse_batched(
        self,
        conditions: BagelDiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: BagelDiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Pack ``B = conditions.batch_size`` same-shape images into ONE ``_forward_flow``
        per step (forward_batch_size pack-B). B-wide mirror of :meth:`diffuse`.

        All B images share (H, W), so the packed VAE latents ``[B*seq, C]`` reshape to
        ``[B, seq, C]`` and the kernel's per-sample reduction yields ``[B]`` log-probs
        (:meth:`BagelDiffusionStep.denoise` ``n_samples``); each image attends only to
        its own context (block-diagonal via prepare_vae_latent's packed_seqlens). The
        result is a ``[B, K, seq, C]`` / ``[B, S]`` segment — identical in shape to the
        per-sample loop + :meth:`BagelPipeline._batch_segments`.
        """
        bagel = self.model.model
        device = torch.device(self.model.device)
        schedule = schedule.to(device)
        T = int(schedule.shape[0]) - 1
        require(
            T == int(params.num_inference_steps),
            f"BagelDiffusionStage._diffuse_batched: schedule length {schedule.shape[0]} != "
            f"num_inference_steps+1 ({int(params.num_inference_steps) + 1})",
        )
        sigma_max = schedule[1] if int(schedule.shape[0]) > 1 else schedule[0]
        sde_set: Set[int] = set(int(i) for i in (params.sde_indices or []))
        sde_sorted: List[int] = sorted(sde_set)

        texts = list(conditions.texts)
        image_shapes = [tuple(s) for s in conditions.image_shapes]
        n = len(texts)
        gen, cfg_text, cfg_img = self._build_contexts_batch(texts)
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs(
            gen, cfg_text, cfg_img, image_shapes, device=device
        )
        forward_kwargs = self._forward_kwargs(gen, cfg_text, cfg_img, gi, gi_cfg_text, gi_cfg_img, params)

        if initial_latents is not None:
            x_t = initial_latents.to(device=device, dtype=self.trajectory_dtype).reshape(
                -1, int(initial_latents.shape[-1])
            )
        else:
            x_t = gi["packed_init_noises"].to(device=device, dtype=self.trajectory_dtype)
        c = int(x_t.shape[-1])
        seq = int(x_t.shape[0]) // n  # per-image token count (all B share the shape)

        self.strategy.init_schedule(schedule)
        needed: Set[int] = set(compute_trajectory_positions(sde_set, T))
        needed.add(T)
        stored_pairs: List[Tuple[int, torch.Tensor]] = []  # each frame [B, seq, C]
        if 0 in needed:
            stored_pairs.append((0, x_t.detach().clone().reshape(n, seq, c)))
        sde_logp_list: List[torch.Tensor] = []  # each [B]
        sde_means_list: List[torch.Tensor] = []  # each [B, seq, C]

        with torch.no_grad(), self._autocast_ctx(device):
            for i in range(T):
                t_cur = schedule[i]
                t_next = schedule[i + 1]
                cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(float(t_cur.item()), params)
                step_eta = float(params.eta) if i in sde_set else 0.0
                x_t, log_prob, prev_mean = self.step.step_with_logp(
                    bagel,
                    self.strategy,
                    x_t=x_t,
                    prev_sample=None,
                    t_cur=t_cur,
                    t_next=t_next,
                    sigma_max=sigma_max,
                    eta=step_eta,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    forward_kwargs=forward_kwargs,
                    n_samples=n,
                )
                x_t = x_t.to(dtype=self.trajectory_dtype)  # [B*seq, C]
                if (i + 1) in needed:
                    stored_pairs.append((i + 1, x_t.detach().clone().reshape(n, seq, c)))
                if log_prob is not None:
                    sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))  # [B]
                    if prev_mean is not None:
                        sde_means_list.append(prev_mean.detach().to(dtype=self.trajectory_dtype))  # [B, seq, C]

        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)  # [B, K, seq, C]
        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None  # [B, S]
        sde_means = torch.stack(sde_means_list, dim=1) if sde_means_list else None  # [B, S, seq, C]
        sde_indices = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None
        indices = torch.tensor(positions_collected, dtype=torch.long, device=device)

        return LatentSegment(
            latents=latents_stacked,
            sigmas=schedule,
            indices=indices,
            sde_logp=sde_logp,
            sde_means=sde_means,
            sde_indices=sde_indices,
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        conditions: BagelDiffusionConditions,
        *,
        segment: LatentSegment,
        params: BagelDiffusionParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Recompute per-step log-probs over the SDE window (mirrors SD3 replay).

        Loops ``step.step_with_logp`` (``prev_sample`` = the stored next frame) over
        ``segment.sde_indices`` (or the ``step_indices`` subset). Returns a
        :class:`ReplayResult` with ``log_probs [1, S']`` aligned with
        ``segment.sde_logp`` plus ``prev_sample_means [1, S', seq, C]`` for KL.

        Caller owns ``.train()`` mode + grad scope; this method manages only the
        autocast scope (mirrors ``SD3DiffusionStage.replay``).
        """
        if segment.sde_indices is None or segment.latents is None or segment.sigmas is None:
            raise ValueError("BagelDiffusionStage.replay: segment.sde_indices / latents / sigmas missing")

        bagel = self.model.model
        device = torch.device(self.model.device)
        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = [int(i) for i in step_indices] if step_indices is not None else sorted(sde_set)
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"BagelDiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        schedule = segment.sigmas.to(device)
        sigma_max = schedule[1] if int(schedule.shape[0]) > 1 else schedule[0]

        # Rebuild the KV contexts from text once for the whole replay (deterministic →
        # matches the rollout-time build, so the recomputed log-probs align, ratio ≈ 1).
        _, forward_kwargs = self._resolve_contexts(conditions, params=params, device=device)

        log_probs: List[torch.Tensor] = []
        prev_sample_means: List[torch.Tensor] = []
        with self._autocast_ctx(device):
            for step_idx in target:
                t_cur = schedule[step_idx]
                t_next = schedule[step_idx + 1]
                cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(float(t_cur.item()), params)
                x_t = segment.latents_at(step_idx)[0].to(device)  # [seq, C]
                prev_sample = segment.latents_at(step_idx + 1)[0].to(device)
                _, log_prob, prev_mean = self.step.step_with_logp(
                    bagel,
                    self.strategy,
                    x_t=x_t,
                    prev_sample=prev_sample,
                    t_cur=t_cur,
                    t_next=t_next,
                    sigma_max=sigma_max,
                    eta=float(params.eta),
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    forward_kwargs=forward_kwargs,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"BagelDiffusionStage.replay: strategy returned None log-prob at step={step_idx} "
                        f"(deterministic mode); replay requires a stochastic SDE strategy (eta>0)."
                    )
                log_probs.append(log_prob)
                prev_sample_means.append(prev_mean)

        log_probs_t = torch.stack(log_probs, dim=0).unsqueeze(0).to(dtype=self.logprob_dtype)  # [1, S']
        means_t = torch.stack(prev_sample_means, dim=0).unsqueeze(0).to(dtype=self.trajectory_dtype)  # [1, S', seq, C]
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    # ------------------------------------------------------------------
    # Single-step velocity (forward-process algorithms: DiffusionNFT et al.)
    # ------------------------------------------------------------------

    def build_forward_kwargs(
        self,
        conditions: BagelDiffusionConditions,
        *,
        params: BagelDiffusionParams,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Rebuild the KV contexts from text → the step-invariant ``_forward_flow`` kwargs.

        The expensive part (the und-path KV prefill) done ONCE; a caller scoring
        several steps against the same conditioning then reuses the result via
        :meth:`predict_velocity_at` (no per-step rebuild). Used by
        :class:`~unirl.algorithms.bagel_flow_unigrpo.BagelFlowUniGRPO`'s velocity-MSE
        loop, which evaluates ``v_theta`` / ``v_ref`` across the SDE window.
        """
        _, forward_kwargs = self._resolve_contexts(conditions, params=params, device=device)
        return forward_kwargs

    def predict_velocity_at(
        self,
        forward_kwargs: Dict[str, Any],
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: BagelDiffusionParams,
    ) -> torch.Tensor:
        """Single ``(x_t, sigma)`` CFG velocity reusing prebuilt ``forward_kwargs``.

        Pairs with :meth:`build_forward_kwargs` so a caller scoring multiple steps
        against one conditioning pays the und-path prefill once. ``sample`` is packed
        ``[seq, C]`` (or ``[1, seq, C]``; the unit batch dim is squeezed). Grad flows
        through the velocity forward (gen experts) only — the contexts inside
        ``forward_kwargs`` are detached constants built under ``no_grad``.
        """
        device = torch.device(self.model.device)
        t_val = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(t_val, params)
        sample = sample.to(device)
        if sample.dim() == 3:  # [1, seq, C] → [seq, C] (navit bs=1)
            sample = sample[0]
        with self._autocast_ctx(device):
            return self.step.predict_velocity(
                self.model.model,
                x_t=sample,
                t_cur=sigma,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                forward_kwargs=forward_kwargs,
            )

    def predict_noise_at_step(
        self,
        conditions: BagelDiffusionConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: BagelDiffusionParams,
    ) -> torch.Tensor:
        """Single ``(x_t, sigma)`` velocity forward — no scheduler iteration.

        Completes the ``DiffusionStage`` protocol (used by forward-process algorithms
        like DiffusionNFT). Rebuilds the KV contexts from the conditions' text
        (:meth:`build_forward_kwargs`) then runs the same navit ``_forward_flow`` call
        (:meth:`predict_velocity_at`) ``diffuse`` / ``replay`` use, so CFG handling is
        identical. ``sample`` is packed ``[seq, C]`` (or ``[1, seq, C]``). Callers that
        score multiple steps (e.g. the velocity-MSE loop) should instead build the
        kwargs once via :meth:`build_forward_kwargs` and loop :meth:`predict_velocity_at`.
        """
        device = torch.device(self.model.device)
        forward_kwargs = self.build_forward_kwargs(conditions, params=params, device=device)
        return self.predict_velocity_at(forward_kwargs, sample=sample, sigma=sigma, params=params)

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """The MoT transformer (``bundle.transformer`` == ``model.language_model``).

        This is the FSDP wrap target / LoRA injection root — the same module the
        vendored ``_forward_flow`` runs on, so sharding it shards the gen forward.
        """
        return self.model.transformer


__all__ = [
    "BagelDiffusionParams",
    "BagelDiffusionStage",
    "BagelDiffusionStep",
]
