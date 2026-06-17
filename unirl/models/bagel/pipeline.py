"""BagelPipeline — RolloutReq → RolloutResp end-to-end for BAGEL-7B-MoT (T2I).

Four-tier flow, per-sample (navit ``bs=1``)::

    Texts ─▶ BagelDiffusionConditions(text) ─diffuse─▶ LatentSegment ─vae_decode─▶ Images

Per prompt the pipeline stores only the conditioning **text** on
:class:`BagelDiffusionConditions`; the diffusion stage rebuilds the three KV-cache
contexts the sampler needs from that text (``BagelDiffusionStage._build_contexts``,
mirroring ``InterleaveInferencer.interleave_inference`` for plain T2I, ``think=False``,
no input image):

- ``gen``      = init + text(prompt)          (conditional)
- ``cfg_text`` = init snapshot before the text (unconditional / text-CFG)
- ``cfg_img``  = init + text(prompt)          (== gen for pure T2I)

It runs ``diffusion.diffuse`` once per sample, accumulates the per-sample latents
into one batched ``LatentSegment``, decodes them, and packs one ``"image"`` track.
Keeping the (large, opaque) KV caches out of the conditions — rebuilt on the worker
instead — is what keeps them off the DP round-trip / driver cuda:0.

Central-runtime contract (same as SD3 — NOT a flow_grpo port):

- **σ schedule**: the hosting engine pins ``req.sigmas`` via
  :func:`unirl.sde.runtime.ensure_req_sigmas` (built from :meth:`build_schedule_policy`)
  BEFORE ``generate``; this pipeline reads it verbatim and passes it to ``diffuse``.
- **initial noise x_T**: driver-authored via :class:`NoiseRecipe` (per-sample,
  ``r{rollout_id}:{sample_id}``-keyed, byte-identical across engines). The pipeline
  resolves it from the request and hands each sample its slice; :meth:`latent_shape`
  declares the packed ``(seq, C)`` geometry so the driver can author the recipe.
- **SDE steps**: ``params.sde_indices`` (driver-resolved via
  ``resolve_sde_indices`` → ``AllSDEScheduler``); shared per rollout across the group.

``BagelBundle`` is imported lazily (it pulls the vendored modeling + flash_attn);
this keeps ``BagelPipeline`` importable on CPU for fake-stage tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack, _track_with_field
from unirl.types.sampling import get_ar_params, get_diffusion_params
from unirl.types.segments.latent import LatentSegment

from .ar import BagelARStage
from .conditions import BagelARConditions, BagelDiffusionConditions
from .diffusion import BagelDiffusionParams, BagelDiffusionStage
from .vae import BagelVAEDecodeStage, bagel_latent_shape

if TYPE_CHECKING:
    from .bundle import BagelBundle


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a DictConfig / dict / dataclass, falling back to ``default``."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            val = cfg.get(key, default)
            return default if val is None else val
        except Exception:
            return default
    return getattr(cfg, key, default)


class BagelPipeline(Pipeline):
    """BAGEL-7B-MoT T2I generate pipeline (trainside A1)."""

    def __init__(
        self,
        *,
        bundle: "BagelBundle",
        diffusion: Optional[BagelDiffusionStage] = None,
        vae_decode: Optional[BagelVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp32",
        logprob_precision: str = "fp32",
        shift: float = 3.0,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        if diffusion is None:
            diffusion = BagelDiffusionStage(
                model=bundle,
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else BagelVAEDecodeStage(bundle)
        # Navit pack size for the P*N fan-out (AR chains / diffusion images / VAE
        # decode), set by the hosting engine via set_forward_batch_size. None or 1 =
        # the legacy per-sample (navit bs=1) path; >1 opts into packed batching.
        self.forward_batch_size: Optional[int] = None
        self.autocast_precision = autocast_precision
        # FlowMatch time-shift for the σ schedule policy (read by the hosting engine
        # via build_schedule_policy → ensure_req_sigmas). Bagel uses static shift.
        self.shift = shift

    def set_forward_batch_size(self, forward_batch_size: Optional[int]) -> None:
        """Receive the engine's ``rollout.forward_batch_size`` as ONE knob for the
        P*N fan-out pack size (AR chains / diffusion images / VAE decode).

        Only activates packed batching when ``> 1``; ``None`` / ``1`` keeps the legacy
        per-sample (navit bs=1) path AND the VAE's own default decode chunk, so it is
        non-breaking. Larger values also raise the VAE decode chunk — note the VAE runs
        fp32 upsample-convs, so very large values can OOM the decode (prior default 4).
        """
        if forward_batch_size is None or int(forward_batch_size) <= 1:
            return
        self.forward_batch_size = int(forward_batch_size)
        self.vae_decode.decode_batch_size = int(forward_batch_size)
        # BagelUniPipeline also fans out N thinking chains on its AR stage; hand it the
        # same pack size. The base T2I pipeline has no .ar (getattr → None → skipped).
        ar = getattr(self, "ar", None)
        if ar is not None:
            ar.forward_batch_size = int(forward_batch_size)

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]:
        """Packed per-sample x_T shape ``(seq, p²·z)`` for the driver NoiseRecipe.

        Bagel's x_T is packed navit ``[h·w, p²·z]`` (the ``packed_init_noises`` shape),
        NOT spatial ``[C, H, W]``; ``seq = (H // (vae_downsample·patch))²`` for a square
        image. Returning a concrete shape (rather than raising) opts Bagel into the
        driver-authored, cross-engine-reproducible x_T recipe (same as SD3).
        """
        cfg = _cfg_get(model_config, "config", model_config)
        patch = int(_cfg_get(cfg, "latent_patch_size", 2))
        vae_ds = int(_cfg_get(cfg, "vae_downsample", 8))
        z = int(_cfg_get(cfg, "latent_channels", 16))
        H, W = int(sampling_spec.height), int(sampling_spec.width)
        return bagel_latent_shape((H, W), latent_downsample=vae_ds * patch, latent_patch_size=patch, latent_channels=z)

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Static-shift FlowMatch σ policy (BAGEL uses no dynamic shifting).

        The hosting engine calls this once and pins ``req.sigmas`` via
        ``ensure_req_sigmas``; ``get_sigma_schedule(num_inference_steps, shift)`` then
        produces the byte-identical schedule the (former) ``bagel_timesteps`` did.
        """
        return FlowMatchSchedulePolicy.static_only(float(self.shift))

    @classmethod
    def from_config(cls, config: Any, *, strategy: Optional[StepStrategy] = None) -> "BagelPipeline":
        """Build the full pipeline from a :class:`BagelPipelineConfig`."""
        from .bundle import BagelBundle

        bundle = BagelBundle.from_config(config)
        return cls(
            bundle=bundle,
            strategy=strategy,
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            shift=float(config.shift),
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run BAGEL T2I per-sample and pack one ``"image"`` track."""
        if req.sigmas is None:
            raise ValueError(
                "BagelPipeline.generate: req.sigmas is None. The hosting engine must call "
                "unirl.sde.runtime.ensure_req_sigmas(req, policy) before generate "
                "(policy = pipeline.build_schedule_policy())."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"BagelPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        params = get_diffusion_params(req.sampling_params)
        if not isinstance(params, BagelDiffusionParams):
            raise TypeError(
                f"BagelPipeline.generate: sampling_params must be BagelDiffusionParams, got {type(params).__name__}"
            )

        prompts = list(texts.texts)
        n = len(prompts)
        sample_ids = list(req.sample_ids) if req.sample_ids else [f"s{i}" for i in range(n)]
        image_shape = (int(params.height), int(params.width))
        device = torch.device(self.bundle.device)
        schedule = req.sigmas.to(device)

        # Driver-authoritative per-sample x_T (NoiseRecipe), [n, seq, C] or None
        # (engine draws its own — tests / no driver recipe). Per-sample-unique +
        # per-rollout-fresh via the driver's r{rollout_id}:{sample_id} group ids.
        initial = NoiseRecipe.from_rollout_req(req).resolve(device=device, dtype=torch.float32)

        shapes: List[Tuple[int, int]] = []
        segments: List[LatentSegment] = []
        for i, prompt in enumerate(prompts):
            # Conditions carry only the text; the stage rebuilds the KV contexts from
            # it inside diffuse/replay (keeps the opaque caches off the track/driver).
            cond_i = BagelDiffusionConditions.for_sample(text=prompt, image_shape=image_shape)
            x0_i = initial[i] if initial is not None else None
            seg_i = self.diffusion.diffuse(cond_i, schedule=schedule, params=params, initial_latents=x0_i)
            segments.append(seg_i)
            shapes.append(image_shape)

        segment = self._batch_segments(segments)
        conditions = BagelDiffusionConditions(texts=list(prompts), image_shapes=shapes)
        images = self.vae_decode.decode(segment, image_shape=image_shape)

        return RolloutResp(
            tracks={
                "image": RolloutTrack(
                    sample_ids=sample_ids,
                    parent_ids=list(req.group_ids) if req.group_ids else None,
                    conditions=conditions.to_dict(),
                    segment=segment,
                    decoded=images,
                ),
            }
        )

    @staticmethod
    def _batch_segments(segments: List[LatentSegment]) -> LatentSegment:
        """Stack per-sample 1-row segments into one ``[N, ...]`` segment.

        With the per-rollout SDE window (``resolve_sde_indices(rollout_id)`` is shared
        across the group), every sample's ``sde_indices`` / ``indices`` / ``sigmas`` is
        identical, so the shared fields are taken from ``segments[0]`` and only the
        per-sample ``latents`` / ``sde_logp`` stack along the batch axis. Segment rows
        are 1:1 with track samples by construction (no explicit row→sample mapping —
        UniRL's Segment dropped ``sample_indices`` / ``positions``).
        """
        if len(segments) == 1:
            return segments[0]
        latents = torch.cat([s.latents for s in segments], dim=0)  # [N, K, seq, C]
        sde_logp = (
            torch.cat([s.sde_logp for s in segments], dim=0) if segments[0].sde_logp is not None else None
        )  # [N, S]
        # μ_old per SDE step [N, S, seq, C] — required by BagelFlowUniGRPO(ratio_norm=True).
        # Stack it the same way as sde_logp; omitting it leaves the batched segment's
        # sde_means=None, which makes RatioNorm raise at replay (the per-sample segments
        # from diffuse DO carry it, only this merge dropped it).
        sde_means = (
            torch.cat([s.sde_means for s in segments], dim=0) if segments[0].sde_means is not None else None
        )  # [N, S, seq, C]
        return LatentSegment(
            latents=latents,
            sigmas=segments[0].sigmas,
            indices=segments[0].indices,
            sde_logp=sde_logp,
            sde_means=sde_means,
            sde_indices=segments[0].sde_indices,
        )


# Default planning/"think" system prompt — copied as a plain string from the
# vendored ``InterleaveInferencer.GEN_THINK_SYSTEM_PROMPT`` so this module stays
# flash-free. Recipes override via the ``think_system_prompt`` pipeline kwarg.
_DEFAULT_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then "
    "generate the image. The planning process is enclosed within <think> </think> "
    "tags, i.e. <think> planning process here </think> image here"
)


class BagelUniPipeline(BagelPipeline):
    """BAGEL unified reasoning->image pipeline (UniGRPO).

    One shared ``BagelBundle`` drives both halves on the SAME MoT transformer:
    the AR ``.ar`` stage (und/text experts) generates the reasoning ("thinking")
    text, then the ``.diffusion`` stage (gen/image experts) renders an image
    conditioned on prompt + thinking. ``generate`` fans out ``P -> P*N -> P*N*M``
    (N = ``ar.samples_per_prompt`` thinking chains, M =
    ``diffusion.samples_per_prompt`` images/chain; UniGRPO uses M=1) and returns
    a 2-track ``{"ar", "image"}`` RolloutResp with explicit lineage, so AR-GRPO
    groups by prompt and the image advantage is shared per chain (M=1).

    Subclasses :class:`BagelPipeline` to reuse the diffusion machinery
    (``diffuse`` / ``_batch_segments`` / VAE decode / ``build_schedule_policy`` /
    ``latent_shape``); only the AR phase, the fan-out/lineage, and conditioning the
    image on the thinking text are added. The image conditioning text
    (``{prompt}\n{thinking}``) is stored on the conditions and the KV contexts are
    rebuilt from it in the stage (see :class:`BagelDiffusionConditions`).
    """

    def __init__(
        self,
        *,
        bundle: "BagelBundle",
        ar: Optional[BagelARStage] = None,
        diffusion: Optional[BagelDiffusionStage] = None,
        vae_decode: Optional[BagelVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp32",
        logprob_precision: str = "fp32",
        shift: float = 3.0,
        think_system_prompt: Optional[str] = _DEFAULT_THINK_SYSTEM_PROMPT,
    ) -> None:
        super().__init__(
            bundle=bundle,
            diffusion=diffusion,
            vae_decode=vae_decode,
            strategy=strategy,
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
            shift=shift,
        )
        self.ar = (
            ar
            if ar is not None
            else BagelARStage(
                model=bundle,
                autocast_precision=autocast_precision,
                logprob_precision=logprob_precision,
            )
        )
        self.think_system_prompt = think_system_prompt

    @classmethod
    def from_config(cls, config: Any, *, strategy: Optional[StepStrategy] = None) -> "BagelUniPipeline":
        """Build the full unified pipeline (bundle + ar + diffusion) from a config."""
        from .bundle import BagelBundle

        bundle = BagelBundle.from_config(config)
        return cls(
            bundle=bundle,
            strategy=strategy,
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            shift=float(config.shift),
        )

    def _think_prompt(self, user_prompt: str) -> str:
        """Full prompt the AR stage conditions on (system framing + user prompt)."""
        sp = self.think_system_prompt
        return f"{sp}\n\n{user_prompt}" if sp else user_prompt

    @staticmethod
    def _image_prompt(user_prompt: str, thinking: str) -> str:
        """Text the image is conditioned on: the user prompt plus the thinking plan."""
        t = (thinking or "").strip()
        return f"{user_prompt}\n{t}" if t else user_prompt

    def _decode_thinking(self, segment: Any) -> Texts:
        """Detokenize the AR ``TextSegment`` (varlen-packed) into per-sample texts."""
        tok = self.bundle.tokenizer
        if segment.tokens is None or segment.cu_seqlens is None:
            return Texts(texts=[])
        # BAGEL's lm_head vocab (152064) is padded above the tokenizer's real id
        # range (len(tok) = 151665): ids 151665..152063 are alignment padding slots
        # with NO token. ``generate_text`` samples over the full logit width, so a
        # long thinking chain can draw one of these padding ids; the slow tokenizer's
        # ``_convert_id_to_token`` returns None for them and ``convert_tokens_to_string``
        # then trips on ``"".join`` ("expected str, got NoneType"). Drop out-of-range
        # ids before decode — they carry no text. (Real special tokens < len(tok) are
        # still handled by skip_special_tokens.)
        vocab_n = len(tok)
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        out: List[str] = []
        for k in range(len(cu) - 1):
            ids = [i for i in segment.tokens[cu[k] : cu[k + 1]].tolist() if 0 <= i < vocab_n]
            out.append(tok.decode(ids, skip_special_tokens=True) if ids else "")
        return Texts(texts=out)

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run reasoning->image and pack a 2-track ``{"ar", "image"}`` resp."""
        if req.sigmas is None:
            raise ValueError(
                "BagelUniPipeline.generate: req.sigmas is None. The hosting engine must pin "
                "it via unirl.sde.runtime.ensure_req_sigmas(req, policy) before generate."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"BagelUniPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        ar_params = get_ar_params(req.sampling_params)
        if ar_params is None:
            raise TypeError("BagelUniPipeline.generate: sampling_params must carry an 'ar' (ARSamplingParams) block.")
        params = get_diffusion_params(req.sampling_params)
        if not isinstance(params, BagelDiffusionParams):
            raise TypeError(
                f"BagelUniPipeline.generate: diffusion sampling_params must be BagelDiffusionParams, "
                f"got {type(params).__name__}"
            )

        prompts = list(texts.texts)
        n_rewrites = int(ar_params.samples_per_prompt)
        n_images = int(params.samples_per_prompt)

        # Level 1: P -> P*N thinking chains (root "ar" track, groups by prompt).
        ar_shell = req.make_root_track(track_name="ar", branch=n_rewrites)
        think_prompts = [self._think_prompt(p) for p in prompts for _ in range(n_rewrites)]
        ar_conditions = BagelARConditions(prompts=think_prompts)
        ar_segment = self.ar.autoregress(ar_conditions, sampling_params=ar_params)
        thinking = self._decode_thinking(ar_segment)
        if len(thinking.texts) != len(ar_shell.sample_ids):
            raise RuntimeError(
                f"BagelUniPipeline.generate: AR produced {len(thinking.texts)} thinking text(s) "
                f"but the AR track expects {len(ar_shell.sample_ids)} (= P*N)."
            )
        ar_track = _track_with_field(ar_shell, "segment", ar_segment)
        ar_track = _track_with_field(ar_track, "decoded", thinking)
        ar_track = _track_with_field(ar_track, "conditions", ar_conditions.to_dict())

        # Level 2: P*N -> P*N*M images (fork "image" from "ar"; UniGRPO M=1).
        img_shell = ar_track.fork_track(parent_name="ar", child_name="image", branch=n_images)
        n_ar = len(ar_shell.sample_ids)
        image_texts = [
            self._image_prompt(prompts[i // n_rewrites], thinking.texts[i])
            for i in range(n_ar)
            for _ in range(n_images)
        ]

        device = torch.device(self.bundle.device)
        image_shape = (int(params.height), int(params.width))
        schedule = req.sigmas.to(device)

        shapes: List[Tuple[int, int]] = []
        segments: List[LatentSegment] = []
        # Conditions carry only the conditioning text ({prompt}\n{thinking}); the stage
        # rebuilds the KV contexts from it inside diffuse/replay, keeping the opaque
        # caches off the track/driver (see BagelDiffusionConditions). Each image draws
        # its own x_T inside diffuse (initial_latents=None).
        fbs = self.forward_batch_size
        if fbs is None or fbs <= 1:
            for image_text in image_texts:  # legacy navit bs=1 (untouched)
                cond_i = BagelDiffusionConditions.for_sample(text=image_text, image_shape=image_shape)
                segments.append(self.diffusion.diffuse(cond_i, schedule=schedule, params=params, initial_latents=None))
                shapes.append(image_shape)
        else:
            # forward_batch_size pack-B: navit-pack up to fbs same-shape images per
            # diffuse call (one _forward_flow per step over the chunk). _batch_segments
            # cats the [B, ...] chunk segments back into the [P*N, ...] image track.
            for start in range(0, len(image_texts), fbs):
                chunk = image_texts[start : start + fbs]
                cond_b = BagelDiffusionConditions(texts=list(chunk), image_shapes=[image_shape] * len(chunk))
                segments.append(self.diffusion.diffuse(cond_b, schedule=schedule, params=params, initial_latents=None))
                shapes.extend([image_shape] * len(chunk))

        segment = self._batch_segments(segments)
        img_conditions = BagelDiffusionConditions(texts=list(image_texts), image_shapes=shapes)
        images = self.vae_decode.decode(segment, image_shape=image_shape)

        image_track = _track_with_field(img_shell, "segment", segment)
        image_track = _track_with_field(image_track, "decoded", images)
        image_track = _track_with_field(image_track, "conditions", img_conditions.to_dict())

        return RolloutResp(tracks={"ar": ar_track, "image": image_track})


__all__ = ["BagelPipeline", "BagelUniPipeline"]
